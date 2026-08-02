"""
RL TRAINING WITH WORK ZONE (GHOST VEHICLE) + PARETO SET SELECTION
================================================================

Adds:
✅ Random weight vectors for multi-objective reward terms
✅ Train one policy per weight vector
✅ Evaluate each policy on raw objectives
✅ Compute Pareto-optimal (non-dominated) policies using paretoset
✅ Save CSV + plots for Pareto front

Notes:
- Pareto is computed on *objectives* (waiting, queue, speed, etc.), NOT on reward.
- Reward is just a scalarization to train.
"""

import os
import sys
import csv
from datetime import datetime
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

import pandas as pd
from paretoset import paretoset
import matplotlib.pyplot as plt
from matplotlib import gridspec

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")


# ============================================================
# WORK ZONE CONFIGURATION
# ============================================================

GHOST_CONFIG = {
    "edge": "E#9",
    "lane": 1,
    "pos": 320.0,
    "length": 60.0,
    "duration": 999999.0,
    "queue_start": 200.0,
    "merge_start": 170.0,
    "force_start": 20.0,
    "queue_speed": 1.0,
    "merge_speed": 2.5,
    "downstream_speed": 1.7,
    "downstream_length": 90.0,
}


# ============================================================
# QUEUE TAIL DETECTION
# ============================================================

def get_queue_tail_pos(sumo, edge_id, lane_index, rear_pos, speed_thresh=1.0):
    lane_id = f"{edge_id}_{lane_index}"
    try:
        vids = sumo.lane.getLastStepVehicleIDs(lane_id)
    except:
        return None

    tail_pos = None
    for vid in vids:
        if vid == "ghost":
            continue
        try:
            p = sumo.vehicle.getLanePosition(vid)
            v = sumo.vehicle.getSpeed(vid)
            if p < rear_pos and v < speed_thresh:
                if tail_pos is None or p < tail_pos:
                    tail_pos = p
        except:
            pass

    return tail_pos


# ============================================================
# GLOBAL WEIGHTS (updated per run)
# ============================================================

# Positive weights for each term:
# waiting, queue, speed, pressure, ttc, queue_length
CURRENT_W = np.array([0.35, 0.20, 1.60, 0.10, 0.40, 0.15], dtype=float)


def set_current_weights(w: np.ndarray):
    global CURRENT_W
    CURRENT_W = np.array(w, dtype=float)


# ============================================================
# REWARD (scalarization of multiple objectives)
# ============================================================

def my_reward_fn_with_ttc_and_queue(traffic_signal):
    """
    Returns scalar reward using CURRENT_W, but we also want the underlying
    objective metrics for Pareto evaluation (done in separate eval loop).
    """

    # Raw metrics from SUMO-RL TrafficSignal
    waiting = sum(traffic_signal.get_accumulated_waiting_time_per_lane())
    queue = traffic_signal.get_total_queued()
    speed = traffic_signal.get_average_speed()
    pressure = traffic_signal.get_pressure()

    # TTC conflicts from SSM device
    ttc_conflicts = 0
    try:
        sumo = traffic_signal.sumo
        vehicle_ids = sumo.vehicle.getIDList()
        for veh_id in vehicle_ids:
            try:
                min_ttc = sumo.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                if min_ttc not in ("", "NA"):
                    if float(min_ttc) < 1.5:
                        ttc_conflicts += 1
            except:
                continue
    except:
        ttc_conflicts = 0

    # Work zone queue length
    queue_length = 0.0
    try:
        sumo = traffic_signal.sumo
        rear_pos = GHOST_CONFIG["pos"] - GHOST_CONFIG["length"]
        tail_pos = get_queue_tail_pos(sumo, GHOST_CONFIG["edge"], GHOST_CONFIG["lane"], rear_pos)
        if tail_pos is not None:
            queue_length = rear_pos - tail_pos
    except:
        queue_length = 0.0

    # Normalization (same as your design)
    normalized_waiting = waiting / 100.0
    normalized_queue = min(queue / 50.0, 1.0)
    normalized_pressure = max(min(pressure / 20.0, 1.0), -1.0)
    normalized_ttc = min(ttc_conflicts / 10.0, 1.0)
    normalized_queue_length = min(queue_length / 200.0, 1.0)

    # Scalarized reward with CURRENT_W
    # (penalties are negative, speed is bonus)
    w_wait, w_queue, w_speed, w_press, w_ttc, w_qlen = CURRENT_W

    reward = (
        -w_wait * normalized_waiting
        -w_queue * normalized_queue
        +w_speed * speed
        -w_press * abs(normalized_pressure)   # pressure magnitude penalty is usually safer
        -w_ttc * normalized_ttc
        -w_qlen * normalized_queue_length
    )

    return float(reward)


# ============================================================
# ENVIRONMENT (your WorkZoneSumoEnv unchanged except we keep it)
# ============================================================

class WorkZoneSumoEnv(SumoEnvironment):
    def __init__(self,
                 state_file='saved_state.xml',
                 warmup_steps=120,
                 restart_every=50,
                 enable_data_collection=True,
                 csv_output_dir='./training_data/',
                 **kwargs):

        self.state_file = state_file
        self.warmup_steps = warmup_steps
        self.restart_every = restart_every
        self.enable_data_collection = enable_data_collection
        self.csv_output_dir = csv_output_dir

        self._warmed_up = False
        self._reset_count = 0
        self.crash_count = 0
        self._ghost_created = False

        self.csv_file = None
        self.csv_writer = None
        self._csv_closed = False

        self.slowed_for_queue = set()
        self.issued_merge_cmd = set()
        self.pinned_after_merge = set()
        self.slowed_downstream = set()

        super().__init__(**kwargs)

    def _setup_csv_logging(self):
        if not self.enable_data_collection:
            return

        os.makedirs(self.csv_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(self.csv_output_dir, f"trajectories_{timestamp}.csv")

        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'episode', 'time_s', 'vehicle_id', 'position_m', 'speed_mps', 'lane',
            'tail_pos_m', 'rear_pos_m', 'queue_length_m', 'ttc_conflicts'
        ])
        print(f"📊 CSV logging enabled: {csv_path}")

    def _create_ghost_vehicle(self):
        if self._ghost_created:
            return
        try:
            sumo = self.sumo
            if "ghost" in sumo.vehicle.getIDList():
                self._ghost_created = True
                return

            if "ghost" not in sumo.vehicletype.getIDList():
                sumo.vehicletype.copy("DEFAULT_VEHTYPE", "ghost")
                sumo.vehicletype.setMaxSpeed("ghost", 0.01)
                sumo.vehicletype.setColor("ghost", (0, 0, 0, 255))
                sumo.vehicletype.setLength("ghost", GHOST_CONFIG["length"])
                sumo.vehicletype.setWidth("ghost", 2.6)
                sumo.vehicletype.setMinGap("ghost", 0.5)

            if "ghostRoute" not in sumo.route.getIDList():
                sumo.route.add("ghostRoute", [GHOST_CONFIG["edge"]])

            sumo.vehicle.add(
                vehID="ghost",
                routeID="ghostRoute",
                typeID="ghost",
                depart=0,
                departLane=GHOST_CONFIG["lane"],
                departPos=str(GHOST_CONFIG["pos"]),
                departSpeed="0"
            )

            sumo.vehicle.setStop(
                "ghost",
                edgeID=GHOST_CONFIG["edge"],
                pos=GHOST_CONFIG["pos"],
                laneIndex=GHOST_CONFIG["lane"],
                duration=GHOST_CONFIG["duration"]
            )

            self._ghost_created = True
            print(f"🚧 Ghost vehicle created at {GHOST_CONFIG['edge']} lane {GHOST_CONFIG['lane']}, pos {GHOST_CONFIG['pos']}m")
        except Exception as e:
            print(f"⚠️ Could not create ghost vehicle: {e}")

    def _ensure_ghost_exists(self):
        try:
            if "ghost" not in self.sumo.vehicle.getIDList():
                self._ghost_created = False
                self._create_ghost_vehicle()
            else:
                self._ghost_created = True
        except:
            self._ghost_created = False
            self._create_ghost_vehicle()

    def _control_work_zone_behavior(self):
        try:
            sumo = self.sumo
            edge_id = GHOST_CONFIG["edge"]
            ghost_pos = GHOST_CONFIG["pos"]

            vehicles = sumo.edge.getLastStepVehicleIDs(edge_id)
            for vid in vehicles:
                if vid == "ghost":
                    continue

                try:
                    lane = sumo.vehicle.getLaneIndex(vid)
                    pos = sumo.vehicle.getLanePosition(vid)
                    dist = ghost_pos - pos

                    # Keep vehicles on lane 1 after passing
                    if lane == 1 and pos >= ghost_pos and vid not in self.pinned_after_merge:
                        try:
                            sumo.vehicle.changeLane(vid, 1, 99999)
                            self.pinned_after_merge.add(vid)
                        except:
                            pass

                    # Downstream congestion on lane 1
                    if lane == 1 and ghost_pos <= pos <= ghost_pos + GHOST_CONFIG["downstream_length"]:
                        if vid not in self.slowed_downstream:
                            try:
                                sumo.vehicle.slowDown(vid, GHOST_CONFIG["downstream_speed"], 3.0)
                                self.slowed_downstream.add(vid)
                            except:
                                pass

                    if dist <= 0:
                        continue

                    # Queue formation on lane 1
                    if lane == GHOST_CONFIG["lane"] and dist <= GHOST_CONFIG["queue_start"] and vid not in self.slowed_for_queue:
                        try:
                            sumo.vehicle.setLaneChangeMode(vid, 0)
                            sumo.vehicle.slowDown(vid, GHOST_CONFIG["queue_speed"], 4.0)
                            self.slowed_for_queue.add(vid)
                        except:
                            pass

                    # Smooth merge (as you wrote, though note: changing lane 1->1 does nothing)
                    if lane == GHOST_CONFIG["lane"] and dist <= GHOST_CONFIG["merge_start"] and vid not in self.issued_merge_cmd:
                        try:
                            sumo.vehicle.setLaneChangeMode(vid, 1621)
                            sumo.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", "0.8")
                            sumo.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", "1.0")
                            sumo.vehicle.changeLane(vid, 1, 8.0)
                            sumo.vehicle.slowDown(vid, GHOST_CONFIG["merge_speed"], 3.0)
                            self.issued_merge_cmd.add(vid)
                        except:
                            pass

                    # Panic merge
                    if lane == GHOST_CONFIG["lane"] and dist <= GHOST_CONFIG["force_start"] and vid not in self.issued_merge_cmd:
                        try:
                            sumo.vehicle.changeLane(vid, 1, 3.0)
                            self.issued_merge_cmd.add(vid)
                        except:
                            pass

                except:
                    continue

        except:
            pass

    def _collect_trajectory_data(self):
        if not self.enable_data_collection or self.csv_writer is None:
            return

        try:
            sumo = self.sumo
            sim_time = sumo.simulation.getTime()
            edge_id = GHOST_CONFIG["edge"]
            rear_pos = GHOST_CONFIG["pos"] - GHOST_CONFIG["length"]

            tail_pos = get_queue_tail_pos(sumo, edge_id, GHOST_CONFIG["lane"], rear_pos)
            queue_length = (rear_pos - tail_pos) if tail_pos is not None else 0.0

            # TTC conflicts
            ttc_conflicts = 0
            try:
                for veh_id in sumo.vehicle.getIDList():
                    try:
                        min_ttc = sumo.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                        if min_ttc not in ("", "NA"):
                            if float(min_ttc) < 1.5:
                                ttc_conflicts += 1
                    except:
                        continue
            except:
                pass

            vehicles = sumo.edge.getLastStepVehicleIDs(edge_id)
            for vid in vehicles:
                if vid == "ghost":
                    continue
                try:
                    lane = sumo.vehicle.getLaneIndex(vid)
                    pos = sumo.vehicle.getLanePosition(vid)
                    speed = sumo.vehicle.getSpeed(vid)

                    self.csv_writer.writerow([
                        self._reset_count,
                        f"{sim_time:.2f}",
                        vid,
                        f"{pos:.3f}",
                        f"{speed:.3f}",
                        lane,
                        f"{tail_pos:.3f}" if tail_pos is not None else "",
                        f"{rear_pos:.3f}",
                        f"{queue_length:.3f}",
                        ttc_conflicts
                    ])
                except:
                    pass
        except:
            pass

    def _start_simulation(self):
        super()._start_simulation()
        self._create_ghost_vehicle()
        if self.csv_file is None:
            self._setup_csv_logging()
        if not self._warmed_up and self.sumo is not None:
            self._do_warmup()
            self._warmed_up = True

    def _do_warmup(self):
        print(f"\n{'='*70}")
        print(f"🔥 WARMUP PHASE - Stabilizing Traffic with Work Zone")
        print(f"{'='*70}")
        print(f"Running {self.warmup_steps} steps...")

        for i in range(self.warmup_steps):
            try:
                self._control_work_zone_behavior()
                self.sumo.simulationStep()
                if i % 5 == 0:
                    self._collect_trajectory_data()
            except Exception as e:
                print(f"⚠️ Warmup failed at step {i}: {e}")
                return

        try:
            self.sumo.simulation.saveState(self.state_file)
            print(f"\n✅ State saved with ghost vehicle: {self.state_file}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")

    def reset(self, **kwargs):
        self._reset_count += 1

        self.slowed_for_queue.clear()
        self.issued_merge_cmd.clear()
        self.pinned_after_merge.clear()
        self.slowed_downstream.clear()

        if self._reset_count % self.restart_every == 0:
            print(f"\n🔄 Periodic restart (episode {self._reset_count})")
            self._ghost_created = False
            return self._hard_reset(**kwargs)

        try:
            result = super().reset(**kwargs)
            self._ensure_ghost_exists()
            return result
        except Exception as e:
            print(f"\n⚠️ Reset failed: {e}")
            self.crash_count += 1
            self._ghost_created = False
            return self._hard_reset(**kwargs)

    def _hard_reset(self, **kwargs):
        try:
            if self.sumo is not None:
                try:
                    import traci
                    traci.close()
                except:
                    pass
        except:
            pass

        self._warmed_up = False
        self._ghost_created = False
        self._start_simulation()
        return super().reset(**kwargs)

    def _sumo_reset(self):
        if self.sumo is None:
            self._start_simulation()
            return

        if os.path.exists(self.state_file):
            try:
                self.sumo.simulation.loadState(self.state_file)
                self._ensure_ghost_exists()
                return
            except Exception as e:
                print(f"  (State load failed: {e})")

        super()._sumo_reset()
        self._create_ghost_vehicle()

    def step(self, action):
        try:
            self._control_work_zone_behavior()
            result = super().step(action)
            self._collect_trajectory_data()
            return result
        except Exception as e:
            print(f"\n⚠️ Step crashed: {e}")
            self.crash_count += 1
            obs = self.observation_space.sample()
            return obs, -10.0, True, True, {'crash': True}

    def close(self):
        if self.csv_file is not None and not self._csv_closed:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                self._csv_closed = True
                print(f"📁 CSV file closed successfully")
            except Exception as e:
                print(f"⚠️ Error closing CSV: {e}")
            finally:
                self.csv_file = None
                self.csv_writer = None

        try:
            super().close()
        except:
            pass


# ============================================================
# PARETO: weight generation, eval, selection
# ============================================================

def sample_weight_vectors(n: int, seed: int = 0):
    """
    Create n positive weight vectors for [wait, queue, speed, pressure, ttc, qlen]
    using a Dirichlet distribution. Then we scale them so the speed term tends
    to remain influential (optional).

    You can remove the scaling if you want "pure" Dirichlet.
    """
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(alpha=np.ones(6), size=n)

    # Optional: scale so speed doesn't become tiny too often
    # (keeps training from collapsing into "only penalties")
    W[:, 2] = np.clip(W[:, 2] * 2.0, 0.0, 1.0)
    W = W / W.sum(axis=1, keepdims=True)

    return W


def evaluate_policy_objectives(env: WorkZoneSumoEnv, model: DQN, eval_steps: int = 2000):
    """
    Run a single evaluation rollout and compute average objectives.

    Objectives:
      waiting (min), queue (min), speed (max), |pressure| (min), ttc (min), queue_len (min)

    NOTE: These are computed from TraCI / traffic_signal inside SUMO-RL.
    For a strict evaluation, we use the env's internal traffic signal object.
    """
    obs, info = env.reset()

    # In single_agent=True, env.ts_ids has one ID; traffic signal is env.traffic_signals[ts_id]
    ts_id = env.ts_ids[0]
    ts = env.traffic_signals[ts_id]

    waiting_list = []
    queue_list = []
    speed_list = []
    pressure_list = []
    ttc_list = []
    qlen_list = []

    for _ in range(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # --- pull raw objectives each step ---
        waiting = sum(ts.get_accumulated_waiting_time_per_lane())
        queue = ts.get_total_queued()
        speed = ts.get_average_speed()
        pressure = ts.get_pressure()

        # TTC conflicts
        ttc_conflicts = 0
        try:
            sumo = ts.sumo
            for veh_id in sumo.vehicle.getIDList():
                try:
                    min_ttc = sumo.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                    if min_ttc not in ("", "NA"):
                        if float(min_ttc) < 1.5:
                            ttc_conflicts += 1
                except:
                    continue
        except:
            ttc_conflicts = 0

        # queue length near work zone
        queue_length = 0.0
        try:
            sumo = ts.sumo
            rear_pos = GHOST_CONFIG["pos"] - GHOST_CONFIG["length"]
            tail_pos = get_queue_tail_pos(sumo, GHOST_CONFIG["edge"], GHOST_CONFIG["lane"], rear_pos)
            if tail_pos is not None:
                queue_length = rear_pos - tail_pos
        except:
            queue_length = 0.0

        waiting_list.append(waiting)
        queue_list.append(queue)
        speed_list.append(speed)
        pressure_list.append(abs(pressure))
        ttc_list.append(ttc_conflicts)
        qlen_list.append(queue_length)

        if terminated or truncated:
            break

    # Return averages
    return {
        "waiting": float(np.mean(waiting_list)) if waiting_list else np.inf,
        "queue": float(np.mean(queue_list)) if queue_list else np.inf,
        "speed": float(np.mean(speed_list)) if speed_list else 0.0,
        "pressure_abs": float(np.mean(pressure_list)) if pressure_list else np.inf,
        "ttc_conflicts": float(np.mean(ttc_list)) if ttc_list else np.inf,
        "queue_length": float(np.mean(qlen_list)) if qlen_list else np.inf,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------- SUMO cmd ----------
    SSM_AND_PHYSICS = (
        "--device.ssm.probability 1.0 "
        "--device.ssm.measures TTC "
        "--collision.action remove "
        "--no-step-log "
        "--lateral-resolution 0.8 "
        "--step-length 0.2"
    )

    # --------- Pareto sweep settings ----------
    N_WEIGHT_VECTORS = 20          # how many different weight vectors to try
    TRAIN_TIMESTEPS = 5_000        # per weight vector (increase for better convergence)
    EVAL_STEPS = 2000              # evaluation rollout length

    weights = sample_weight_vectors(N_WEIGHT_VECTORS, seed=42)

    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./training_data/", exist_ok=True)
    os.makedirs("./pareto/", exist_ok=True)

    results = []

    for i, w in enumerate(weights, start=1):
        print("\n" + "=" * 80)
        print(f"🔁 TRAINING RUN {i}/{N_WEIGHT_VECTORS}")
        print(f"   weights [wait, queue, speed, pressure, ttc, qlen] = {np.round(w, 4)}")
        print("=" * 80)

        # set global weights used by reward_fn
        set_current_weights(w)

        env = WorkZoneSumoEnv(
            net_file=os.path.join(_CONFIG_DIR, 'net.net.xml'),
            route_file=os.path.join(_CONFIG_DIR, 'rou.route.xml'),
            use_gui=False,
            single_agent=True,
            num_seconds=120,
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=50,
            reward_fn=my_reward_fn_with_ttc_and_queue,
            time_to_teleport=-1,
            max_depart_delay=-1,
            additional_sumo_cmd=SSM_AND_PHYSICS,
            state_file=f'workzone_state_{i}.xml',
            warmup_steps=120,
            restart_every=50,
            enable_data_collection=True,
            csv_output_dir='./training_data/',
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path='./models/',
            name_prefix=f'dqn_w{i:02d}',
            verbose=1
        )

        model = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate=1e-3,
            learning_starts=0,
            buffer_size=50000,
            batch_size=32,
            train_freq=1,
            target_update_interval=500,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=1,
            tensorboard_log="./logs/",
        )

        # Train
        model.learn(
            total_timesteps=TRAIN_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=10,
            progress_bar=True,
            tb_log_name=f"DQN_WorkZone_W{i:02d}"
        )

        model_path = f"models/dqn_workzone_w{i:02d}.zip"
        model.save(model_path)

        # Evaluate on objectives (Pareto works on objectives)
        obj = evaluate_policy_objectives(env, model, eval_steps=EVAL_STEPS)

        # Save row
        row = {
            "run": i,
            "model_path": model_path,
            "w_wait": w[0],
            "w_queue": w[1],
            "w_speed": w[2],
            "w_pressure": w[3],
            "w_ttc": w[4],
            "w_qlen": w[5],
            **obj
        }
        results.append(row)

        # Cleanup
        env.close()

    # ============================================================
    # PARETO SET COMPUTATION
    # ============================================================

    df = pd.DataFrame(results)

    # Objectives for Pareto:
    # waiting min, queue min, speed max, pressure_abs min, ttc_conflicts min, queue_length min
    obj_cols = ["waiting", "queue", "speed", "pressure_abs", "ttc_conflicts", "queue_length"]
    sense = ["min", "min", "max", "min", "min", "min"]

    mask = paretoset(df[obj_cols], sense=sense)
    df["pareto"] = mask

    # Save all results
    all_csv = "./pareto/all_runs.csv"
    df.to_csv(all_csv, index=False)

    pareto_df = df[df["pareto"]].copy().sort_values(["ttc_conflicts", "waiting"])
    pareto_csv = "./pareto/pareto_front.csv"
    pareto_df.to_csv(pareto_csv, index=False)

    print("\n" + "=" * 80)
    print("✅ PARETO FRONT RESULTS")
    print("=" * 80)
    print(f"All runs saved: {all_csv}")
    print(f"Pareto front saved: {pareto_csv}")
    print("\nPareto-optimal runs:")
    print(pareto_df[["run"] + obj_cols + ["w_wait","w_queue","w_speed","w_pressure","w_ttc","w_qlen"]].to_string(index=False))

    # ============================================================
    # 6-PANEL PARETO GRAPH
    # ============================================================

    fig = plt.figure(figsize=(16, 12), facecolor="#1e1e2e")
    fig.suptitle("Pareto Multi-Objective RL — Work Zone Traffic Control",
                 fontsize=18, color="white", fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.96, top=0.90, bottom=0.07)

    pairs = [
        ("waiting",        "speed",          "Waiting Time (s)",    "Avg Speed (m/s)"),
        ("waiting",        "queue_length",   "Waiting Time (s)",    "Work Zone Queue (m)"),
        ("waiting",        "ttc_conflicts",  "Waiting Time (s)",    "TTC Conflicts"),
        ("queue",          "speed",          "Queue Count",         "Avg Speed (m/s)"),
        ("ttc_conflicts",  "speed",          "TTC Conflicts",       "Avg Speed (m/s)"),
        ("queue_length",   "ttc_conflicts",  "Work Zone Queue (m)", "TTC Conflicts"),
    ]

    for idx, (xkey, ykey, xlabel, ylabel) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#444466")

        mask_off = ~df["pareto"]
        mask_on  =  df["pareto"]

        ax.scatter(df.loc[mask_off, xkey], df.loc[mask_off, ykey],
                   c="#5555aa", s=60, edgecolors="#3333aa", linewidths=0.8,
                   alpha=0.7, label="Non-Pareto", zorder=2)

        ax.scatter(df.loc[mask_on, xkey], df.loc[mask_on, ykey],
                   c="#ff6b6b", s=100, edgecolors="white", linewidths=1.5,
                   alpha=1.0, label="Pareto-Optimal", zorder=3)

        pareto_sorted = df.loc[mask_on].sort_values(by=xkey)
        ax.plot(pareto_sorted[xkey], pareto_sorted[ykey],
                color="#ff6b6b", linewidth=1.2, alpha=0.5, zorder=1)

        ax.set_xlabel(xlabel, color="#aaaacc", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)
        ax.set_title(f"{xlabel} vs {ylabel}", color="white", fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color="#333355", alpha=0.4)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=11, frameon=True, facecolor="#2a2a3e",
               edgecolor="#555577", labelcolor="white",
               bbox_to_anchor=(0.5, 0.005))

    pareto_count = df["pareto"].sum()
    fig.text(0.5, 0.935,
             f"Total candidates: {len(df)}   |   Pareto-optimal: {pareto_count}   |   Training: {N_WEIGHT_VECTORS} agents x {TRAIN_TIMESTEPS:,} steps",
             ha="center", fontsize=10, color="#aaaacc", style="italic")

    save_path = "./pareto/pareto_graph.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n📈 Pareto graph saved: {save_path}")
    plt.show()

    print("\n✅ Done.")