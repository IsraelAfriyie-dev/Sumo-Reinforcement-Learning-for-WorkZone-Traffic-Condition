import traci
import sumolib
import os

# ===============================
# INPUTS
# ===============================
cfg_file = r"C:\Users\Asus ROG\Desktop\sumo-IA\sumo_rl\nets\construction\sumo-xml\network_IA.sumocfg"

ghost = {"edge": "E#9", "lane": 1, "pos": 320.0, "duration": 1200.0}

use_gui = True
step_length = 0.2

QUEUE_START = 200.0
MERGE_START = 170.0
FORCE_START = 20.0

QUEUE_SPEED = 1.0
MERGE_SPEED = 2.5

DOWNSTREAM_CONGESTION_LENGTH = 90.0
DOWNSTREAM_SPEED = 1.7

# ===============================
# QUEUE TAIL DETECTION
# ===============================
def get_queue_tail_pos(edge_id, lane_index, rear_pos, speed_thresh=1.0):
    """
    Queue tail position (m) on a lane:
    the most-upstream vehicle with speed < speed_thresh and position < rear_pos.
    """
    lane_id = f"{edge_id}_{lane_index}"
    try:
        vids = traci.lane.getLastStepVehicleIDs(lane_id)
    except:
        return None

    tail_pos = None
    for vid in vids:
        if vid == "ghost":
            continue
        try:
            p = traci.vehicle.getLanePosition(vid)
            v = traci.vehicle.getSpeed(vid)
            if p < rear_pos and v < speed_thresh:
                if tail_pos is None or p < tail_pos:
                    tail_pos = p
        except:
            pass

    return tail_pos


# ===============================
# SIMULATION
# ===============================
def run():
    sumo = sumolib.checkBinary("sumo-gui" if use_gui else "sumo")

    traci.start([
        sumo,
        "-c", cfg_file,
        "--step-length", str(step_length),
        "--lateral-resolution", "0.8",
        "--error-log", "sumo_error.log",
        "--log", "sumo_log.txt",
        "--message-log", "sumo_message.log"
    ])

    # ---- ghost vehicle
    if "ghost" not in traci.vehicletype.getIDList():
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "ghost")
        traci.vehicletype.setMaxSpeed("ghost", 0.01)

        traci.vehicletype.setColor("ghost", (0, 0, 0, 255))   # black
        traci.vehicletype.setLength("ghost", 60.0)            # long work-zone blocker
        traci.vehicletype.setWidth("ghost", 2.6)
        traci.vehicletype.setMinGap("ghost", 0.5)

    if "ghostRoute" not in traci.route.getIDList():
        traci.route.add("ghostRoute", [ghost["edge"]])

    if "ghost" not in traci.vehicle.getIDList():
        traci.vehicle.add(
            vehID="ghost",
            routeID="ghostRoute",
            typeID="ghost",
            depart=0,
            departLane=ghost["lane"],
            departPos=str(ghost["pos"]),
            departSpeed="0"
        )
        traci.vehicle.setStop(
            "ghost",
            edgeID=ghost["edge"],
            pos=ghost["pos"],
            laneIndex=ghost["lane"],
            duration=ghost["duration"]
        )

    # ---- blinking markers at START and END of ghost
    ghost_len = traci.vehicletype.getLength("ghost")
    start_pos = max(0.0, ghost["pos"] - ghost_len)  # rear position
    end_pos = ghost["pos"]                          # front position

    def place_or_move_poi(poi_id, edge_id, pos, lane_idx, color, layer=300):
        try:
            x, y = traci.simulation.convert2D(edge_id, pos, lane_idx)
            if poi_id not in traci.poi.getIDList():
                traci.poi.add(poi_id, x, y, color, layer=layer)
            else:
                traci.poi.setPosition(poi_id, x, y)
                traci.poi.setColor(poi_id, color)
        except:
            pass

    # ---- data collection: ONE CSV FILE ONLY
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Single CSV for vehicle trajectories WITH tail_pos_m and rear_pos_m
    vehicle_csv = os.path.join(script_dir, "vehicle_trajectories.csv")
    print(f"\n📊 Saving vehicle data to: {vehicle_csv}")
    fv = open(vehicle_csv, "w")
    fv.write("time_s,vehicle_id,position_m,speed_mps,lane,tail_pos_m,rear_pos_m\n")

    rear_pos = ghost["pos"] - ghost_len  # upstream start of blockage

    slowed_for_queue = set()
    issued_merge_cmd = set()
    pinned_after_merge = set()
    slowed_downstream = set()

    print("🚗 Running simulation...")

    while traci.simulation.getTime() < ghost["duration"] + 20:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # ---- calculate queue tail position for this timestep
        tail_pos = get_queue_tail_pos(ghost["edge"], ghost["lane"], rear_pos, speed_thresh=1.0)
        
        # Convert None to empty string for CSV (or you can use 0.0)
        tail_pos_str = f"{tail_pos:.3f}" if tail_pos is not None else ""

        # ---- record individual vehicle data with tail_pos_m and rear_pos_m
        vehicles = traci.edge.getLastStepVehicleIDs(ghost["edge"])
        for vid in vehicles:
            if vid == "ghost":
                continue
            try:
                lane = traci.vehicle.getLaneIndex(vid)
                pos = traci.vehicle.getLanePosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                # Write with tail_pos and rear_pos for each vehicle entry
                fv.write(f"{sim_time:.2f},{vid},{pos:.3f},{speed:.3f},{lane},{tail_pos_str},{rear_pos:.3f}\n")
            except:
                pass

        # ---- blink markers
        blink_on = (int(sim_time * 2) % 2 == 0)
        if blink_on:
            c = (255, 0, 0, 255)      # visible (red)
        else:
            c = (255, 255, 0, 0)      # invisible (transparent)

        place_or_move_poi("block_start", ghost["edge"], start_pos, ghost["lane"], c)
        place_or_move_poi("block_end",   ghost["edge"], end_pos,   ghost["lane"], c)

        for vid in vehicles:
            if vid == "ghost":
                continue

            lane = traci.vehicle.getLaneIndex(vid)
            pos = traci.vehicle.getLanePosition(vid)
            dist = ghost["pos"] - pos

            # Keep vehicles on lane 1 after passing the blockage
            if lane == 1 and pos >= ghost["pos"] and vid not in pinned_after_merge:
                try:
                    traci.vehicle.changeLane(vid, 1, 99999)
                    pinned_after_merge.add(vid)
                except:
                    pass

            # Downstream congestion on lane 1 after blockage (apply once)
            if lane == 1 and ghost["pos"] <= pos <= ghost["pos"] + DOWNSTREAM_CONGESTION_LENGTH:
                if vid not in slowed_downstream:
                    try:
                        traci.vehicle.slowDown(vid, DOWNSTREAM_SPEED, 3.0)
                        slowed_downstream.add(vid)
                    except:
                        pass

            if dist <= 0:
                continue

            # Queue formation (apply once)
            if lane == ghost["lane"] and dist <= QUEUE_START and vid not in slowed_for_queue:
                try:
                    traci.vehicle.setLaneChangeMode(vid, 0)
                    traci.vehicle.slowDown(vid, QUEUE_SPEED, 4.0)
                    slowed_for_queue.add(vid)
                except:
                    pass

            # Smooth merge command (issue once)
            if lane == ghost["lane"] and dist <= MERGE_START and vid not in issued_merge_cmd:
                try:
                    traci.vehicle.setLaneChangeMode(vid, 1621)
                    traci.vehicle.setParameter(vid, "laneChangeModel.lcAssertive", "0.8")
                    traci.vehicle.setParameter(vid, "laneChangeModel.lcImpatience", "0.2")
                    traci.vehicle.setParameter(vid, "laneChangeModel.lcCooperative", "1.0")

                    traci.vehicle.changeLane(vid, 1, 8.0)
                    traci.vehicle.slowDown(vid, MERGE_SPEED, 3.0)

                    issued_merge_cmd.add(vid)
                except:
                    pass

            # Panic merge (don't spam)
            if lane == ghost["lane"] and dist <= FORCE_START and vid not in issued_merge_cmd:
                try:
                    traci.vehicle.changeLane(vid, 1, 3.0)
                    issued_merge_cmd.add(vid)
                except:
                    pass

    fv.close()
    traci.close()

    print("✅ Simulation complete!")
    print(f"📁 Vehicle CSV file: {vehicle_csv}")
   

if __name__ == "__main__":
    run()