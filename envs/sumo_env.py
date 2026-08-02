"""
SUMO Work Zone Traffic Control Environment

This module provides a Gymnasium-compatible environment for traffic signal control
in a SUMO work zone simulation. It integrates with TraCI to interact with SUMO
and implements the RL interface for training DQN agents.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Try to import SUMO tools
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("SUMO_HOME environment variable not set. Please install SUMO first.")

try:
    import traci
    import sumolib
except ImportError:
    raise ImportError("SUMO tools not found. Please ensure SUMO is properly installed.")

from envs.reward_functions import RewardFunction, default_reward
from envs.observations import ObservationFunction, default_observation


class SumoWorkZoneEnv(gym.Env):
    """
    A Gymnasium-compatible environment for traffic signal control in SUMO work zones.
    
    This environment wraps the SUMO simulation and provides:
    - State space: traffic metrics (queue, speed, waiting time, etc.)
    - Action space: discrete phase control (keep/switch)
    - Reward function: configurable multi-objective rewards
    
    Attributes:
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space (discrete)
        num_lanes: Number of lanes in the controlled intersection
        tls_id: Traffic light system ID
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        tls_id: str = "TL1",
        num_seconds: int = 10000,
        delta_time: int = 5,
        min_green: int = 5,
        max_green: int = 50,
        use_gui: bool = False,
        single_agent: bool = True,
        reward_fn: Optional[RewardFunction] = None,
        observation_fn: Optional[ObservationFunction] = None,
        sumo_seed: Any = "random",
        additional_sumo_cmd: Optional[List[str]] = None,
        time_to_teleport: int = -1,
        max_depart_delay: int = -1,
        sumo_cfg_file: Optional[str] = None,
        additional_files: Optional[List[str]] = None,
    ):
        """
        Initialize the SUMO Work Zone Traffic Control environment.
        
        Args:
            net_file: Path to SUMO network file (.net.xml)
            route_file: Path to route file (.rou.xml)
            tls_id: Traffic light system ID to control
            num_seconds: Total simulation duration in seconds
            delta_time: Time step in seconds between decisions
            min_green: Minimum green time for each phase
            max_green: Maximum green time for each phase
            use_gui: Whether to use SUMO GUI
            single_agent: Whether to use single-agent mode
            reward_fn: Custom reward function (default: default_reward)
            observation_fn: Custom observation function (default: default_observation)
            sumo_seed: Random seed for SUMO simulation
            additional_sumo_cmd: Additional SUMO command line arguments
            time_to_teleport: Time before teleporting stuck vehicles
            max_depart_delay: Maximum departure delay before removing vehicles
            sumo_cfg_file: Path to SUMO config file (.sumocfg)
            additional_files: List of additional XML files to load
        """
        super().__init__()
        
        # Store configuration
        self.net_file = net_file
        self.route_file = route_file
        self.tls_id = tls_id
        self.num_seconds = num_seconds
        self.delta_time = delta_time
        self.min_green = min_green
        self.max_green = max_green
        self.use_gui = use_gui
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.time_to_teleport = time_to_teleport
        self.max_depart_delay = max_depart_delay
        self.sumo_cfg_file = sumo_cfg_file
        self.additional_files = additional_files or []
        
        # Set reward and observation functions
        self.reward_fn = reward_fn or default_reward
        self.observation_fn = observation_fn or default_observation
        
        # Parse network to get lane information
        self.net = sumolib.net.readNet(net_file)
        self._parse_network()
        
        # Initialize state
        self.current_step = 0
        self.elapsed_time_in_phase = 0
        self.current_phase = 0
        self.sumo_started = False
        
        # Define spaces based on network
        self._define_spaces()
        
        # Additional SUMO command options
        self.additional_sumo_cmd = additional_sumo_cmd or []
    
    def _parse_network(self):
        """Parse the SUMO network to extract lane information."""
        self.lanes = []
        self.lane_capacities = {}
        
        # Get controlled lanes from traffic light
        try:
            tls = self.net.getTLS(self.tls_id)
            for connection in tls.getConnections():
                lane = connection.getFromLane()
                lane_id = lane.getID()
                if lane_id not in self.lanes:
                    self.lanes.append(lane_id)
                    # Estimate lane capacity based on length
                    self.lane_capacities[lane_id] = lane.getLength() / 5.0  # ~5m per vehicle
        except Exception:
            # Fallback: get all lanes
            for edge in self.net.getEdges():
                for lane in edge.getLanes():
                    lane_id = lane.getID()
                    if lane_id not in self.lanes:
                        self.lanes.append(lane_id)
                        self.lane_capacities[lane_id] = lane.getLength() / 5.0
        
        self.num_lanes = len(self.lanes)
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        # Observation space: [phase_one_hot, min_green, queue_lengths, speeds, waiting_times, counts]
        # Phase one-hot: number of phases (typically 2-8)
        # min_green: 1 binary value
        # Per lane: queue, speed, waiting, count (4 values per lane)
        num_phases = 2  # Keep or switch (simplified action space)
        obs_dim = num_phases + 1 + (self.num_lanes * 4)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 0 = keep current phase, 1 = switch to next phase
        self.action_space = spaces.Discrete(2)
        
        self.num_phases = num_phases
    
    def _start_sumo(self):
        """Start the SUMO simulation."""
        if self.sumo_started:
            return
        
        # Determine SUMO binary
        if self.use_gui:
            sumo_binary = "sumo-gui"
        else:
            sumo_binary = "sumo"
        
        # Build command
        sumo_cmd = [sumo_binary, "-n", self.net_file]
        
        if self.route_file:
            sumo_cmd.extend(["-r", self.route_file])
        
        if self.sumo_cfg_file:
            sumo_cmd.extend(["-c", self.sumo_cfg_file])
        
        # Add additional files
        for add_file in self.additional_files:
            if os.path.exists(add_file):
                sumo_cmd.extend(["--additional-files", add_file])
        
        # Add additional options
        sumo_cmd.extend(self.additional_sumo_cmd)
        
        # Configure options
        if self.time_to_teleport != -1:
            sumo_cmd.extend(["--time-to-teleport", str(self.time_to_teleport)])
        if self.max_depart_delay != -1:
            sumo_cmd.extend(["--max-depart-delay", str(self.max_depart_delay)])
        
        # Set seed
        if self.sumo_seed != "random":
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        
        sumo_cmd.extend(["--no-step-log"])
        sumo_cmd.append("--step-length")
        sumo_cmd.append(str(self.delta_time))
        
        # Start TraCI
        traci.start(sumo_cmd)
        self.sumo_started = True
        
        # Initialize phase tracking
        self.current_phase = self._get_current_phase()
        self.elapsed_time_in_phase = 0
    
    def _get_current_phase(self) -> int:
        """Get the current traffic light phase index."""
        try:
            phase_index = traci.trafficlight.getPhase(self.tls_id)
            return phase_index % self.num_phases
        except:
            return 0
    
    def _collect_state(self) -> Dict[str, np.ndarray]:
        """
        Collect current state information from SUMO.
        
        Returns:
            Dictionary containing traffic metrics
        """
        state = {
            "phase": self.current_phase,
            "min_green_elapsed": 1.0 if self.elapsed_time_in_phase >= self.min_green else 0.0,
            "lanes": {},
        }
        
        for lane_id in self.lanes:
            try:
                # Queue length (vehicles with speed < 0.1 m/s)
                queue = traci.lane.getLastStepHaltingNumber(lane_id)
                
                # Average speed
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                if speed < 0:
                    speed = 0
                
                # Waiting time
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                # Vehicle count
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                
                # Normalize by capacity
                capacity = self.lane_capacities.get(lane_id, 10.0)
                
                state["lanes"][lane_id] = {
                    "queue": queue / capacity,
                    "speed": speed / 20.0,  # Normalize by typical max speed
                    "waiting_time": waiting_time / 100.0,  # Normalize
                    "vehicle_count": vehicle_count / capacity,
                    "raw_queue": queue,
                    "raw_speed": speed,
                    "raw_waiting_time": waiting_time,
                    "raw_vehicle_count": vehicle_count,
                }
            except:
                state["lanes"][lane_id] = {
                    "queue": 0.0,
                    "speed": 0.0,
                    "waiting_time": 0.0,
                    "vehicle_count": 0.0,
                    "raw_queue": 0,
                    "raw_speed": 0.0,
                    "raw_waiting_time": 0.0,
                    "raw_vehicle_count": 0,
                }
        
        # Collect work zone metrics
        state["workzone"] = self._collect_workzone_metrics()
        
        # Collect safety metrics
        state["safety"] = self._collect_safety_metrics()
        
        return state
    
    def _collect_workzone_metrics(self) -> Dict[str, float]:
        """Collect metrics specific to the work zone."""
        metrics = {
            "spillback_risk": 0.0,
            "merge_conflicts": 0,
            "workzone_queue": 0,
        }
        
        # Check for spillback at rerouter edges (work zone)
        try:
            rerouter_edges = ["E#9", "R2"]
            for edge_id in rerouter_edges:
                for lane_id in traci.edge.getLaneIDs(edge_id):
                    queue = traci.lane.getLastStepHaltingNumber(lane_id)
                    metrics["workzone_queue"] += queue
                    
                    # Check for spillback (queue near intersection)
                    if queue > 5:  # Threshold for spillback
                        metrics["spillback_risk"] = 1.0
        except:
            pass
        
        return metrics
    
    def _collect_safety_metrics(self) -> Dict[str, float]:
        """Collect safety-related metrics."""
        metrics = {
            "ttc_conflicts": 0,
            "avg_ttc": 10.0,
            "drac_conflicts": 0,
        }
        
        try:
            for veh_id in traci.vehicle.getIDList():
                try:
                    ttc = traci.vehicle.getParameter(veh_id, "device.ssm.minTTC")
                    if ttc not in ("", "NA", None):
                        ttc_val = float(ttc)
                        if ttc_val < 1.5:  # TTC threshold
                            metrics["ttc_conflicts"] += 1
                        metrics["avg_ttc"] = min(metrics["avg_ttc"], ttc_val)
                except:
                    pass
        except:
            pass
        
        return metrics
    
    def _compute_reward(self, state: Dict[str, np.ndarray]) -> float:
        """Compute reward from current state."""
        return self.reward_fn(state)
    
    def _apply_action(self, action: int):
        """
        Apply action to the traffic signal.
        
        Args:
            action: 0 = keep current phase, 1 = switch phase
        """
        if action == 1:  # Switch phase
            try:
                current = traci.trafficlight.getPhase(self.tls_id)
                next_phase = (current + 1) % self.num_phases
                traci.trafficlight.setPhase(self.tls_id, next_phase)
            except:
                pass
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Close any existing SUMO connection
        if self.sumo_started:
            try:
                traci.close()
            except:
                pass
            self.sumo_started = False
        
        # Set random seed if provided
        if seed is not None:
            self.sumo_seed = seed
        
        # Start SUMO
        self._start_sumo()
        
        # Reset state
        self.current_step = 0
        self.elapsed_time_in_phase = 0
        self.current_phase = self._get_current_phase()
        
        # Collect initial state
        state = self._collect_state()
        observation = self.observation_fn(state, self)
        
        info = {
            "tls_id": self.tls_id,
            "num_lanes": self.num_lanes,
            "current_step": self.current_step,
        }
        
        return observation, info
    
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0 = keep phase, 1 = switch phase)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action
        self._apply_action(action)
        
        # Simulate one step
        traci.simulationStep()
        
        # Update state
        self.current_step += 1
        self.elapsed_time_in_phase += self.delta_time
        
        # Check if phase changed
        new_phase = self._get_current_phase()
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.elapsed_time_in_phase = 0
        
        # Collect state
        state = self._collect_state()
        observation = self.observation_fn(state, self)
        
        # Compute reward
        reward = self._compute_reward(state)
        
        # Check termination conditions
        terminated = self.current_step >= self.num_seconds // self.delta_time
        
        # Check truncation (e.g., no vehicles left)
        truncated = False
        if len(traci.vehicle.getIDList()) == 0 and self.current_step > 10:
            truncated = True
        
        # Build info
        info = {
            "step": self.current_step,
            "current_phase": self.current_phase,
            "elapsed_time_in_phase": self.elapsed_time_in_phase,
            "total_vehicles": len(traci.vehicle.getIDList()),
            "arrived_vehicles": traci.simulation.getArrivedNumber(),
            "state": state,
        }
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment and stop SUMO."""
        if self.sumo_started:
            try:
                traci.close()
            except:
                pass
            self.sumo_started = False
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            # SUMO GUI handles rendering automatically when use_gui=True
            pass
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self


class MultiAgentSumoWorkZoneEnv(gym.Env):
    """
    Multi-agent version of the SUMO Work Zone environment.
    
    This environment manages multiple traffic light intersections,
    each controlled by its own agent.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        tls_ids: List[str],
        **kwargs,
    ):
        """
        Initialize the multi-agent environment.
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to route file
            tls_ids: List of traffic light system IDs to control
            **kwargs: Additional arguments passed to SumoWorkZoneEnv
        """
        super().__init__()
        
        self.tls_ids = tls_ids
        self.num_agents = len(tls_ids)
        
        # Create individual environments for each agent
        self.envs = {
            tls_id: SumoWorkZoneEnv(
                net_file=net_file,
                route_file=route_file,
                tls_id=tls_id,
                **kwargs,
            )
            for tls_id in tls_ids
        }
        
        # Define joint action/observation spaces
        self._define_spaces()
    
    def _define_spaces(self):
        """Define joint spaces for multi-agent setting."""
        # Each agent has the same observation and action space
        base_env = list(self.envs.values())[0]
        
        # Joint observation: concatenate all agents' observations
        self.observation_space = spaces.Tuple(
            [base_env.observation_space for _ in range(self.num_agents)]
        )
        
        # Joint action: tuple of all agents' actions
        self.action_space = spaces.Tuple(
            [base_env.action_space for _ in range(self.num_agents)]
        )
    
    def reset(self, seed=None, options=None):
        """Reset all agent environments."""
        observations = []
        infos = []
        
        for tls_id in self.tls_ids:
            obs, info = self.envs[tls_id].reset(seed=seed, options=options)
            observations.append(obs)
            infos.append(info)
        
        return tuple(observations), {"agents": infos}
    
    def step(self, actions):
        """Step all agent environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, tls_id in enumerate(self.tls_ids):
            obs, reward, terminated, truncated, info = self.envs[tls_id].step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            tuple(observations),
            tuple(rewards),
            tuple(terminateds),
            tuple(truncateds),
            {"agents": infos},
        )
    
    def close(self):
        """Close all agent environments."""
        for env in self.envs.values():
            env.close()