"""
Observation Functions for SUMO Work Zone Traffic Control

This module provides various observation (state) functions for defining
the state space that RL agents receive from the environment.

State features include:
- Queue length (normalized)
- Average speed (normalized)
- Waiting time (normalized)
- Signal phase (one-hot encoded)
- Vehicle count (normalized)
- Work zone metrics
"""

from typing import Dict, Callable, List
import numpy as np


# Type alias for observation functions
ObservationFunction = Callable[[Dict, "SumoWorkZoneEnv"], np.ndarray]


# =============================================================================
# Default Observation Function
# =============================================================================

def default_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Default observation function for traffic signal control.
    
    Returns a feature vector containing:
    1. Phase one-hot encoding (num_phases dimensions)
    2. Minimum green elapsed flag (1 dimension)
    3. Per-lane features (4 dimensions each: queue, speed, waiting, count)
    
    Args:
        state: Dictionary containing traffic state from environment
        env: The environment instance
        
    Returns:
        Numpy array observation vector
    """
    features = []
    
    # 1. Phase one-hot encoding
    current_phase = state.get("phase", 0)
    num_phases = env.num_phases
    phase_one_hot = np.zeros(num_phases, dtype=np.float32)
    phase_one_hot[current_phase % num_phases] = 1.0
    features.extend(phase_one_hot)
    
    # 2. Minimum green elapsed flag
    min_green_elapsed = state.get("min_green_elapsed", 0.0)
    features.append(min_green_elapsed)
    
    # 3. Per-lane features
    lane_data = state.get("lanes", {})
    for lane_id in env.lanes:
        if lane_id in lane_data:
            metrics = lane_data[lane_id]
            features.append(metrics.get("queue", 0.0))
            features.append(metrics.get("speed", 0.0))
            features.append(metrics.get("waiting_time", 0.0))
            features.append(metrics.get("vehicle_count", 0.0))
        else:
            # Default values for missing lanes
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Minimal Observation Function
# =============================================================================

def minimal_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Minimal observation function with just essential features.
    
    Returns:
    - Phase one-hot (2 dimensions)
    - Total queue length (1 dimension)
    - Average speed (1 dimension)
    
    Total: 4 dimensions
    """
    features = []
    
    # Phase one-hot (2 phases: green for NS, green for EW)
    current_phase = state.get("phase", 0)
    phase_one_hot = np.zeros(2, dtype=np.float32)
    phase_one_hot[current_phase % 2] = 1.0
    features.extend(phase_one_hot)
    
    # Aggregate lane metrics
    lane_data = state.get("lanes", {})
    total_queue = sum(m.get("queue", 0.0) for m in lane_data.values())
    speeds = [m.get("speed", 0.0) for m in lane_data.values() if m.get("vehicle_count", 0) > 0]
    avg_speed = np.mean(speeds) if speeds else 0.0
    
    features.append(total_queue)
    features.append(avg_speed)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Extended Observation Function
# =============================================================================

def extended_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Extended observation function with all available features.
    
    Returns:
    - Phase one-hot encoding
    - Minimum green elapsed flag
    - Per-lane features (queue, speed, waiting, count)
    - Work zone metrics (spillback risk, merge conflicts, workzone queue)
    - Safety metrics (TTC conflicts, average TTC)
    
    Args:
        state: Dictionary containing traffic state
        env: Environment instance
        
    Returns:
        Extended observation vector
    """
    features = []
    
    # Base features from default observation
    base_features = default_observation(state, env)
    features.extend(base_features)
    
    # Work zone features
    workzone = state.get("workzone", {})
    features.append(workzone.get("spillback_risk", 0.0))
    features.append(float(workzone.get("merge_conflicts", 0)))
    features.append(workzone.get("workzone_queue", 0.0) / 20.0)  # Normalized
    
    # Safety features
    safety = state.get("safety", {})
    features.append(float(safety.get("ttc_conflicts", 0)) / 10.0)  # Normalized
    features.append(safety.get("avg_ttc", 10.0) / 10.0)  # Normalized TTC
    features.append(float(safety.get("drac_conflicts", 0)) / 10.0)  # Normalized
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Pressure-Based Observation
# =============================================================================

def pressure_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Traffic pressure-based observation.
    
    Uses incoming/outgoing vehicle differences to compute pressure,
    which can be more informative for signal control than raw counts.
    
    Args:
        state: Traffic state dictionary
        env: Environment instance
        
    Returns:
        Pressure-based observation vector
    """
    features = []
    
    # Phase encoding (reduced)
    current_phase = state.get("phase", 0)
    phase_one_hot = np.zeros(2, dtype=np.float32)
    phase_one_hot[current_phase % 2] = 1.0
    features.extend(phase_one_hot)
    
    # Min green elapsed
    features.append(state.get("min_green_elapsed", 0.0))
    
    # Traffic pressure per lane
    lane_data = state.get("lanes", {})
    for lane_id in env.lanes:
        if lane_id in lane_data:
            metrics = lane_data[lane_id]
            # Pressure = vehicles approaching - vehicles clearing
            # Simplified: use queue and speed as proxy
            pressure = metrics.get("queue", 0.0) * (1.0 - metrics.get("speed", 0.0))
            features.append(pressure)
        else:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Incoming-Outgoing Observation
# =============================================================================

def in_out_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Incoming vs outgoing vehicle count observation.
    
    This observation explicitly captures the difference between
    vehicles entering and exiting the intersection, which is
    useful for detecting congestion buildup.
    
    Args:
        state: Traffic state dictionary
        env: Environment instance
        
    Returns:
        Incoming-outgoing observation vector
    """
    features = []
    
    # Phase encoding
    current_phase = state.get("phase", 0)
    phase_one_hot = np.zeros(env.num_phases, dtype=np.float32)
    phase_one_hot[current_phase % env.num_phases] = 1.0
    features.extend(phase_one_hot)
    
    # Min green
    features.append(state.get("min_green_elapsed", 0.0))
    
    # Split lanes into incoming and outgoing groups
    # For a typical 4-way intersection:
    # - Incoming lanes: E#4, E#10, E#12, E#17
    # - Outgoing lanes: E#1, E#11, E#15, E#18
    incoming_edges = ["E#4", "E#10", "E#12", "E#17"]
    outgoing_edges = ["E#1", "E#11", "E#15", "E#18"]
    
    lane_data = state.get("lanes", {})
    
    incoming_count = 0
    outgoing_count = 0
    incoming_queue = 0
    outgoing_queue = 0
    
    for lane_id, metrics in lane_data.items():
        edge_id = lane_id.rsplit("_", 1)[0] if "_" in lane_id else lane_id
        
        if edge_id in incoming_edges:
            incoming_count += metrics.get("raw_vehicle_count", 0)
            incoming_queue += metrics.get("raw_queue", 0)
        elif edge_id in outgoing_edges:
            outgoing_count += metrics.get("raw_vehicle_count", 0)
            outgoing_queue += metrics.get("raw_queue", 0)
    
    # Normalize
    features.append(incoming_count / 20.0)
    features.append(outgoing_count / 20.0)
    features.append(incoming_queue / 10.0)
    features.append(outgoing_queue / 10.0)
    features.append((incoming_count - outgoing_count) / 20.0)  # Net pressure
    
    # Pad to match expected size
    while len(features) < env.observation_space.shape[0]:
        features.append(0.0)
    
    return np.array(features[:env.observation_space.shape[0]], dtype=np.float32)


# =============================================================================
# Queue-Only Observation
# =============================================================================

def queue_only_observation(state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
    """
    Minimal queue-length only observation.
    
    Returns only normalized queue lengths for each lane.
    Useful for very simple baselines or debugging.
    """
    features = []
    
    lane_data = state.get("lanes", {})
    for lane_id in env.lanes:
        if lane_id in lane_data:
            features.append(lane_data[lane_id].get("queue", 0.0))
        else:
            features.append(0.0)
    
    return np.array(features, dtype=np.float32)


# =============================================================================
# Configurable Observation Builder
# =============================================================================

class ConfigurableObservation:
    """
    Build custom observation functions with selectable features.
    
    Example:
        obs_fn = ConfigurableObservation(
            include_phase=True,
            include_min_green=True,
            include_queue=True,
            include_speed=True,
            include_waiting=True,
            include_vehicle_count=True,
            include_workzone=True,
            include_safety=True,
        )
    """
    
    def __init__(
        self,
        include_phase: bool = True,
        include_min_green: bool = True,
        include_queue: bool = True,
        include_speed: bool = True,
        include_waiting: bool = True,
        include_vehicle_count: bool = True,
        include_workzone: bool = False,
        include_safety: bool = False,
        normalize: bool = True,
    ):
        """
        Initialize configurable observation builder.
        
        Args:
            include_phase: Include one-hot phase encoding
            include_min_green: Include min green elapsed flag
            include_queue: Include queue lengths
            include_speed: Include average speeds
            include_waiting: Include waiting times
            include_vehicle_count: Include vehicle counts
            include_workzone: Include work zone metrics
            include_safety: Include safety metrics
            normalize: Normalize all values to [0, 1]
        """
        self.options = {
            "phase": include_phase,
            "min_green": include_min_green,
            "queue": include_queue,
            "speed": include_speed,
            "waiting": include_waiting,
            "vehicle_count": include_vehicle_count,
            "workzone": include_workzone,
            "safety": include_safety,
        }
        self.normalize = normalize
    
    def __call__(self, state: Dict, env: "SumoWorkZoneEnv") -> np.ndarray:
        """Build observation from state."""
        features = []
        
        # Phase encoding
        if self.options["phase"]:
            current_phase = state.get("phase", 0)
            num_phases = env.num_phases
            phase_one_hot = np.zeros(num_phases, dtype=np.float32)
            phase_one_hot[current_phase % num_phases] = 1.0
            features.extend(phase_one_hot)
        
        # Min green elapsed
        if self.options["min_green"]:
            features.append(state.get("min_green_elapsed", 0.0))
        
        # Lane features
        lane_data = state.get("lanes", {})
        for lane_id in env.lanes:
            if lane_id in lane_data:
                metrics = lane_data[lane_id]
            else:
                metrics = {}
            
            if self.options["queue"]:
                features.append(metrics.get("queue", 0.0))
            if self.options["speed"]:
                features.append(metrics.get("speed", 0.0))
            if self.options["waiting"]:
                features.append(metrics.get("waiting_time", 0.0))
            if self.options["vehicle_count"]:
                features.append(metrics.get("vehicle_count", 0.0))
        
        # Work zone features
        if self.options["workzone"]:
            workzone = state.get("workzone", {})
            features.append(workzone.get("spillback_risk", 0.0))
            features.append(float(workzone.get("merge_conflicts", 0)))
            features.append(workzone.get("workzone_queue", 0.0) / 20.0)
        
        # Safety features
        if self.options["safety"]:
            safety = state.get("safety", {})
            features.append(float(safety.get("ttc_conflicts", 0)) / 10.0)
            features.append(safety.get("avg_ttc", 10.0) / 10.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_output_dim(self, env: "SumoWorkZoneEnv") -> int:
        """Calculate output dimension for a given environment."""
        dim = 0
        
        if self.options["phase"]:
            dim += env.num_phases
        if self.options["min_green"]:
            dim += 1
        
        lane_features = sum([
            self.options["queue"],
            self.options["speed"],
            self.options["waiting"],
            self.options["vehicle_count"],
        ])
        dim += env.num_lanes * lane_features
        
        if self.options["workzone"]:
            dim += 3
        if self.options["safety"]:
            dim += 2
        
        return dim


# =============================================================================
# Dictionary of available observation functions
# =============================================================================

OBSERVATION_FUNCTIONS = {
    "default": default_observation,
    "minimal": minimal_observation,
    "extended": extended_observation,
    "pressure": pressure_observation,
    "in_out": in_out_observation,
    "queue_only": queue_only_observation,
}


def get_observation_function(name: str) -> ObservationFunction:
    """
    Get an observation function by name.
    
    Args:
        name: Name of the observation function
        
    Returns:
        Observation function
        
    Raises:
        ValueError: If name is not recognized
    """
    if name not in OBSERVATION_FUNCTIONS:
        raise ValueError(
            f"Unknown observation function: {name}. "
            f"Available: {list(OBSERVATION_FUNCTIONS.keys())}"
        )
    return OBSERVATION_FUNCTIONS[name]