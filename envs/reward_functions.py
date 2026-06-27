"""
Reward Functions for SUMO Work Zone Traffic Control

This module provides various reward functions for training RL agents
to optimize traffic signal control in work zone scenarios.

Reward components:
- Penalize: long queues, excessive waiting time, stop-and-go behavior, merge conflicts
- Encourage: smoother traffic flow, lower delay, improved throughput, safer merging
"""

from typing import Dict, Callable
import numpy as np


# Type alias for reward functions
RewardFunction = Callable[[Dict], float]


# =============================================================================
# Default Multi-Objective Reward Function
# =============================================================================

def default_reward(state: Dict) -> float:
    """
    Default multi-objective reward function for work zone traffic control.
    
    This reward combines multiple traffic metrics to encourage:
    - Lower waiting times
    - Shorter queues
    - Higher average speeds
    - Better throughput
    - Safer driving (fewer TTC conflicts)
    - Reduced work zone spillback
    - Less stop-and-go behavior
    
    Args:
        state: Dictionary containing traffic state metrics
        
    Returns:
        Computed reward value
    """
    reward = 0.0
    
    # Weights (can be configured)
    WAITING_TIME_WEIGHT = -0.1
    QUEUE_LENGTH_WEIGHT = -0.2
    SPEED_WEIGHT = 0.1
    THROUGHPUT_WEIGHT = 0.01
    SAFETY_WEIGHT = -0.3
    SPILLBACK_WEIGHT = -0.2
    STOP_GO_WEIGHT = -0.1
    
    # Extract lane metrics
    lane_data = state.get("lanes", {})
    total_queue = 0
    total_speed = 0
    total_waiting = 0
    total_vehicles = 0
    speed_variance = 0.0
    speeds = []
    
    for lane_id, metrics in lane_data.items():
        raw_queue = metrics.get("raw_queue", 0)
        raw_speed = metrics.get("raw_speed", 0)
        raw_waiting = metrics.get("raw_waiting_time", 0)
        raw_count = metrics.get("raw_vehicle_count", 0)
        
        total_queue += raw_queue
        total_speed += raw_speed
        total_waiting += raw_waiting
        total_vehicles += raw_count
        
        if raw_count > 0:
            speeds.append(raw_speed)
    
    # 1. Waiting Time Component (penalize high waiting times)
    if total_vehicles > 0:
        avg_waiting = total_waiting / total_vehicles
    else:
        avg_waiting = 0
    reward += WAITING_TIME_WEIGHT * avg_waiting
    
    # 2. Queue Length Component (penalize long queues)
    reward += QUEUE_LENGTH_WEIGHT * total_queue
    
    # 3. Speed Component (reward higher speeds)
    if speeds:
        avg_speed = np.mean(speeds)
        reward += SPEED_WEIGHT * avg_speed
        
        # Stop-and-go penalty: penalize high variance in speeds
        if len(speeds) > 1:
            speed_variance = np.std(speeds)
            reward += STOP_GO_WEIGHT * speed_variance
    
    # 4. Throughput Component (reward vehicles that have passed)
    # This is typically computed from info, but we can estimate from total vehicles
    reward += THROUGHPUT_WEIGHT * total_vehicles
    
    # 5. Safety Component (penalize TTC conflicts)
    safety_metrics = state.get("safety", {})
    ttc_conflicts = safety_metrics.get("ttc_conflicts", 0)
    reward += SAFETY_WEIGHT * ttc_conflicts
    
    # 6. Work Zone Spillback Component (penalize spillback at work zone)
    workzone_metrics = state.get("workzone", {})
    spillback_risk = workzone_metrics.get("spillback_risk", 0)
    workzone_queue = workzone_metrics.get("workzone_queue", 0)
    reward += SPILLBACK_WEIGHT * (spillback_risk + workzone_queue / 10.0)
    
    return reward


# =============================================================================
# Efficiency-Focused Reward
# =============================================================================

def efficiency_reward(state: Dict) -> float:
    """
    Reward function focused on traffic efficiency.
    
    Optimizes for:
    - Minimal waiting time
    - Short queues
    - High average speeds
    
    Best for scenarios where throughput is the primary concern.
    """
    lane_data = state.get("lanes", {})
    
    total_queue = 0
    total_speed = 0
    total_waiting = 0
    vehicle_count = 0
    
    for lane_id, metrics in lane_data.items():
        total_queue += metrics.get("raw_queue", 0)
        total_speed += metrics.get("raw_speed", 0)
        total_waiting += metrics.get("raw_waiting_time", 0)
        vehicle_count += metrics.get("raw_vehicle_count", 0)
    
    # Compute efficiency metrics
    avg_speed = total_speed / max(1, len(lane_data))
    avg_waiting = total_waiting / max(1, vehicle_count)
    
    # Combined reward: maximize speed, minimize queue and waiting
    reward = 0.5 * avg_speed - 0.3 * total_queue - 0.2 * avg_waiting
    
    return reward


# =============================================================================
# Safety-Focused Reward
# =============================================================================

def safety_reward(state: Dict) -> float:
    """
    Reward function focused on traffic safety.
    
    Optimizes for:
    - Few TTC conflicts
    - Smooth traffic flow
    - Minimal hard braking
    
    Best for high-risk work zone scenarios.
    """
    safety_metrics = state.get("safety", {})
    lane_data = state.get("lanes", {})
    workzone_metrics = state.get("workzone", {})
    
    # Safety metrics
    ttc_conflicts = safety_metrics.get("ttc_conflicts", 0)
    avg_ttc = safety_metrics.get("avg_ttc", 10.0)
    
    # Work zone safety
    spillback_risk = workzone_metrics.get("spillback_risk", 0)
    merge_conflicts = workzone_metrics.get("merge_conflicts", 0)
    
    # Flow smoothness
    speeds = [m.get("raw_speed", 0) for m in lane_data.values() if m.get("raw_vehicle_count", 0) > 0]
    speed_variance = np.std(speeds) if len(speeds) > 1 else 0
    
    # Combined reward
    reward = (
        -5.0 * ttc_conflicts +           # Strong penalty for conflicts
        0.5 * max(0, avg_ttc - 1.5) -    # Reward for higher TTC
        2.0 * spillback_risk -           # Penalty for spillback risk
        1.0 * merge_conflicts -          # Penalty for merge conflicts
        1.0 * speed_variance             # Penalty for unstable flow
    )
    
    return reward


# =============================================================================
# Multi-Objective Reward (Pareto-Optimizable)
# =============================================================================

def multi_objective_reward(state: Dict, weights: Dict) -> float:
    """
    Configurable multi-objective reward function.
    
    This reward function allows specifying weights for different objectives,
    enabling Pareto-optimal training across multiple metrics.
    
    Args:
        state: Traffic state dictionary
        weights: Dictionary of weights for each objective:
            - waiting_time: weight for waiting time (negative = penalty)
            - queue_length: weight for queue length (negative = penalty)
            - speed: weight for average speed (positive = reward)
            - throughput: weight for throughput (positive = reward)
            - safety: weight for TTC conflicts (negative = penalty)
            - spillback: weight for work zone spillback (negative = penalty)
    
    Returns:
        Weighted sum of all objective values
    """
    # Default weights
    default_weights = {
        "waiting_time": -0.1,
        "queue_length": -0.2,
        "speed": 0.1,
        "throughput": 0.01,
        "safety": -0.3,
        "spillback": -0.2,
    }
    
    # Update with provided weights
    if weights is None:
        weights = default_weights
    else:
        weights = {**default_weights, **weights}
    
    lane_data = state.get("lanes", {})
    safety_metrics = state.get("safety", {})
    workzone_metrics = state.get("workzone", {})
    
    # Compute individual objectives
    total_queue = sum(m.get("raw_queue", 0) for m in lane_data.values())
    total_waiting = sum(m.get("raw_waiting_time", 0) for m in lane_data.values())
    total_vehicles = sum(m.get("raw_vehicle_count", 0) for m in lane_data.values())
    speeds = [m.get("raw_speed", 0) for m in lane_data.values() if m.get("raw_vehicle_count", 0) > 0]
    avg_speed = np.mean(speeds) if speeds else 0
    ttc_conflicts = safety_metrics.get("ttc_conflicts", 0)
    spillback = workzone_metrics.get("spillback_risk", 0) + workzone_metrics.get("workzone_queue", 0) / 10.0
    
    # Compute weighted reward
    reward = (
        weights.get("waiting_time", -0.1) * total_waiting / max(1, total_vehicles) +
        weights.get("queue_length", -0.2) * total_queue +
        weights.get("speed", 0.1) * avg_speed +
        weights.get("throughput", 0.01) * total_vehicles +
        weights.get("safety", -0.3) * ttc_conflicts +
        weights.get("spillback", -0.2) * spillback
    )
    
    return reward


# =============================================================================
# Simple Queue-Based Reward
# =============================================================================

def queue_penalty_reward(state: Dict) -> float:
    """
    Simple reward based on queue lengths.
    
    Penalizes total queue length, encouraging the agent to keep traffic flowing.
    Suitable for quick experiments and baseline comparisons.
    """
    lane_data = state.get("lanes", {})
    total_queue = sum(m.get("raw_queue", 0) for m in lane_data.values())
    
    # Negative reward proportional to queue length
    return -total_queue


# =============================================================================
# Delay-Based Reward
# =============================================================================

def delay_reward(state: Dict) -> float:
    """
    Reward based on total vehicle delay.
    
    Penalizes cumulative waiting time, which directly relates to
    user inconvenience and fuel waste.
    """
    lane_data = state.get("lanes", {})
    total_delay = sum(m.get("raw_waiting_time", 0) for m in lane_data.values())
    
    # Negative reward proportional to total delay
    return -total_delay / 100.0  # Scale down to reasonable range


# =============================================================================
# Composite Reward with Tunable Weights
# =============================================================================

class CompositeReward:
    """
    Configurable composite reward function with tunable weights.
    
    This class allows dynamic adjustment of reward weights during training,
    which can be useful for curriculum learning or multi-phase training.
    """
    
    def __init__(
        self,
        waiting_time_weight: float = -0.1,
        queue_length_weight: float = -0.2,
        speed_weight: float = 0.1,
        throughput_weight: float = 0.01,
        safety_weight: float = -0.3,
        spillback_weight: float = -0.2,
        stop_go_weight: float = -0.1,
    ):
        """
        Initialize composite reward function.
        
        Args:
            waiting_time_weight: Weight for waiting time penalty
            queue_length_weight: Weight for queue length penalty
            speed_weight: Weight for average speed reward
            throughput_weight: Weight for throughput reward
            safety_weight: Weight for TTC safety penalty
            spillback_weight: Weight for work zone spillback penalty
            stop_go_weight: Weight for stop-and-go penalty
        """
        self.weights = {
            "waiting_time": waiting_time_weight,
            "queue_length": queue_length_weight,
            "speed": speed_weight,
            "throughput": throughput_weight,
            "safety": safety_weight,
            "spillback": spillback_weight,
            "stop_go": stop_go_weight,
        }
    
    def __call__(self, state: Dict) -> float:
        """Compute composite reward from state."""
        return multi_objective_reward(state, self.weights)
    
    def update_weight(self, objective: str, weight: float):
        """Update the weight for a specific objective."""
        if objective in self.weights:
            self.weights[objective] = weight
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()


# =============================================================================
# Utility Functions
# =============================================================================

def normalize_reward(reward: float, scale: float = 100.0) -> float:
    """
    Normalize reward to a standard range.
    
    Args:
        reward: Raw reward value
        scale: Scaling factor
        
    Returns:
        Normalized reward
    """
    return reward / scale


def compute_pareto_weights(
    efficiency_weight: float = 1.0,
    safety_weight: float = 1.0,
) -> Dict[str, float]:
    """
    Generate reward weights for Pareto-frontier training.
    
    Args:
        efficiency_weight: Relative weight for efficiency objectives
        safety_weight: Relative weight for safety objectives
        
    Returns:
        Dictionary of weights for multi_objective_reward
    """
    total_weight = efficiency_weight + safety_weight
    efficiency_factor = efficiency_weight / total_weight
    safety_factor = safety_weight / total_weight
    
    return {
        "waiting_time": -0.2 * efficiency_factor,
        "queue_length": -0.3 * efficiency_factor,
        "speed": 0.15 * efficiency_factor,
        "throughput": 0.05 * efficiency_factor,
        "safety": -0.3 * safety_factor,
        "spillback": -0.2 * safety_factor,
    }


# Dictionary of available reward functions
REWARD_FUNCTIONS = {
    "default": default_reward,
    "efficiency": efficiency_reward,
    "safety": safety_reward,
    "queue_penalty": queue_penalty_reward,
    "delay": delay_reward,
}


def get_reward_function(name: str) -> RewardFunction:
    """
    Get a reward function by name.
    
    Args:
        name: Name of the reward function
        
    Returns:
        Reward function
        
    Raises:
        ValueError: If name is not recognized
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function: {name}. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name]