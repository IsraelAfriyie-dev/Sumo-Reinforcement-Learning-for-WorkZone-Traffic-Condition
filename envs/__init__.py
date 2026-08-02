"""
SUMO Work Zone Traffic Control - Environment Module

This module provides the Gymnasium-compatible environment for SUMO traffic simulation
with reinforcement learning for work zone traffic control.
"""

from envs.sumo_env import SumoWorkZoneEnv
from envs.reward_functions import (
    default_reward,
    efficiency_reward,
    safety_reward,
    multi_objective_reward,
    RewardFunction,
)
from envs.observations import (
    default_observation,
    minimal_observation,
    extended_observation,
    ObservationFunction,
)

__all__ = [
    "SumoWorkZoneEnv",
    "default_reward",
    "efficiency_reward",
    "safety_reward",
    "multi_objective_reward",
    "RewardFunction",
    "default_observation",
    "minimal_observation",
    "extended_observation",
    "ObservationFunction",
]

__version__ = "1.0.0"