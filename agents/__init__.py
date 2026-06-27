"""
SUMO Work Zone Traffic Control - Agents Module

This module provides reinforcement learning agents for traffic signal control,
including DQN (Deep Q-Network) implementation and utilities.
"""

from agents.dqn_agent import DQNAgent, DQNPolicy

__all__ = [
    "DQNAgent",
    "DQNPolicy",
]

__version__ = "1.0.0"