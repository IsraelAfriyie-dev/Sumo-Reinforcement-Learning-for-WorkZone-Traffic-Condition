"""
Experience Replay Buffer for DQN Training

This module implements a replay buffer for storing and sampling
experiences during reinforcement learning training.
"""

import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN agent.
    
    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training.
    
    Uses a circular buffer for efficient memory usage.
    
    Attributes:
        capacity: Maximum number of transitions to store
        batch_size: Number of transitions to sample per training step
        buffer: Internal storage for transitions
    """
    
    def __init__(self, capacity: int, batch_size: int = 32):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Number of transitions to sample per training step
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, self.batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_samples: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_samples: Minimum number of samples required
            
        Returns:
            True if buffer has enough samples
        """
        return len(self.buffer) >= min_samples
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
        self.position = 0
    
    def get_statistics(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fill_ratio": len(self.buffer) / self.capacity if self.capacity > 0 else 0,
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions based on their TD error (priority),
    allowing the agent to learn more from surprising transitions.
    
    Uses proportional prioritization with sum-tree implementation
    for efficient sampling.
    """
    
    def __init__(
        self,
        capacity: int,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Number of transitions to sample per training step
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta per sampling
            epsilon: Small constant to avoid zero priority
        """
        super().__init__(capacity, batch_size)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Sum-tree for efficient sampling
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
    ):
        """
        Add a transition with priority based on TD error.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            td_error: TD error for prioritization (uses max_priority if None)
        """
        priority = self.max_priority if td_error is None else (abs(td_error) + self.epsilon) ** self.alpha
        
        self.tree.add(priority, (state, action, reward, next_state, done))
        
        # Store transition separately for retrieval
        self.buffer.append((state, action, reward, next_state, done))
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with prioritization.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        batch = []
        indices = []
        weights = []
        
        segment = self.tree.total() / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, priority), data = self.tree.get(s)
            indices.append(idx)
            weights.append(priority)
            batch.append(data)
        
        # Normalize weights
        weights = np.array(weights)
        weights = (weights / (self.tree.total() + self.epsilon)) ** (-self.beta)
        weights = weights / weights.max()
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            td_errors: New TD errors for these transitions
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return number of valid transitions."""
        return self.tree.n


class SumTree:
    """
    Sum-Tree data structure for prioritized sampling.
    
    A binary tree where each leaf stores a priority value,
    and each internal node stores the sum of its children's priorities.
    
    Allows O(log n) sampling and O(log n) priority updates.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree.
        
        Args:
            capacity: Number of leaf nodes (transitions)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.n = 0  # Number of stored transitions
    
    def add(self, priority: float, data: Tuple):
        """
        Add a transition with given priority.
        
        Args:
            priority: Priority value
            data: Transition data
        """
        idx = self.capacity + self.n
        self.tree[idx] = priority
        self._update(idx)
        self.n = min(self.n + 1, self.capacity)
        
        # Store data
        if not hasattr(self, 'data'):
            self.data = {}
        self.data[idx] = data
    
    def get(self, s: float) -> Tuple[Tuple[int, float], Tuple]:
        """
        Sample a transition based on priority.
        
        Args:
            s: Random value in [0, total_priority]
            
        Returns:
            Tuple of ((index, priority), data)
        """
        idx = self._find(1, s)
        data_idx = idx - self.capacity
        return (idx, self.tree[idx]), self.data.get(data_idx, (None, None, None, None, None))
    
    def _update(self, idx: int):
        """Update priority and propagate up the tree."""
        change = self.tree[idx]
        parent = idx // 2
        
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2
    
    def _find(self, idx: int, s: float) -> int:
        """Find the leaf containing value s."""
        left = 2 * idx
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if self.tree[left] >= s:
            return self._find(left, s)
        else:
            return self._find(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[1]
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return self.n


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent scenarios.
    
    Stores transitions for multiple agents and allows
    sampling per-agent or joint batches.
    """
    
    def __init__(self, capacity: int, batch_size: int, num_agents: int):
        """
        Initialize multi-agent replay buffer.
        
        Args:
            capacity: Maximum transitions per agent
            batch_size: Batch size per agent
            num_agents: Number of agents
        """
        self.num_agents = num_agents
        self.buffers = [ReplayBuffer(capacity, batch_size) for _ in range(num_agents)]
    
    def push(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ):
        """
        Add transitions for all agents.
        """
        for i in range(self.num_agents):
            self.buffers[i].push(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
    
    def sample(self, agent_id: Optional[int] = None) -> List[Tuple]:
        """
        Sample batch(es) from buffer(s).
        
        Args:
            agent_id: If specified, sample only for this agent
            
        Returns:
            List of batches or single batch if agent_id specified
        """
        if agent_id is not None:
            return self.buffers[agent_id].sample()
        
        return [buffer.sample() for buffer in self.buffers]
    
    def __len__(self) -> int:
        """Return size of each buffer."""
        return len(self.buffers[0])
    
    def is_ready(self, min_samples: int) -> bool:
        """Check if all buffers are ready."""
        return all(buffer.is_ready(min_samples) for buffer in self.buffers)
    
    def clear(self):
        """Clear all buffers."""
        for buffer in self.buffers:
            buffer.clear()