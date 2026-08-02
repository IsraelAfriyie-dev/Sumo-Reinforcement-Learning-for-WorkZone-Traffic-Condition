"""
DQN Agent for SUMO Work Zone Traffic Control

This module implements a Deep Q-Network (DQN) agent for traffic signal control.
It uses experience replay and target networks for stable learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import copy

from training.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 64
    target_update_frequency: int = 1000
    replay_buffer_size: int = 100000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    gradient_steps: int = 1
    learning_starts: int = 500
    hidden_layers: List[int] = None
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128]


class DQNNetwork(nn.Module):
    """
    Deep Q-Network neural network.
    
    Architecture:
    - Input: state dimension
    - Hidden layers: configurable sizes with ReLU activation
    - Output: number of actions (Q-values for each action)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int] = None,
        activation: str = "relu",
    ):
        """
        Initialize DQN network.
        
        Args:
            input_dim: Dimension of input state
            output_dim: Number of possible actions
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = self.network(x)
        return self.output(x)


class DQNPolicy:
    """
    DQN Policy class for action selection.
    
    Provides epsilon-greedy action selection.
    """
    
    def __init__(
        self,
        q_network: DQNNetwork,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        exploration_fraction: float = 0.1,
        device: str = "auto",
    ):
        """
        Initialize DQN policy.
        
        Args:
            q_network: The Q-network
            exploration_initial_eps: Initial exploration rate
            exploration_final_eps: Final exploration rate
            exploration_fraction: Fraction of training for exploration decay
            device: Device to run on ("auto", "cpu", or "cuda")
        """
        self.q_network = q_network
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.q_network.to(self.device)
        
        # Exploration settings
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        
        self.total_timesteps = 0
    
    def get_epsilon(self, current_step: int, total_timesteps: int) -> float:
        """
        Compute current exploration rate.
        
        Uses linear decay from exploration_initial_eps to exploration_final_eps
        over exploration_fraction of total training.
        
        Args:
            current_step: Current training step
            total_timesteps: Total planned training steps
            
        Returns:
            Current epsilon value
        """
        exploration_steps = total_timesteps * self.exploration_fraction
        
        if current_step >= exploration_steps:
            return self.exploration_final_eps
        
        # Linear decay
        ratio = current_step / max(1, exploration_steps)
        return self.exploration_initial_eps - ratio * (
            self.exploration_initial_eps - self.exploration_final_eps
        )
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (numpy array)
            epsilon: Exploration rate (if None, uses internal calculation)
            deterministic: If True, always select best action (no exploration)
            
        Returns:
            Tuple of (selected_action, info_dict)
        """
        if deterministic:
            return self._get_best_action(state), {"policy": "greedy"}
        
        if epsilon is None:
            epsilon = self.exploration_final_eps
        
        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            # Random action
            action = np.random.randint(0, self.q_network.output.out_features)
            info = {"policy": "exploration", "epsilon": epsilon}
        else:
            # Greedy action
            action = self._get_best_action(state)
            info = {"policy": "greedy", "epsilon": epsilon}
        
        return action, info
    
    def _get_best_action(self, state: np.ndarray) -> int:
        """Get the action with highest Q-value."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax(dim=1).item()


class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for traffic signal control.
    
    This agent implements:
    - Experience replay for stable learning
    - Target network for TD learning
    - Epsilon-greedy exploration
    - Configurable neural network architecture
    
    Attributes:
        q_network: Main Q-network
        target_network: Target Q-network
        policy: Policy for action selection
        replay_buffer: Experience replay buffer
        optimizer: Optimizer for Q-network
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[DQNConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: DQN configuration (uses defaults if None)
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Use default config if not provided
        if config is None:
            config = DQNConfig()
        self.config = config
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.q_network = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=config.hidden_layers,
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_layers=config.hidden_layers,
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize policy
        self.policy = DQNPolicy(
            q_network=self.q_network,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            exploration_fraction=config.exploration_fraction,
            device=self.device,
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_size,
            batch_size=config.batch_size,
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
        )
        
        # Training state
        self.total_timesteps = 0
        self.training_step = 0
        self.update_step = 0
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """
        Select action for given state.
        
        Args:
            state: Current state
            deterministic: If True, use greedy policy
            
        Returns:
            Tuple of (action, info_dict)
        """
        return self.policy.select_action(
            state,
            deterministic=deterministic,
        )
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_timesteps += 1
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training was performed, None otherwise
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.config.learning_starts:
            return None
        
        # Check if we should train this step
        if self.training_step % self.config.gradient_steps != 0:
            self.training_step += 1
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            # Double DQN: use online network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.config.discount_factor * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network if needed
        self.update_step += 1
        if self.update_step % self.config.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
        
        return loss.item()
    
    def update_epsilon(self, total_timesteps: int) -> float:
        """
        Update and return current exploration rate.
        
        Args:
            total_timesteps: Total training steps planned
            
        Returns:
            Current epsilon value
        """
        return self.policy.get_epsilon(
            self.total_timesteps,
            total_timesteps,
        )
    
    def save(self, path: str):
        """
        Save agent state to file.
        
        Args:
            path: Path to save file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "total_timesteps": self.total_timesteps,
            "training_step": self.training_step,
            "update_step": self.update_step,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """
        Load agent state from file.
        
        Args:
            path: Path to saved file
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_timesteps = checkpoint.get("total_timesteps", 0)
        self.training_step = checkpoint.get("training_step", 0)
        self.update_step = checkpoint.get("update_step", 0)
        
        # Update policy's network reference
        self.policy.q_network = self.q_network
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in given state.
        
        Args:
            state: Input state
            
        Returns:
            Array of Q-values for each action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy()[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_timesteps": self.total_timesteps,
            "training_step": self.training_step,
            "update_step": self.update_step,
            "replay_buffer_size": len(self.replay_buffer),
            "device": str(self.device),
            "learning_rate": self.config.learning_rate,
            "epsilon": self.policy.get_epsilon(self.total_timesteps, self.total_timesteps),
        }


class MultiAgentDQN:
    """
    Multi-agent DQN for controlling multiple traffic lights.
    
    Each traffic light is controlled by its own DQN agent,
    but they share the same network architecture and training logic.
    """
    
    def __init__(
        self,
        state_dims: List[int],
        action_dim: int,
        config: Optional[DQNConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize multi-agent DQN.
        
        Args:
            state_dims: List of state dimensions for each agent
            action_dim: Action dimension (same for all agents)
            config: DQN configuration
            device: Device to run on
        """
        self.num_agents = len(state_dims)
        self.action_dim = action_dim
        
        # Create one agent per traffic light
        self.agents = [
            DQNAgent(state_dim, action_dim, config, device)
            for state_dim in state_dims
        ]
        
        self.device = self.agents[0].device
    
    def select_actions(
        self,
        states: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[List[int], Dict]:
        """
        Select actions for all agents.
        
        Args:
            states: List of states, one per agent
            deterministic: If True, use greedy policy
            
        Returns:
            Tuple of (list of actions, info dict)
        """
        actions = []
        infos = []
        
        for i, state in enumerate(states):
            action, info = self.agents[i].select_action(state, deterministic)
            actions.append(action)
            infos.append(info)
        
        return actions, {"agents": infos}
    
    def store_transitions(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
    ):
        """
        Store transitions for all agents.
        """
        for i in range(self.num_agents):
            self.agents[i].store_transition(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
    
    def train_step(self) -> Optional[float]:
        """
        Perform training step for all agents.
        
        Returns:
            Average loss across all agents
        """
        losses = []
        for agent in self.agents:
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        return np.mean(losses) if losses else None
    
    def save(self, path: str):
        """Save all agents."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            agent.save(path.replace(".pt", f"_agent_{i}.pt"))
    
    def load(self, path: str):
        """Load all agents."""
        for i in range(self.num_agents):
            agent_path = path.replace(".pt", f"_agent_{i}.pt")
            if os.path.exists(agent_path):
                self.agents[i].load(agent_path)