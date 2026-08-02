"""
Tests for DQN Agent

These tests verify the DQN agent implementation.
"""

import pytest
import numpy as np
import torch
import tempfile
import os

from agents.dqn_agent import DQNAgent, DQNConfig, DQNNetwork, DQNPolicy


class TestDQNNetwork:
    """Tests for DQN neural network."""
    
    def test_network_forward(self):
        """Test network forward pass."""
        network = DQNNetwork(input_dim=10, output_dim=2, hidden_layers=[32, 16])
        
        x = torch.randn(5, 10)
        output = network(x)
        
        assert output.shape == (5, 2)
    
    def test_network_output_dim(self):
        """Test network produces correct output dimension."""
        network = DQNNetwork(input_dim=20, output_dim=4)
        
        x = torch.randn(1, 20)
        output = network(x)
        
        assert output.shape == (1, 4)
    
    def test_network_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "tanh", "leaky_relu"]:
            network = DQNNetwork(
                input_dim=8, 
                output_dim=2, 
                hidden_layers=[16],
                activation=activation
            )
            
            x = torch.randn(2, 8)
            output = network(x)
            
            assert output.shape == (2, 2)
            assert not torch.isnan(output).any()


class TestDQNPolicy:
    """Tests for DQN policy."""
    
    def test_greedy_action_selection(self):
        """Test greedy action selection."""
        network = DQNNetwork(input_dim=10, output_dim=2)
        policy = DQNPolicy(network, exploration_final_eps=0.0)
        
        state = np.random.randn(10)
        action, info = policy.select_action(state, deterministic=True)
        
        assert action in [0, 1]
        assert info["policy"] == "greedy"
    
    def test_exploration_action_selection(self):
        """Test exploration behavior."""
        network = DQNNetwork(input_dim=10, output_dim=2)
        policy = DQNPolicy(
            network, 
            exploration_initial_eps=1.0,
            exploration_final_eps=0.0
        )
        
        # With epsilon=1.0, should select random action
        state = np.random.randn(10)
        actions = [policy.select_action(state, epsilon=1.0)[0] for _ in range(20)]
        
        # Should have some variation (not all same)
        assert len(set(actions)) > 1
    
    def test_epsilon_decay(self):
        """Test epsilon decay over training."""
        network = DQNNetwork(input_dim=10, output_dim=2)
        policy = DQNPolicy(
            network,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            exploration_fraction=0.1
        )
        
        total_steps = 1000
        exploration_steps = total_steps * 0.1
        
        # At start
        eps_start = policy.get_epsilon(0, total_steps)
        assert eps_start == 1.0
        
        # At end of exploration
        eps_mid = policy.get_epsilon(int(exploration_steps), total_steps)
        assert eps_mid == 0.1
        
        # After exploration
        eps_end = policy.get_epsilon(total_steps, total_steps)
        assert eps_end == 0.1


class TestDQNAgent:
    """Tests for DQN agent."""
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = DQNAgent(state_dim=10, action_dim=2)
        
        assert agent.state_dim == 10
        assert agent.action_dim == 2
        assert agent.q_network is not None
        assert agent.target_network is not None
        assert agent.replay_buffer is not None
    
    def test_agent_action_selection(self):
        """Test agent can select actions."""
        agent = DQNAgent(state_dim=10, action_dim=2)
        
        state = np.random.randn(10)
        action, info = agent.select_action(state)
        
        assert action in [0, 1]
        assert "policy" in info
    
    def test_agent_store_transition(self):
        """Test agent stores transitions."""
        agent = DQNAgent(state_dim=10, action_dim=2)
        
        state = np.random.randn(10)
        action = 0
        reward = 1.0
        next_state = np.random.randn(10)
        done = False
        
        initial_size = len(agent.replay_buffer)
        agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.replay_buffer) == initial_size + 1
    
    def test_agent_training_step(self):
        """Test agent training step."""
        config = DQNConfig(learning_starts=10, batch_size=4)
        agent = DQNAgent(state_dim=10, action_dim=2, config=config)
        
        # Fill replay buffer
        for _ in range(15):
            state = np.random.randn(10)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = np.random.rand() > 0.9
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Training step
        loss = agent.train_step()
        
        # Loss should be a finite number
        assert loss is not None
        assert np.isfinite(loss)
    
    def test_agent_save_load(self):
        """Test agent save and load."""
        agent1 = DQNAgent(state_dim=10, action_dim=2)
        
        # Store some transitions
        for _ in range(50):
            agent1.store_transition(
                np.random.randn(10),
                np.random.randint(0, 2),
                np.random.randn(),
                np.random.randn(10),
                False
            )
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        
        try:
            agent1.save(path)
            
            # Load into new agent
            agent2 = DQNAgent(state_dim=10, action_dim=2)
            agent2.load(path)
            
            # Verify weights match
            for p1, p2 in zip(agent1.q_network.parameters(), agent2.q_network.parameters()):
                assert torch.allclose(p1, p2)
            
            assert agent2.total_timesteps == agent1.total_timesteps
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_agent_q_values(self):
        """Test agent returns Q-values."""
        agent = DQNAgent(state_dim=10, action_dim=2)
        
        state = np.random.randn(10)
        q_values = agent.get_q_values(state)
        
        assert q_values.shape == (2,)
        assert np.isfinite(q_values).all()
    
    def test_agent_statistics(self):
        """Test agent statistics."""
        agent = DQNAgent(state_dim=10, action_dim=2)
        
        # Store some transitions
        for _ in range(20):
            agent.store_transition(
                np.random.randn(10),
                np.random.randint(0, 2),
                np.random.randn(),
                np.random.randn(10),
                False
            )
        
        stats = agent.get_statistics()
        
        assert "total_timesteps" in stats
        assert "replay_buffer_size" in stats
        assert "device" in stats
        assert stats["replay_buffer_size"] == 20


class TestReplayBuffer:
    """Tests for replay buffer."""
    
    def test_buffer_push_and_sample(self):
        """Test buffer push and sample."""
        from training.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100, batch_size=4)
        
        # Push transitions
        for i in range(20):
            buffer.push(
                np.array([float(i)] * 10),
                i % 2,
                float(i),
                np.array([float(i + 1)] * 10),
                False
            )
        
        assert len(buffer) == 20
        
        # Sample
        states, actions, rewards, next_states, dones = buffer.sample()
        
        assert states.shape == (4, 10)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)
        assert next_states.shape == (4, 10)
        assert dones.shape == (4,)
    
    def test_buffer_capacity(self):
        """Test buffer respects capacity."""
        from training.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=10, batch_size=2)
        
        # Push more than capacity
        for i in range(20):
            buffer.push(
                np.array([i]),
                0,
                1.0,
                np.array([i + 1]),
                False
            )
        
        # Should be at capacity
        assert len(buffer) == 10
    
    def test_buffer_clear(self):
        """Test buffer clear."""
        from training.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=10, batch_size=2)
        
        for i in range(5):
            buffer.push(np.array([i]), 0, 1.0, np.array([i + 1]), False)
        
        assert len(buffer) == 5
        
        buffer.clear()
        
        assert len(buffer) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])