"""
Tests for SUMO Environment

These tests verify the environment wrapper and basic functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock SUMO before importing env module
import sys
sys.modules['traci'] = MagicMock()
sys.modules['sumolib'] = MagicMock()


class TestSumoWorkZoneEnv:
    """Tests for SumoWorkZoneEnv class."""
    
    def test_env_initialization(self):
        """Test environment initializes with correct parameters."""
        from envs.sumo_env import SumoWorkZoneEnv
        
        # Mock the network
        mock_net = MagicMock()
        mock_lane = MagicMock()
        mock_lane.getID.return_value = "test_lane_0"
        mock_lane.getLength.return_value = 100.0
        mock_lane.getConnections.return_value = []
        mock_net.getEdges.return_value = []
        mock_net.getTLS.return_value = MagicMock(getConnections=lambda: [])
        
        with patch('sumolib.net.readNet', return_value=mock_net):
            env = SumoWorkZoneEnv(
                net_file="test.net.xml",
                route_file="test.rou.xml",
                tls_id="TL1",
                num_seconds=1000,
                delta_time=5,
            )
            
            assert env.tls_id == "TL1"
            assert env.num_seconds == 1000
            assert env.delta_time == 5
            assert env.observation_space is not None
            assert env.action_space is not None
    
    def test_action_space(self):
        """Test action space is discrete with 2 actions."""
        from envs.sumo_env import SumoWorkZoneEnv
        from gymnasium import spaces
        
        mock_net = MagicMock()
        mock_net.getTLS.return_value = MagicMock(getConnections=lambda: [])
        mock_net.getEdges.return_value = []
        
        with patch('sumolib.net.readNet', return_value=mock_net):
            env = SumoWorkZoneEnv(
                net_file="test.net.xml",
                route_file="test.rou.xml",
            )
            
            assert isinstance(env.action_space, spaces.Discrete)
            assert env.action_space.n == 2  # Keep or switch
    
    def test_observation_space(self):
        """Test observation space is a Box."""
        from envs.sumo_env import SumoWorkZoneEnv
        from gymnasium import spaces
        
        mock_net = MagicMock()
        mock_net.getTLS.return_value = MagicMock(getConnections=lambda: [])
        mock_net.getEdges.return_value = []
        
        with patch('sumolib.net.readNet', return_value=mock_net):
            env = SumoWorkZoneEnv(
                net_file="test.net.xml",
                route_file="test.rou.xml",
            )
            
            assert isinstance(env.observation_space, spaces.Box)
            assert env.observation_space.dtype == np.float32


class TestRewardFunctions:
    """Tests for reward functions."""
    
    def test_default_reward_computation(self):
        """Test default reward function computes valid values."""
        from envs.reward_functions import default_reward
        
        # Create a mock state
        state = {
            "phase": 0,
            "min_green_elapsed": 1.0,
            "lanes": {
                "lane_0": {
                    "raw_queue": 2,
                    "raw_speed": 5.0,
                    "raw_waiting_time": 10.0,
                    "raw_vehicle_count": 5,
                    "queue": 0.2,
                    "speed": 0.25,
                    "waiting_time": 0.1,
                    "vehicle_count": 0.5,
                },
                "lane_1": {
                    "raw_queue": 1,
                    "raw_speed": 8.0,
                    "raw_waiting_time": 5.0,
                    "raw_vehicle_count": 3,
                    "queue": 0.1,
                    "speed": 0.4,
                    "waiting_time": 0.05,
                    "vehicle_count": 0.3,
                },
            },
            "workzone": {
                "spillback_risk": 0.0,
                "merge_conflicts": 0,
                "workzone_queue": 0,
            },
            "safety": {
                "ttc_conflicts": 0,
                "avg_ttc": 10.0,
            },
        }
        
        reward = default_reward(state)
        
        # Reward should be a finite number
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_efficiency_reward(self):
        """Test efficiency reward function."""
        from envs.reward_functions import efficiency_reward
        
        state = {
            "lanes": {
                "lane_0": {
                    "raw_queue": 0,
                    "raw_speed": 15.0,
                    "raw_waiting_time": 0.0,
                    "raw_vehicle_count": 5,
                },
            },
        }
        
        reward = efficiency_reward(state)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_safety_reward(self):
        """Test safety reward function."""
        from envs.reward_functions import safety_reward
        
        state = {
            "lanes": {},
            "safety": {
                "ttc_conflicts": 2,
                "avg_ttc": 0.8,
            },
            "workzone": {
                "spillback_risk": 0.0,
                "merge_conflicts": 0,
            },
        }
        
        reward = safety_reward(state)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
        # Should have negative reward due to TTC conflicts
        assert reward < 0
    
    def test_queue_penalty_reward(self):
        """Test simple queue penalty reward."""
        from envs.reward_functions import queue_penalty_reward
        
        state = {
            "lanes": {
                "lane_0": {"raw_queue": 5},
                "lane_1": {"raw_queue": 3},
            },
        }
        
        reward = queue_penalty_reward(state)
        assert reward == -8  # Negative sum of queues


class TestObservationFunctions:
    """Tests for observation functions."""
    
    def test_default_observation(self):
        """Test default observation function."""
        from envs.observations import default_observation
        
        # Create mock environment
        mock_env = Mock()
        mock_env.num_phases = 2
        mock_env.num_lanes = 2
        mock_env.lanes = ["lane_0", "lane_1"]
        
        state = {
            "phase": 0,
            "min_green_elapsed": 1.0,
            "lanes": {
                "lane_0": {
                    "queue": 0.2,
                    "speed": 0.5,
                    "waiting_time": 0.1,
                    "vehicle_count": 0.5,
                },
                "lane_1": {
                    "queue": 0.1,
                    "speed": 0.3,
                    "waiting_time": 0.05,
                    "vehicle_count": 0.3,
                },
            },
        }
        
        obs = default_observation(state, mock_env)
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        # Should have: 2 (phase one-hot) + 1 (min_green) + 2*4 (lane features) = 11
        assert len(obs) == 11
    
    def test_minimal_observation(self):
        """Test minimal observation function."""
        from envs.observations import minimal_observation
        
        mock_env = Mock()
        mock_env.num_phases = 2
        mock_env.lanes = ["lane_0"]
        
        state = {
            "phase": 1,
            "lanes": {
                "lane_0": {
                    "queue": 0.5,
                    "speed": 0.3,
                    "vehicle_count": 2,
                },
            },
        }
        
        obs = minimal_observation(state, mock_env)
        
        assert len(obs) == 4  # 2 phase + queue + speed
    
    def test_extended_observation(self):
        """Test extended observation includes work zone and safety."""
        from envs.observations import extended_observation
        
        mock_env = Mock()
        mock_env.num_phases = 2
        mock_env.num_lanes = 1
        mock_env.lanes = ["lane_0"]
        
        state = {
            "phase": 0,
            "min_green_elapsed": 1.0,
            "lanes": {
                "lane_0": {
                    "queue": 0.2,
                    "speed": 0.5,
                    "waiting_time": 0.1,
                    "vehicle_count": 0.5,
                },
            },
            "workzone": {
                "spillback_risk": 0.5,
                "merge_conflicts": 1,
                "workzone_queue": 5,
            },
            "safety": {
                "ttc_conflicts": 2,
                "avg_ttc": 1.0,
                "drac_conflicts": 1,
            },
        }
        
        obs = extended_observation(state, mock_env)
        
        # Should be longer than default
        from envs.observations import default_observation
        default_obs = default_observation(state, mock_env)
        assert len(obs) > len(default_obs)


class TestConfigurableObservation:
    """Tests for ConfigurableObservation class."""
    
    def test_phase_only(self):
        """Test observation with only phase encoding."""
        from envs.observations import ConfigurableObservation
        
        mock_env = Mock()
        mock_env.num_phases = 4
        mock_env.num_lanes = 0
        mock_env.lanes = []
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = (4,)
        
        obs_fn = ConfigurableObservation(
            include_phase=True,
            include_min_green=False,
            include_queue=False,
            include_speed=False,
            include_waiting=False,
            include_vehicle_count=False,
        )
        
        state = {"phase": 2, "min_green_elapsed": 0.0, "lanes": {}}
        obs = obs_fn(state, mock_env)
        
        assert len(obs) == 4  # Just phase one-hot
    
    def test_output_dimension(self):
        """Test output dimension calculation."""
        from envs.observations import ConfigurableObservation
        
        mock_env = Mock()
        mock_env.num_phases = 2
        mock_env.num_lanes = 4
        
        obs_fn = ConfigurableObservation(
            include_phase=True,
            include_min_green=True,
            include_queue=True,
            include_speed=True,
            include_waiting=False,
            include_vehicle_count=False,
        )
        
        dim = obs_fn.get_output_dim(mock_env)
        # 2 (phase) + 1 (min_green) + 4 lanes * 2 features (queue, speed) = 11
        assert dim == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])