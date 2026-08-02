"""
Tests for Reward Functions

These tests verify the various reward function implementations.
"""

import pytest
import numpy as np
from envs.reward_functions import (
    default_reward,
    efficiency_reward,
    safety_reward,
    multi_objective_reward,
    queue_penalty_reward,
    delay_reward,
    CompositeReward,
    compute_pareto_weights,
    get_reward_function,
)


class TestRewardFunctions:
    """Tests for individual reward functions."""
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        return {
            "phase": 0,
            "min_green_elapsed": 1.0,
            "lanes": {
                "lane_0": {
                    "raw_queue": 5,
                    "raw_speed": 8.0,
                    "raw_waiting_time": 30.0,
                    "raw_vehicle_count": 10,
                    "queue": 0.5,
                    "speed": 0.4,
                    "waiting_time": 0.3,
                    "vehicle_count": 1.0,
                },
                "lane_1": {
                    "raw_queue": 3,
                    "raw_speed": 12.0,
                    "raw_waiting_time": 15.0,
                    "raw_vehicle_count": 8,
                    "queue": 0.3,
                    "speed": 0.6,
                    "waiting_time": 0.15,
                    "vehicle_count": 0.8,
                },
            },
            "workzone": {
                "spillback_risk": 0.0,
                "merge_conflicts": 0,
                "workzone_queue": 2,
            },
            "safety": {
                "ttc_conflicts": 0,
                "avg_ttc": 10.0,
                "drac_conflicts": 0,
            },
        }
    
    def test_default_reward_returns_float(self, sample_state):
        """Test default reward returns a float."""
        reward = default_reward(sample_state)
        assert isinstance(reward, (int, float))
    
    def test_default_reward_is_finite(self, sample_state):
        """Test default reward is finite."""
        reward = default_reward(sample_state)
        assert np.isfinite(reward)
    
    def test_default_reward_empty_lanes(self):
        """Test default reward with no lanes."""
        state = {
            "lanes": {},
            "workzone": {},
            "safety": {},
        }
        reward = default_reward(state)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_efficiency_reward(self, sample_state):
        """Test efficiency reward."""
        reward = efficiency_reward(sample_state)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_safety_reward_with_conflicts(self):
        """Test safety reward penalizes conflicts."""
        state = {
            "lanes": {},
            "safety": {
                "ttc_conflicts": 5,
                "avg_ttc": 0.5,
            },
            "workzone": {
                "spillback_risk": 0.0,
                "merge_conflicts": 0,
            },
        }
        reward = safety_reward(state)
        # Should have strong negative penalty
        assert reward < 0
    
    def test_safety_reward_no_conflicts(self):
        """Test safety reward with no conflicts."""
        state = {
            "lanes": {},
            "safety": {
                "ttc_conflicts": 0,
                "avg_ttc": 10.0,
            },
            "workzone": {
                "spillback_risk": 0.0,
                "merge_conflicts": 0,
            },
        }
        reward = safety_reward(state)
        # Should be positive (rewarding high TTC)
        assert reward > 0
    
    def test_queue_penalty_reward(self, sample_state):
        """Test queue penalty is negative sum of queues."""
        reward = queue_penalty_reward(sample_state)
        # 5 + 3 = 8, so reward should be -8
        assert reward == -8
    
    def test_queue_penalty_empty(self):
        """Test queue penalty with no queues."""
        state = {"lanes": {}}
        reward = queue_penalty_reward(state)
        assert reward == 0
    
    def test_delay_reward(self, sample_state):
        """Test delay reward."""
        reward = delay_reward(sample_state)
        # (30 + 15) / 100 = 0.45, so reward = -0.45
        assert abs(reward - (-0.45)) < 0.01
    
    def test_multi_objective_reward(self, sample_state):
        """Test multi-objective reward with custom weights."""
        weights = {
            "waiting_time": -0.2,
            "queue_length": -0.3,
            "speed": 0.2,
            "throughput": 0.01,
            "safety": -0.5,
            "spillback": -0.2,
        }
        reward = multi_objective_reward(sample_state, weights)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
    
    def test_multi_objective_default_weights(self, sample_state):
        """Test multi-objective with default weights."""
        reward = multi_objective_reward(sample_state, None)
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)


class TestCompositeReward:
    """Tests for CompositeReward class."""
    
    def test_initialization(self):
        """Test CompositeReward initialization."""
        reward_fn = CompositeReward(
            waiting_time_weight=-0.2,
            queue_length_weight=-0.3,
        )
        assert reward_fn.weights["waiting_time"] == -0.2
        assert reward_fn.weights["queue_length"] == -0.3
    
    def test_callable(self):
        """Test CompositeReward is callable."""
        reward_fn = CompositeReward()
        state = {
            "lanes": {
                "lane_0": {"raw_queue": 5, "raw_speed": 10.0, "raw_waiting_time": 20.0, "raw_vehicle_count": 5},
            },
            "workzone": {},
            "safety": {},
        }
        reward = reward_fn(state)
        assert isinstance(reward, (int, float))
    
    def test_update_weight(self):
        """Test updating reward weights."""
        reward_fn = CompositeReward(waiting_time_weight=-0.1)
        reward_fn.update_weight("waiting_time", -0.5)
        assert reward_fn.weights["waiting_time"] == -0.5
    
    def test_get_weights(self):
        """Test getting current weights."""
        reward_fn = CompositeReward(waiting_time_weight=-0.2)
        weights = reward_fn.get_weights()
        assert "waiting_time" in weights
        assert weights["waiting_time"] == -0.2


class TestParetoWeights:
    """Tests for Pareto weight computation."""
    
    def test_pareto_weights_efficiency_focused(self):
        """Test Pareto weights for efficiency-focused training."""
        weights = compute_pareto_weights(efficiency_weight=1.0, safety_weight=0.0)
        assert weights["waiting_time"] < 0
        assert weights["queue_length"] < 0
        assert weights["safety"] == 0
    
    def test_pareto_weights_safety_focused(self):
        """Test Pareto weights for safety-focused training."""
        weights = compute_pareto_weights(efficiency_weight=0.0, safety_weight=1.0)
        assert weights["safety"] < 0
        assert weights["spillback"] < 0
        assert weights["waiting_time"] == 0
    
    def test_pareto_weights_balanced(self):
        """Test Pareto weights for balanced training."""
        weights = compute_pareto_weights(efficiency_weight=1.0, safety_weight=1.0)
        assert weights["waiting_time"] < 0
        assert weights["safety"] < 0
        # Both should have same total magnitude
        assert abs(weights["waiting_time"]) == abs(weights["safety"])


class TestRewardFunctionRegistry:
    """Tests for reward function registry."""
    
    def test_get_reward_function_default(self):
        """Test getting default reward function."""
        fn = get_reward_function("default")
        assert fn == default_reward
    
    def test_get_reward_function_efficiency(self):
        """Test getting efficiency reward function."""
        fn = get_reward_function("efficiency")
        assert fn == efficiency_reward
    
    def test_get_reward_function_safety(self):
        """Test getting safety reward function."""
        fn = get_reward_function("safety")
        assert fn == safety_reward
    
    def test_get_reward_function_invalid(self):
        """Test getting invalid reward function raises error."""
        with pytest.raises(ValueError):
            get_reward_function("invalid_name")
    
    def test_all_reward_functions_exist(self):
        """Test all named reward functions exist."""
        from envs.reward_functions import REWARD_FUNCTIONS
        
        for name in ["default", "efficiency", "safety", "queue_penalty", "delay"]:
            assert name in REWARD_FUNCTIONS
            fn = REWARD_FUNCTIONS[name]
            assert callable(fn)


class TestRewardEdgeCases:
    """Tests for edge cases in reward functions."""
    
    def test_division_by_zero_in_waiting(self):
        """Test handling of zero vehicles in waiting time calculation."""
        state = {
            "lanes": {
                "lane_0": {
                    "raw_queue": 0,
                    "raw_speed": 0,
                    "raw_waiting_time": 0,
                    "raw_vehicle_count": 0,
                },
            },
            "workzone": {},
            "safety": {},
        }
        # Should not raise division by zero
        reward = default_reward(state)
        assert isinstance(reward, (int, float))
    
    def test_negative_ttc(self):
        """Test handling of negative TTC values."""
        state = {
            "lanes": {},
            "workzone": {},
            "safety": {
                "ttc_conflicts": -1,  # Invalid negative value
                "avg_ttc": -5.0,
            },
        }
        reward = safety_reward(state)
        assert isinstance(reward, (int, float))
    
    def test_large_values(self):
        """Test handling of very large values."""
        state = {
            "lanes": {
                "lane_0": {
                    "raw_queue": 10000,
                    "raw_speed": 100.0,
                    "raw_waiting_time": 100000.0,
                    "raw_vehicle_count": 1000,
                },
            },
            "workzone": {
                "spillback_risk": 1.0,
                "merge_conflicts": 100,
                "workzone_queue": 500,
            },
            "safety": {
                "ttc_conflicts": 100,
                "avg_ttc": 0.1,
            },
        }
        reward = default_reward(state)
        # Should handle large values without overflow
        assert np.isfinite(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])