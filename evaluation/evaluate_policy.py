"""
Policy Evaluation Script for SUMO Work Zone Traffic Control

This script evaluates trained RL agents against baselines (e.g., fixed-timing)
and generates performance reports and visualizations.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import yaml

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import SumoWorkZoneEnv, default_reward
from evaluation.metrics import (
    MetricsCollector,
    compute_performance_metrics,
    plot_training_curves,
    plot_comparison,
    print_evaluation_report,
)


class PolicyEvaluator:
    """
    Evaluator for traffic signal control policies.
    
    Supports evaluation of:
    - Trained DQN agents
    - Fixed-timing baselines
    - Random policies
    - SUMO default controllers
    """
    
    def __init__(
        self,
        net_file: str,
        route_file: str,
        tls_id: str = "TL1",
        delta_time: int = 5,
        min_green: int = 5,
    ):
        """
        Initialize evaluator.
        
        Args:
            net_file: Path to SUMO network file
            route_file: Path to route file
            tls_id: Traffic light system ID
            delta_time: Simulation step size
            min_green: Minimum green time
        """
        self.net_file = net_file
        self.route_file = route_file
        self.tls_id = tls_id
        self.delta_time = delta_time
        self.min_green = min_green
        
        # Metrics collector
        self.metrics_collector = MetricsCollector()
    
    def evaluate_dqn(
        self,
        model_path: str,
        num_episodes: int = 10,
        simulation_seconds: int = 1000,
        seeds: Optional[List[int]] = None,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate a trained DQN agent.
        
        Args:
            model_path: Path to saved DQN model
            num_episodes: Number of evaluation episodes
            simulation_seconds: Simulation duration per episode
            seeds: List of random seeds (one per episode)
            deterministic: Use deterministic policy (no exploration)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load agent
        try:
            from agents import DQNAgent
            agent = DQNAgent(state_dim=1, action_dim=2)  # Placeholder dims
            agent.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
        
        if seeds is None:
            seeds = [42 + i for i in range(num_episodes)]
        
        all_metrics = []
        
        for ep_idx, seed in enumerate(seeds):
            # Create environment
            env = SumoWorkZoneEnv(
                net_file=self.net_file,
                route_file=self.route_file,
                tls_id=self.tls_id,
                num_seconds=simulation_seconds,
                delta_time=self.delta_time,
                min_green=self.min_green,
                use_gui=False,
                single_agent=True,
                reward_fn=default_reward,
                sumo_seed=seed,
            )
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            rewards = []
            queue_lengths = []
            speeds = []
            waiting_times = []
            ttc_conflicts = 0
            
            while not (done or truncated):
                action, _ = agent.select_action(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                rewards.append(reward)
                
                # Collect metrics from state
                state = info.get("state", {})
                lane_data = state.get("lanes", {})
                
                total_queue = sum(m.get("raw_queue", 0) for m in lane_data.values())
                total_speed = sum(m.get("raw_speed", 0) for m in lane_data.values())
                total_waiting = sum(m.get("raw_waiting_time", 0) for m in lane_data.values())
                num_lanes = max(1, len(lane_data))
                
                queue_lengths.append(total_queue / num_lanes)
                speeds.append(total_speed / num_lanes)
                waiting_times.append(total_waiting / max(1, sum(m.get("raw_vehicle_count", 0) for m in lane_data.values())))
                
                safety = state.get("safety", {})
                ttc_conflicts += safety.get("ttc_conflicts", 0)
                
                step += 1
            
            env.close()
            
            # Compute episode metrics
            throughput = info.get("arrived_vehicles", 0)
            episode_metrics = compute_performance_metrics(
                rewards, queue_lengths, speeds, waiting_times, throughput, ttc_conflicts
            )
            episode_metrics["episode"] = ep_idx
            all_metrics.append(episode_metrics)
        
        # Aggregate metrics
        return self._aggregate_metrics(all_metrics)
    
    def evaluate_fixed_timing(
        self,
        num_episodes: int = 10,
        simulation_seconds: int = 1000,
        seeds: Optional[List[int]] = None,
        phase_duration: int = 30,
    ) -> Dict[str, float]:
        """
        Evaluate fixed-timing baseline policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            simulation_seconds: Simulation duration per episode
            seeds: List of random seeds
            phase_duration: Duration for each phase
            
        Returns:
            Dictionary of evaluation metrics
        """
        if seeds is None:
            seeds = [42 + i for i in range(num_episodes)]
        
        all_metrics = []
        
        for ep_idx, seed in enumerate(seeds):
            # Create environment
            env = SumoWorkZoneEnv(
                net_file=self.net_file,
                route_file=self.route_file,
                tls_id=self.tls_id,
                num_seconds=simulation_seconds,
                delta_time=self.delta_time,
                min_green=phase_duration,  # Use phase_duration as min_green
                use_gui=False,
                single_agent=True,
                reward_fn=default_reward,
                sumo_seed=seed,
            )
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            rewards = []
            queue_lengths = []
            speeds = []
            waiting_times = []
            ttc_conflicts = 0
            steps_in_phase = 0
            current_phase = 0
            
            while not (done or truncated):
                # Fixed timing: switch phase every phase_duration
                if steps_in_phase >= phase_duration // self.delta_time:
                    current_phase = 1 - current_phase  # Toggle between 0 and 1
                    steps_in_phase = 0
                
                action = current_phase
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                rewards.append(reward)
                
                # Collect metrics
                state = info.get("state", {})
                lane_data = state.get("lanes", {})
                
                total_queue = sum(m.get("raw_queue", 0) for m in lane_data.values())
                total_speed = sum(m.get("raw_speed", 0) for m in lane_data.values())
                total_waiting = sum(m.get("raw_waiting_time", 0) for m in lane_data.values())
                num_lanes = max(1, len(lane_data))
                
                queue_lengths.append(total_queue / num_lanes)
                speeds.append(total_speed / num_lanes)
                waiting_times.append(total_waiting / max(1, sum(m.get("raw_vehicle_count", 0) for m in lane_data.values())))
                
                safety = state.get("safety", {})
                ttc_conflicts += safety.get("ttc_conflicts", 0)
                
                step += 1
                steps_in_phase += 1
            
            env.close()
            
            throughput = info.get("arrived_vehicles", 0)
            episode_metrics = compute_performance_metrics(
                rewards, queue_lengths, speeds, waiting_times, throughput, ttc_conflicts
            )
            episode_metrics["episode"] = ep_idx
            all_metrics.append(episode_metrics)
        
        return self._aggregate_metrics(all_metrics)
    
    def evaluate_random(
        self,
        num_episodes: int = 10,
        simulation_seconds: int = 1000,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate random policy baseline.
        
        Args:
            num_episodes: Number of evaluation episodes
            simulation_seconds: Simulation duration per episode
            seeds: List of random seeds
            
        Returns:
            Dictionary of evaluation metrics
        """
        if seeds is None:
            seeds = [42 + i for i in range(num_episodes)]
        
        all_metrics = []
        
        for ep_idx, seed in enumerate(seeds):
            np.random.seed(seed)
            
            env = SumoWorkZoneEnv(
                net_file=self.net_file,
                route_file=self.route_file,
                tls_id=self.tls_id,
                num_seconds=simulation_seconds,
                delta_time=self.delta_time,
                min_green=self.min_green,
                use_gui=False,
                single_agent=True,
                reward_fn=default_reward,
                sumo_seed=seed,
            )
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            step = 0
            
            rewards = []
            queue_lengths = []
            speeds = []
            waiting_times = []
            ttc_conflicts = 0
            
            while not (done or truncated):
                action = np.random.randint(0, 2)  # Random action
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                rewards.append(reward)
                
                state = info.get("state", {})
                lane_data = state.get("lanes", {})
                
                total_queue = sum(m.get("raw_queue", 0) for m in lane_data.values())
                total_speed = sum(m.get("raw_speed", 0) for m in lane_data.values())
                total_waiting = sum(m.get("raw_waiting_time", 0) for m in lane_data.values())
                num_lanes = max(1, len(lane_data))
                
                queue_lengths.append(total_queue / num_lanes)
                speeds.append(total_speed / num_lanes)
                waiting_times.append(total_waiting / max(1, sum(m.get("raw_vehicle_count", 0) for m in lane_data.values())))
                
                safety = state.get("safety", {})
                ttc_conflicts += safety.get("ttc_conflicts", 0)
                
                step += 1
            
            env.close()
            
            throughput = info.get("arrived_vehicles", 0)
            episode_metrics = compute_performance_metrics(
                rewards, queue_lengths, speeds, waiting_times, throughput, ttc_conflicts
            )
            episode_metrics["episode"] = ep_idx
            all_metrics.append(episode_metrics)
        
        return self._aggregate_metrics(all_metrics)
    
    def _aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across episodes."""
        if not all_metrics:
            return {}
        
        aggregated = {}
        keys = [k for k in all_metrics[0].keys() if k != "episode"]
        
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
            aggregated[f"min_{key}"] = np.min(values)
            aggregated[f"max_{key}"] = np.max(values)
        
        return aggregated
    
    def compare_policies(
        self,
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        simulation_seconds: int = 1000,
        output_dir: str = "outputs/evaluation",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare trained DQN agent with baselines.
        
        Args:
            model_path: Path to trained DQN model
            num_episodes: Number of evaluation episodes per policy
            simulation_seconds: Simulation duration per episode
            output_dir: Directory for output files
            
        Returns:
            Dictionary mapping policy names to their metrics
        """
        results = {}
        
        print("\n" + "=" * 60)
        print("Evaluating Fixed-Timing Baseline...")
        print("=" * 60)
        results["Fixed-Timing"] = self.evaluate_fixed_timing(
            num_episodes=num_episodes,
            simulation_seconds=simulation_seconds,
        )
        
        print("\n" + "=" * 60)
        print("Evaluating Random Policy...")
        print("=" * 60)
        results["Random"] = self.evaluate_random(
            num_episodes=num_episodes,
            simulation_seconds=simulation_seconds,
        )
        
        if model_path and os.path.exists(model_path):
            print("\n" + "=" * 60)
            print("Evaluating Trained DQN Agent...")
            print("=" * 60)
            results["DQN"] = self.evaluate_dqn(
                model_path=model_path,
                num_episodes=num_episodes,
                simulation_seconds=simulation_seconds,
            )
        
        # Generate comparison plot
        os.makedirs(output_dir, exist_ok=True)
        
        # Create simplified results for plotting
        plot_results = {}
        for name, metrics in results.items():
            plot_results[name] = {
                "avg_queue_length": metrics.get("avg_avg_queue", 0),
                "avg_speed": metrics.get("avg_avg_speed", 0),
                "avg_waiting_time": metrics.get("avg_avg_waiting", 0),
                "throughput": metrics.get("avg_throughput", 0),
                "ttc_conflicts": metrics.get("avg_ttc_conflicts", 0),
                "total_reward": metrics.get("avg_total_reward", 0),
            }
        
        plot_path = os.path.join(output_dir, "policy_comparison.png")
        plot_comparison(plot_results, save_path=plot_path)
        
        # Save results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        
        return results


def evaluate_policy(
    model_path: str,
    net_file: str = "data/networks/workzone/net.net.xml",
    route_file: str = "data/networks/workzone/rou.route.xml",
    num_episodes: int = 10,
    simulation_seconds: int = 1000,
    output_dir: str = "outputs/evaluation",
) -> Dict[str, float]:
    """
    Evaluate a trained policy.
    
    Args:
        model_path: Path to trained model
        net_file: Path to SUMO network file
        route_file: Path to route file
        num_episodes: Number of evaluation episodes
        simulation_seconds: Simulation duration per episode
        output_dir: Directory for outputs
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = PolicyEvaluator(
        net_file=net_file,
        route_file=route_file,
    )
    
    results = evaluator.compare_policies(
        model_path=model_path,
        num_episodes=num_episodes,
        simulation_seconds=simulation_seconds,
        output_dir=output_dir,
    )
    
    # Print reports
    for policy_name, metrics in results.items():
        print_evaluation_report(policy_name, metrics)
    
    return results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate traffic signal control policies")
    
    # Model path
    parser.add_argument("--model", type=str, help="Path to trained DQN model")
    parser.add_argument("--no-model", action="store_true", help="Evaluate only baselines")
    
    # Paths
    parser.add_argument("--net-file", type=str, default="data/networks/workzone/net.net.xml")
    parser.add_argument("--route-file", type=str, default="data/networks/workzone/rou.route.xml")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation")
    
    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--simulation-seconds", type=int, default=1000, help="Simulation duration")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    
    args = parser.parse_args()
    
    model_path = None if args.no_model else args.model
    
    results = evaluate_policy(
        model_path=model_path,
        net_file=args.net_file,
        route_file=args.route_file,
        num_episodes=args.episodes,
        simulation_seconds=args.simulation_seconds,
        output_dir=args.output_dir,
    )
    
    print("\n" + "=" * 60)
    print("Evaluation complete! Results saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()