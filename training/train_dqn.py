"""
DQN Training Script for SUMO Work Zone Traffic Control

This script provides a complete training loop for the DQN agent,
including logging, checkpointing, and evaluation.
"""

import os
import sys
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import yaml

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import SumoWorkZoneEnv, default_reward
from agents import DQNAgent
from agents.dqn_agent import DQNConfig


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Paths
    net_file: str = "data/networks/workzone/net.net.xml"
    route_file: str = "data/networks/workzone/rou.route.xml"
    output_dir: str = "outputs"
    
    # Simulation
    num_episodes: int = 500
    max_steps_per_episode: int = 1000
    delta_time: int = 5
    min_green: int = 5
    sumo_seed: int = 42
    
    # DQN hyperparameters
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 64
    target_update_frequency: int = 1000
    replay_buffer_size: int = 100000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    learning_starts: int = 500
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    
    # Logging and saving
    save_frequency: int = 50
    log_frequency: int = 10
    eval_frequency: int = 50
    eval_episodes: int = 5
    tensorboard: bool = True
    
    # Environment
    tls_id: str = "TL1"
    use_gui: bool = False
    time_to_teleport: int = -1
    max_depart_delay: int = -1


class TrainingLogger:
    """Logger for training metrics."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")
        
        # Initialize CSV file
        with open(self.metrics_file, "w") as f:
            f.write("episode,step,reward,loss,epsilon,avg_queue,avg_speed,avg_waiting\n")
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.writer = None
                self.use_tensorboard = False
        else:
            self.writer = None
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_losses = []
    
    def log_step(self, episode: int, step: int, reward: float, loss: Optional[float], 
                 epsilon: float, metrics: Dict[str, float]):
        """Log metrics for a single step."""
        with open(self.metrics_file, "a") as f:
            f.write(f"{episode},{step},{reward},{loss if loss else ''},{epsilon},")
            f.write(f"{metrics.get('avg_queue', '')},{metrics.get('avg_speed', '')},")
            f.write(f"{metrics.get('avg_waiting', '')}\n")
        
        if self.writer:
            self.writer.add_scalar("train/step_reward", reward, step)
            if loss is not None:
                self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/epsilon", epsilon, step)
    
    def log_episode(self, episode: int, episode_reward: float, episode_length: int,
                    metrics: Dict[str, float]):
        """Log metrics for completed episode."""
        self.episode_rewards.append(episode_reward)
        
        if self.writer:
            self.writer.add_scalar("train/episode_reward", episode_reward, episode)
            self.writer.add_scalar("train/episode_length", episode_length, episode)
            for key, value in metrics.items():
                self.writer.add_scalar(f"train/{key}", value, episode)
    
    def log_evaluation(self, episode: int, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, episode)
    
    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()


def collect_metrics(state: Dict) -> Dict[str, float]:
    """
    Collect aggregate metrics from state.
    
    Args:
        state: State dictionary from environment
        
    Returns:
        Dictionary of aggregate metrics
    """
    lane_data = state.get("lanes", {})
    
    total_queue = 0
    total_speed = 0
    total_waiting = 0
    total_vehicles = 0
    
    for lane_id, metrics in lane_data.items():
        total_queue += metrics.get("raw_queue", 0)
        total_speed += metrics.get("raw_speed", 0)
        total_waiting += metrics.get("raw_waiting_time", 0)
        total_vehicles += metrics.get("raw_vehicle_count", 0)
    
    num_lanes = max(1, len(lane_data))
    
    return {
        "avg_queue": total_queue / num_lanes,
        "avg_speed": total_speed / num_lanes,
        "avg_waiting": total_waiting / max(1, total_vehicles) if total_vehicles > 0 else 0,
        "total_queue": total_queue,
        "total_vehicles": total_vehicles,
    }


def evaluate_agent(
    agent: DQNAgent,
    env: SumoWorkZoneEnv,
    num_episodes: int = 5,
) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Args:
        agent: DQN agent to evaluate
        env: Environment for evaluation
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    eval_rewards = []
    eval_queues = []
    eval_speeds = []
    eval_waiting = []
    eval_throughput = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            state = info.get("state", {})
            metrics = collect_metrics(state)
            eval_queues.append(metrics["avg_queue"])
            eval_speeds.append(metrics["avg_speed"])
            eval_waiting.append(metrics["avg_waiting"])
        
        eval_rewards.append(episode_reward)
        eval_throughput.append(info.get("arrived_vehicles", 0))
    
    env.close()
    
    return {
        "avg_reward": np.mean(eval_rewards),
        "std_reward": np.std(eval_rewards),
        "avg_queue": np.mean(eval_queues),
        "avg_speed": np.mean(eval_speeds),
        "avg_waiting": np.mean(eval_waiting),
        "avg_throughput": np.mean(eval_throughput),
    }


def train_dqn(config: TrainingConfig) -> DQNAgent:
    """
    Train DQN agent for traffic signal control.
    
    Args:
        config: Training configuration
        
    Returns:
        Trained DQN agent
    """
    print("=" * 60)
    print("DQN Training for SUMO Work Zone Traffic Control")
    print("=" * 60)
    print(f"Episodes: {config.num_episodes}")
    print(f"Steps per episode: {config.max_steps_per_episode}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Hidden layers: {config.hidden_layers}")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(config.sumo_seed)
    torch.manual_seed(config.sumo_seed)
    
    # Create output directories
    model_dir = os.path.join(config.output_dir, "models")
    log_dir = os.path.join(config.output_dir, "logs")
    plot_dir = os.path.join(config.output_dir, "plots")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create environment to get state/action dimensions
    print("\nInitializing environment...")
    env = SumoWorkZoneEnv(
        net_file=config.net_file,
        route_file=config.route_file,
        tls_id=config.tls_id,
        num_seconds=config.max_steps_per_episode * config.delta_time,
        delta_time=config.delta_time,
        min_green=config.min_green,
        use_gui=config.use_gui,
        single_agent=True,
        reward_fn=default_reward,
        sumo_seed=config.sumo_seed,
        time_to_teleport=config.time_to_teleport,
        max_depart_delay=config.max_depart_delay,
    )
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create DQN agent
    dqn_config = DQNConfig(
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        batch_size=config.batch_size,
        target_update_frequency=config.target_update_frequency,
        replay_buffer_size=config.replay_buffer_size,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        learning_starts=config.learning_starts,
        hidden_layers=config.hidden_layers,
    )
    
    agent = DQNAgent(state_dim, action_dim, dqn_config)
    
    # Create logger
    logger = TrainingLogger(log_dir, config.tensorboard)
    
    # Training loop
    print("\nStarting training...")
    total_steps = 0
    best_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in range(1, config.num_episodes + 1):
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0
        episode_loss = []
        episode_metrics = []
        
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Select action
            epsilon = agent.update_epsilon(config.num_episodes * config.max_steps_per_episode)
            action, action_info = agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done or truncated)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Collect metrics
            state = info.get("state", {})
            metrics = collect_metrics(state)
            episode_metrics.append(metrics)
            
            # Log step
            if step % config.log_frequency == 0:
                logger.log_step(episode, total_steps, reward, loss, epsilon, metrics)
            
            # Update for next step
            obs = next_obs
            episode_reward += reward
            total_steps += 1
            step += 1
        
        # Log episode metrics
        avg_metrics = {
            "avg_queue": np.mean([m["avg_queue"] for m in episode_metrics]),
            "avg_speed": np.mean([m["avg_speed"] for m in episode_metrics]),
            "avg_waiting": np.mean([m["avg_waiting"] for m in episode_metrics]),
            "total_queue": np.mean([m["total_queue"] for m in episode_metrics]),
        }
        
        logger.log_episode(episode, episode_reward, step, avg_metrics)
        
        # Print progress
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode {episode}/{config.num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epsilon: {epsilon:.4f} | "
              f"Steps: {step}")
        
        # Save model
        if episode % config.save_frequency == 0:
            model_path = os.path.join(model_dir, f"dqn_episode_{episode}.pt")
            agent.save(model_path)
            print(f"  Model saved to {model_path}")
        
        # Evaluate agent
        if episode % config.eval_frequency == 0:
            print("  Running evaluation...")
            eval_env = SumoWorkZoneEnv(
                net_file=config.net_file,
                route_file=config.route_file,
                tls_id=config.tls_id,
                num_seconds=config.max_steps_per_episode * config.delta_time,
                delta_time=config.delta_time,
                min_green=config.min_green,
                use_gui=False,
                single_agent=True,
                reward_fn=default_reward,
                sumo_seed=config.sumo_seed + episode,
            )
            
            eval_metrics = evaluate_agent(agent, eval_env, config.eval_episodes)
            logger.log_evaluation(episode, eval_metrics)
            
            print(f"  Evaluation - Avg Reward: {eval_metrics['avg_reward']:.2f}, "
                  f"Avg Queue: {eval_metrics['avg_queue']:.2f}, "
                  f"Avg Speed: {eval_metrics['avg_speed']:.2f}")
            
            # Track best model
            if eval_metrics['avg_reward'] > best_reward:
                best_reward = eval_metrics['avg_reward']
                best_model_path = os.path.join(model_dir, "dqn_best.pt")
                agent.save(best_model_path)
                print(f"  New best model saved!")
                no_improvement_count = 0
            else:
                no_improvement_count += config.eval_frequency
            
            # Early stopping
            if no_improvement_count >= config.num_episodes:
                print(f"\nEarly stopping: No improvement for {no_improvement_count} episodes")
                break
    
    # Save final model
    final_model_path = os.path.join(model_dir, "dqn_final.pt")
    agent.save(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Save training config
    config_path = os.path.join(log_dir, "training_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    
    # Save statistics
    stats_path = os.path.join(log_dir, "training_statistics.json")
    with open(stats_path, "w") as f:
        json.dump({
            "total_episodes": episode,
            "total_steps": total_steps,
            "best_reward": best_reward,
            "final_epsilon": epsilon,
        }, f, indent=2)
    
    logger.close()
    env.close()
    
    return agent


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent for SUMO traffic control")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="Hidden layer sizes")
    
    # Path arguments
    parser.add_argument("--net-file", type=str, default="data/networks/workzone/net.net.xml")
    parser.add_argument("--route-file", type=str, default="data/networks/workzone/rou.route.xml")
    parser.add_argument("--output-dir", type=str, default="outputs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            hidden_layers=args.hidden,
            net_file=args.net_file,
            route_file=args.route_file,
            output_dir=args.output_dir,
            sumo_seed=args.seed,
            use_gui=args.gui,
            tensorboard=not args.no_tensorboard,
        )
    
    # Run training
    train_dqn(config)


if __name__ == "__main__":
    main()