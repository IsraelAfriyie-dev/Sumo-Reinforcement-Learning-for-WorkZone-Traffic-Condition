"""
Metrics Module for Traffic Control Evaluation

This module provides utilities for collecting and computing
traffic performance metrics during evaluation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TrafficMetrics:
    """Container for traffic performance metrics."""
    # Efficiency metrics
    avg_waiting_time: float = 0.0
    total_waiting_time: float = 0.0
    max_waiting_time: float = 0.0
    
    # Queue metrics
    avg_queue_length: float = 0.0
    max_queue_length: float = 0.0
    total_queued_vehicles: float = 0.0
    
    # Speed metrics
    avg_speed: float = 0.0
    max_speed: float = 0.0
    min_speed: float = 0.0
    
    # Throughput metrics
    throughput: int = 0
    vehicles_completed: int = 0
    vehicles_remaining: int = 0
    
    # Safety metrics
    ttc_conflicts: int = 0
    avg_ttc: float = 10.0
    drac_conflicts: int = 0
    
    # Work zone metrics
    spillback_events: int = 0
    workzone_queue: float = 0.0
    merge_conflicts: int = 0
    
    # Reward metrics
    total_reward: float = 0.0
    avg_reward: float = 0.0
    std_reward: float = 0.0
    
    # Episode info
    episode_length: int = 0
    num_steps: int = 0


class MetricsCollector:
    """
    Collects and aggregates traffic metrics during evaluation.
    
    This class tracks various traffic metrics across evaluation episodes
    and provides methods to compute statistics and generate reports.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.episode_metrics = []
        self.step_metrics = []
        self.current_episode = None
    
    def start_episode(self, episode_id: int):
        """Start tracking a new episode."""
        self.current_episode = {
            "episode_id": episode_id,
            "rewards": [],
            "waiting_times": [],
            "queue_lengths": [],
            "speeds": [],
            "vehicles": [],
            "ttc_values": [],
            "workzone_queues": [],
        }
    
    def record_step(
        self,
        reward: float,
        waiting_time: float,
        queue_length: float,
        speed: float,
        num_vehicles: int,
        ttc_conflicts: int = 0,
        workzone_queue: float = 0.0,
    ):
        """Record metrics for a single step."""
        if self.current_episode is None:
            self.start_episode(0)
        
        self.current_episode["rewards"].append(reward)
        self.current_episode["waiting_times"].append(waiting_time)
        self.current_episode["queue_lengths"].append(queue_length)
        self.current_episode["speeds"].append(speed)
        self.current_episode["vehicles"].append(num_vehicles)
        self.current_episode["ttc_values"].append(ttc_conflicts)
        self.current_episode["workzone_queues"].append(workzone_queue)
    
    def end_episode(self, throughput: int, total_steps: int):
        """Finalize and store episode metrics."""
        if self.current_episode is None:
            return
        
        episode = self.current_episode
        
        metrics = TrafficMetrics(
            avg_waiting_time=np.mean(episode["waiting_times"]) if episode["waiting_times"] else 0,
            max_waiting_time=np.max(episode["waiting_times"]) if episode["waiting_times"] else 0,
            avg_queue_length=np.mean(episode["queue_lengths"]) if episode["queue_lengths"] else 0,
            max_queue_length=np.max(episode["queue_lengths"]) if episode["queue_lengths"] else 0,
            avg_speed=np.mean(episode["speeds"]) if episode["speeds"] else 0,
            max_speed=np.max(episode["speeds"]) if episode["speeds"] else 0,
            min_speed=np.min(episode["speeds"]) if episode["speeds"] else 0,
            throughput=throughput,
            vehicles_completed=throughput,
            ttc_conflicts=int(np.sum(episode["ttc_values"])),
            workzone_queue=np.mean(episode["workzone_queues"]) if episode["workzone_queues"] else 0,
            total_reward=np.sum(episode["rewards"]),
            avg_reward=np.mean(episode["rewards"]) if episode["rewards"] else 0,
            std_reward=np.std(episode["rewards"]) if episode["rewards"] else 0,
            episode_length=total_steps,
            num_steps=total_steps,
        )
        
        self.episode_metrics.append(metrics)
        self.current_episode = None
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all episodes.
        
        Returns:
            Dictionary of aggregated metrics
        """
        if not self.episode_metrics:
            return {}
        
        return {
            "num_episodes": len(self.episode_metrics),
            "avg_episode_reward": np.mean([m.total_reward for m in self.episode_metrics]),
            "std_episode_reward": np.std([m.total_reward for m in self.episode_metrics]),
            "avg_waiting_time": np.mean([m.avg_waiting_time for m in self.episode_metrics]),
            "avg_queue_length": np.mean([m.avg_queue_length for m in self.episode_metrics]),
            "avg_speed": np.mean([m.avg_speed for m in self.episode_metrics]),
            "total_throughput": np.sum([m.throughput for m in self.episode_metrics]),
            "avg_throughput": np.mean([m.throughput for m in self.episode_metrics]),
            "total_ttc_conflicts": np.sum([m.ttc_conflicts for m in self.episode_metrics]),
            "avg_ttc_conflicts": np.mean([m.ttc_conflicts for m in self.episode_metrics]),
            "avg_workzone_queue": np.mean([m.workzone_queue for m in self.episode_metrics]),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to pandas DataFrame.
        
        Returns:
            DataFrame with episode metrics
        """
        data = []
        for i, m in enumerate(self.episode_metrics):
            data.append({
                "episode": i,
                "total_reward": m.total_reward,
                "avg_reward": m.avg_reward,
                "std_reward": m.std_reward,
                "avg_waiting_time": m.avg_waiting_time,
                "max_waiting_time": m.max_waiting_time,
                "avg_queue_length": m.avg_queue_length,
                "max_queue_length": m.max_queue_length,
                "avg_speed": m.avg_speed,
                "throughput": m.throughput,
                "ttc_conflicts": m.ttc_conflicts,
                "workzone_queue": m.workzone_queue,
                "episode_length": m.episode_length,
            })
        
        return pd.DataFrame(data)
    
    def save(self, path: str):
        """Save metrics to file."""
        summary = self.get_summary()
        df = self.to_dataframe()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path.replace(".csv", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        df.to_csv(path, index=False)
    
    def load(self, path: str):
        """Load metrics from file."""
        df = pd.read_csv(path)
        
        self.episode_metrics = []
        for _, row in df.iterrows():
            metrics = TrafficMetrics(
                total_reward=row["total_reward"],
                avg_reward=row["avg_reward"],
                std_reward=row["std_reward"],
                avg_waiting_time=row["avg_waiting_time"],
                max_waiting_time=row["max_waiting_time"],
                avg_queue_length=row["avg_queue_length"],
                max_queue_length=row["max_queue_length"],
                avg_speed=row["avg_speed"],
                throughput=row["throughput"],
                ttc_conflicts=row["ttc_conflicts"],
                workzone_queue=row["workzone_queue"],
                episode_length=row["episode_length"],
            )
            self.episode_metrics.append(metrics)


def compute_performance_metrics(
    rewards: List[float],
    queue_lengths: List[float],
    speeds: List[float],
    waiting_times: List[float],
    throughput: int,
    ttc_conflicts: int = 0,
) -> Dict[str, float]:
    """
    Compute performance metrics from raw data.
    
    Args:
        rewards: List of step rewards
        queue_lengths: List of queue length measurements
        speeds: List of speed measurements
        waiting_times: List of waiting time measurements
        throughput: Total vehicles that completed trip
        ttc_conflicts: Total TTC safety conflicts
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Reward metrics
    metrics["total_reward"] = sum(rewards)
    metrics["avg_reward"] = np.mean(rewards) if rewards else 0
    metrics["std_reward"] = np.std(rewards) if rewards else 0
    
    # Queue metrics
    metrics["avg_queue"] = np.mean(queue_lengths) if queue_lengths else 0
    metrics["max_queue"] = np.max(queue_lengths) if queue_lengths else 0
    metrics["total_queue"] = sum(queue_lengths) if queue_lengths else 0
    
    # Speed metrics
    metrics["avg_speed"] = np.mean(speeds) if speeds else 0
    metrics["max_speed"] = np.max(speeds) if speeds else 0
    metrics["min_speed"] = np.min(speeds) if speeds else 0
    metrics["std_speed"] = np.std(speeds) if speeds else 0
    
    # Waiting time metrics
    metrics["avg_waiting"] = np.mean(waiting_times) if waiting_times else 0
    metrics["max_waiting"] = np.max(waiting_times) if waiting_times else 0
    
    # Throughput
    metrics["throughput"] = throughput
    
    # Safety
    metrics["ttc_conflicts"] = ttc_conflicts
    
    # Efficiency score (combined metric)
    if metrics["avg_speed"] > 0:
        metrics["efficiency_score"] = (
            metrics["avg_speed"] / (1 + metrics["avg_queue"]) * 
            (1 / (1 + metrics["avg_waiting"]))
        )
    else:
        metrics["efficiency_score"] = 0
    
    return metrics


def plot_training_curves(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot training curves from metrics DataFrame.
    
    Args:
        metrics_df: DataFrame with training metrics
        save_path: Path to save plot (if None, doesn't save)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Training Metrics Over Episodes", fontsize=14, fontweight="bold")
    
    # Reward curve
    ax = axes[0, 0]
    ax.plot(metrics_df["episode"], metrics_df["total_reward"], alpha=0.5, label="Episode Reward")
    if "avg_reward" in metrics_df.columns:
        # Rolling average
        window = min(10, len(metrics_df))
        rolling_reward = metrics_df["total_reward"].rolling(window).mean()
        ax.plot(metrics_df["episode"], rolling_reward, label=f"Rolling Avg ({window})", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Queue length curve
    ax = axes[0, 1]
    ax.plot(metrics_df["episode"], metrics_df["avg_queue_length"], label="Avg Queue")
    if "max_queue_length" in metrics_df.columns:
        ax.plot(metrics_df["episode"], metrics_df["max_queue_length"], label="Max Queue", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Queue Length")
    ax.set_title("Queue Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Speed curve
    ax = axes[0, 2]
    ax.plot(metrics_df["episode"], metrics_df["avg_speed"], color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Speed (m/s)")
    ax.set_title("Average Speed")
    ax.grid(True, alpha=0.3)
    
    # Throughput curve
    ax = axes[1, 0]
    ax.plot(metrics_df["episode"], metrics_df["throughput"], color="purple")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Throughput (vehicles)")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)
    
    # Waiting time curve
    ax = axes[1, 1]
    ax.plot(metrics_df["episode"], metrics_df["avg_waiting_time"], color="orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Waiting Time (s)")
    ax.set_title("Average Waiting Time")
    ax.grid(True, alpha=0.3)
    
    # TTC conflicts
    ax = axes[1, 2]
    ax.plot(metrics_df["episode"], metrics_df["ttc_conflicts"], color="red")
    ax.set_xlabel("Episode")
    ax.set_ylabel("TTC Conflicts")
    ax.set_title("Safety Conflicts (TTC)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    plt.close()


def plot_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    Plot comparison between multiple models/policies.
    
    Args:
        results_dict: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to plot (if None, uses default)
        save_path: Path to save plot
        show: Whether to display plot
    """
    if metrics is None:
        metrics = ["avg_queue_length", "avg_speed", "avg_waiting_time", "throughput", "ttc_conflicts"]
    
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    metric_labels = {
        "avg_queue_length": "Avg Queue Length",
        "avg_speed": "Avg Speed (m/s)",
        "avg_waiting_time": "Avg Waiting Time (s)",
        "throughput": "Throughput",
        "ttc_conflicts": "TTC Conflicts",
        "total_reward": "Total Reward",
        "avg_reward": "Avg Reward",
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results_dict.get(name, {}).get(metric, 0) for name in model_names]
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor="black")
        
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    plt.close()


def print_evaluation_report(
    model_name: str,
    metrics: Dict[str, float],
    comparison_metrics: Optional[Dict[str, float]] = None,
):
    """
    Print a formatted evaluation report.
    
    Args:
        model_name: Name of the evaluated model
        metrics: Dictionary of metrics
        comparison_metrics: Optional baseline metrics for comparison
    """
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 70)
    
    # Efficiency metrics
    print("\n📊 EFFICIENCY METRICS")
    print("-" * 40)
    print(f"  Average Queue Length:    {metrics.get('avg_queue_length', 0):.2f}")
    print(f"  Maximum Queue Length:    {metrics.get('max_queue_length', 0):.2f}")
    print(f"  Average Speed:           {metrics.get('avg_speed', 0):.2f} m/s")
    print(f"  Maximum Speed:           {metrics.get('max_speed', 0):.2f} m/s")
    print(f"  Average Waiting Time:    {metrics.get('avg_waiting_time', 0):.2f} s")
    print(f"  Maximum Waiting Time:    {metrics.get('max_waiting_time', 0):.2f} s")
    
    # Throughput
    print("\n🚗 THROUGHPUT")
    print("-" * 40)
    print(f"  Total Throughput:        {metrics.get('throughput', 0)} vehicles")
    print(f"  Avg per Episode:         {metrics.get('avg_throughput', 0):.1f} vehicles")
    
    # Safety
    print("\n⚠️  SAFETY METRICS")
    print("-" * 40)
    print(f"  TTC Conflicts:           {metrics.get('ttc_conflicts', 0)}")
    print(f"  Avg TTC:                 {metrics.get('avg_ttc', 10.0):.2f} s")
    
    # Work zone
    print("\n🚧 WORK ZONE")
    print("-" * 40)
    print(f"  Work Zone Queue:         {metrics.get('workzone_queue', 0):.2f}")
    print(f"  Spillback Events:        {metrics.get('spillback_events', 0)}")
    
    # Reward
    print("\n🎯 REWARD")
    print("-" * 40)
    print(f"  Total Reward:            {metrics.get('total_reward', 0):.2f}")
    print(f"  Average Reward:          {metrics.get('avg_reward', 0):.2f}")
    print(f"  Efficiency Score:        {metrics.get('efficiency_score', 0):.4f}")
    
    # Comparison with baseline
    if comparison_metrics:
        print("\n📈 COMPARISON WITH BASELINE")
        print("-" * 40)
        
        for metric in ["avg_queue_length", "avg_waiting_time", "ttc_conflicts"]:
            baseline = comparison_metrics.get(metric, 0)
            current = metrics.get(metric, 0)
            
            if baseline != 0:
                if metric in ["avg_queue_length", "avg_waiting_time", "ttc_conflicts"]:
                    # Lower is better
                    change = (current - baseline) / baseline * 100
                    direction = "↓" if change < 0 else "↑"
                    print(f"  {metric}: {current:.2f} vs {baseline:.2f} ({change:+.1f}% {direction})")
                else:
                    # Higher is better
                    change = (current - baseline) / baseline * 100
                    direction = "↑" if change > 0 else "↓"
                    print(f"  {metric}: {current:.2f} vs {baseline:.2f} ({change:+.1f}% {direction})")
    
    print("\n" + "=" * 70)