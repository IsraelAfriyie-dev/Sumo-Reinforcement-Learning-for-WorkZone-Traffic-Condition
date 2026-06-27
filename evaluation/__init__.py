"""
SUMO Work Zone Traffic Control - Evaluation Module

This module provides evaluation utilities for trained RL agents,
including metrics computation and visualization.
"""

from evaluation.evaluate_policy import evaluate_policy, PolicyEvaluator
from evaluation.metrics import (
    MetricsCollector,
    compute_performance_metrics,
    plot_training_curves,
    plot_comparison,
)

__all__ = [
    "evaluate_policy",
    "PolicyEvaluator",
    "MetricsCollector",
    "compute_performance_metrics",
    "plot_training_curves",
    "plot_comparison",
]

__version__ = "1.0.0"