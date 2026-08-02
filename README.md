

# SUMO Work Zone Traffic Control with Deep Q-Network (DQN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A reinforcement learning framework for traffic signal control in SUMO work zone environments using Deep Q-Networks (DQN).

## Overview

This project implements a DQN agent that learns to optimize traffic flow, reduce delays, and improve safety near work zones. The agent observes traffic conditions and decides when to change traffic signal phases to minimize vehicle delays and prevent congestion spillback.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the agent
python training/train_dqn.py --episodes 100

# Evaluate the trained agent
python evaluation/evaluate_policy.py --model outputs/models/dqn_final.pt
```

## Project Structure

```
sumo-workzone-rl/
├── configs/               # Configuration files
│   ├── default_config.yaml    # DQN hyperparameters
│   └── sumo_config.yaml       # SUMO simulation settings
├── data/
│   └── networks/workzone/     # SUMO network files
├── envs/                  # Environment module
│   ├── sumo_env.py        # SUMO environment wrapper
│   ├── reward_functions.py
│   └── observations.py
├── agents/                # RL agents
│   └── dqn_agent.py       # DQN implementation
├── training/              # Training utilities
│   ├── train_dqn.py       # Training script
│   └── replay_buffer.py
├── evaluation/            # Evaluation utilities
│   ├── evaluate_policy.py
│   └── metrics.py
├── outputs/               # Training outputs
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── requirements.txt
```

## Key Features

- **Gymnasium-compatible environment** for easy integration
- **DQN with experience replay** and target networks
- **Multi-objective reward function** penalizing queues, waiting time, and TTC conflicts
- **Work zone-specific metrics** for spillback and merge conflicts
- **Configurable via YAML** - no code changes needed for hyperparameters
- **Comprehensive evaluation** with fixed-timing and random baselines

## Documentation

For detailed documentation, see [docs/README.md](docs/README.md).

## State Space

| Feature | Description | Dimensions |
|---------|-------------|------------|
| Phase One-Hot | Current signal phase | 2 |
| Min Green | Minimum green elapsed | 1 |
| Queue Length | Normalized queue per lane | num_lanes |
| Average Speed | Normalized speed per lane | num_lanes |
| Waiting Time | Normalized waiting per lane | num_lanes |
| Vehicle Count | Normalized count per lane | num_lanes |

## Action Space

- **0**: Keep current phase
- **1**: Switch to next phase

## Reward Function

**Penalizes:**
- Long queues
- Excessive waiting time
- Stop-and-go behavior
- MTI conflicts
- Work zone spillback

**Encourages:**
- Higher average speeds
- Better throughput

## Requirements

- Python 3.8+
- SUMO 1.18+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Installation

See [docs/README.md](docs/README.md#installation) for detailed installation instructions.

## License

MIT License



