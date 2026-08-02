# SUMO Work Zone Traffic Control with Deep Q-Network (DQN)

A reinforcement learning framework for traffic signal control in SUMO work zone environments. This project implements a DQN agent that learns to optimize traffic flow, reduce delays, and improve safety near work zones.

<p align="center">
  <img src="https://sumo.dlr.de/docs/images/Sumo-gui.png" alt="SUMO GUI" width="400"/>
</p>

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Function](#reward-function)
- [DQN Architecture](#dqn-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)

## Overview

This project addresses the challenge of traffic signal control in work zone areas, where lane closures create bottleneck conditions and increased safety risks. The DQN agent learns to make intelligent traffic signal decisions to:

- Minimize vehicle delays and queue lengths
- Reduce stop-and-go behavior
- Improve throughput through work zones
- Enhance safety by reducing TTC (Time-To-Collision) conflicts

## Project Structure

```
sumo-workzone-rl/
├── configs/               # YAML configuration files
│   ├── default_config.yaml    # DQN hyperparameters
│   └── sumo_config.yaml       # SUMO simulation settings
├── data/                  # Data files
│   └── networks/
│       └── workzone/      # SUMO network files (.net.xml, .rou.xml, etc.)
├── envs/                  # Environment module
│   ├── sumo_env.py        # Main environment wrapper
│   ├── reward_functions.py    # Reward function implementations
│   └── observations.py    # State observation functions
├── agents/                # RL agents
│   └── dqn_agent.py       # DQN agent implementation
├── training/              # Training utilities
│   ├── train_dqn.py       # Training script
│   └── replay_buffer.py   # Experience replay buffer
├── evaluation/            # Evaluation utilities
│   ├── evaluate_policy.py # Evaluation script
│   └── metrics.py         # Performance metrics
├── outputs/               # Training outputs (created during training)
│   ├── logs/              # Training logs and TensorBoard data
│   ├── plots/             # Generated visualizations
│   └── models/            # Saved model checkpoints
├── docs/                  # Documentation
├── tests/                 # Unit tests
└── requirements.txt       # Python dependencies
```

## Installation

### Prerequisites

1. **Python 3.8+**
2. **SUMO Traffic Simulation** (version 1.18.0 or later)

### Install SUMO

```bash
# Ubuntu/Debian
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Set SUMO_HOME
export SUMO_HOME="/usr/share/sumo"
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

# Optional: Enable Libsumo for ~8x speed boost
export LIBSUMO_AS_TRACI=1
```

### Install Python Dependencies

```bash
# Clone repository
git clone https://github.com/IsraelAfriyie-dev/Sumo-Reinforcement-Learning-for-WorkZone-Traffic-Condition
cd Sumo-Reinforcement-Learning-for-WorkZone-Traffic-Condition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Installation

```bash
python -c "import traci; print('SUMO/TraCI installed successfully')"
```

### 2. Train a DQN Agent

```bash
python training/train_dqn.py --episodes 100
```

This will:
- Load the SUMO network from `data/networks/workzone/`
- Train the DQN agent for 100 episodes
- Save checkpoints to `outputs/models/`
- Log metrics to `outputs/logs/`

### 3. Evaluate the Trained Agent

```bash
python evaluation/evaluate_policy.py --model outputs/models/dqn_final.pt --episodes 10
```

## State Space

The state observed by the agent at each time step includes:

| Feature | Description | Dimensions |
|---------|-------------|------------|
| Phase One-Hot | Current traffic signal phase | 2 |
| Min Green Elapsed | Whether minimum green time has passed | 1 |
| Queue Length | Normalized queue per lane | num_lanes |
| Average Speed | Normalized average speed per lane | num_lanes |
| Waiting Time | Normalized waiting time per lane | num_lanes |
| Vehicle Count | Normalized vehicle count per lane | num_lanes |

**Total State Dimensions:** 2 + 1 + (4 × num_lanes)

For the default work zone network with 8 lanes, this results in a 35-dimensional state space.

### State Normalization

All features are normalized to [0, 1] range:
- Queue: `queue / lane_capacity`
- Speed: `speed / 20.0` (typical max speed)
- Waiting: `waiting_time / 100.0`
- Vehicle count: `count / lane_capacity`

## Action Space

The action space is discrete with 2 actions:

| Action | Description |
|--------|-------------|
| 0 | **Keep Current Phase** - Maintain the current signal state |
| 1 | **Switch Phase** - Change to the next phase in the sequence |

The agent decides whether to change the signal phase every `delta_time` seconds (default: 5 seconds).

## Reward Function

The reward function is multi-objective, combining several traffic metrics:

### Penalizes (Negative Rewards)

| Metric | Description |
|--------|-------------|
| Waiting Time | Cumulative waiting time of all vehicles |
| Queue Length | Number of vehicles with speed < 0.1 m/s |
| Stop-and-Go | Variance in vehicle speeds |
| TTC Conflicts | Time-To-Collision < 1.5 seconds |
| Spillback | Queue buildup approaching work zone |

### Encourages (Positive Rewards)

| Metric | Description |
|--------|-------------|
| Average Speed | Higher speeds indicate better flow |
| Throughput | More vehicles completing their trips |

### Default Weights

```python
WAITING_TIME_WEIGHT = -0.1
QUEUE_LENGTH_WEIGHT = -0.2
SPEED_WEIGHT = 0.1
THROUGHPUT_WEIGHT = 0.01
SAFETY_WEIGHT = -0.3
SPILLBACK_WEIGHT = -0.2
STOP_GO_WEIGHT = -0.1
```

## DQN Architecture

### Neural Network

```
Input Layer (state_dim)
    ↓
Dense Layer 1 (128 units, ReLU activation)
    ↓
Dense Layer 2 (128 units, ReLU activation)
    ↓
Output Layer (action_dim Q-values)
```

### Training Algorithm

1. **Experience Replay**: Store transitions in a replay buffer of size 100,000
2. **Target Network**: Separate network for computing TD targets, updated every 1,000 steps
3. **Double DQN**: Use online network to select actions, target network to evaluate
4. **Epsilon-Greedy Exploration**: Linear decay from 1.0 to 0.01 over 10% of training

### Key Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Discount Factor (γ) | 0.99 | Future reward importance |
| Batch Size | 64 | Training batch size |
| Target Update Freq | 1000 | Steps between target updates |
| Replay Buffer Size | 100,000 | Experience replay capacity |
| Initial Epsilon | 1.0 | Starting exploration rate |
| Final Epsilon | 0.01 | Minimum exploration rate |
| Exploration Fraction | 0.1 | Fraction of training for decay |

## Training

### Training Command

```bash
python training/train_dqn.py \
    --episodes 500 \
    --steps 1000 \
    --lr 0.001 \
    --batch-size 64 \
    --hidden 128 128 \
    --net-file data/networks/workzone/net.net.xml \
    --route-file data/networks/workzone/rou.route.xml \
    --output-dir outputs
```

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `--episodes` | Number of training episodes |
| `--steps` | Maximum steps per episode |
| `--lr` | Learning rate |
| `--batch-size` | Training batch size |
| `--hidden` | Hidden layer sizes |
| `--net-file` | Path to SUMO network file |
| `--route-file` | Path to route file |
| `--output-dir` | Output directory |
| `--seed` | Random seed |
| `--gui` | Use SUMO GUI (slower) |
| `--no-tensorboard` | Disable TensorBoard logging |

### Using Configuration File

```bash
python training/train_dqn.py --config configs/default_config.yaml
```

## Evaluation

### Evaluation Command

```bash
python evaluation/evaluate_policy.py \
    --model outputs/models/dqn_final.pt \
    --net-file data/networks/workzone/net.net.xml \
    --route-file data/networks/workzone/rou.route.xml \
    --episodes 10 \
    --simulation-seconds 1000 \
    --output-dir outputs/evaluation
```

### Metrics Reported

| Metric | Description | Unit |
|--------|-------------|------|
| Average Queue Length | Mean vehicles queued per lane | vehicles |
| Average Speed | Mean vehicle speed | m/s |
| Average Waiting Time | Mean waiting time per vehicle | seconds |
| Throughput | Total vehicles completed | vehicles |
| TTC Conflicts | Safety conflicts (TTC < 1.5s) | count |
| Work Zone Queue | Vehicles queued at work zone | vehicles |
| Total Reward | Cumulative episode reward | - |

## Configuration

### DQN Configuration (`configs/default_config.yaml`)

```yaml
dqn:
  learning_rate: 0.001
  discount_factor: 0.99
  batch_size: 64
  target_update_frequency: 1000
  replay_buffer_size: 100000
  exploration_fraction: 0.1
  exploration_initial_eps: 1.0
  exploration_final_eps: 0.01

training:
  num_episodes: 500
  max_steps_per_episode: 1000
  save_frequency: 50
  log_frequency: 10
  eval_frequency: 50
```

### SUMO Configuration (`configs/sumo_config.yaml`)

Contains paths to SUMO files and simulation parameters.

## Expected Outputs

### During Training

```
Episode 1/500 | Reward: -45.23 | Loss: 0.00 | Epsilon: 1.0000 | Steps: 200
Episode 2/500 | Reward: -38.45 | Loss: 0.12 | Epsilon: 0.9980 | Steps: 200
...
Episode 500/500 | Reward: 12.34 | Loss: 0.05 | Epsilon: 0.0100 | Steps: 200
```

### Output Files

| Directory | Contents |
|-----------|----------|
| `outputs/models/` | Model checkpoints (`dqn_episode_*.pt`, `dqn_final.pt`, `dqn_best.pt`) |
| `outputs/logs/` | Training logs, TensorBoard events, configuration |
| `outputs/plots/` | Generated visualization plots |
| `outputs/evaluation/` | Evaluation results and comparisons |

### TensorBoard

View training progress:
```bash
tensorboard --logdir outputs/logs
```

## SUMO Network Files

This project requires the following SUMO files (located in `data/networks/workzone/`):

| File | Description | Required |
|------|-------------|----------|
| `net.net.xml` | Network definition | ✓ |
| `rou.route.xml` | Vehicle routes and flows | ✓ |
| `network_IA.sumocfg` | SUMO configuration | ✓ |
| `laneClosure.add.xml` | Lane closures/rerouters | Recommended |
| `network_IA.add.xml` | Additional elements (detectors) | Optional |
| `tls.rl.add.xml` | Traffic light programs | Optional |

### Creating Your Own Network

If you need to create a custom work zone scenario:

1. Use [SUMO NetEdit](https://sumo.dlr.de/docs/netedit.html) to create/edit networks
2. Define vehicle routes in a `.rou.xml` file
3. Configure lane closures using rerouters in an `.add.xml` file
4. Update paths in `configs/sumo_config.yaml`

## Troubleshooting

### SUMO_HOME Not Set

```bash
export SUMO_HOME="/usr/share/sumo"  # Adjust path as needed
```

### Connection Error with SUMO

Make sure no other SUMO process is running:
```bash
pkill -f sumo
```

### Out of Memory

Reduce batch size or replay buffer size in config.

### Training Not Converging

- Try different learning rates (0.0001 - 0.01)
- Adjust reward function weights
- Increase network capacity (more hidden units)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sumo_workzone_rl,
  title={SUMO Work Zone Traffic Control with DQN},
  author={Israel Afriyie},
  year={2024},
  url={https://github.com/IsraelAfriyie-dev/Sumo-Reinforcement-Learning-for-WorkZone-Traffic-Condition}
}
```

## License

MIT License - see repository for details.

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [DQN Paper](https://www.nature.com/articles/nature14236)