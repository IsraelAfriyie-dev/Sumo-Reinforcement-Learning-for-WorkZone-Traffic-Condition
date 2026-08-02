# SUMO Work Zone Traffic Control - Architecture Documentation

## Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent for traffic signal control in SUMO work zone environments. The system is designed to optimize traffic flow, reduce delays, and improve safety near work zones.

## Project Structure

```
sumo-workzone-rl/
├── configs/               # Configuration files
│   ├── default_config.yaml
│   └── sumo_config.yaml
├── data/                  # Data files
│   └── networks/
│       └── workzone/      # SUMO network files
├── envs/                  # Environment module
│   ├── __init__.py
│   ├── sumo_env.py        # SUMO environment wrapper
│   ├── reward_functions.py
│   └── observations.py
├── agents/                # RL agents
│   ├── __init__.py
│   └── dqn_agent.py       # DQN implementation
├── training/              # Training utilities
│   ├── __init__.py
│   ├── train_dqn.py       # Training script
│   └── replay_buffer.py
├── evaluation/            # Evaluation utilities
│   ├── __init__.py
│   ├── evaluate_policy.py
│   └── metrics.py
├── outputs/               # Training outputs
│   ├── logs/
│   ├── plots/
│   └── models/
├── docs/                  # Documentation
├── tests/                 # Unit tests
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Components

### 1. Environment (`envs/`)

#### SumoWorkZoneEnv
The main environment class that wraps SUMO simulation via TraCI.

**Features:**
- Gymnasium-compatible interface
- Configurable state and reward functions
- Work zone-specific metrics (spillback, merge conflicts)
- Safety metrics (TTC conflicts)

**Key Methods:**
- `reset()`: Initialize simulation
- `step(action)`: Execute action and return observation/reward
- `close()`: Clean up simulation

#### State Space
The state includes traffic metrics from all lanes:
- Phase one-hot encoding (2 dimensions)
- Minimum green elapsed flag (1 dimension)
- Per-lane features (4 dimensions per lane):
  - Queue length (normalized)
  - Average speed (normalized)
  - Waiting time (normalized)
  - Vehicle count (normalized)

Total: 2 + 1 + (4 × num_lanes) dimensions

#### Action Space
Discrete action space with 2 actions:
- **0**: Keep current phase
- **1**: Switch to next phase

#### Reward Function
Multi-objective reward combining:
- **Penalize**:
  - Long queues
  - Excessive waiting time
  - Stop-and-go behavior
  - TTC conflicts (safety)
  - Work zone spillback

- **Encourage**:
  - Higher average speeds
  - Better throughput
  - Smoother traffic flow

### 2. Agents (`agents/`)

#### DQNAgent
Deep Q-Network implementation with:
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration
- Double DQN for reduced overestimation

**Neural Network Architecture:**
```
Input (state_dim)
    ↓
Hidden Layer 1 (128 units, ReLU)
    ↓
Hidden Layer 2 (128 units, ReLU)
    ↓
Output (action_dim Q-values)
```

**Training Algorithm:**
1. Store transitions in replay buffer
2. Sample random batch
3. Compute TD targets using target network
4. Update Q-network with gradient descent
5. Periodically sync target network

### 3. Training (`training/`)

#### ReplayBuffer
Efficient circular buffer for experience replay.

**Features:**
- O(1) insertion
- O(batch_size) sampling
- Automatic capacity management

#### Training Loop
1. Initialize environment and agent
2. For each episode:
   - Reset environment
   - For each step:
     - Select action (ε-greedy)
     - Execute action in environment
     - Store transition
     - Train agent
     - Log metrics
3. Periodically evaluate and save model

### 4. Evaluation (`evaluation/`)

#### Metrics
Collected metrics include:
- **Efficiency**: Queue length, waiting time, speed
- **Throughput**: Vehicles completed
- **Safety**: TTC conflicts
- **Work Zone**: Spillback events, merge conflicts

#### Comparison
Evaluates trained agent against:
- Fixed-timing baseline
- Random policy

## Configuration

### DQN Hyperparameters
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
```

### Reward Weights
```yaml
reward:
  waiting_time_weight: -0.1
  queue_length_weight: -0.2
  speed_weight: 0.1
  safety_weight: -0.3
  spillback_weight: -0.2
```

## Usage

### Training
```bash
python training/train_dqn.py --episodes 500 --lr 0.001
```

### Evaluation
```bash
python evaluation/evaluate_policy.py --model outputs/models/dqn_final.pt
```

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- SUMO 1.18+
- Gymnasium 0.28+
- Stable-Baselines3 2.0+