# Minari Library Documentation

## Overview
Minari is a Python library for conducting research in offline reinforcement learning. It provides a standard format for offline RL datasets with utilities for data collection, storage, and manipulation. The library is akin to an offline version of Gymnasium.

**Latest Version**: 0.5.3  
**Documentation**: https://minari.farama.org  
**GitHub**: https://github.com/Farama-Foundation/Minari  
**Python Requirements**: >=3.8

## Installation

### Basic Installation
```bash
pip install minari
```

### Full Installation (with all dependencies)
```bash
pip install "minari[all]"
```

### From Source
```bash
git clone https://github.com/Farama-Foundation/Minari.git --single-branch
cd Minari
pip install -e ".[all]"
```

## Core Concepts

### Dataset Structure
Minari datasets are stored in HDF5 or Arrow format with the following structure:
- **Root Directory**: `~/.minari/datasets/` (default, configurable via `MINARI_DATASETS_PATH`)
- **Dataset ID Format**: `(namespace/)(env_name/)dataset_name(-v[version])`
  - Example: `D4RL/door/human-v2`

### Episode Data Structure
Each episode contains:
- **observations**: np.ndarray, shape `(total_steps + 1, *obs_shape)` - includes initial and final observation
- **actions**: np.ndarray, shape `(total_steps, *action_shape)`
- **rewards**: np.ndarray, shape `(total_steps,)`
- **terminations**: np.ndarray, shape `(total_steps,)` - boolean flags
- **truncations**: np.ndarray, shape `(total_steps,)` - boolean flags
- **infos**: dict (optional) - additional step information

## Basic Usage

### Loading a Dataset

```python
import minari

# Load a dataset (downloads if not present locally)
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)

# Access dataset metadata
print(f"Total episodes: {dataset.total_episodes}")
print(f"Total steps: {dataset.total_steps}")
print(f"Observation space: {dataset.spec.observation_space}")
print(f"Action space: {dataset.spec.action_space}")
```

### Iterating Through Episodes

```python
# Iterate through all episodes
for episode_data in dataset.iterate_episodes():
    observations = episode_data.observations  # shape: (steps+1, *obs_shape)
    actions = episode_data.actions            # shape: (steps, *action_shape)
    rewards = episode_data.rewards            # shape: (steps,)
    terminations = episode_data.terminations  # shape: (steps,)
    truncations = episode_data.truncations    # shape: (steps,)
    infos = episode_data.infos                # dict
    
    # Process episode data
    print(f"Episode {episode_data.id}: {episode_data.total_timesteps} steps")
```

### Sampling Episodes

```python
# Set random seed for reproducibility
dataset.set_seed(seed=123)

# Sample random episodes
sampled_episodes = dataset.sample_episodes(n_episodes=10)

# sampled_episodes is a list of EpisodeData objects
for ep in sampled_episodes:
    print(f"Episode ID: {ep.id}, Steps: {ep.total_timesteps}")
```

### Filtering Episodes

```python
# Filter episodes based on custom condition
def high_reward_filter(episode: EpisodeData) -> bool:
    return episode.rewards.mean() > 2.0

filtered_dataset = dataset.filter_episodes(high_reward_filter)
print(f"Filtered episodes: {filtered_dataset.total_episodes}")

# Filter for terminated episodes only
terminated_dataset = dataset.filter_episodes(
    lambda ep: ep.terminations[-1]
)
```

## EpisodeData Class

The `EpisodeData` object returned by iteration/sampling has these attributes:

```python
@dataclass
class EpisodeData:
    id: int                    # Episode identifier
    seed: Optional[int]        # Random seed used for episode
    total_timesteps: int       # Number of timesteps in episode
    observations: np.ndarray   # Observations (steps+1, *obs_shape)
    actions: np.ndarray        # Actions (steps, *action_shape)
    rewards: np.ndarray        # Rewards (steps,)
    terminations: np.ndarray   # Termination flags (steps,)
    truncations: np.ndarray    # Truncation flags (steps,)
```

## MinariDataset Methods

### Key Methods

```python
# Sample episodes
episodes = dataset.sample_episodes(n_episodes=5)

# Iterate through episodes
for ep in dataset.iterate_episodes():
    pass

# Filter episodes
filtered = dataset.filter_episodes(condition=lambda ep: ep.rewards.sum() > 100)

# Recover the environment used to create the dataset
env = dataset.recover_environment()

# Recover evaluation environment (if specified)
eval_env = dataset.recover_environment(eval_env=True)

# Get dataset specification
spec = dataset.spec
```

### Dataset Properties

```python
dataset.total_episodes    # Total number of episodes
dataset.total_steps       # Total number of steps across all episodes
dataset.episode_indices   # Available episode indices
dataset.spec             # MinariDatasetSpec with metadata
```

## CLI Commands

Minari provides CLI tools for dataset management:

```bash
# List remote datasets available on Farama server
minari list remote

# List local datasets
minari list local

# Download a dataset
minari download D4RL/door/human-v2

# Delete a local dataset
minari delete D4RL/door/human-v2

# Show version
minari --version

# Combine multiple datasets
minari combine dataset1-v0 dataset2-v0 --new-name combined-v0
```

## Supported Spaces

Minari supports the following Gymnasium spaces:
- **Box**: Continuous spaces
- **Discrete**: Discrete action/observation spaces
- **MultiBinary**: Binary vector spaces
- **MultiDiscrete**: Multiple discrete spaces
- **Dict**: Dictionary of spaces (nested)
- **Tuple**: Tuple of spaces (nested)

## Working with Different Data Formats

### Multivariate Observations/Actions

```python
# For environments with Dict observation spaces
for episode in dataset.iterate_episodes():
    # Observations might be nested dictionaries
    if isinstance(episode.observations, dict):
        for key, values in episode.observations.items():
            print(f"{key}: {values.shape}")
```

### Handling Infos

```python
# Access additional information stored in infos
for episode in dataset.iterate_episodes():
    if episode.infos:
        # infos is a dict with keys corresponding to info keys
        for key in episode.infos.keys():
            print(f"Info key: {key}")
```

## Creating Datasets (Data Collection)

### Using DataCollector Wrapper

```python
import gymnasium as gym
from minari import DataCollector

# Wrap environment with DataCollector
env = gym.make('CartPole-v1')
env = DataCollector(env, record_infos=True)

# Collect episodes
for _ in range(100):
    env.reset(seed=42)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

# Create dataset from collected data
dataset = env.create_dataset(
    dataset_id="cartpole/random-v0",
    algorithm_name="Random-Policy",
    author="Your Name",
    author_email="your.email@example.com"
)
```

### Creating from Buffers

```python
# If you have episode buffers as dictionaries
episode_buffers = [
    {
        'observations': obs_array,  # shape: (steps+1, *obs_shape)
        'actions': act_array,       # shape: (steps, *action_shape)
        'rewards': rew_array,       # shape: (steps,)
        'terminations': term_array, # shape: (steps,)
        'truncations': trunc_array, # shape: (steps,)
    }
    # ... more episodes
]

dataset = minari.create_dataset_from_buffers(
    dataset_id="my-env/my-dataset-v0",
    buffer=episode_buffers,
    env="MyEnv-v0",  # or environment object or EnvSpec
    algorithm_name="My Algorithm"
)
```

## Advanced Usage

### Remote Storage

Set custom remote storage:

```bash
# Google Cloud Platform
export MINARI_REMOTE="gcs://my-bucket"

# Hugging Face Hub
export MINARI_REMOTE="hf://my-username"
```

### Dataset Metadata

```python
# Access comprehensive metadata
spec = dataset.spec

print(f"Dataset ID: {spec.dataset_id}")
print(f"Environment: {spec.env_spec}")
print(f"Total episodes: {spec.total_episodes}")
print(f"Total steps: {spec.total_steps}")
print(f"Observation space: {spec.observation_space}")
print(f"Action space: {spec.action_space}")
print(f"Combined datasets: {spec.combined_datasets}")
```

## Common Use Cases for MBRL

### Extracting State-Action-Next State Trajectories

```python
import numpy as np

def extract_transitions(dataset):
    """Extract (s, a, s', r, done) transitions for dynamics modeling."""
    all_states = []
    all_actions = []
    all_next_states = []
    all_rewards = []
    all_dones = []
    
    for episode in dataset.iterate_episodes():
        states = episode.observations[:-1]      # All states except last
        next_states = episode.observations[1:]  # All states except first
        actions = episode.actions
        rewards = episode.rewards
        dones = episode.terminations | episode.truncations
        
        all_states.append(states)
        all_actions.append(actions)
        all_next_states.append(next_states)
        all_rewards.append(rewards)
        all_dones.append(dones)
    
    return (
        np.concatenate(all_states),
        np.concatenate(all_actions),
        np.concatenate(all_next_states),
        np.concatenate(all_rewards),
        np.concatenate(all_dones)
    )

# Use it
states, actions, next_states, rewards, dones = extract_transitions(dataset)
print(f"Total transitions: {len(states)}")
```

### Creating Rolling Windows for Time Series

```python
def create_time_series_windows(episode, lookback=10, horizon=1):
    """Create overlapping windows from episode data."""
    obs = episode.observations[:-1]  # Exclude final observation
    actions = episode.actions
    
    windows = []
    for i in range(len(obs) - lookback - horizon + 1):
        context_obs = obs[i:i+lookback]
        context_actions = actions[i:i+lookback]
        future_obs = obs[i+lookback:i+lookback+horizon]
        
        windows.append({
            'context_obs': context_obs,
            'context_actions': context_actions,
            'target_obs': future_obs
        })
    
    return windows

# Process all episodes
all_windows = []
for episode in dataset.iterate_episodes():
    windows = create_time_series_windows(episode, lookback=20, horizon=5)
    all_windows.extend(windows)
```

## Important Notes

1. **Observations Include Initial State**: The observations array has `steps+1` elements because it includes both the initial observation (from reset) and all subsequent observations.

2. **Actions Match Steps**: The actions array has `steps` elements, one for each step taken.

3. **Episode Must Be Complete**: Episodes must be either terminated or truncated before being saved to a dataset.

4. **Memory Considerations**: When working with large datasets, use `iterate_episodes()` instead of loading all episodes at once with `sample_episodes()`.

5. **Data Types**: All arrays are NumPy arrays. Convert to torch tensors or other formats as needed.

## Common D4RL Datasets Available

Popular datasets you can download:
- `D4RL/door/human-v2` - Door opening task with human demonstrations
- `D4RL/door/expert-v2` - Door opening with expert policy
- `D4RL/door/cloned-v2` - Door opening with cloned policy
- `D4RL/pen/human-v2` - Pen manipulation
- `D4RL/hammer/human-v2` - Hammer manipulation
- Many more available via `minari list remote`

## Troubleshooting

### Dataset Not Found
```python
# Download dataset if not present
dataset = minari.load_dataset("D4RL/door/human-v2", download=True)
```

### Custom Dataset Path
```python
import os
os.environ['MINARI_DATASETS_PATH'] = '/path/to/my/datasets'
```

### Space Serialization Issues
If you encounter issues with custom spaces, ensure they're supported by Minari or implement custom serialization.
