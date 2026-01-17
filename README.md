# TSMBRL

**Time Series Foundation Models for Model-Based Reinforcement Learning**

A testbed for evaluating Time Series Foundation Models (TSFMs) on Model-Based RL dynamics modeling tasks.

## Overview

TSMBRL provides tools to:
- Load offline RL datasets from [Minari](https://minari.farama.org/)
- Format RL trajectories for TSFM input with lookback windows and action covariates
- Run inference with [Chronos-2](https://github.com/amazon-science/chronos-forecasting) and other TSFMs
- Compute forecasting metrics (MSE, MAE, RMSE) and probabilistic metrics (CRPS, calibration, coverage)
- Visualize predictions and compare models

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (recommended for faster inference)

### Option 1: Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/abenechehab/tsmbrl.git
cd tsmbrl

# Create and activate conda environment
conda create -n tsmbrl python=3.10 -y
conda activate tsmbrl

# Install PyTorch with CUDA support (adjust cuda version as needed)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install the package in editable mode
pip install -e .
```

### Option 2: Python Virtual Environment

```bash
# Clone the repository
git clone https://github.com/abenechehab/tsmbrl.git
cd tsmbrl

# Create and activate virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
# .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install the package in editable mode
pip install -e .
```

### With Development Dependencies

```bash
# After activating your environment
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check that the package is installed
python -c "import tsmbrl; print(tsmbrl.__version__)"

# Run tests
pytest tests/ -v
```

## Quick Start

### Basic Usage

```python
from tsmbrl.data.minari_loader import MinariDataLoader
from tsmbrl.data.formatters import TimeSeriesFormatter, prepare_evaluation_data
from tsmbrl.models.model_registry import get_model
from tsmbrl.metrics.forecasting_metrics import multi_step_metrics

# Load dataset
loader = MinariDataLoader("D4RL/door/human-v2", download=True)

# Create formatter
formatter = TimeSeriesFormatter(
    lookback=50,
    horizon=10,
    obs_dim=loader.obs_dim,
    act_dim=loader.act_dim
)

# Prepare evaluation windows
windows = prepare_evaluation_data(loader, formatter, max_episodes=10)

# Load model
model = get_model("chronos2", device="cuda")

# Run predictions
for window in windows:
    # Unified predict method - pass future_covariates to use actions
    result = model.predict(
        context=window.context_observations,
        prediction_length=10,
        future_covariates=window.future_actions,  # Actions as covariates
        quantile_levels=[0.1, 0.5, 0.9],  # Optional: for probabilistic predictions
    )
    # result contains 'mean' and optionally 'quantiles', 'quantile_levels'
```

### CLI Usage

```bash
# Run inference with actions as covariates
python -m tsmbrl.inference \
    --dataset door-human \
    --model chronos2 \
    --lookback 50 \
    --horizon 10 \
    --with-actions \
    --output results/experiment.json

# Without actions (baseline)
python -m tsmbrl.inference \
    --dataset door-human \
    --model chronos2 \
    --horizon 10 \
    --output results/baseline.json

# Point predictions only (no quantiles)
python -m tsmbrl.inference \
    --dataset door-human \
    --model chronos2 \
    --no-probabilistic \
    --output results/point_only.json
```

## Project Structure

```
tsmbrl/
├── src/tsmbrl/
│   ├── data/           # Data loading and formatting
│   │   ├── minari_loader.py
│   │   ├── formatters.py
│   │   └── dataset_registry.py
│   ├── models/         # TSFM wrappers
│   │   ├── base_tsfm.py
│   │   ├── chronos2_wrapper.py
│   │   └── model_registry.py
│   ├── metrics/        # Evaluation metrics
│   │   ├── forecasting_metrics.py
│   │   └── probabilistic_metrics.py
│   ├── visualization/  # Plotting tools
│   ├── config/         # Configuration
│   └── inference.py    # Main CLI
├── tests/              # Unit tests
├── examples/           # Usage examples
└── scripts/            # Batch scripts
```

## Supported Datasets

Any dataset available through Minari, including D4RL datasets:
- `door-human`, `door-expert`, `door-cloned`
- `pen-human`, `pen-expert`, `pen-cloned`
- `hammer-human`, `hammer-expert`
- And more...

## Supported Models

- `chronos2` - Amazon Chronos-2 (full model)
- `chronos2-small` - Amazon Chronos-2 (small variant)

## Key Features

### Action Covariates
TSMBRL supports using actions as covariates for dynamics modeling:
- **Past actions**: Included in context as past covariates
- **Future actions**: Known future actions used as future covariates (MBRL assumption)

### Probabilistic Forecasting
Evaluate uncertainty quantification with:
- CRPS (Continuous Ranked Probability Score)
- Calibration error per quantile
- Coverage and interval width metrics

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT License
