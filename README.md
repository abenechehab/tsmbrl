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

### From Source

```bash
git clone https://github.com/tsmbrl/tsmbrl.git
cd tsmbrl
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
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
    result = model.predict_with_actions(
        context_obs=window.context_observations,
        context_actions=window.context_actions,
        future_actions=window.future_actions,
        prediction_length=10
    )
```

### CLI Usage

```bash
# Run inference experiment
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
    --no-actions \
    --output results/baseline.json
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
