# TSMBRL Repository Structure

## Overview
This repository implements a testbed for evaluating Time Series Foundation Models (TSFMs) on Model-Based Reinforcement Learning (MBRL) dynamics modeling tasks. The structure is organized to support data loading from Minari, TSFM inference, evaluation metrics, and visualization.

## Directory Tree

```
tsmbrl/
├── README.md                          # Project overview and quick start
├── LICENSE                            # License file (e.g., MIT)
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
├── pyproject.toml                     # Modern Python project configuration
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
│
├── docs/                              # Documentation (optional)
│   ├── usage.md                       # Usage guide
│   └── api.md                         # API documentation
│
├── scripts/                           # Executable scripts
│   ├── run_experiments.sh             # Batch run experiments across datasets/models
│   ├── aggregate_results.py           # Collect and aggregate metrics from experiments
│   └── generate_figures.py            # Create publication-quality figures
│
├── examples/                          # Example usage scripts
│   ├── basic_usage.py                 # Simple end-to-end example
│   ├── chronos2_inference.py          # Chronos2-specific example
│   └── plot_predictions.py            # Visualization example
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_data_utils.py             # Test data loading/formatting
│   ├── test_models.py                 # Test TSFM wrappers
│   ├── test_metrics.py                # Test evaluation metrics
│   └── test_inference.py              # Test inference pipeline
│
├── results/                           # Experiment results (gitignored)
│   ├── raw/                           # Raw prediction outputs
│   ├── metrics/                       # Computed metrics (JSON/CSV)
│   └── figures/                       # Generated plots
│
└── src/tsmbrl/                        # Main source code package
    ├── __init__.py                    # Package initialization
    │
    ├── data/                          # Data loading and formatting
    │   ├── __init__.py
    │   ├── minari_loader.py           # Load datasets from Minari
    │   ├── formatters.py              # Format RL data for TSFM input
    │   │                              # - Create lookback windows
    │   │                              # - Handle past/future covariates
    │   │                              # - Support different context sizes
    │   └── dataset_registry.py        # Registry of available datasets
    │
    ├── models/                        # TSFM wrapper implementations
    │   ├── __init__.py
    │   ├── base_tsfm.py               # Abstract base class for TSFMs
    │   │                              # - Defines common interface
    │   │                              # - predict() method signature
    │   │                              # - predict_probabilistic() for uncertainty
    │   ├── chronos2_wrapper.py        # Chronos-2 implementation
    │   └── model_registry.py          # Registry of available models
    │
    ├── metrics/                       # Evaluation metrics
    │   ├── __init__.py
    │   ├── forecasting_metrics.py     # MSE, MAE, RMSE for single/multi-step
    │   ├── probabilistic_metrics.py   # CRPS, calibration, coverage metrics
    │   └── metric_utils.py            # Helper functions for metric computation
    │
    ├── inference.py                   # Main inference script/module
    │                                  # - CLI interface for running experiments
    │                                  # - Load dataset, TSFM, run inference
    │                                  # - Compute and save metrics
    │
    ├── visualization/                 # Plotting and visualization
    │   ├── __init__.py
    │   ├── plot_predictions.py        # Plot TSFM predictions vs ground truth
    │   ├── plot_uncertainty.py        # Visualize uncertainty estimates
    │   ├── plot_comparisons.py        # Compare multiple TSFMs
    │   └── figure_config.py           # Publication-quality figure settings
    │
    ├── config/                        # Configuration management
    │   ├── __init__.py
    │   ├── experiment_config.py       # Experiment configuration dataclasses
    │   └── default_configs.yaml       # Default hyperparameters
    │
    └── utils/                         # Utility functions
        ├── __init__.py
        ├── logging_utils.py           # Logging configuration
        └── file_utils.py              # File I/O helpers
```

## Module Descriptions

### 📦 `src/tsmbrl/data/` - Data Loading and Formatting

#### `minari_loader.py`
**Purpose**: Load offline RL datasets from Minari

**Key Classes/Functions**:
```python
class MinariDataLoader:
    """Load and manage Minari datasets."""
    
    def __init__(self, dataset_id: str, download: bool = True)
    def get_episode(self, episode_idx: int) -> EpisodeData
    def get_all_episodes(self) -> List[EpisodeData]
    def get_metadata(self) -> Dict[str, Any]
```

**Responsibilities**:
- Download/load Minari datasets
- Extract observations, actions, rewards
- Handle different observation/action spaces (Box, Dict, etc.)
- Provide metadata (obs/action dimensions, episode counts)

#### `formatters.py`
**Purpose**: Transform RL data into TSFM-compatible format

**Key Classes/Functions**:
```python
class TimeSeriesFormatter:
    """Format RL trajectories for TSFM inference."""
    
    def create_lookback_windows(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        lookback: int,
        horizon: int
    ) -> Dict[str, np.ndarray]
    
    def format_with_covariates(
        self,
        observations: np.ndarray,
        past_actions: np.ndarray,
        future_actions: Optional[np.ndarray] = None
    ) -> pd.DataFrame  # For DataFrame API (Chronos2)
    
    def to_tensor_format(
        self,
        observations: np.ndarray,
        actions: Optional[np.ndarray] = None
    ) -> torch.Tensor  # For Tensor API
```

**Responsibilities**:
- Create sliding windows (lookback + horizon)
- Format past actions as past covariates
- Format future actions as future covariates
- Handle multivariate observations (flatten if needed)
- Support both DataFrame and Tensor formats

#### `dataset_registry.py`
**Purpose**: Maintain registry of available datasets

**Example**:
```python
DATASETS = {
    "door-human": "D4RL/door/human-v2",
    "door-expert": "D4RL/door/expert-v2",
    "pen-human": "D4RL/pen/human-v2",
    # ... more datasets
}

def get_dataset_id(name: str) -> str:
    """Get Minari dataset ID from short name."""
    return DATASETS[name]
```

---

### 🤖 `src/tsmbrl/models/` - TSFM Wrappers

#### `base_tsfm.py`
**Purpose**: Abstract base class defining unified TSFM interface

**Key Class**:
```python
from abc import ABC, abstractmethod

class BaseTSFM(ABC):
    """Abstract base class for all TSFM wrappers."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None  # Loaded by subclass
    
    @abstractmethod
    def load_model(self):
        """Load the pretrained model."""
        pass
    
    @abstractmethod
    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate point predictions.
        
        Returns:
            predictions: (prediction_length, n_features) array
        """
        pass
    
    @abstractmethod
    def predict_probabilistic(
        self,
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic predictions.
        
        Returns:
            dict with keys:
                - 'mean': Mean predictions
                - 'quantiles': Quantile predictions (prediction_length, n_quantiles)
        """
        pass
    
    @property
    def is_probabilistic(self) -> bool:
        """Whether this TSFM supports probabilistic forecasting."""
        return True
    
    @property
    def supports_covariates(self) -> bool:
        """Whether this TSFM supports covariates."""
        return False
```

#### `chronos2_wrapper.py`
**Purpose**: Chronos-2 specific implementation

**Key Class**:
```python
class Chronos2TSFM(BaseTSFM):
    """Wrapper for Chronos-2 foundation model."""
    
    def __init__(self, model_name: str = "amazon/chronos-2", device: str = "cuda"):
        super().__init__(model_name, device)
        self.load_model()
    
    def load_model(self):
        """Load Chronos-2 pipeline."""
        from chronos import Chronos2Pipeline
        self.model = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )
    
    def predict(self, context, prediction_length, **kwargs):
        """Single-step or multi-step point predictions."""
        # Implementation with DataFrame or Tensor API
        pass
    
    def predict_probabilistic(self, context, prediction_length, 
                             quantile_levels=[0.1, 0.5, 0.9], **kwargs):
        """Probabilistic predictions with quantiles."""
        # Implementation
        pass
    
    @property
    def supports_covariates(self) -> bool:
        return True  # Chronos-2 supports covariates
```

#### `model_registry.py`
**Purpose**: Registry of available models

**Example**:
```python
from typing import Type

MODEL_REGISTRY: Dict[str, Type[BaseTSFM]] = {
    "chronos2": Chronos2TSFM,
    "chronos2-small": lambda: Chronos2TSFM("amazon/chronos-2-small"),
    # ... more models
}

def get_model(model_name: str, **kwargs) -> BaseTSFM:
    """Get TSFM instance by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)
```

---

### 📊 `src/tsmbrl/metrics/` - Evaluation Metrics

#### `forecasting_metrics.py`
**Purpose**: Standard forecasting metrics

**Key Functions**:
```python
def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Squared Error."""
    pass

def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    pass

def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Squared Error."""
    pass

def single_step_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """MSE for single-step predictions."""
    pass

def multi_step_mse(predictions: np.ndarray, targets: np.ndarray) -> List[float]:
    """MSE for each step in multi-step predictions."""
    pass
```

#### `probabilistic_metrics.py`
**Purpose**: Metrics for probabilistic forecasts

**Key Functions**:
```python
def crps(
    quantiles: np.ndarray, 
    quantile_levels: List[float],
    targets: np.ndarray
) -> float:
    """Continuous Ranked Probability Score."""
    pass

def calibration_error(
    quantiles: np.ndarray,
    quantile_levels: List[float],
    targets: np.ndarray
) -> Dict[float, float]:
    """
    Calibration error for each quantile level.
    Returns: {quantile_level: coverage_error}
    """
    pass

def coverage(
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    targets: np.ndarray
) -> float:
    """Empirical coverage of prediction intervals."""
    pass
```

---

### 🎯 `src/tsmbrl/inference.py` - Main Inference Script

**Purpose**: CLI tool for running experiments

**Usage**:
```bash
python -m tsmbrl.inference \
    --dataset door-human \
    --model chronos2 \
    --lookback 50 \
    --horizon 10 \
    --with-actions \
    --output results/door-human_chronos2.json
```

**Key Functions**:
```python
def run_inference(
    dataset_name: str,
    model_name: str,
    lookback: int,
    horizon: int,
    use_actions: bool,
    output_path: str
) -> Dict[str, Any]:
    """
    Run inference experiment.
    
    Returns:
        dict with keys:
            - 'metrics': Computed metrics
            - 'predictions': Raw predictions (optional)
            - 'metadata': Experiment configuration
    """
    # 1. Load dataset
    # 2. Load model
    # 3. Format data with appropriate lookback/covariates
    # 4. Run predictions
    # 5. Compute metrics
    # 6. Save results
    pass

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    # Add arguments
    args = parser.parse_args()
    run_inference(**vars(args))
```

---

### 📈 `src/tsmbrl/visualization/` - Plotting

#### `plot_predictions.py`
**Purpose**: Visualize TSFM predictions vs ground truth

**Key Functions**:
```python
def plot_trajectory_predictions(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],  # {model_name: predictions}
    model_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot predictions from multiple models on a single trajectory.
    
    Args:
        ground_truth: (timesteps, n_features) ground truth observations
        predictions: Dict mapping model names to prediction arrays
        model_names: List of model names to plot
        save_path: Path to save figure
    """
    pass
```

#### `plot_uncertainty.py`
**Purpose**: Visualize uncertainty estimates

**Key Functions**:
```python
def plot_prediction_intervals(
    ground_truth: np.ndarray,
    mean_predictions: np.ndarray,
    quantiles: np.ndarray,
    quantile_levels: List[float],
    save_path: Optional[str] = None
):
    """
    Plot predictions with uncertainty bands.
    """
    pass
```

---

## Configuration Files

### `pyproject.toml`
Modern Python project configuration:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_backend"

[project]
name = "tsmbrl"
version = "0.1.0"
description = "Time Series Foundation Models for Model-Based RL"
authors = [{name = "Your Name", email = "your.email@example.com"}]
requires-python = ">=3.8"
dependencies = [
    "minari>=0.5.0",
    "chronos-forecasting>=2.0",
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "matplotlib>=3.5.0",
    "scipy>=1.9.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "flake8>=5.0",
    "mypy>=0.990",
    "pre-commit>=2.20",
]

[project.scripts]
tsmbrl-inference = "tsmbrl.inference:main"
```

### `requirements.txt`
```txt
# Core dependencies
minari>=0.5.0
chronos-forecasting>=2.0
torch>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
matplotlib>=3.5.0
scipy>=1.9.0
pyyaml>=6.0

# Development dependencies (optional)
pytest>=7.0
pytest-cov>=4.0
black>=22.0
flake8>=5.0
mypy>=0.990
pre-commit>=2.20
```

### `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Results
results/
*.log

# Data
data/
*.hdf5

# Models
models/checkpoints/

# OS
.DS_Store
Thumbs.db
```

### `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.990
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

---

## Example Workflow

### 1. Installation
```bash
cd tsmbrl
pip install -e .
```

### 2. Run Single Experiment
```bash
python -m tsmbrl.inference \
    --dataset door-human \
    --model chronos2 \
    --lookback 50 \
    --horizon 10 \
    --with-actions \
    --output results/experiment1.json
```

### 3. Run Multiple Experiments
```bash
bash scripts/run_experiments.sh
```

### 4. Aggregate Results
```bash
python scripts/aggregate_results.py --input results/raw/ --output results/summary.csv
```

### 5. Generate Figures
```bash
python scripts/generate_figures.py --results results/summary.csv --output results/figures/
```

---

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Use Google-style docstrings
- Run `black` for formatting
- Run `flake8` for linting

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=tsmbrl tests/

# Run specific test
pytest tests/test_data_utils.py::test_minari_loader
```

### Adding New TSFM
1. Create `src/tsmbrl/models/{model_name}_wrapper.py`
2. Implement `BaseTSFM` interface
3. Register in `model_registry.py`
4. Add tests in `tests/test_models.py`
5. Update documentation

### Adding New Metric
1. Add function to `src/tsmbrl/metrics/forecasting_metrics.py` or `probabilistic_metrics.py`
2. Add tests in `tests/test_metrics.py`
3. Update `inference.py` to compute new metric
4. Document in README

---

## Key Design Principles

1. **Modularity**: Each component (data, models, metrics) is independent
2. **Extensibility**: Easy to add new TSFMs, datasets, or metrics
3. **Type Safety**: Use type hints throughout
4. **Testing**: Comprehensive unit tests for all modules
5. **Documentation**: Clear docstrings and examples
6. **Reproducibility**: Configuration files and random seeds
7. **Clean Code**: Follow PEP 8 and use linters

---

## Notes for Claude Code

When implementing this structure:
1. **Start with base classes first** (`base_tsfm.py`, data loaders)
2. **Implement one TSFM wrapper completely** (Chronos2) before others
3. **Test each module independently** as you build
4. **Use small example datasets** for testing before full experiments
5. **Focus on clarity over optimization** initially
6. **Add comprehensive docstrings** to all classes/functions
7. **Create simple examples** in `examples/` directory as you go

This structure supports the paper plan and makes it easy to:
- Compare different TSFMs
- Evaluate with/without action covariates
- Compute both deterministic and probabilistic metrics
- Generate publication-quality figures
