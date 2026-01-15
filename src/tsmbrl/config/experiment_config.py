"""Configuration dataclasses for TSMBRL experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class DataConfig:
    """
    Configuration for data loading and formatting.

    Attributes:
        dataset_id: Minari dataset identifier (e.g., "D4RL/door/human-v2")
        lookback: Number of past timesteps for context window
        horizon: Number of future timesteps to predict
        use_actions: Whether to include actions as covariates
        flatten_observations: Whether to flatten Dict observations
        max_episodes: Maximum number of episodes to load (None for all)
        windows_per_episode: Maximum windows to create per episode
        stride: Step size between consecutive windows
    """

    dataset_id: str
    lookback: int = 50
    horizon: int = 10
    use_actions: bool = True
    flatten_observations: bool = True
    max_episodes: Optional[int] = None
    windows_per_episode: int = 20
    stride: int = 1


@dataclass
class ModelConfig:
    """
    Configuration for TSFM model.

    Attributes:
        model_name: Model identifier (e.g., "chronos2", "chronos2-small")
        device: Device for inference ("cuda" or "cpu")
        torch_dtype: Torch dtype for model weights (e.g., "bfloat16", "float32")
        cache_dir: Optional directory for model cache
    """

    model_name: str = "chronos2"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    cache_dir: Optional[str] = None


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a TSMBRL experiment.

    Combines data, model, and output settings into a single configuration.

    Attributes:
        dataset_name: Short dataset name (e.g., "door-human") or full Minari ID
        model_name: Model identifier
        lookback: Context window length
        horizon: Prediction horizon
        use_actions_as_covariates: Whether to use actions as TSFM covariates
        device: Inference device
        quantile_levels: Quantile levels for probabilistic forecasting
        compute_probabilistic_metrics: Whether to compute CRPS, calibration, etc.
        max_episodes: Maximum episodes to process
        windows_per_episode: Maximum windows per episode
        output_dir: Directory for results
        save_predictions: Whether to save raw predictions
        seed: Random seed for reproducibility
    """

    # Dataset settings
    dataset_name: str
    max_episodes: Optional[int] = None
    windows_per_episode: int = 20

    # Model settings
    model_name: str = "chronos2"
    device: str = "cuda"

    # Forecasting settings
    lookback: int = 50
    horizon: int = 10
    use_actions_as_covariates: bool = True

    # Evaluation settings
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    compute_probabilistic_metrics: bool = True

    # Output settings
    output_dir: Union[str, Path] = field(default_factory=lambda: Path("results"))
    save_predictions: bool = False

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "use_actions_as_covariates": self.use_actions_as_covariates,
            "device": self.device,
            "quantile_levels": self.quantile_levels,
            "compute_probabilistic_metrics": self.compute_probabilistic_metrics,
            "max_episodes": self.max_episodes,
            "windows_per_episode": self.windows_per_episode,
            "output_dir": str(self.output_dir),
            "save_predictions": self.save_predictions,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**data)
