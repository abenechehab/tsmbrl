"""Abstract base class for Time Series Foundation Model wrappers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


class BaseTSFM(ABC):
    """
    Abstract base class for all TSFM wrappers.

    This class defines the common interface that all TSFM implementations must follow.
    A single predict() method handles all cases: point/probabilistic, with/without
    covariates, univariate/multivariate.

    Attributes:
        model_name: Name/identifier of the model
        device: Device to run inference on ('cuda' or 'cpu')
        model: The loaded model instance (set by subclass)
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize the TSFM wrapper.

        Args:
            model_name: Name or path of the pretrained model
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model: Any = None

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the pretrained model.

        This method should be implemented by subclasses to load the specific
        TSFM model from HuggingFace or local storage.
        """
        pass

    @abstractmethod
    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        future_covariates: Optional[np.ndarray] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate predictions.

        This unified method handles all prediction scenarios:
        - If future_covariates provided and model supports them → use covariates
        - If model is multivariate → joint prediction, else loop over dimensions
        - If quantile_levels provided and model is probabilistic → include quantiles

        Args:
            context: Historical observations
                - Shape (lookback,) for univariate
                - Shape (lookback, features) for multivariate
            prediction_length: Number of future timesteps to predict
            future_covariates: Known future values (e.g., actions in MBRL)
                - Shape (prediction_length, covariate_dim)
            quantile_levels: Quantile levels for probabilistic forecast (e.g., [0.1, 0.5, 0.9])
                - If None, only point predictions are returned
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing:
                - 'mean': Point predictions, shape (prediction_length,) or (prediction_length, features)
                - 'quantiles': (optional) Quantile predictions if quantile_levels provided
                    Shape (prediction_length, n_quantiles) or (prediction_length, n_quantiles, features)
                - 'quantile_levels': (optional) The quantile levels used
        """
        pass

    @property
    def is_probabilistic(self) -> bool:
        """Whether this TSFM supports probabilistic forecasting."""
        return True

    @property
    def supports_covariates(self) -> bool:
        """Whether this TSFM supports future covariates."""
        return False

    @property
    def supports_multivariate(self) -> bool:
        """Whether this TSFM supports joint multivariate forecasting."""
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}', "
            f"probabilistic={self.is_probabilistic}, "
            f"covariates={self.supports_covariates})"
        )
