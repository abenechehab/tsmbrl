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
    It supports both point predictions and probabilistic predictions with quantiles.

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
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate point predictions (mean/median).

        Args:
            context: Historical observations
                - Tensor/ndarray: Shape (lookback,) for univariate or (lookback, features)
                - DataFrame: With columns [timestamp, id, target, covariates...]
            prediction_length: Number of future timesteps to predict
            future_covariates: Known future values (e.g., planned actions in MBRL)
            **kwargs: Additional model-specific parameters

        Returns:
            Point predictions with shape (prediction_length,) or (prediction_length, features)
        """
        pass

    @abstractmethod
    def predict_probabilistic(
        self,
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic predictions with quantiles.

        Args:
            context: Historical observations (same format as predict())
            prediction_length: Number of future timesteps to predict
            quantile_levels: List of quantile levels for probabilistic forecast
            future_covariates: Known future values
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary containing:
                - 'mean': Mean predictions, shape (prediction_length,) or (prediction_length, features)
                - 'quantiles': Quantile predictions, shape (prediction_length, num_quantiles) or
                              (prediction_length, num_quantiles, features)
                - 'quantile_levels': List of quantile levels used
        """
        pass

    @property
    def is_probabilistic(self) -> bool:
        """Whether this TSFM supports probabilistic forecasting."""
        return True

    @property
    def supports_covariates(self) -> bool:
        """Whether this TSFM supports past/future covariates."""
        return False

    @property
    def supports_multivariate(self) -> bool:
        """Whether this TSFM supports multivariate forecasting."""
        return False

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}', "
            f"probabilistic={self.is_probabilistic}, "
            f"covariates={self.supports_covariates})"
        )
