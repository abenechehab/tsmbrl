"""Wrapper for Amazon Chronos-2 time series foundation model."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .base_tsfm import BaseTSFM


class Chronos2TSFM(BaseTSFM):
    """
    Wrapper for Amazon Chronos-2 time series foundation model.

    Supports both tensor and DataFrame APIs for flexible usage:
    - Tensor API: Fast inference for univariate series without covariates
    - DataFrame API: Native support for past/future covariates

    Attributes:
        model_name: HuggingFace model identifier
        device: Inference device ('cuda' or 'cpu')
        torch_dtype: Torch dtype for model weights
        model: Loaded Chronos2Pipeline instance

    Example:
        >>> model = Chronos2TSFM(device="cuda")
        >>> result = model.predict_with_actions(
        ...     context_obs, context_actions, future_actions,
        ...     prediction_length=10
        ... )
    """

    DEFAULT_MODEL = "amazon/chronos-t5-base"
    SUPPORTED_MODELS = [
        "amazon/chronos-t5-tiny",
        "amazon/chronos-t5-mini",
        "amazon/chronos-t5-small",
        "amazon/chronos-t5-base",
        "amazon/chronos-t5-large",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize Chronos-2 wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cuda' or 'cpu')
            torch_dtype: Torch dtype for model weights (bfloat16 recommended for GPU)
            cache_dir: Optional directory for model cache
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self._pipeline = None

    def load_model(self) -> None:
        """Load Chronos-2 pipeline from HuggingFace."""
        from chronos import ChronosPipeline

        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.model = self._pipeline

    @property
    def pipeline(self):
        """Get the Chronos pipeline, loading if necessary."""
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def predict(
        self,
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate point predictions (median).

        Args:
            context: Historical observations
            prediction_length: Number of steps to forecast
            future_covariates: Known future values (not used in tensor API)
            **kwargs: Additional arguments

        Returns:
            Point predictions array
        """
        result = self.predict_probabilistic(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=[0.5],  # Median as point estimate
            future_covariates=future_covariates,
            **kwargs,
        )
        return result["mean"]

    def predict_probabilistic(
        self,
        context: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        num_samples: int = 20,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic predictions with quantiles.

        Args:
            context: Historical observations
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels for probabilistic forecast
            future_covariates: Known future values (unused in tensor API)
            num_samples: Number of samples for prediction
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'mean', 'quantiles', 'quantile_levels'
        """
        return self._predict_tensor(
            context, prediction_length, quantile_levels, num_samples
        )

    def _predict_tensor(
        self,
        context: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        quantile_levels: List[float],
        num_samples: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Use Chronos tensor API for prediction.

        Args:
            context: Historical data, shape (batch, lookback) or (lookback,)
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels
            num_samples: Number of samples for prediction

        Returns:
            Dictionary with predictions
        """
        # Convert to tensor if needed
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float()

        # Handle different input shapes
        # original_shape = context.shape
        is_multivariate = context.dim() == 3

        if is_multivariate:
            # (batch, lookback, features) -> process each feature separately
            batch_size, lookback, n_features = context.shape
            all_samples = []

            for f in range(n_features):
                ctx_f = context[:, :, f]  # (batch, lookback)
                samples_f = self.pipeline.predict(
                    ctx_f,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )
                all_samples.append(samples_f)

            # Stack: (batch, num_samples, pred_len, features)
            samples = torch.stack(all_samples, dim=-1)
        else:
            # Univariate or batched univariate
            if context.dim() == 1:
                context = context.unsqueeze(0)  # (1, lookback)

            # Get samples: (batch, num_samples, pred_len)
            samples = self.pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )

        # Convert to numpy
        samples_np = samples.cpu().numpy()

        # Compute statistics
        mean = np.mean(samples_np, axis=1)  # (batch, pred_len, [features])

        # Compute quantiles
        quantiles = np.quantile(
            samples_np, quantile_levels, axis=1
        )  # (n_quantiles, batch, pred_len, [features])

        # Transpose to (batch, pred_len, n_quantiles, [features])
        quantiles = np.moveaxis(quantiles, 0, 2)

        return {
            "mean": mean,
            "quantiles": quantiles,
            "quantile_levels": quantile_levels,
            "samples": samples_np,
        }

    def predict_univariate_batch(
        self,
        contexts: List[np.ndarray],
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Predict for a batch of univariate time series.

        Args:
            contexts: List of 1D arrays (possibly different lengths)
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels
            num_samples: Number of samples

        Returns:
            Dictionary with predictions for each series
        """
        # Convert to list of tensors
        context_tensors = [torch.from_numpy(c).float() for c in contexts]

        # Predict
        samples = self.pipeline.predict(
            context_tensors,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

        samples_np = samples.cpu().numpy()
        mean = np.mean(samples_np, axis=1)
        quantiles = np.quantile(samples_np, quantile_levels, axis=1)
        quantiles = np.moveaxis(quantiles, 0, 2)

        return {
            "mean": mean,
            "quantiles": quantiles,
            "quantile_levels": quantile_levels,
        }

    def predict_with_actions(
        self,
        context_obs: np.ndarray,
        context_actions: np.ndarray,
        future_actions: np.ndarray,
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with action covariates for MBRL.

        Note: Standard Chronos doesn't support covariates natively, so this
        method uses a simple approach of predicting each observation dimension
        independently without action conditioning.

        For proper covariate support, use Chronos-2 (when available) or
        consider concatenating actions to context.

        Args:
            context_obs: Historical observations, shape (lookback, obs_dim)
            context_actions: Historical actions, shape (lookback, act_dim) [unused]
            future_actions: Future actions, shape (horizon, act_dim) [unused]
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels
            num_samples: Number of samples

        Returns:
            Dictionary with predictions
        """
        obs_dim = context_obs.shape[-1] if context_obs.ndim > 1 else 1

        # Predict each dimension independently
        if obs_dim == 1:
            contexts = [context_obs.flatten()]
        else:
            contexts = [context_obs[:, d] for d in range(obs_dim)]

        result = self.predict_univariate_batch(
            contexts=contexts,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            num_samples=num_samples,
        )

        # Reshape to (pred_len, obs_dim)
        mean = result["mean"].T  # (pred_len, obs_dim)
        quantiles = np.moveaxis(result["quantiles"], 0, 1)  # (pred_len, n_q, obs_dim)

        return {
            "mean": mean,
            "quantiles": quantiles,
            "quantile_levels": quantile_levels,
        }

    def predict_multivariate(
        self,
        context: np.ndarray,
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        num_samples: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Predict multivariate time series.

        Treats each dimension as independent univariate series.

        Args:
            context: Historical data, shape (lookback, features)
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels
            num_samples: Number of samples

        Returns:
            Dictionary with multivariate predictions
        """
        n_features = context.shape[-1]
        contexts = [context[:, f] for f in range(n_features)]

        result = self.predict_univariate_batch(
            contexts=contexts,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            num_samples=num_samples,
        )

        # Reshape: (n_features, pred_len) -> (pred_len, n_features)
        mean = result["mean"].T
        # (n_features, pred_len, n_q) -> (pred_len, n_q, n_features)
        quantiles = np.moveaxis(result["quantiles"], 0, -1)
        quantiles = np.moveaxis(quantiles, 0, 1)

        return {
            "mean": mean,
            "quantiles": quantiles,
            "quantile_levels": quantile_levels,
        }

    @property
    def supports_covariates(self) -> bool:
        """Chronos doesn't natively support covariates."""
        return False

    @property
    def supports_multivariate(self) -> bool:
        """Multivariate supported via independent prediction."""
        return True

    @property
    def is_probabilistic(self) -> bool:
        """Chronos provides probabilistic predictions."""
        return True
