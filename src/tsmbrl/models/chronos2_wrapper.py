"""Wrapper for Amazon Chronos-2 time series foundation model."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .base_tsfm import BaseTSFM


class Chronos2TSFM(BaseTSFM):
    """
    Wrapper for Amazon Chronos-2 time series foundation model.

    Supports two prediction modes:
    - Tensor API: Fast inference for univariate series (no covariates)
    - DataFrame API: Full covariate support via predict_with_actions()

    Example:
        >>> model = Chronos2TSFM(device="cuda")
        >>> # Without actions (tensor API)
        >>> result = model.predict_probabilistic(context, prediction_length=10)
        >>> # With actions as covariates (DataFrame API)
        >>> result = model.predict_with_actions(
        ...     context_obs, context_actions, future_actions, prediction_length=10
        ... )
    """

    DEFAULT_MODEL = "amazon/chronos-2"
    SUPPORTED_MODELS = [
        "amazon/chronos-2",
        "amazon/chronos-2-small",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Chronos-2 wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cuda' or 'cpu')
            torch_dtype: Torch dtype for model weights
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
        self._pipeline = None

    def load_model(self) -> None:
        """Load Chronos-2 pipeline from HuggingFace."""
        from chronos import Chronos2Pipeline

        self._pipeline = Chronos2Pipeline.from_pretrained(
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
        context: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Generate point predictions (median).

        Args:
            context: Historical observations, shape (lookback,) or (lookback, features)
            prediction_length: Number of steps to forecast
            future_covariates: Not used in tensor API (use predict_with_actions instead)
            **kwargs: Additional arguments passed to predict_probabilistic

        Returns:
            Point predictions array, shape (prediction_length,) or (prediction_length, features)
        """
        result = self.predict_probabilistic(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            **kwargs,
        )
        return result["mean"]

    def predict_probabilistic(
        self,
        context: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9],
        future_covariates: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Generate probabilistic predictions using tensor API.

        For action-conditioned predictions, use predict_with_actions() instead.

        Args:
            context: Historical observations, shape (lookback,) or (lookback, features)
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels for probabilistic forecast
            future_covariates: Not used (use predict_with_actions for covariates)
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'mean', 'quantiles', 'quantile_levels'
        """
        # Convert to tensor
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float()

        # Handle multivariate by predicting each dimension independently
        if context.dim() == 2:
            n_features = context.shape[1]
            all_means = []
            all_quantiles = []

            for f in range(n_features):
                ctx_f = context[:, f]
                quantiles_f, mean_f = self.pipeline.predict_quantiles(
                    context=ctx_f,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                )
                all_means.append(mean_f.cpu().numpy())
                all_quantiles.append(quantiles_f.cpu().numpy())

            # Stack: (prediction_length, features)
            mean = np.stack([m[0] for m in all_means], axis=-1)
            # (prediction_length, n_quantiles, features)
            quantiles = np.stack([q[0] for q in all_quantiles], axis=-1)

        else:
            # Univariate
            quantiles_t, mean_t = self.pipeline.predict_quantiles(
                context=context,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            )
            mean = mean_t[0].cpu().numpy()  # (prediction_length,)
            quantiles = quantiles_t[0].cpu().numpy()  # (prediction_length, n_quantiles)

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
    ) -> Dict[str, np.ndarray]:
        """
        Predict with actions as future covariates using DataFrame API.

        This method uses Chronos-2's native covariate support to condition
        predictions on known future actions (the MBRL setting).

        Args:
            context_obs: Historical observations, shape (lookback, obs_dim)
            context_actions: Historical actions, shape (lookback, act_dim)
            future_actions: Known future actions, shape (horizon, act_dim)
            prediction_length: Number of steps to forecast
            quantile_levels: Quantile levels for probabilistic forecast

        Returns:
            Dictionary with 'mean', 'quantiles', 'quantile_levels'
        """
        lookback = context_obs.shape[0]
        obs_dim = context_obs.shape[1] if context_obs.ndim > 1 else 1
        act_dim = context_actions.shape[1] if context_actions.ndim > 1 else 1

        # Ensure 2D
        if context_obs.ndim == 1:
            context_obs = context_obs.reshape(-1, 1)
        if context_actions.ndim == 1:
            context_actions = context_actions.reshape(-1, 1)
        if future_actions.ndim == 1:
            future_actions = future_actions.reshape(-1, 1)

        # Build context DataFrame
        context_rows = []
        for t in range(lookback):
            for d in range(obs_dim):
                row = {
                    "timestamp": t,
                    "id": f"obs_{d}",
                    "target": context_obs[t, d],
                }
                # Add actions as past covariates
                for a in range(act_dim):
                    row[f"action_{a}"] = context_actions[t, a]
                context_rows.append(row)

        context_df = pd.DataFrame(context_rows)

        # Build future DataFrame with actions as covariates
        future_rows = []
        for t in range(prediction_length):
            for d in range(obs_dim):
                row = {
                    "timestamp": lookback + t,
                    "id": f"obs_{d}",
                }
                # Add future actions as covariates
                for a in range(act_dim):
                    row[f"action_{a}"] = future_actions[t, a]
                future_rows.append(row)

        future_df = pd.DataFrame(future_rows)

        # Predict using DataFrame API
        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Parse output DataFrame into arrays
        mean = np.zeros((prediction_length, obs_dim))
        quantiles = np.zeros((prediction_length, len(quantile_levels), obs_dim))

        for d in range(obs_dim):
            dim_df = pred_df[pred_df["id"] == f"obs_{d}"].sort_values("timestamp")
            mean[:, d] = dim_df["mean"].values

            for q_idx, q_level in enumerate(quantile_levels):
                q_col = str(q_level)
                if q_col in dim_df.columns:
                    quantiles[:, q_idx, d] = dim_df[q_col].values

        # Squeeze if univariate
        if obs_dim == 1:
            mean = mean.squeeze(-1)
            quantiles = quantiles.squeeze(-1)

        return {
            "mean": mean,
            "quantiles": quantiles,
            "quantile_levels": quantile_levels,
        }

    @property
    def supports_covariates(self) -> bool:
        """Chronos-2 supports covariates via DataFrame API."""
        return True

    @property
    def supports_multivariate(self) -> bool:
        """Multivariate supported via independent prediction per dimension."""
        return True

    @property
    def is_probabilistic(self) -> bool:
        """Chronos-2 provides probabilistic predictions."""
        return True
