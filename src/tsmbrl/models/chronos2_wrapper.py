"""Wrapper for Amazon Chronos-2 time series foundation model."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .base_tsfm import BaseTSFM


class Chronos2TSFM(BaseTSFM):
    """
    Wrapper for Amazon Chronos-2 time series foundation model.

    Provides a unified predict() method that automatically selects the best
    prediction strategy based on input and model capabilities:
    - With future_covariates → DataFrame API with covariate support
    - Without covariates → fast tensor API
    - Multivariate data → loops over dimensions (Chronos-2 is univariate)

    Example:
        >>> model = Chronos2TSFM(device="cuda")
        >>> # Without covariates
        >>> result = model.predict(context, prediction_length=10)
        >>> # With actions as covariates
        >>> result = model.predict(
        ...     context_obs, prediction_length=10,
        ...     future_covariates=future_actions,
        ...     quantile_levels=[0.1, 0.5, 0.9]
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
        future_covariates: Optional[np.ndarray] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate predictions.

        Automatically selects the prediction strategy:
        - If future_covariates provided → use DataFrame API with covariates
        - Otherwise → use fast tensor API
        - For multivariate context → loop over dimensions

        Args:
            context: Historical observations
                - Shape (lookback,) for univariate
                - Shape (lookback, features) for multivariate
            prediction_length: Number of steps to forecast
            future_covariates: Known future values (e.g., actions)
                - Shape (prediction_length, covariate_dim)
            quantile_levels: Quantile levels for probabilistic forecast
                - If None, only point predictions returned
            **kwargs: Additional arguments

        Returns:
            Dictionary with 'mean' and optionally 'quantiles', 'quantile_levels'
        """
        # Use DataFrame API if covariates provided
        if future_covariates is not None and self.supports_covariates:
            return self._predict_with_covariates(
                context, prediction_length, future_covariates, quantile_levels
            )

        # Use tensor API (faster, no covariates)
        return self._predict_tensor(context, prediction_length, quantile_levels)

    def _predict_tensor(
        self,
        context: Union[np.ndarray, torch.Tensor],
        prediction_length: int,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Predict using fast tensor API (no covariates)."""
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
                if quantile_levels:
                    quantiles_f, mean_f = self.pipeline.predict_quantiles(
                        context=ctx_f,
                        prediction_length=prediction_length,
                        quantile_levels=quantile_levels,
                    )
                    all_means.append(mean_f[0].cpu().numpy())
                    all_quantiles.append(quantiles_f[0].cpu().numpy())
                else:
                    # Just get median as point prediction
                    quantiles_f, mean_f = self.pipeline.predict_quantiles(
                        context=ctx_f,
                        prediction_length=prediction_length,
                        quantile_levels=[0.5],
                    )
                    all_means.append(mean_f[0].cpu().numpy())

            # Stack: (prediction_length, features)
            mean = np.stack(all_means, axis=-1)

            result: Dict[str, Any] = {"mean": mean}

            if quantile_levels:
                # (prediction_length, n_quantiles, features)
                quantiles = np.stack(all_quantiles, axis=-1)
                result["quantiles"] = quantiles
                result["quantile_levels"] = quantile_levels

            return result

        else:
            # Univariate
            if quantile_levels:
                quantiles_t, mean_t = self.pipeline.predict_quantiles(
                    context=context,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                )
                return {
                    "mean": mean_t[0].cpu().numpy(),
                    "quantiles": quantiles_t[0].cpu().numpy(),
                    "quantile_levels": quantile_levels,
                }
            else:
                quantiles_t, mean_t = self.pipeline.predict_quantiles(
                    context=context,
                    prediction_length=prediction_length,
                    quantile_levels=[0.5],
                )
                return {"mean": mean_t[0].cpu().numpy()}

    def _predict_with_covariates(
        self,
        context: np.ndarray,
        prediction_length: int,
        future_covariates: np.ndarray,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Predict using DataFrame API with covariates."""
        # Ensure numpy
        if isinstance(context, torch.Tensor):
            context = context.cpu().numpy()

        lookback = context.shape[0]
        obs_dim = context.shape[1] if context.ndim > 1 else 1
        cov_dim = future_covariates.shape[1] if future_covariates.ndim > 1 else 1

        # Ensure 2D
        if context.ndim == 1:
            context = context.reshape(-1, 1)
        if future_covariates.ndim == 1:
            future_covariates = future_covariates.reshape(-1, 1)

        # Build context DataFrame
        context_rows = []
        for t in range(lookback):
            for d in range(obs_dim):
                row = {
                    "timestamp": t,
                    "id": f"obs_{d}",
                    "target": context[t, d],
                }
                # Add placeholder covariates for context (use zeros or could use past actions)
                for c in range(cov_dim):
                    row[f"cov_{c}"] = 0.0
                context_rows.append(row)

        context_df = pd.DataFrame(context_rows)

        # Build future DataFrame with covariates
        future_rows = []
        for t in range(prediction_length):
            for d in range(obs_dim):
                row = {
                    "timestamp": lookback + t,
                    "id": f"obs_{d}",
                }
                for c in range(cov_dim):
                    row[f"cov_{c}"] = future_covariates[t, c]
                future_rows.append(row)

        future_df = pd.DataFrame(future_rows)

        # Determine quantile levels
        q_levels = quantile_levels if quantile_levels else [0.5]

        # Predict using DataFrame API
        pred_df = self.pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=q_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # Parse output DataFrame into arrays
        mean = np.zeros((prediction_length, obs_dim))
        quantiles = np.zeros((prediction_length, len(q_levels), obs_dim))

        for d in range(obs_dim):
            dim_df = pred_df[pred_df["id"] == f"obs_{d}"].sort_values("timestamp")
            mean[:, d] = dim_df["mean"].values

            for q_idx, q_level in enumerate(q_levels):
                q_col = str(q_level)
                if q_col in dim_df.columns:
                    quantiles[:, q_idx, d] = dim_df[q_col].values

        # Squeeze if univariate
        if obs_dim == 1:
            mean = mean.squeeze(-1)
            quantiles = quantiles.squeeze(-1)

        result: Dict[str, Any] = {"mean": mean}

        if quantile_levels:
            result["quantiles"] = quantiles
            result["quantile_levels"] = quantile_levels

        return result

    @property
    def supports_covariates(self) -> bool:
        """Chronos-2 supports covariates via DataFrame API."""
        return True

    @property
    def supports_multivariate(self) -> bool:
        """Multivariate handled by looping over dimensions."""
        return False  # Chronos-2 is fundamentally univariate

    @property
    def is_probabilistic(self) -> bool:
        """Chronos-2 provides probabilistic predictions."""
        return True
