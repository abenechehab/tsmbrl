"""Tests for model wrappers."""

import numpy as np
import pytest

from tsmbrl.models.base_tsfm import BaseTSFM
from tsmbrl.models.model_registry import get_model_info, list_models


class TestBaseTSFM:
    """Tests for BaseTSFM abstract class."""

    def test_base_tsfm_abstract(self):
        """Test that BaseTSFM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTSFM("test_model")

    def test_base_tsfm_implementation(self):
        """Test implementing BaseTSFM."""

        class DummyTSFM(BaseTSFM):
            def load_model(self):
                self.model = "dummy"

            def predict(
                self, context, prediction_length, future_covariates=None, **kwargs
            ):
                return np.zeros((prediction_length,))

            def predict_probabilistic(
                self,
                context,
                prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
                future_covariates=None,
                **kwargs,
            ):
                return {
                    "mean": np.zeros((prediction_length,)),
                    "quantiles": np.zeros((prediction_length, len(quantile_levels))),
                    "quantile_levels": quantile_levels,
                }

        model = DummyTSFM("dummy_model", device="cpu")
        assert model.model_name == "dummy_model"
        assert model.device == "cpu"
        assert model.is_probabilistic is True
        assert model.supports_covariates is False

    def test_base_tsfm_repr(self):
        """Test string representation."""

        class DummyTSFM(BaseTSFM):
            def load_model(self):
                pass

            def predict(self, context, prediction_length, **kwargs):
                return np.zeros((prediction_length,))

            def predict_probabilistic(self, context, prediction_length, **kwargs):
                return {}

        model = DummyTSFM("test", device="cpu")
        repr_str = repr(model)
        assert "DummyTSFM" in repr_str
        assert "test" in repr_str


class TestModelRegistry:
    """Tests for model registry."""

    def test_list_models(self):
        """Test listing available models."""
        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "chronos2" in models

    def test_get_model_info(self):
        """Test getting model info."""
        info = get_model_info("chronos2")
        assert "hf_name" in info or "description" in info

    def test_get_model_info_unknown(self):
        """Test getting info for unknown model."""
        info = get_model_info("unknown_model")
        assert "Unknown" in info.get("description", "")


class TestChronos2Wrapper:
    """Tests for Chronos2 wrapper (requires model to be available)."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context data."""
        return np.random.randn(50)

    def test_wrapper_properties(self):
        """Test wrapper properties without loading model."""
        from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

        # Just test initialization, not model loading
        model = Chronos2TSFM.__new__(Chronos2TSFM)
        model.model_name = "amazon/chronos-t5-base"
        model.device = "cpu"
        model._pipeline = None

        assert model.supports_covariates is False
        assert model.supports_multivariate is True
        assert model.is_probabilistic is True

    def test_supported_models(self):
        """Test supported model list."""
        from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

        assert len(Chronos2TSFM.SUPPORTED_MODELS) > 0
        assert "amazon/chronos-t5-base" in Chronos2TSFM.SUPPORTED_MODELS
