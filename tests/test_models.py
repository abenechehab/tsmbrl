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
                self,
                context,
                prediction_length,
                future_covariates=None,
                quantile_levels=None,
                **kwargs,
            ):
                result = {"mean": np.zeros((prediction_length,))}
                if quantile_levels:
                    result["quantiles"] = np.zeros(
                        (prediction_length, len(quantile_levels))
                    )
                    result["quantile_levels"] = quantile_levels
                return result

        model = DummyTSFM("dummy_model", device="cpu")
        assert model.model_name == "dummy_model"
        assert model.device == "cpu"
        assert model.is_probabilistic is True
        assert model.supports_covariates is False

    def test_base_tsfm_predict_point(self):
        """Test point predictions (no quantiles)."""

        class DummyTSFM(BaseTSFM):
            def load_model(self):
                pass

            def predict(
                self,
                context,
                prediction_length,
                future_covariates=None,
                quantile_levels=None,
                **kwargs,
            ):
                result = {"mean": np.ones((prediction_length,))}
                if quantile_levels:
                    result["quantiles"] = np.ones(
                        (prediction_length, len(quantile_levels))
                    )
                    result["quantile_levels"] = quantile_levels
                return result

        model = DummyTSFM("test", device="cpu")
        context = np.random.randn(50)

        # Point prediction only
        result = model.predict(context, prediction_length=10)
        assert "mean" in result
        assert result["mean"].shape == (10,)
        assert "quantiles" not in result

        # With quantiles
        result = model.predict(
            context, prediction_length=10, quantile_levels=[0.1, 0.5, 0.9]
        )
        assert "mean" in result
        assert "quantiles" in result
        assert result["quantiles"].shape == (10, 3)

    def test_base_tsfm_repr(self):
        """Test string representation."""

        class DummyTSFM(BaseTSFM):
            def load_model(self):
                pass

            def predict(self, context, prediction_length, **kwargs):
                return {"mean": np.zeros((prediction_length,))}

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
        model.model_name = "amazon/chronos-2"
        model.device = "cpu"
        model._pipeline = None

        assert model.supports_covariates is True
        assert model.supports_multivariate is False  # Loops over dimensions
        assert model.is_probabilistic is True

    def test_supported_models(self):
        """Test supported model list."""
        from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

        assert len(Chronos2TSFM.SUPPORTED_MODELS) > 0
        assert "amazon/chronos-2" in Chronos2TSFM.SUPPORTED_MODELS

    def test_predict_interface(self):
        """Test that predict method has correct signature."""
        from tsmbrl.models.chronos2_wrapper import Chronos2TSFM
        import inspect

        sig = inspect.signature(Chronos2TSFM.predict)
        params = list(sig.parameters.keys())

        assert "context" in params
        assert "prediction_length" in params
        assert "future_covariates" in params
        assert "quantile_levels" in params
