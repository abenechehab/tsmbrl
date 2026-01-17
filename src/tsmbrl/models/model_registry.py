"""Registry of available TSFM models."""

from typing import Callable, Dict, List, Type

from .base_tsfm import BaseTSFM

# Registry mapping model names to factory functions
MODEL_REGISTRY: Dict[str, Callable[..., BaseTSFM]] = {}


def register_model(name: str) -> Callable[[Type[BaseTSFM]], Type[BaseTSFM]]:
    """
    Decorator to register a model class.

    Args:
        name: Name to register the model under

    Returns:
        Decorator function

    Example:
        >>> @register_model("my_model")
        ... class MyModelTSFM(BaseTSFM):
        ...     pass
    """

    def decorator(cls: Type[BaseTSFM]) -> Type[BaseTSFM]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseTSFM:
    """
    Get a TSFM instance by name.

    Args:
        name: Model name (e.g., "chronos2", "chronos2-small")
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instantiated TSFM wrapper

    Raises:
        ValueError: If model name is not recognized

    Example:
        >>> model = get_model("chronos2", device="cuda")
        >>> predictions = model.predict(context, prediction_length=10)
    """
    from .chronos2_wrapper import Chronos2TSFM

    # Built-in model variants
    model_configs = {
        "chronos2": {"model_name": "amazon/chronos-2"},
        "chronos2-small": {"model_name": "amazon/chronos-2-small"},
    }

    if name in model_configs:
        config = model_configs[name]
        config.update(kwargs)
        return Chronos2TSFM(**config)
    elif name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name](**kwargs)
    else:
        raise ValueError(f"Unknown model: '{name}'. Available models: {list_models()}")


def list_models() -> List[str]:
    """
    List all available model names.

    Returns:
        List of model names
    """
    built_in = [
        "chronos2",
        "chronos2-small",
    ]
    registered = list(MODEL_REGISTRY.keys())
    return built_in + registered


def get_model_info(name: str) -> Dict[str, str]:
    """
    Get information about a model.

    Args:
        name: Model name

    Returns:
        Dictionary with model information
    """
    model_configs = {
        "chronos2": {
            "hf_name": "amazon/chronos-2",
            "description": "Chronos-2 foundation model with covariate support",
        },
        "chronos2-small": {
            "hf_name": "amazon/chronos-2-small",
            "description": "Chronos-2 small variant for faster inference",
        },
    }

    if name in model_configs:
        return model_configs[name]
    elif name in MODEL_REGISTRY:
        return {"description": f"Custom registered model: {name}"}
    else:
        return {"description": f"Unknown model: {name}"}
