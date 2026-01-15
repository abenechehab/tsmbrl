"""File I/O utilities for TSMBRL."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles NumPy arrays and scalar types.

    Converts numpy arrays to lists and numpy scalars to Python types
    for JSON serialization.

    Example:
        >>> data = {"array": np.array([1, 2, 3]), "scalar": np.float64(3.14)}
        >>> json.dumps(data, cls=NumpyEncoder)
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Save experiment results to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        results: Dictionary containing experiment results
        path: Output file path
        indent: JSON indentation level (default: 2)

    Example:
        >>> results = {"mse": 0.01, "predictions": np.array([1, 2, 3])}
        >>> save_results(results, "results/experiment.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=indent)


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Dictionary containing loaded results

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)

    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
