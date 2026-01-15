"""Registry of available datasets for TSMBRL."""

from typing import Dict, List

# Registry mapping short names to Minari dataset IDs
DATASET_REGISTRY: Dict[str, str] = {
    # D4RL Door environment (39-dim obs, 28-dim act)
    "door-human": "D4RL/door/human-v2",
    "door-expert": "D4RL/door/expert-v2",
    "door-cloned": "D4RL/door/cloned-v2",
    # D4RL Pen environment (45-dim obs, 24-dim act)
    "pen-human": "D4RL/pen/human-v2",
    "pen-expert": "D4RL/pen/expert-v2",
    "pen-cloned": "D4RL/pen/cloned-v2",
    # D4RL Hammer environment (46-dim obs, 26-dim act)
    "hammer-human": "D4RL/hammer/human-v2",
    "hammer-expert": "D4RL/hammer/expert-v2",
    "hammer-cloned": "D4RL/hammer/cloned-v2",
    # D4RL Relocate environment (39-dim obs, 30-dim act)
    "relocate-human": "D4RL/relocate/human-v2",
    "relocate-expert": "D4RL/relocate/expert-v2",
    "relocate-cloned": "D4RL/relocate/cloned-v2",
}


def get_dataset_id(name: str) -> str:
    """
    Get Minari dataset ID from short name.

    If the name is not in the registry, it's assumed to be a full
    Minari dataset ID and returned as-is.

    Args:
        name: Short dataset name (e.g., "door-human") or full Minari ID

    Returns:
        Full Minari dataset identifier

    Example:
        >>> get_dataset_id("door-human")
        'D4RL/door/human-v2'
        >>> get_dataset_id("D4RL/custom/dataset-v1")
        'D4RL/custom/dataset-v1'
    """
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    # Assume it's already a full Minari ID
    return name


def list_datasets() -> List[str]:
    """
    List all registered dataset short names.

    Returns:
        List of dataset short names
    """
    return list(DATASET_REGISTRY.keys())


def get_dataset_info(name: str) -> Dict[str, str]:
    """
    Get information about a dataset.

    Args:
        name: Short dataset name or full Minari ID

    Returns:
        Dictionary with dataset information
    """
    dataset_id = get_dataset_id(name)
    return {
        "short_name": name if name in DATASET_REGISTRY else None,
        "minari_id": dataset_id,
    }


def register_dataset(short_name: str, minari_id: str) -> None:
    """
    Register a new dataset mapping.

    Args:
        short_name: Short name for the dataset
        minari_id: Full Minari dataset identifier

    Raises:
        ValueError: If short_name already exists
    """
    if short_name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{short_name}' already registered")
    DATASET_REGISTRY[short_name] = minari_id


def unregister_dataset(short_name: str) -> None:
    """
    Remove a dataset from the registry.

    Args:
        short_name: Short name to remove

    Raises:
        KeyError: If short_name not in registry
    """
    del DATASET_REGISTRY[short_name]
