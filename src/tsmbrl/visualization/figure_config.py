"""Publication-quality figure configuration for TSMBRL."""

from typing import Dict, Optional

import matplotlib.pyplot as plt

# Publication style configuration
FIGURE_CONFIG: Dict[str, any] = {
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
}

# Color palette
COLORS: Dict[str, str] = {
    "ground_truth": "#2C3E50",  # Dark blue-gray
    "prediction": "#E74C3C",  # Red
    "context": "#3498DB",  # Blue
    "uncertainty_fill": "#E74C3C",  # Red for uncertainty bands
    "uncertainty_alpha": 0.3,
    "grid": "#BDC3C7",  # Light gray
}

# Model-specific colors
MODEL_COLORS: Dict[str, str] = {
    "chronos2": "#E74C3C",  # Red
    "chronos2-small": "#3498DB",  # Blue
    "chronos2-tiny": "#2ECC71",  # Green
    "chronos2-mini": "#9B59B6",  # Purple
    "chronos2-base": "#E74C3C",  # Red
    "chronos2-large": "#F39C12",  # Orange
    "baseline": "#95A5A6",  # Gray
}

# Line styles for multiple models
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# Marker styles
MARKERS = ["o", "s", "^", "D", "v", "p", "*"]


def apply_style() -> None:
    """Apply publication style to matplotlib."""
    plt.rcParams.update(FIGURE_CONFIG)


def reset_style() -> None:
    """Reset matplotlib to default style."""
    plt.rcParams.update(plt.rcParamsDefault)


def get_model_color(model_name: str) -> str:
    """
    Get color for a model.

    Args:
        model_name: Name of the model

    Returns:
        Hex color code
    """
    # Check for exact match
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    # Check for partial match
    for key in MODEL_COLORS:
        if key in model_name.lower():
            return MODEL_COLORS[key]

    # Default gray
    return "#7F8C8D"


def get_model_linestyle(idx: int) -> str:
    """
    Get line style for model index.

    Args:
        idx: Model index

    Returns:
        Matplotlib line style
    """
    return LINE_STYLES[idx % len(LINE_STYLES)]


def get_model_marker(idx: int) -> str:
    """
    Get marker style for model index.

    Args:
        idx: Model index

    Returns:
        Matplotlib marker style
    """
    return MARKERS[idx % len(MARKERS)]


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[tuple] = None,
    apply_pub_style: bool = True,
) -> tuple:
    """
    Create a figure with publication style.

    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        apply_pub_style: Whether to apply publication style

    Returns:
        Tuple of (figure, axes)
    """
    if apply_pub_style:
        apply_style()

    if figsize is None:
        # Scale figure size based on subplots
        base_width = 6
        base_height = 4
        figsize = (base_width * ncols, base_height * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_figure(
    fig,
    path: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    transparent: bool = False,
) -> None:
    """
    Save figure with publication settings.

    Args:
        fig: Matplotlib figure
        path: Output path
        dpi: Resolution
        bbox_inches: Bounding box setting
        transparent: Whether to use transparent background
    """
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
    )
    plt.close(fig)
