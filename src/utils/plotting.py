"""
Matplotlib plotting utilities and style configuration.
"""

from typing import Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def setup_matplotlib_style(style_dict: Optional[dict] = None) -> None:
    """
    Configure matplotlib with consistent styling.

    Args:
        style_dict: Dictionary of matplotlib rcParams
    """
    default_style = {
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 8),
        'lines.linewidth': 2,
        'axes.grid': False,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    }
    
    if style_dict:
        default_style.update(style_dict)
    
    plt.rcParams.update(default_style)


def save_figure(
    fig: Figure,
    path: Path,
    formats: Tuple[str, ...] = ('svg', 'png'),
    dpi: int = 150,
) -> None:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Output path (without extension)
        formats: Tuple of format extensions
        dpi: DPI for raster formats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
