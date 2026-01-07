"""
Configuration management for the analysis pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class Config:
    """
    Configuration for DESI BAO analysis pipeline.

    Attributes:
        data_dir: Path to data directory
        output_dir: Path to output directory
        desi_data_url: Base URL for DESI data (configurable via DESI_DATA_URL env var)
        plot_style: Matplotlib style settings
        analysis_params: Analysis-specific parameters
    """

    data_dir: Path = Path("artifacts/data")
    output_dir: Path = Path("docs")
    desi_data_url: str = field(default_factory=lambda: os.getenv(
        "DESI_DATA_URL",
        "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base_w_wa"
    ))
    
    plot_style: Dict[str, Any] = field(default_factory=lambda: {
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (10, 8),
        'lines.linewidth': 2
    })
    
    analysis_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_mcmc_samples': 1_000_000,
        'w0_prior': (-3.0, 1.0),
        'wa_prior': (-3.0, 2.0),
        'survey_volume': 10.0,  # (Gpc/h)^3
        'bao_scale': 100.0,  # Mpc/h
        'dao_amplitude_range': (0.01, 0.05),
    })

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'desi_data_url': self.desi_data_url,
            'plot_style': self.plot_style,
            'analysis_params': self.analysis_params,
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Config object
    """
    if config_path and config_path.exists():
        return Config.load(config_path)
    return Config()
