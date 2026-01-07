"""
desibao: Late-2025 Cosmology & Dark Sector Analysis

A professional software package for the analysis of DESI Y3 BAO data,
Supernova systematics, and Dark Matter phenomenology.
"""

__version__ = "1.0.0"
__author__ = "DESI Analysis Team"

from . import cosmology
from . import analysis
from . import utils

__all__ = [
    "cosmology",
    "analysis",
    "utils",
]
