"""
Analysis module for DESI BAO, Supernova, and Dark Matter analyses.

This module provides high-level analysis functions for:
- Wedge analysis (canonical quintessence vs phantom)
- DAO stress testing
- Supernova age bias corrections
- Dark matter parameter scans
"""

from .wedge import wedge_analysis, wedge_mask

__all__ = [
    "wedge_analysis",
    "wedge_mask",
]
