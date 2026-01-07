"""
Wedge analysis for canonical quintessence vs phantom dark energy.

This module analyzes MCMC chains to determine the posterior mass
in the canonical quintessence wedge (w(z) >= -1 for all z).
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import multivariate_normal


def wedge_mask(w0: NDArray, wa: NDArray) -> NDArray:
    """
    Create boolean mask for canonical quintessence wedge.

    Canonical quintessence requires:
    - w0 >= -1
    - w0 + wa >= -1

    Args:
        w0: Array of w0 values
        wa: Array of wa values

    Returns:
        Boolean mask indicating samples in wedge
    """
    return (w0 >= -1.0) & ((w0 + wa) >= -1.0)


def wedge_prior_fraction(
    w0_bounds: Tuple[float, float], wa_bounds: Tuple[float, float]
) -> float:
    """
    Calculate prior volume fraction in canonical wedge.

    Uses numerical integration over the prior box to find
    the fraction that satisfies wedge constraints.

    Args:
        w0_bounds: (min, max) for w0 prior
        wa_bounds: (min, max) for wa prior

    Returns:
        Fraction of prior volume in wedge
    """
    w0min, w0max = w0_bounds
    wamin, wamax = wa_bounds

    # Wedge constraint: w0 >= -1
    a = max(-1.0, w0min)
    b = w0max
    if b <= a:
        return 0.0

    # For each w0 in [a, b], find valid wa range
    grid = np.linspace(a, b, 20001)
    # w0 + wa >= -1  =>  wa >= -1 - w0
    low = np.maximum(wamin, -1.0 - grid)
    # wa also <= wamax
    width = np.maximum(0.0, wamax - low)

    area_wedge = np.trapezoid(width, grid)
    area_box = (w0max - w0min) * (wamax - wamin)

    return float(area_wedge / area_box) if area_box > 0 else 0.0


def wedge_analysis(
    w0: NDArray,
    wa: NDArray,
    weights: Optional[NDArray] = None,
    w0_bounds: Tuple[float, float] = (-3.0, 1.0),
    wa_bounds: Tuple[float, float] = (-3.0, 2.0),
) -> dict:
    """
    Perform comprehensive wedge analysis on parameter samples.

    Args:
        w0: Array of w0 samples
        wa: Array of wa samples
        weights: Optional sample weights (for MCMC chains)
        w0_bounds: Prior bounds on w0
        wa_bounds: Prior bounds on wa

    Returns:
        Dictionary containing:
        - n_samples: Total number of samples
        - n_canonical: Number in wedge
        - post_wedge: Posterior mass in wedge
        - prior_wedge: Prior mass in wedge
        - evidence_ratio: P(W|D) / P(W)
        - log_evidence_ratio: log of evidence ratio
    """
    if weights is None:
        weights = np.ones(len(w0))

    # Normalize weights
    weights = weights / np.sum(weights)

    # Apply wedge mask
    mask = wedge_mask(w0, wa)

    # Posterior mass in wedge
    post_wedge = np.sum(weights[mask])

    # Prior mass in wedge
    prior_wedge = wedge_prior_fraction(w0_bounds, wa_bounds)

    # Evidence ratio (Bayes factor for wedge vs box)
    evidence_ratio = post_wedge / prior_wedge if prior_wedge > 0 else 0.0
    log_evidence_ratio = np.log(evidence_ratio) if evidence_ratio > 0 else -np.inf

    return {
        "n_samples": len(w0),
        "n_canonical": int(np.sum(mask)),
        "post_wedge": float(post_wedge),
        "prior_wedge": float(prior_wedge),
        "evidence_ratio": float(evidence_ratio),
        "log_evidence_ratio": float(log_evidence_ratio),
    }


def generate_wedge_boundary(w0_range: Tuple[float, float] = (-2.0, 0.0)) -> Tuple[NDArray, NDArray]:
    """
    Generate boundary lines for canonical quintessence wedge.

    Args:
        w0_range: Range of w0 values for boundary

    Returns:
        Tuple of (w0_array, wa_boundary) for plotting
    """
    w0_vals = np.linspace(max(-1.0, w0_range[0]), w0_range[1], 100)
    # Boundary: w0 + wa = -1  =>  wa = -1 - w0
    wa_boundary = -1.0 - w0_vals
    return w0_vals, wa_boundary
