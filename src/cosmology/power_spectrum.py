"""
Power spectrum calculations and Dark Acoustic Oscillation (DAO) models.

This module provides tools for computing matter power spectra,
BAO features, and DAO contamination models.
"""

from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray


def linear_power_spectrum(
    k: NDArray, amplitude: float = 2000.0, n_s: float = 0.96, k_pivot: float = 0.05
) -> NDArray:
    """
    Simple parametric linear matter power spectrum (Eisenstein-Hu no-wiggle-like).

    Note: This uses a power-law approximation P(k) ∝ k^(n_s - 4) with exponential cutoff,
    not the standard definition where n_s is the primordial spectral index.

    Args:
        k: Wavenumber array in h/Mpc
        amplitude: Power spectrum amplitude
        n_s: Effective spectral slope (not the primordial spectral index)
        k_pivot: Pivot scale in h/Mpc

    Returns:
        P(k) in (Mpc/h)^3
    """
    k_cutoff = 0.2
    # Note: For actual cosmology, use proper Eisenstein-Hu or CLASS/CAMB
    effective_slope = n_s - 4.0  # Convert to matter power slope
    return amplitude * (k / k_pivot) ** effective_slope * np.exp(-((k / k_cutoff) ** 2))


def bao_wiggles(
    k: NDArray, r_s: float = 100.0, sigma_damp: float = 8.0, amplitude: float = 0.05
) -> NDArray:
    """
    Generate Baryon Acoustic Oscillation wiggles.

    Wiggles = 1 + A * sin(k * r_s) * exp(-0.5 * (k * σ)^2)

    Args:
        k: Wavenumber array in h/Mpc
        r_s: Sound horizon scale in Mpc/h
        sigma_damp: Damping scale in Mpc/h
        amplitude: Wiggle amplitude

    Returns:
        Multiplicative wiggle pattern
    """
    return 1 + amplitude * np.sin(k * r_s) * np.exp(-0.5 * (k * sigma_damp) ** 2)


def power_spectrum_with_bao(
    k: NDArray,
    alpha: float = 1.0,
    r_s: float = 100.0,
    sigma_damp: float = 8.0,
    **smooth_kwargs,
) -> NDArray:
    """
    Compute matter power spectrum with BAO features.

    P_BAO(k) = P_smooth(k) * [1 + A * sin(k * r_s * α) * exp(...)]

    Args:
        k: Wavenumber array in h/Mpc
        alpha: BAO dilation parameter (α = 1 for fiducial cosmology)
        r_s: Sound horizon scale
        sigma_damp: BAO damping scale
        **smooth_kwargs: Arguments for smooth power spectrum

    Returns:
        P(k) with BAO features
    """
    p_smooth = linear_power_spectrum(k, **smooth_kwargs)
    wiggles = bao_wiggles(k, r_s * alpha, sigma_damp)
    return p_smooth * wiggles


def apply_dao_modulation(
    power_spectrum: NDArray,
    k: NDArray,
    A_dao: float,
    r_dao_ratio: float = 0.98,
    r_s: float = 100.0,
    sigma_dao: float = 8.0,
) -> NDArray:
    """
    Apply Dark Acoustic Oscillation contamination to power spectrum.

    DAO adds an additional oscillatory feature slightly offset from BAO:
    P_DAO(k) = P(k) * [1 + A_DAO * sin(k * r_DAO) * exp(...)]

    Args:
        power_spectrum: Input power spectrum array
        k: Wavenumber array
        A_dao: DAO amplitude (typical range: 0.01-0.05)
        r_dao_ratio: Ratio of DAO scale to BAO scale (< 1 for blue-shifted)
        r_s: Standard sound horizon
        sigma_dao: DAO damping scale

    Returns:
        Power spectrum with DAO contamination
    """
    r_dao = r_s * r_dao_ratio
    dao_term = 1 + A_dao * np.sin(k * r_dao) * np.exp(-0.5 * (k * sigma_dao) ** 2)
    return power_spectrum * dao_term


def compute_pk_errors(
    k: NDArray, power_spectrum: NDArray, volume: float = 10.0, n_modes_factor: float = 5.0
) -> NDArray:
    """
    Estimate cosmic variance limited errors on P(k) measurements.

    σ_P / P ≈ 1 / sqrt(N_modes)
    N_modes = 4π k² Δk V / (2π)³

    Args:
        k: Wavenumber array
        power_spectrum: P(k) values
        volume: Survey effective volume in (Gpc/h)^3
        n_modes_factor: Conservative factor for shot noise

    Returns:
        Error estimates σ_P
    """
    dk = np.gradient(k)
    # Convert volume to h^-3 Mpc^3
    vol_mpc = volume * 1e9
    n_modes = 4 * np.pi * k**2 * dk * vol_mpc / (2 * np.pi) ** 3
    sigma_p = power_spectrum / np.sqrt(n_modes) * n_modes_factor
    return sigma_p


def chi_squared_pk(
    k: NDArray,
    data: NDArray,
    model: NDArray,
    sigma: NDArray,
) -> float:
    """
    Compute chi-squared for power spectrum fit.

    Args:
        k: Wavenumber array
        data: Observed P(k)
        model: Model P(k)
        sigma: Errors on P(k)

    Returns:
        Chi-squared value
    """
    return np.sum(((data - model) / sigma) ** 2)
