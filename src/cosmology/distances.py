"""
Distance calculations for cosmological analyses.

Implements comoving distance, luminosity distance, and distance modulus
computations for various cosmological models.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from .models import CosmoParams, C_LIGHT


def comoving_distance(
    z: Union[float, NDArray], cosmo: CosmoParams
) -> Union[float, NDArray]:
    """
    Compute comoving distance in Mpc.

    DC = (c/H0) * ∫[0 to z] dz' / E(z')

    Args:
        z: Redshift (scalar or array)
        cosmo: Cosmological parameters

    Returns:
        Comoving distance in Mpc
    """
    integrand = lambda zp: 1.0 / cosmo.E(zp)

    if np.isscalar(z):
        dc, _ = quad(integrand, 0, z)
        return C_LIGHT / (100 * cosmo.h) * dc
    else:
        dc = np.array([quad(integrand, 0, zi)[0] for zi in z])
        return C_LIGHT / (100 * cosmo.h) * dc


def luminosity_distance(
    z: Union[float, NDArray], cosmo: CosmoParams
) -> Union[float, NDArray]:
    """
    Compute luminosity distance in Mpc.

    DL = (1 + z) * DC

    Args:
        z: Redshift (scalar or array)
        cosmo: Cosmological parameters

    Returns:
        Luminosity distance in Mpc
    """
    dc = comoving_distance(z, cosmo)
    return (1.0 + z) * dc


def dist_mod(z: Union[float, NDArray], cosmo: CosmoParams) -> Union[float, NDArray]:
    """
    Compute distance modulus for Type Ia Supernovae.

    μ = 5 * log10(DL / 10 pc) = 5 * log10(DL) + 25
    where DL is in Mpc.

    Args:
        z: Redshift (scalar or array)
        cosmo: Cosmological parameters

    Returns:
        Distance modulus in magnitudes
    """
    dl = luminosity_distance(z, cosmo)
    return 5 * np.log10(dl) + 25


def angular_diameter_distance(
    z: Union[float, NDArray], cosmo: CosmoParams
) -> Union[float, NDArray]:
    """
    Compute angular diameter distance in Mpc.

    DA = DC / (1 + z)

    Args:
        z: Redshift (scalar or array)
        cosmo: Cosmological parameters

    Returns:
        Angular diameter distance in Mpc
    """
    dc = comoving_distance(z, cosmo)
    return dc / (1.0 + z)
