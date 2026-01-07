"""
Cosmological models and dark energy parameterizations.

This module implements various dark energy models including CPL parameterization,
quintessence, and phantom dark energy.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray

# Physical constants (all in SI-derived units used in cosmology)
# Speed of light in km/s
C_LIGHT_KM_S = 299792.458  # km/s
# Fiducial Hubble constant in km/s/Mpc
H0_FID_KM_S_MPC = 67.4  # km/s/Mpc
# Fiducial matter density (dimensionless)
OM_FID = 0.315
# Fiducial sound horizon at drag epoch in Mpc
RD_FID_MPC = 147.09  # Mpc

# For convenience, keep old names as aliases
C_LIGHT = C_LIGHT_KM_S
H0_FID = H0_FID_KM_S_MPC
RD_FID = RD_FID_MPC


@dataclass
class CosmoParams:
    """
    Cosmological parameters for a flat Universe with dark energy.

    Uses the Chevallier-Polarski-Linder (CPL) parameterization:
    w(a) = w0 + wa * (1 - a)

    Attributes:
        h: Reduced Hubble constant (H0 / 100 km/s/Mpc)
        Om0: Matter density parameter at z=0
        w0: Dark energy equation of state at z=0
        wa: Dark energy equation of state evolution parameter
    """

    h: float = 0.674
    Om0: float = 0.315
    w0: float = -1.0
    wa: float = 0.0

    def validate(self) -> None:
        """Validate cosmological parameters are physically reasonable."""
        if not 0 < self.h < 2:
            raise ValueError(f"h must be in (0, 2), got {self.h}")
        if not 0 < self.Om0 < 1:
            raise ValueError(f"Om0 must be in (0, 1), got {self.Om0}")

    def E(self, z: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Compute normalized Hubble parameter E(z) = H(z) / H0.

        For CPL dark energy with evolution:
        w(z) = w0 + wa * z / (1 + z)

        The dark energy density evolves as:
        ρ_de(z) / ρ_de(0) = (1+z)^(3(1+w0+wa)) * exp(-3 wa z / (1+z))

        Args:
            z: Redshift (scalar or array)

        Returns:
            E(z) = sqrt(Ωm(1+z)³ + Ωde(z))
        """
        zp1 = 1.0 + z
        # Dark energy evolution factor
        f_de = (zp1 ** (3 * (1 + self.w0 + self.wa))) * np.exp(
            -3 * self.wa * z / zp1
        )
        # Total Hubble evolution
        return np.sqrt(self.Om0 * zp1**3 + (1 - self.Om0) * f_de)


def dark_energy_evolution(
    z: Union[float, NDArray], w0: float = -1.0, wa: float = 0.0
) -> Union[float, NDArray]:
    """
    Compute dark energy equation of state w(z) using CPL parameterization.

    w(z) = w0 + wa * z / (1 + z)

    Args:
        z: Redshift
        w0: Equation of state at z=0
        wa: Evolution parameter

    Returns:
        w(z): Equation of state at redshift z
    """
    return w0 + wa * z / (1.0 + z)


def is_canonical_quintessence(w0: float, wa: float) -> bool:
    """
    Check if dark energy parameters correspond to canonical quintessence.

    Canonical quintessence requires w(z) >= -1 for all z >= 0.
    For CPL: This requires w0 >= -1 AND w0 + wa >= -1

    Args:
        w0: Equation of state at z=0
        wa: Evolution parameter

    Returns:
        True if parameters are in canonical quintessence wedge
    """
    return (w0 >= -1.0) and ((w0 + wa) >= -1.0)


def is_phantom_de(w0: float, wa: float) -> bool:
    """
    Check if dark energy parameters indicate phantom dark energy (w < -1).

    Args:
        w0: Equation of state at z=0
        wa: Evolution parameter

    Returns:
        True if any epoch has w < -1
    """
    # Check at z=0 and z->infinity
    return (w0 < -1.0) or ((w0 + wa) < -1.0)
