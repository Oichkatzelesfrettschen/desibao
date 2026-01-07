"""
Tests for cosmology models and dark energy parameterizations.
"""

import pytest
import numpy as np
from src.cosmology.models import (
    CosmoParams,
    dark_energy_evolution,
    is_canonical_quintessence,
    is_phantom_de,
)


class TestCosmoParams:
    """Test CosmoParams dataclass and methods."""

    def test_default_params(self):
        """Test default cosmological parameters."""
        cosmo = CosmoParams()
        assert cosmo.h == 0.674
        assert cosmo.Om0 == 0.315
        assert cosmo.w0 == -1.0
        assert cosmo.wa == 0.0

    def test_lcdm_evolution(self):
        """Test LCDM (w=-1, wa=0) Hubble evolution."""
        cosmo = CosmoParams(w0=-1.0, wa=0.0)
        
        # At z=0, E(0) should be 1
        assert np.isclose(cosmo.E(0), 1.0)
        
        # At high z, should approach matter-dominated
        z_high = 1000
        E_high = cosmo.E(z_high)
        E_matter = np.sqrt(cosmo.Om0 * (1 + z_high)**3)
        assert np.isclose(E_high, E_matter, rtol=1e-3)

    def test_quintessence_evolution(self):
        """Test quintessence (w>-1) evolution."""
        cosmo = CosmoParams(w0=-0.8, wa=0.0)
        
        # Evolution should be different from LCDM
        cosmo_lcdm = CosmoParams(w0=-1.0, wa=0.0)
        z = 0.5
        assert not np.isclose(cosmo.E(z), cosmo_lcdm.E(z))

    def test_phantom_evolution(self):
        """Test phantom (w<-1) evolution."""
        cosmo = CosmoParams(w0=-1.2, wa=0.0)
        
        # Should still produce valid E(z)
        z = np.linspace(0, 2, 10)
        E = cosmo.E(z)
        assert np.all(E > 0)
        assert np.all(np.isfinite(E))

    def test_array_input(self):
        """Test that E(z) works with array inputs."""
        cosmo = CosmoParams()
        z = np.array([0, 0.5, 1.0, 2.0])
        E = cosmo.E(z)
        
        assert isinstance(E, np.ndarray)
        assert len(E) == len(z)
        assert np.all(E > 0)

    def test_validation(self):
        """Test parameter validation."""
        # Invalid h
        with pytest.raises(ValueError):
            cosmo = CosmoParams(h=-0.5)
            cosmo.validate()
        
        # Invalid Om0
        with pytest.raises(ValueError):
            cosmo = CosmoParams(Om0=1.5)
            cosmo.validate()


class TestDarkEnergyEvolution:
    """Test dark energy evolution functions."""

    def test_lcdm(self):
        """Test LCDM (constant w=-1)."""
        z = np.array([0, 0.5, 1.0])
        w = dark_energy_evolution(z, w0=-1.0, wa=0.0)
        assert np.allclose(w, -1.0)

    def test_quintessence(self):
        """Test evolving quintessence."""
        w0, wa = -0.8, 0.2
        z = 0.5
        w = dark_energy_evolution(z, w0, wa)
        expected = w0 + wa * z / (1 + z)
        assert np.isclose(w, expected)

    def test_phantom_crossing(self):
        """Test models that cross phantom divide."""
        w0, wa = -1.1, 0.3
        z = np.array([0, 1.0])
        w = dark_energy_evolution(z, w0, wa)
        # At z=0: w = -1.1 (phantom)
        assert w[0] < -1.0
        # At z=1: w = -1.1 + 0.3*1/2 = -0.95 (quintessence)
        assert w[1] > -1.0


class TestQuintessenceClassification:
    """Test quintessence and phantom classification."""

    def test_canonical_quintessence(self):
        """Test canonical quintessence identification."""
        assert is_canonical_quintessence(w0=-0.8, wa=0.0)
        assert is_canonical_quintessence(w0=-1.0, wa=0.0)
        assert not is_canonical_quintessence(w0=-1.2, wa=0.0)

    def test_wedge_boundary(self):
        """Test boundary of quintessence wedge."""
        # On boundary: w0 = -1
        assert is_canonical_quintessence(w0=-1.0, wa=0.0)
        # On boundary: w0 + wa = -1
        assert is_canonical_quintessence(w0=-0.5, wa=-0.5)
        # Just outside wedge
        assert not is_canonical_quintessence(w0=-1.01, wa=0.0)

    def test_phantom_de(self):
        """Test phantom dark energy identification."""
        assert is_phantom_de(w0=-1.2, wa=0.0)
        assert is_phantom_de(w0=-1.0, wa=-0.1)
        assert not is_phantom_de(w0=-0.8, wa=0.0)
