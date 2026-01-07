"""
Tests for wedge analysis functionality.
"""

import pytest
import numpy as np
from src.analysis.wedge import (
    wedge_mask,
    wedge_prior_fraction,
    wedge_analysis,
    generate_wedge_boundary,
)


class TestWedgeMask:
    """Test wedge mask generation."""

    def test_lcdm(self):
        """LCDM should be in wedge."""
        w0 = np.array([-1.0])
        wa = np.array([0.0])
        mask = wedge_mask(w0, wa)
        assert mask[0]

    def test_phantom(self):
        """Phantom DE should not be in wedge."""
        # Correct test
        w0 = np.array([-1.2, -1.0, -0.5])
        wa = np.array([0.0, -0.5, -0.4])
        mask = wedge_mask(w0, wa)
        assert not mask[0]  # w0 < -1
        assert not mask[1]  # w0 + wa < -1
        assert mask[2]  # w0=-0.5, wa=-0.4, sum=-0.9 >= -1

    def test_array_input(self):
        """Test with array inputs."""
        w0 = np.linspace(-2, 0, 100)
        wa = np.zeros(100)
        mask = wedge_mask(w0, wa)
        
        # Only w0 >= -1 should be in wedge
        expected = w0 >= -1.0
        assert np.array_equal(mask, expected)


class TestWedgePriorFraction:
    """Test prior volume calculation."""

    def test_full_box_in_wedge(self):
        """If entire prior is in wedge, fraction = 1."""
        w0_bounds = (-0.9, 0.0)
        wa_bounds = (0.0, 1.0)
        frac = wedge_prior_fraction(w0_bounds, wa_bounds)
        assert frac == pytest.approx(1.0, rel=0.01)

    def test_no_overlap(self):
        """If no prior overlaps wedge, fraction = 0."""
        w0_bounds = (-3.0, -1.5)
        wa_bounds = (-2.0, -1.0)
        frac = wedge_prior_fraction(w0_bounds, wa_bounds)
        assert frac == pytest.approx(0.0, abs=0.01)

    def test_partial_overlap(self):
        """Test partial overlap."""
        w0_bounds = (-3.0, 1.0)
        wa_bounds = (-3.0, 2.0)
        frac = wedge_prior_fraction(w0_bounds, wa_bounds)
        # Should be ~0.3 (30%) for DESI-like priors
        assert 0.25 < frac < 0.35


class TestWedgeAnalysis:
    """Test full wedge analysis."""

    def test_all_lcdm_samples(self):
        """All LCDM samples should be in wedge."""
        n = 1000
        w0 = np.ones(n) * -1.0
        wa = np.zeros(n)
        
        result = wedge_analysis(w0, wa)
        assert result['n_samples'] == n
        assert result['n_canonical'] == n
        assert result['post_wedge'] == pytest.approx(1.0)

    def test_no_canonical_samples(self):
        """All phantom samples."""
        n = 1000
        w0 = np.ones(n) * -1.5
        wa = np.zeros(n)
        
        result = wedge_analysis(w0, wa)
        assert result['n_canonical'] == 0
        assert result['post_wedge'] == 0.0

    def test_weighted_samples(self):
        """Test with importance weights."""
        w0 = np.array([-1.5, -0.8, -0.8])
        wa = np.zeros(3)
        weights = np.array([1.0, 2.0, 2.0])
        
        result = wedge_analysis(w0, wa, weights=weights)
        # 2 out of 3 samples in wedge, with weight 4/5
        assert result['post_wedge'] == pytest.approx(0.8)


class TestGenerateWedgeBoundary:
    """Test wedge boundary generation for plotting."""

    def test_boundary_generation(self):
        """Test boundary line generation."""
        w0_vals, wa_boundary = generate_wedge_boundary()
        
        assert len(w0_vals) == 100
        assert len(wa_boundary) == 100
        
        # Check relationship: wa = -1 - w0
        assert np.allclose(wa_boundary, -1.0 - w0_vals)
        
        # All w0 should be >= -1
        assert np.all(w0_vals >= -1.0)
