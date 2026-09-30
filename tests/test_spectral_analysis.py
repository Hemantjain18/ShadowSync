"""Tests for spectral analysis, continuum removal, and diagnostic band parameter extraction."""

import numpy as np
import pytest
from lunar_matcher.spectral.analysis import (
    continuum_remove,
    band_parameters,
    compute_indices,
    classify_pixels,
    run_mineralogy_pipeline,
)
from lunar_matcher.spectral.simulate import build_simulated_cube


def test_band_parameters_recovery():
    # Synthetic spectrum with known Gaussian absorption at 930 nm
    w = np.linspace(800.0, 2600.0, 200, dtype=np.float32)
    known_center = 930.0
    known_depth = 0.20
    sigma = 40.0

    continuum = 0.30 + 0.10 * (w - 800) / 1800
    absorption = known_depth * np.exp(-0.5 * ((w - known_center) / sigma) ** 2)
    spectrum = continuum - absorption

    # Extract band parameters on (820, 1100 nm) window
    res = band_parameters(spectrum, w, (820.0, 1100.0))

    assert res["valid"] is True
    assert res["center_nm"] is not None
    # Recover center within +/- 15 nm
    assert abs(res["center_nm"] - known_center) <= 15.0
    # Depth should be close to known depth
    assert abs(res["depth"] - known_depth) <= 0.05
    assert res["area"] > 0.0


def test_clipped_window_returns_invalid():
    w = np.linspace(800.0, 2600.0, 100, dtype=np.float32)
    spectrum = np.full_like(w, 0.25)

    # Window starts below instrument range (700 nm < 800 nm)
    res_clipped_left = band_parameters(spectrum, w, (700.0, 1000.0))
    assert res_clipped_left["valid"] is False

    # Window ends above instrument range (2800 nm > 2600 nm)
    res_clipped_right = band_parameters(spectrum, w, (2400.0, 2800.0))
    assert res_clipped_right["valid"] is False


def test_run_mineralogy_pipeline():
    cube_data = build_simulated_cube(seed=1, height=32, width=32)
    report = run_mineralogy_pipeline(cube_data, illumination_confidence=0.85)

    assert report["source"] == "simulated"
    assert "classes" in report
    assert len(report["classes"]) > 0
    assert "indices" in report
    assert "mean_spectrum" in report
    assert "diagnostic_windows" in report
    assert "quality" in report

    # Check total coverage sums to ~100%
    total_cov = sum(c["coverage_pct"] for c in report["classes"])
    assert 99.0 <= total_cov <= 101.0
