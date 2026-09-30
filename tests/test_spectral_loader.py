"""Tests for IIRS hyperspectral cube loader and synthetic cube simulation."""

import numpy as np
import pytest
from lunar_matcher.spectral.simulate import build_simulated_cube
from lunar_matcher.spectral.cube_loader import load_iirs_cube, get_default_wavelengths


def test_default_wavelengths():
    w = get_default_wavelengths(256, 800.0, 16.85)
    assert len(w) == 256
    assert np.all(np.diff(w) > 0), "Wavelengths must be monotonically increasing"
    assert np.isclose(w[0], 800.0)
    assert np.isclose(w[-1], 800.0 + 255 * 16.85)


def test_build_simulated_cube():
    res = build_simulated_cube(seed=42, height=64, width=64)

    assert "cube" in res
    assert "wavelengths_nm" in res
    assert "provenance" in res

    cube = res["cube"]
    w = res["wavelengths_nm"]

    # Check shapes
    assert cube.shape == (64, 64, 256)
    assert len(w) == 256

    # Check wavelength monotonicity
    assert np.all(np.diff(w) > 0)

    # Check provenance
    assert res["provenance"]["source"] == "simulated"
    assert "demo tile has no spectral data" in res["provenance"]["notes"].lower()

    # Valid reflectance ranges
    assert np.nanmin(cube) >= 0.0
    assert np.nanmax(cube) <= 1.0


def test_load_npz_cube(tmp_path):
    w = np.linspace(800.0, 3000.0, 64, dtype=np.float32)
    fake_cube = np.random.uniform(0.1, 0.4, (32, 32, 64)).astype(np.float32)
    fake_cube[0, 0, :] = -9999.0  # fill value

    npz_path = str(tmp_path / "test_cube.npz")
    np.savez(
        npz_path,
        cube=fake_cube,
        wavelengths_nm=w,
        provenance={"source": "simulated", "thermal_corrected": True, "notes": "Test cube"},
    )

    loaded = load_iirs_cube(npz_path)
    assert loaded["cube"].shape == (32, 32, 64)
    assert np.isnan(loaded["cube"][0, 0, :]).all(), "Fill value -9999 should be converted to NaN"
    assert len(loaded["wavelengths_nm"]) == 64
    assert loaded["provenance"]["source"] == "simulated"
    assert loaded["provenance"]["thermal_corrected"] is True
