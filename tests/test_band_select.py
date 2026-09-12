"""Unit tests for matching/band_select.py with synthetic hyperspectral cubes."""

import pytest
import numpy as np
import cv2

from lunar_matcher.matching.band_select import select_reference_band, apply_transform_to_cube


@pytest.fixture
def synthetic_cube():
    """Generates a synthetic (32, 32, 16) hyperspectral cube with distinct spectral signatures."""
    h, w, b = 32, 32, 16
    cube = np.zeros((h, w, b), dtype=np.float32)

    for band in range(b):
        # Baseline regolith reflectance with spectral slope
        baseline = 100.0 + band * 5.0
        cube[:, :, band] = baseline
        # Sharp distinct feature at center (x=16, y=16)
        cube[16, 16, band] += 100.0

    return cube


def test_select_reference_band_pca(synthetic_cube):
    ref_pca = select_reference_band(synthetic_cube, method="pca")
    assert ref_pca.shape == (32, 32)
    assert ref_pca.dtype == np.uint8
    assert ref_pca.max() > ref_pca.min()
    # Crater region should have high contrast in 1st PC
    assert ref_pca[16, 16] != ref_pca[0, 0]


def test_select_reference_band_single(synthetic_cube):
    ref_band = select_reference_band(synthetic_cube, method="single_band", band_idx=8)
    assert ref_band.shape == (32, 32)
    assert ref_band.dtype == np.uint8
    assert ref_band.max() > ref_band.min()


def test_apply_transform_to_cube_affine(synthetic_cube):
    # Translate by 4px in X and 2px in Y
    affine_matrix = np.array([
        [1.0, 0.0, 4.0],
        [0.0, 1.0, 2.0]
    ], dtype=np.float32)

    warped = apply_transform_to_cube(synthetic_cube, affine_matrix)
    assert warped.shape == synthetic_cube.shape
    assert warped.dtype == synthetic_cube.dtype
    # The center of the crater should have shifted from (16, 16) to (18, 20)
    assert warped[18, 20, 8] > warped[16, 16, 8]


def test_apply_transform_to_cube_homography(synthetic_cube):
    homography = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    warped = apply_transform_to_cube(synthetic_cube, homography)
    assert warped.shape == synthetic_cube.shape
