"""IIRS Spectral analysis, simulation, and cube loading module."""

from .cube_loader import load_iirs_cube
from .simulate import build_simulated_cube
from .analysis import (
    preprocess_cube,
    continuum_remove,
    band_parameters,
    compute_indices,
    classify_pixels,
    run_mineralogy_pipeline,
)
from .library import MINERAL_LIBRARY, get_mineral_reference

__all__ = [
    "load_iirs_cube",
    "build_simulated_cube",
    "preprocess_cube",
    "continuum_remove",
    "band_parameters",
    "compute_indices",
    "classify_pixels",
    "run_mineralogy_pipeline",
    "MINERAL_LIBRARY",
    "get_mineral_reference",
]
