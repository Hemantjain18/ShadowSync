"""Matching module for lunar multi-sensor image registration."""

from lunar_matcher.matching.illumination import normalize_and_score, apply_clahe, match_histograms
from lunar_matcher.matching.band_select import select_reference_band, apply_transform_to_cube
from lunar_matcher.matching.benchmark import run_benchmark

__all__ = [
    "normalize_and_score",
    "apply_clahe",
    "match_histograms",
    "select_reference_band",
    "apply_transform_to_cube",
    "run_benchmark",
]
