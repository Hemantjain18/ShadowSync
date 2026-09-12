"""Comprehensive end-to-end pipeline tests for lunar-matcher."""

import os
import pytest
import numpy as np
import cv2

from lunar_matcher.georef.pyramid import pair_planner, build_pyramid
from lunar_matcher.matching.illumination import normalize_and_score, compute_illumination_confidence
from lunar_matcher.matching.benchmark import run_benchmark
from lunar_matcher.fitting.ransac import fit_transform
from lunar_matcher.validate.rmse import compute_control_point_rmse, generate_validation_report


def test_scale_bridging_pair_planner():
    # OHRC (0.25m) to TMC (5.0m) -> 20x gap -> Direct
    plan_ohrc_tmc = pair_planner("OHRC", "TMC2")
    assert plan_ohrc_tmc["strategy"] == "direct"
    assert plan_ohrc_tmc["scale_ratio"] == 20.0

    # TMC (5.0m) to IIRS (80m) -> 16x gap -> Direct
    plan_tmc_iirs = pair_planner("TMC2", "IIRS")
    assert plan_tmc_iirs["strategy"] == "direct"
    assert plan_tmc_iirs["scale_ratio"] == 16.0

    # OHRC (0.25m) to IIRS (80m) -> 320x gap -> Must route through TMC pivot
    plan_ohrc_iirs = pair_planner("OHRC", "IIRS")
    assert plan_ohrc_iirs["strategy"] == "pivot"
    assert plan_ohrc_iirs["pivot_sensor"] == "TMC2"
    assert len(plan_ohrc_iirs["chain_steps"]) == 2


def test_illumination_tier3_gating():
    # Safe angle (45 degrees) -> high confidence, no flag
    conf_safe, flag_safe = compute_illumination_confidence(45.0, cutoff_deg=80.0)
    assert conf_safe > 0.7
    assert flag_safe is None

    # Polar shadow grazing angle (85 degrees) -> low confidence, flagged
    conf_shadow, flag_shadow = compute_illumination_confidence(85.0, cutoff_deg=80.0)
    assert conf_shadow < 0.3
    assert flag_shadow is not None
    assert "High solar incidence angle" in flag_shadow


def test_end_to_end_matching_and_ransac():
    # Create two synthetic images related by a known Euclidean shift + small rotation
    size = 128
    np.random.seed(42)
    img_a = np.random.normal(120, 15, (size, size)).astype(np.uint8)
    
    # Draw several distinct craters
    for pt in [(30, 40, 12), (80, 80, 16), (40, 90, 8), (90, 30, 10)]:
        cv2.circle(img_a, (pt[0], pt[1]), pt[2], 40, -1)
        cv2.circle(img_a, (pt[0], pt[1]), pt[2] + 2, 220, 2)

    # Transform: dx = 6, dy = -4
    true_m = np.array([
        [1.0, 0.0, 6.0],
        [0.0, 1.0, -4.0]
    ], dtype=np.float32)

    img_b = cv2.warpAffine(img_a, true_m, (size, size))

    # Benchmark matchers
    df, raw = run_benchmark(img_a, img_b, methods=["SIFT", "AKAZE"])
    assert not df.empty
    assert "Match Count" in df.columns

    # Check SIFT results
    sift_res = raw["SIFT"]
    assert sift_res["count"] >= 4

    # RANSAC fit
    fit_res = fit_transform(sift_res["pts_a"], sift_res["pts_b"], model="affine", min_inlier_count=4, min_inlier_ratio=0.10)
    assert fit_res["success"] is True
    assert fit_res["inlier_count"] >= 4

    # Validation
    pts_test_a = np.array([[30.0, 40.0], [80.0, 80.0]], dtype=np.float32)
    pts_test_b = np.array([[36.0, 36.0], [86.0, 76.0]], dtype=np.float32)
    val_report = compute_control_point_rmse(pts_test_a, pts_test_b, fit_res["transform"])
    assert val_report["rmse_px"] < 2.0
