"""RANSAC-based geometric transformation fitting with strict inlier gating."""

from typing import Dict, Any, Optional, Union
import numpy as np
import cv2


def fit_transform(
    matched_keypoints_a: np.ndarray,
    matched_keypoints_b: np.ndarray,
    model: str = "homography",
    ransac_reproj_threshold: float = 4.0,
    min_inlier_count: int = 8,
    min_inlier_ratio: float = 0.15,
    confidence: float = 0.99,
    max_iters: int = 2000,
) -> Dict[str, Any]:
    """
    Fits a robust geometric transformation between matched keypoints using RANSAC.
    Lunar regolith repetitive crater fields produce high false-positive matches,
    requiring strict inlier count and ratio gating.

    Parameters:
        matched_keypoints_a: (N, 2) array of coordinates in reference image A
        matched_keypoints_b: (N, 2) array of coordinates in moving image B
        model: "homography" (3x3 projective) or "affine" (2x3 similarity/affine partial)
        ransac_reproj_threshold: Maximum reprojection error in pixels to consider a point an inlier
        min_inlier_count: Minimum absolute count of inliers to accept fit (default: 8)
        min_inlier_ratio: Minimum ratio of inliers (inliers / total matches) to accept (default: 0.15)
        confidence: RANSAC confidence level (default: 0.99)
        max_iters: Maximum RANSAC iterations (default: 2000)

    Returns:
        Dict with keys:
            success (bool): True if fit succeeded and passed inlier thresholds
            transform (np.ndarray | None): 3x3 or 2x3 transformation matrix
            model (str): "homography" or "affine"
            inlier_mask (np.ndarray): 1D boolean array indicating inliers
            inlier_count (int): Number of inlier correspondences
            inlier_ratio (float): Ratio of inliers to total matches
            total_matches (int): Initial number of candidate correspondences
            status (str): "success", "insufficient_matches", "insufficient_inliers", "fitting_failed"
            error_message (str | None): Human-readable rejection rationale
    """
    pts_a = np.asarray(matched_keypoints_a, dtype=np.float32).reshape(-1, 2)
    pts_b = np.asarray(matched_keypoints_b, dtype=np.float32).reshape(-1, 2)

    total_matches = len(pts_a)
    if total_matches != len(pts_b):
        raise ValueError(f"Mismatched keypoint array lengths: {total_matches} vs {len(pts_b)}")

    # Minimal points required by geometry
    min_pts_required = 4 if model.lower() == "homography" else 3
    if total_matches < min_pts_required:
        return {
            "success": False,
            "transform": None,
            "model": model,
            "inlier_mask": np.zeros((total_matches,), dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "total_matches": total_matches,
            "status": "insufficient_matches",
            "error_message": (
                f"Insufficient correspondences for {model} estimation: "
                f"got {total_matches}, need at least {min_pts_required}."
            ),
        }

    matrix = None
    mask = None

    if model.lower() == "homography":
        matrix, mask = cv2.findHomography(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    elif model.lower() in ["affine", "affine_partial"]:
        matrix, mask = cv2.estimateAffinePartial2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    else:
        raise ValueError(f"Unsupported transform model '{model}'. Use 'homography' or 'affine'.")

    if matrix is None or mask is None:
        return {
            "success": False,
            "transform": None,
            "model": model,
            "inlier_mask": np.zeros((total_matches,), dtype=bool),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "total_matches": total_matches,
            "status": "fitting_failed",
            "error_message": f"RANSAC optimization failed to converge for {model}.",
        }

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.sum(inlier_mask))
    inlier_ratio = float(inlier_count / total_matches) if total_matches > 0 else 0.0

    # Inlier threshold verification
    if inlier_count < min_inlier_count:
        return {
            "success": False,
            "transform": None,
            "model": model,
            "inlier_mask": inlier_mask,
            "inlier_count": inlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "total_matches": total_matches,
            "status": "insufficient_inliers",
            "error_message": (
                f"Rejected fit: {inlier_count} inliers is below threshold of {min_inlier_count}."
            ),
        }

    if inlier_ratio < min_inlier_ratio:
        return {
            "success": False,
            "transform": None,
            "model": model,
            "inlier_mask": inlier_mask,
            "inlier_count": inlier_count,
            "inlier_ratio": round(inlier_ratio, 4),
            "total_matches": total_matches,
            "status": "insufficient_inliers",
            "error_message": (
                f"Rejected fit: inlier ratio {inlier_ratio*100:.1f}% is below required {min_inlier_ratio*100:.1f}%."
            ),
        }

    return {
        "success": True,
        "transform": matrix,
        "model": model,
        "inlier_mask": inlier_mask,
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "total_matches": total_matches,
        "status": "success",
        "error_message": None,
    }
