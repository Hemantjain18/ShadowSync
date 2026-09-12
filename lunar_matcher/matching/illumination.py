"""Three-tier illumination normalization and solar-angle confidence gating for lunar imagery."""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2


def match_histograms(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the histogram of a source image to that of a reference image.
    Preserves structural features while harmonizing radiometric dynamic range across sensors.
    """
    orig_shape = source.shape
    src_flat = source.ravel()
    ref_flat = reference.ravel()

    # Calculate unique values, inverse indices, and counts
    src_vals, src_idx, src_counts = np.unique(src_flat, return_inverse=True, return_counts=True)
    ref_vals, ref_counts = np.unique(ref_flat, return_counts=True)

    # Compute empirical cumulative distribution functions
    src_quantiles = np.cumsum(src_counts).astype(np.float64) / src_flat.size
    ref_quantiles = np.cumsum(ref_counts).astype(np.float64) / ref_flat.size

    # Interpolate pixel values
    interp_values = np.interp(src_quantiles, ref_quantiles, ref_vals)
    matched = interp_values[src_idx].reshape(orig_shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    if image.dtype != np.uint8:
        # Normalize to uint8
        norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img_u8 = norm.astype(np.uint8)
    else:
        img_u8 = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)


def apply_directional_shadow_correction(
    image: np.ndarray,
    sun_azimuth_deg: float,
    sun_elevation_deg: float,
) -> np.ndarray:
    """
    Tier 2: Metadata-aware directional illumination correction.
    
    Computes expected shadow vector opposite to sun azimuth, extracts directional gradients
    along the shadow terminator, and locally boosts shadow contrast while damping specular highlights.
    """
    img_u8 = image.astype(np.uint8) if image.dtype != np.uint8 else image.copy()
    
    # Shadow direction is opposite to solar azimuth
    shadow_azimuth_rad = np.radians((sun_azimuth_deg + 180.0) % 360.0)
    
    # Directional kernel unit vector
    dx = np.cos(shadow_azimuth_rad)
    dy = np.sin(shadow_azimuth_rad)

    # Compute directional directional derivative (Sobel combination)
    grad_x = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    dir_grad = dx * grad_x + dy * grad_y

    # Low sun elevation (< 30 deg) creates longer, deeper shadows
    elevation_factor = np.clip(1.0 - (sun_elevation_deg / 90.0), 0.2, 1.0)
    
    # Local shadow mask (dark pixels)
    shadow_mask = (img_u8 < 60).astype(np.float32)
    # Blur the mask slightly for smooth transition
    shadow_mask_smooth = cv2.GaussianBlur(shadow_mask, (9, 9), 2.0)

    # Directional boost: enhance micro-ridges in shadowed areas
    boost = dir_grad * (elevation_factor * 0.35) * shadow_mask_smooth
    
    # Combined with base image
    corrected = img_u8.astype(np.float32) + boost
    
    # Adaptive gamma based on elevation to reveal dark crater floors
    if sun_elevation_deg < 25.0:
        gamma = 0.85
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected_u8 = np.clip(corrected, 0, 255).astype(np.uint8)
        corrected_u8 = cv2.LUT(corrected_u8, table)
        return corrected_u8
    
    return np.clip(corrected, 0, 255).astype(np.uint8)


def compute_illumination_confidence(
    incidence_angle_deg: Optional[float],
    cutoff_deg: float = 80.0,
) -> Tuple[float, Optional[str]]:
    """
    Tier 3: Confidence gating based on solar incidence angle.
    
    Solar incidence angle > 80° indicates extreme grazing light near poles or permanently
    shadowed regions (PSRs), leading to high false-positive feature matching.
    """
    if incidence_angle_deg is None:
        # Sun angle missing from sidecar: moderate confidence
        return 0.70, None

    inc = float(incidence_angle_deg)
    
    # Confidence function: high from 0 to 65 deg, gentle drop to 75 deg, sharp drop > 80 deg
    if inc <= 60.0:
        confidence = 1.0 - (inc / 60.0) * 0.15  # 1.0 -> 0.85
    elif inc <= cutoff_deg:
        # Linear decay from 0.85 down to 0.40 at cutoff
        t = (inc - 60.0) / (cutoff_deg - 60.0)
        confidence = 0.85 - t * 0.45
    else:
        # Past cutoff: steep decline
        t = min(1.0, (inc - cutoff_deg) / 10.0)
        confidence = max(0.05, 0.40 - t * 0.35)

    flag_reason = None
    if inc > cutoff_deg:
        flag_reason = (
            f"High solar incidence angle ({inc:.1f}° > {cutoff_deg:.1f}°): "
            "Tile lies in an extreme low-sun or permanently shadowed polar region (PSR). "
            "Feature matching aborted to prevent spurious crater correspondence."
        )

    return round(float(confidence), 3), flag_reason


def normalize_and_score(
    image: np.ndarray,
    sun_angle_metadata: Optional[Dict[str, Any]] = None,
    reference_image: Optional[np.ndarray] = None,
    incidence_cutoff_deg: float = 80.0,
) -> Tuple[np.ndarray, float, Optional[str]]:
    """
    Executes the 3-tier illumination pipeline.

    Tier 1 (baseline): CLAHE + optional histogram matching to reference_image.
    Tier 2 (metadata-aware): Solar azimuth/elevation directional shadow enhancement.
    Tier 3 (confidence gating): Evaluates solar incidence angle and flags extreme shadow regions.

    Parameters:
        image: 2D grayscale array
        sun_angle_metadata: Dict with 'solar_incidence_angle_deg', 'sun_azimuth_deg', 'sun_elevation_deg'
        reference_image: Optional reference image for cross-sensor histogram matching
        incidence_cutoff_deg: Angle threshold above which matching is gated/flagged

    Returns:
        (normalized_image, confidence_score, flag_reason)
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        proc_img = norm.astype(np.uint8)
    else:
        proc_img = image.copy()

    # Parse sun angle metadata
    meta = sun_angle_metadata or {}
    incidence_angle = meta.get("solar_incidence_angle_deg") or meta.get("incidence_deg")
    sun_azimuth = meta.get("sun_azimuth_deg") or meta.get("azimuth_deg")
    sun_elevation = meta.get("sun_elevation_deg") or meta.get("elevation_deg")

    if incidence_angle is None and sun_elevation is not None:
        incidence_angle = max(0.0, 90.0 - sun_elevation)
    elif sun_elevation is None and incidence_angle is not None:
        sun_elevation = max(0.0, 90.0 - incidence_angle)

    # Tier 3: Compute confidence score and check gating
    confidence_score, flag_reason = compute_illumination_confidence(
        incidence_angle_deg=incidence_angle,
        cutoff_deg=incidence_cutoff_deg,
    )

    # Tier 2: Metadata-aware directional correction (if angles present)
    if sun_azimuth is not None and sun_elevation is not None:
        proc_img = apply_directional_shadow_correction(
            proc_img,
            sun_azimuth_deg=sun_azimuth,
            sun_elevation_deg=sun_elevation,
        )

    # Tier 1: Baseline CLAHE (always applied)
    proc_img = apply_clahe(proc_img, clip_limit=3.0, tile_grid_size=(8, 8))

    # Tier 1: Histogram matching (if reference provided)
    if reference_image is not None:
        ref_u8 = reference_image if reference_image.dtype == np.uint8 else cv2.normalize(
            reference_image, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        proc_img = match_histograms(proc_img, ref_u8)

    return proc_img, confidence_score, flag_reason
