"""Spectral analysis, continuum removal, band parameter extraction, and mineral classification for IIRS data."""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from .library import MINERAL_LIBRARY


def preprocess_cube(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    bad_band_mask: Optional[np.ndarray] = None,
    apply_savgol: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess hyperspectral cube:
    - Mask bad bands
    - Remove cosmic ray / readout spikes via median filtering along spectral dimension
    - Apply Savitzky-Golay or moving window smoothing to suppress high-frequency noise
    """
    clean_cube = cube.copy()
    b_dim = clean_cube.shape[-1]

    if bad_band_mask is not None and np.any(bad_band_mask):
        # Linear interpolation over bad bands along spectral dimension
        good_idx = np.where(~bad_band_mask)[0]
        bad_idx = np.where(bad_band_mask)[0]
        if len(good_idx) >= 2:
            for bi in bad_idx:
                left = good_idx[good_idx < bi]
                right = good_idx[good_idx > bi]
                if len(left) > 0 and len(right) > 0:
                    l_i, r_i = left[-1], right[0]
                    weight = (bi - l_i) / (r_i - l_i)
                    clean_cube[:, :, bi] = (1.0 - weight) * clean_cube[:, :, l_i] + weight * clean_cube[:, :, r_i]
                elif len(left) > 0:
                    clean_cube[:, :, bi] = clean_cube[:, :, left[-1]]
                elif len(right) > 0:
                    clean_cube[:, :, bi] = clean_cube[:, :, right[0]]

    # 3-point median filter along spectral dimension for outlier spike removal
    padded = np.pad(clean_cube, ((0, 0), (0, 0), (1, 1)), mode="edge")
    stacked = np.stack([padded[:, :, :-2], padded[:, :, 1:-1], padded[:, :, 2:]], axis=-1)
    clean_cube = np.median(stacked, axis=-1).astype(np.float32)

    # Smoothing along spectral dimension
    if apply_savgol:
        try:
            from scipy.signal import savgol_filter
            window_len = min(9, b_dim if b_dim % 2 != 0 else b_dim - 1)
            if window_len >= 5:
                clean_cube = savgol_filter(clean_cube, window_length=window_len, polyorder=2, axis=-1)
        except Exception:
            # Simple moving average fallback
            kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
            pad_s = np.pad(clean_cube, ((0, 0), (0, 0), (1, 1)), mode="edge")
            clean_cube = (
                pad_s[:, :, :-2] * kernel[0]
                + pad_s[:, :, 1:-1] * kernel[1]
                + pad_s[:, :, 2:] * kernel[2]
            )

    return clean_cube, wavelengths_nm


def continuum_remove(
    spectrum: np.ndarray,
    wavelengths_nm: np.ndarray,
    window_nm: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Remove straight-line continuum between shoulders of a diagnostic absorption window.

    Parameters:
        spectrum: 1D reflectance array (B,)
        wavelengths_nm: 1D wavelength array (B,)
        window_nm: (start_nm, end_nm) shoulder bounds

    Returns:
        (continuum_removed_spectrum, continuum_line, is_valid)
    """
    w_start, w_end = window_nm
    min_w, max_w = wavelengths_nm[0], wavelengths_nm[-1]

    # Check for window clipping by sensor range
    if w_start < min_w - 1e-3 or w_end > max_w + 1e-3:
        return np.ones_like(spectrum), np.ones_like(spectrum), False

    idx_start = int(np.argmin(np.abs(wavelengths_nm - w_start)))
    idx_end = int(np.argmin(np.abs(wavelengths_nm - w_end)))

    if idx_start >= idx_end or (idx_end - idx_start) < 3:
        return np.ones_like(spectrum), np.ones_like(spectrum), False

    # Shoulder values
    r1 = float(spectrum[idx_start])
    r2 = float(spectrum[idx_end])
    w1 = float(wavelengths_nm[idx_start])
    w2 = float(wavelengths_nm[idx_end])

    if r1 <= 0 or r2 <= 0 or (w2 - w1) <= 0:
        return np.ones_like(spectrum), np.ones_like(spectrum), False

    # Calculate straight-line continuum across the full spectrum (1.0 outside window)
    continuum = np.ones_like(spectrum, dtype=np.float32)
    win_w = wavelengths_nm[idx_start : idx_end + 1]
    cont_win = r1 + (r2 - r1) * ((win_w - w1) / (w2 - w1))
    continuum[idx_start : idx_end + 1] = cont_win

    cr_spectrum = np.ones_like(spectrum, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr_win = spectrum[idx_start : idx_end + 1] / cont_win
        cr_spectrum[idx_start : idx_end + 1] = np.nan_to_num(cr_win, nan=1.0)

    return cr_spectrum, continuum, True


def band_parameters(
    spectrum: np.ndarray,
    wavelengths_nm: np.ndarray,
    window_nm: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Extract diagnostic absorption band parameters: center wavelength, depth, and area.
    Uses polynomial fitting around the local minimum of the continuum-removed curve.
    """
    w_start, w_end = window_nm
    cr_spec, cont_line, is_valid = continuum_remove(spectrum, wavelengths_nm, window_nm)

    if not is_valid:
        return {
            "center_nm": None,
            "depth": 0.0,
            "area": 0.0,
            "valid": False,
            "error": "Window clipped or out of wavelength bounds",
        }

    idx_start = int(np.argmin(np.abs(wavelengths_nm - w_start)))
    idx_end = int(np.argmin(np.abs(wavelengths_nm - w_end)))

    cr_win = cr_spec[idx_start : idx_end + 1]
    w_win = wavelengths_nm[idx_start : idx_end + 1]

    # Find raw discrete minimum
    min_local_idx = int(np.argmin(cr_win))
    raw_depth = 1.0 - float(cr_win[min_local_idx])

    if raw_depth <= 0.005:
        # No detectable absorption
        return {
            "center_nm": float(w_win[min_local_idx]),
            "depth": max(0.0, raw_depth),
            "area": 0.0,
            "valid": False,
            "error": "Absorption depth below detection threshold",
        }

    # Sub-band polynomial fit (quadratic) around the minimum to estimate true center
    fit_radius = 3
    fit_start = max(0, min_local_idx - fit_radius)
    fit_end = min(len(w_win), min_local_idx + fit_radius + 1)

    if (fit_end - fit_start) >= 3:
        sub_w = w_win[fit_start:fit_end]
        sub_cr = cr_win[fit_start:fit_end]
        try:
            poly = np.polyfit(sub_w, sub_cr, deg=2)
            # Minimum of a x^2 + b x + c is -b / (2a)
            if poly[0] > 0:
                fitted_center = -poly[1] / (2.0 * poly[0])
                if w_win[0] <= fitted_center <= w_win[-1]:
                    center_nm = float(fitted_center)
                else:
                    center_nm = float(w_win[min_local_idx])
            else:
                center_nm = float(w_win[min_local_idx])
        except Exception:
            center_nm = float(w_win[min_local_idx])
    else:
        center_nm = float(w_win[min_local_idx])

    # Trapezoidal integration for band area: Integral (1 - R_cr) dlambda
    absorption = np.maximum(0.0, 1.0 - cr_win)
    area = float(np.trapezoid(absorption, w_win) if hasattr(np, "trapezoid") else np.trapz(absorption, w_win))

    return {
        "center_nm": round(center_nm, 2),
        "depth": round(max(0.0, raw_depth), 4),
        "area": round(area, 2),
        "valid": True,
    }


def compute_indices(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    thermal_corrected: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Compute lunar mineral diagnostic indices on mean tile spectrum and pixel stats.
    - 1 um band (800-1300 nm)
    - 2 um band (1500-2600 nm)
    - Band Area Ratio (BAR = Area_2um / Area_1um)
    - 1250 nm Plagioclase depth
    - 3 um Hydration depth (2700-3100 nm, gated on thermal correction)
    """
    mean_spec = np.nanmean(cube, axis=(0, 1))

    # 1. Band 1 (1-micron)
    p_b1 = band_parameters(mean_spec, wavelengths_nm, (820.0, 1320.0))

    # 2. Band 2 (2-micron)
    p_b2 = band_parameters(mean_spec, wavelengths_nm, (1600.0, 2550.0))

    # 3. Band Area Ratio (BAR)
    bar = None
    if p_b1["valid"] and p_b2["valid"] and p_b1["area"] > 0:
        bar = round(p_b2["area"] / p_b1["area"], 3)

    # 4. 1250 nm Plagioclase index
    p_plag = band_parameters(mean_spec, wavelengths_nm, (1150.0, 1380.0))

    # 5. 3-micron Hydration band (2700 - 3100 nm)
    hydration_result: Dict[str, Any] = {}
    if thermal_corrected is True:
        p_hyd = band_parameters(mean_spec, wavelengths_nm, (2700.0, 3100.0))
        hydration_result = {
            "depth": p_hyd["depth"],
            "center_nm": p_hyd["center_nm"],
            "valid": p_hyd["valid"],
            "status": "computed",
        }
    else:
        hydration_result = {
            "depth": None,
            "center_nm": None,
            "valid": False,
            "status": "unavailable",
            "reason": (
                "Thermal emission correction unverified. Thermal emission dominates >2500 nm "
                "on lunar daytime surfaces; hydration index requires Level-2 thermal model."
            ),
        }

    return {
        "band1_1um": p_b1,
        "band2_2um": p_b2,
        "band_area_ratio": bar,
        "plagioclase_1250nm": p_plag,
        "hydration_3um": hydration_result,
    }


def classify_pixels(
    cube: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Rule-based mineralogy classifier using diagnostic absorption band parameters and spectral angle mapping.

    Classes:
    0: Low-Ca Pyroxene (Orthopyroxene)
    1: High-Ca Pyroxene (Clinopyroxene)
    2: Olivine-rich
    3: Plagioclase-rich (Anorthosite)
    4: Spinel-like
    5: Featureless / Dark (Ilmenite-rich or Mature)
    6: Unclassified
    """
    h, w, b = cube.shape

    # Find key wavelength indices
    def nearest_idx(w_val: float) -> int:
        return int(np.argmin(np.abs(wavelengths_nm - w_val)))

    idx_820 = nearest_idx(820.0)
    idx_930 = nearest_idx(930.0)
    idx_1015 = nearest_idx(1015.0)
    idx_1050 = nearest_idx(1050.0)
    idx_1250 = nearest_idx(1250.0)
    idx_1300 = nearest_idx(1300.0)
    idx_1600 = nearest_idx(1600.0)
    idx_1900 = nearest_idx(1900.0)
    idx_2020 = nearest_idx(2020.0)
    idx_2120 = nearest_idx(2120.0)
    idx_2500 = nearest_idx(2500.0)

    # Simplified pixel-level spectral ratios for fast 2D map classification
    r_820 = cube[:, :, idx_820]
    r_930 = cube[:, :, idx_930]
    r_1015 = cube[:, :, idx_1015]
    r_1050 = cube[:, :, idx_1050]
    r_1250 = cube[:, :, idx_1250]
    r_1300 = cube[:, :, idx_1300]
    r_1600 = cube[:, :, idx_1600]
    r_1900 = cube[:, :, idx_1900]
    r_2020 = cube[:, :, idx_2020]
    r_2120 = cube[:, :, idx_2120]
    r_2500 = cube[:, :, idx_2500]

    # Depths relative to shoulder baselines
    cont_1um = 0.5 * (r_820 + r_1300)
    depth_930 = np.maximum(0.0, 1.0 - (r_930 / np.maximum(cont_1um, 1e-4)))
    depth_1015 = np.maximum(0.0, 1.0 - (r_1015 / np.maximum(cont_1um, 1e-4)))
    depth_1050 = np.maximum(0.0, 1.0 - (r_1050 / np.maximum(cont_1um, 1e-4)))

    cont_2um = 0.5 * (r_1600 + r_2500)
    depth_1900 = np.maximum(0.0, 1.0 - (r_1900 / np.maximum(cont_2um, 1e-4)))
    depth_2020 = np.maximum(0.0, 1.0 - (r_2020 / np.maximum(cont_2um, 1e-4)))
    depth_2120 = np.maximum(0.0, 1.0 - (r_2120 / np.maximum(cont_2um, 1e-4)))

    depth_1250 = np.maximum(0.0, 1.0 - (r_1250 / np.maximum(0.5 * (r_1050 + r_1600), 1e-4)))
    albedo = np.mean(cube, axis=-1)

    class_map = np.full((h, w), 6, dtype=np.uint8)  # default: Unclassified
    confidence_map = np.full((h, w), 0.50, dtype=np.float32)

    # Classification rules:
    # 0: Low-Ca Pyroxene (deep 930 nm & deep 1900 nm, 930 depth > 1015 depth)
    mask_lpx = (depth_930 > 0.035) & (depth_1900 > 0.035) & (depth_930 > depth_1015)
    class_map[mask_lpx] = 0
    confidence_map[mask_lpx] = np.clip(0.65 + depth_930[mask_lpx] * 0.8, 0.65, 0.95)

    # 1: High-Ca Pyroxene (1015 nm & 2120 nm, 1015 depth > 930 depth)
    mask_hpx = (depth_1015 > 0.035) & (depth_2120 > 0.035) & (depth_1015 >= depth_930)
    class_map[mask_hpx] = 1
    confidence_map[mask_hpx] = np.clip(0.65 + depth_1015[mask_hpx] * 0.8, 0.65, 0.95)

    # 2: Olivine (broad 1050 nm depth, weak/absent 2 um pyroxene band)
    mask_olv = (depth_1050 > 0.04) & (depth_1900 < 0.025) & (depth_2120 < 0.025) & (class_map == 6)
    class_map[mask_olv] = 2
    confidence_map[mask_olv] = np.clip(0.60 + depth_1050[mask_olv] * 0.9, 0.60, 0.92)

    # 3: Plagioclase (high albedo > 0.22, subtle 1250 nm depth, weak mafic bands)
    mask_plag = (albedo > 0.22) & (depth_1250 > 0.012) & (depth_930 < 0.04) & (depth_1015 < 0.04) & (class_map == 6)
    class_map[mask_plag] = 3
    confidence_map[mask_plag] = np.clip(0.60 + depth_1250[mask_plag] * 1.5, 0.60, 0.90)

    # 4: Spinel (strong 2020 nm, absent 1 um band)
    mask_spn = (depth_2020 > 0.04) & (depth_930 < 0.02) & (depth_1015 < 0.02) & (class_map == 6)
    class_map[mask_spn] = 4
    confidence_map[mask_spn] = np.clip(0.65 + depth_2020[mask_spn] * 0.8, 0.65, 0.94)

    # 5: Featureless / Dark (low albedo < 0.14, weak absorption depths)
    mask_dark = (albedo < 0.15) & (depth_930 < 0.03) & (depth_1015 < 0.03) & (class_map == 6)
    class_map[mask_dark] = 5
    confidence_map[mask_dark] = np.clip(0.70 + (0.15 - albedo[mask_dark]), 0.65, 0.88)

    # Calculate class statistics
    total_pixels = h * w
    class_names = [
        ("Low-Ca Pyroxene (Orthopyroxene)", "low_ca_pyroxene", 930.0, 1900.0),
        ("High-Ca Pyroxene (Clinopyroxene)", "high_ca_pyroxene", 1015.0, 2120.0),
        ("Olivine-rich (Dunite/Troctolite)", "olivine", 1050.0, None),
        ("Plagioclase-rich (Anorthosite)", "plagioclase", 1250.0, None),
        ("Spinel-like (Mg-Al Spinel)", "spinel", None, 2020.0),
        ("Featureless / Dark (Ilmenite-rich)", "featureless_dark", None, None),
        ("Unclassified Mixture", "unclassified", None, None),
    ]

    class_stats = []
    for c_id, (label, key, nom_b1, nom_b2) in enumerate(class_names):
        c_mask = class_map == c_id
        count = int(np.sum(c_mask))
        coverage_pct = round((count / total_pixels) * 100.0, 2)
        mean_conf = round(float(np.mean(confidence_map[c_mask])) if count > 0 else 0.0, 3)

        mean_depth = 0.0
        if c_id == 0:
            mean_depth = float(np.mean(depth_930[c_mask])) if count > 0 else 0.0
        elif c_id == 1:
            mean_depth = float(np.mean(depth_1015[c_mask])) if count > 0 else 0.0
        elif c_id == 2:
            mean_depth = float(np.mean(depth_1050[c_mask])) if count > 0 else 0.0
        elif c_id == 3:
            mean_depth = float(np.mean(depth_1250[c_mask])) if count > 0 else 0.0
        elif c_id == 4:
            mean_depth = float(np.mean(depth_2020[c_mask])) if count > 0 else 0.0

        ref_info = MINERAL_LIBRARY.get(key, {})
        class_stats.append({
            "id": c_id,
            "key": key,
            "name": label,
            "coverage_pct": coverage_pct,
            "mean_band1_nm": nom_b1,
            "mean_band2_nm": nom_b2,
            "mean_depth": round(mean_depth, 4),
            "confidence": mean_conf,
            "color_hex": ref_info.get("color_hex", "#475569"),
        })

    # Sort classes by coverage percentage descending
    class_stats.sort(key=lambda x: x["coverage_pct"], reverse=True)

    return class_map, confidence_map, class_stats


def run_mineralogy_pipeline(
    cube_data: Dict[str, Any],
    illumination_confidence: float = 1.0,
) -> Dict[str, Any]:
    """
    Run full spectral analysis pipeline and return one JSON-safe dict.
    """
    cube = cube_data["cube"]
    wavelengths = cube_data["wavelengths_nm"]
    provenance = cube_data.get("provenance", {})
    bad_band_mask = cube_data.get("bad_band_mask")

    # 1. Preprocess
    clean_cube, clean_w = preprocess_cube(cube, wavelengths, bad_band_mask)

    # 2. Compute spectral indices
    indices = compute_indices(clean_cube, clean_w, thermal_corrected=provenance.get("thermal_corrected"))

    # 3. Pixel classification
    class_map, conf_map, classes = classify_pixels(clean_cube, clean_w)

    # Scale reported confidence by illumination confidence if low-sun
    warnings: List[str] = []
    if illumination_confidence < 0.6:
        warnings.append(
            f"Low solar illumination (confidence {illumination_confidence*100:.1f}%): "
            "Shadow-dominated terrain suppresses absorption band depths and reduces classification reliability."
        )
        for c in classes:
            c["confidence"] = round(c["confidence"] * illumination_confidence, 3)

    if provenance.get("source") == "simulated":
        warnings.append(
            "Spectral data is synthetically derived from single-band tile albedo. "
            "Real IIRS Level-2 cube required for definitive mineral verification."
        )

    if provenance.get("thermal_corrected") is not True:
        warnings.append(
            "Thermal emission correction not confirmed; hydration band (2.8-3.0 um) evaluation skipped."
        )

    # Mean spectrum and continuum-removed spectrum across tile
    mean_spec = np.nanmean(clean_cube, axis=(0, 1))
    cr_spec, cont_line, _ = continuum_remove(mean_spec, clean_w, (820.0, 2550.0))

    # Diagnostic window coordinates for frontend plotting
    diagnostic_windows = [
        {"name": "1 µm Mafic Silicate (Pyroxene/Olivine)", "start_nm": 820.0, "end_nm": 1320.0, "color": "rgba(59, 130, 246, 0.15)"},
        {"name": "1.25 µm Plagioclase (Fe²⁺ Feldspar)", "start_nm": 1200.0, "end_nm": 1350.0, "color": "rgba(168, 85, 247, 0.15)"},
        {"name": "2 µm Pyroxene / Spinel", "start_nm": 1600.0, "end_nm": 2400.0, "color": "rgba(16, 185, 129, 0.15)"},
        {"name": "3 µm OH/H₂O Hydration", "start_nm": 2700.0, "end_nm": 3100.0, "color": "rgba(236, 72, 153, 0.15)"},
    ]

    return {
        "source": provenance.get("source", "simulated"),
        "dominant_class": classes[0]["name"] if classes else "Unknown",
        "dominant_coverage_pct": classes[0]["coverage_pct"] if classes else 0.0,
        "classes": classes,
        "indices": indices,
        "mean_spectrum": {
            "wavelengths_nm": [round(float(w), 2) for w in clean_w],
            "reflectance": [round(float(r), 4) for r in mean_spec],
            "continuum_removed": [round(float(cr), 4) for cr in cr_spec],
            "continuum": [round(float(c), 4) for c in cont_line],
        },
        "diagnostic_windows": diagnostic_windows,
        "quality": {
            "thermal_corrected": provenance.get("thermal_corrected"),
            "bad_band_count": int(np.sum(bad_band_mask)) if bad_band_mask is not None else 0,
            "warnings": warnings,
        },
    }
