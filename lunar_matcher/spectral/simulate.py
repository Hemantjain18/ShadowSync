"""Simulated Chandrayaan-2 IIRS hyperspectral cube generator."""

import os
from typing import Dict, Any, Optional
import numpy as np


START_NM = 800.0
SAMPLING_NM = 16.85
N_BANDS = 256


def generate_endmember_spectra(wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate analytical mineral endmember reflectance spectra based on RELAB/USGS reference parameters:
    Continuum slope + Gaussian absorption features.
    """
    w = wavelengths

    def gaussian_band(center_nm: float, fwhm_nm: float, depth: float) -> np.ndarray:
        sigma = fwhm_nm / 2.355
        return depth * np.exp(-0.5 * ((w - center_nm) / sigma) ** 2)

    # 1. Low-Ca Pyroxene (Orthopyroxene / Norite):
    # Distinct narrow absorption at ~930 nm and deep feature at ~1900 nm
    cont_lpx = 0.22 + 0.12 * (w - 800) / 2200
    abs_lpx = gaussian_band(930.0, 160.0, 0.28) + gaussian_band(1900.0, 300.0, 0.32)
    spec_lpx = np.clip(cont_lpx - abs_lpx, 0.05, 0.95)

    # 2. High-Ca Pyroxene (Clinopyroxene / Basaltic Diopside/Augite):
    # Shifted to longer wavelengths: ~1000 nm and ~2100 nm
    cont_hpx = 0.20 + 0.10 * (w - 800) / 2200
    abs_hpx = gaussian_band(1015.0, 190.0, 0.25) + gaussian_band(2120.0, 320.0, 0.29)
    spec_hpx = np.clip(cont_hpx - abs_hpx, 0.05, 0.95)

    # 3. Olivine (Dunite / Troctolite):
    # Very broad composite ~1050 nm absorption band with NO 2-micron feature
    cont_olv = 0.24 + 0.08 * (w - 800) / 2200
    abs_olv = (
        gaussian_band(1050.0, 280.0, 0.30)
        + gaussian_band(880.0, 150.0, 0.10)
        + gaussian_band(1220.0, 160.0, 0.12)
    )
    spec_olv = np.clip(cont_olv - abs_olv, 0.05, 0.95)

    # 4. Plagioclase (Anorthosite):
    # High overall albedo with subtle crystal field absorption at ~1250 nm (Fe2+ in plagioclase)
    cont_plag = 0.45 + 0.15 * (w - 800) / 2200
    abs_plag = gaussian_band(1250.0, 200.0, 0.09)
    spec_plag = np.clip(cont_plag - abs_plag, 0.10, 0.98)

    # 5. Ilmenite / Dark mature mare endmember:
    # Very low albedo, flat to slightly blue continuum, absence of strong mafic silicate bands
    spec_ilm = 0.08 + 0.02 * (w - 800) / 2200

    # 6. Spinel (Mg-Al Spinel / Pink Spinel Anorthosite):
    # No 1 um band, strong absorption centered near ~2000-2050 nm
    cont_spn = 0.35 + 0.10 * (w - 800) / 2200
    abs_spn = gaussian_band(2020.0, 240.0, 0.22)
    spec_spn = np.clip(cont_spn - abs_spn, 0.08, 0.95)

    return {
        "low_ca_pyroxene": spec_lpx,
        "high_ca_pyroxene": spec_hpx,
        "olivine": spec_olv,
        "plagioclase": spec_plag,
        "ilmenite": spec_ilm,
        "spinel": spec_spn,
    }


def build_simulated_cube(
    tile_path: Optional[str] = None,
    seed: int = 0,
    height: int = 512,
    width: int = 512,
) -> Dict[str, Any]:
    """
    Build a physically realistic 256-band Chandrayaan-2 IIRS simulated spectral cube.

    Uses tile brightness as an albedo proxy, modulates it with synthetic spatially
    correlated geological endmember abundance maps, mixes linearly, and adds sensor noise.
    """
    rng = np.random.RandomState(seed)
    wavelengths = START_NM + np.arange(N_BANDS, dtype=np.float32) * SAMPLING_NM

    # Load spatial albedo guide if provided
    albedo_img = None
    if tile_path and os.path.exists(tile_path):
        try:
            import cv2
            albedo_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
            if albedo_img is not None:
                height, width = albedo_img.shape
        except Exception:
            pass

    if albedo_img is None:
        # Synthetic cratered albedo texture
        y, x = np.ogrid[:height, :width]
        dist_c = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
        albedo_img = 120 + 40 * np.sin(dist_c / 25.0) + rng.normal(0, 10, (height, width))
        albedo_img = np.clip(albedo_img, 20, 230).astype(np.float32)
    else:
        albedo_img = albedo_img.astype(np.float32)

    # Normalize albedo to mean ~0.20
    albedo_norm = (albedo_img / np.mean(albedo_img)) * 0.20
    albedo_norm = np.clip(albedo_norm, 0.06, 0.45)

    # Generate smooth geological unit abundance maps via Gaussian smoothed noise
    def smooth_field(scale: float, seed_offset: int) -> np.ndarray:
        r = np.random.RandomState(seed + seed_offset)
        small_h, small_w = max(4, int(height / scale)), max(4, int(width / scale))
        low_res = r.uniform(0.0, 1.0, (small_h, small_w)).astype(np.float32)
        try:
            import cv2
            return cv2.resize(low_res, (width, height), interpolation=cv2.INTER_CUBIC)
        except Exception:
            # Fallback simple bilinear interpolation
            from scipy.ndimage import zoom
            zh = height / small_h
            zw = width / small_w
            return zoom(low_res, (zh, zw), order=1)[:height, :width]

    # Geologic units:
    # Pyroxene-rich basalt in darker regions, Plagioclase in higher albedo, Olivine in localized ejecta
    f_hpx = smooth_field(48.0, 1) * (1.2 - albedo_norm * 2.0)
    f_lpx = smooth_field(40.0, 2) * 0.8
    f_olv = np.exp(-((smooth_field(32.0, 3) - 0.7) ** 2) / 0.05) * 1.5
    f_plag = smooth_field(64.0, 4) * (albedo_norm * 3.0)
    f_ilm = smooth_field(50.0, 5) * (1.5 - albedo_norm * 2.5)
    f_spn = np.exp(-((smooth_field(25.0, 6) - 0.85) ** 2) / 0.02) * 0.6

    stack = np.stack([f_hpx, f_lpx, f_olv, f_plag, f_ilm, f_spn], axis=-1)
    stack = np.maximum(stack, 0.02)
    abundances = stack / np.sum(stack, axis=-1, keepdims=True)  # (H, W, 6)

    # Endmembers
    em = generate_endmember_spectra(wavelengths)
    em_matrix = np.stack([
        em["high_ca_pyroxene"],
        em["low_ca_pyroxene"],
        em["olivine"],
        em["plagioclase"],
        em["ilmenite"],
        em["spinel"],
    ], axis=0)  # (6, B)

    # Linear spectral mixture: (H, W, 6) x (6, B) -> (H, W, B)
    cube = np.tensordot(abundances, em_matrix, axes=([2], [0]))

    # Scale overall cube reflectance by local albedo
    albedo_scale = albedo_norm / np.mean(cube, axis=-1, keepdims=True)
    cube = cube * albedo_scale

    # Add realistic sensor photon and readout noise (SNR ~ 150-250)
    noise = rng.normal(0.0, 0.0015, cube.shape).astype(np.float32)
    cube = np.clip(cube + noise, 0.01, 0.99).astype(np.float32)

    return {
        "cube": cube,
        "wavelengths_nm": wavelengths,
        "bad_band_mask": np.zeros(N_BANDS, dtype=bool),
        "abundances": abundances,
        "provenance": {
            "source": "simulated",
            "processing_level": "Simulated Level-2",
            "thermal_corrected": False,
            "notes": (
                "Synthetic IIRS 256-band hyperspectral cube generated from single-band tile albedo. "
                "Demo tile contains no spectral data."
            ),
        },
    }
