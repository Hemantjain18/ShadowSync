"""Spectral band selection and hyperspectral cube dimensionality reduction."""

from typing import Union, List, Optional
import numpy as np


def get_band_index_for_wavelength(target_nm: float = 1000.0, start_nm: float = 800.0, sampling_nm: float = 16.85, n_bands: int = 256) -> int:
    """
    Computes the nearest IIRS spectral band index for a target wavelength in nanometers.
    For IIRS (800 nm start, 16.85 nm/band), 1000 nm corresponds to band index:
    round((1000 - 800) / 16.85) = round(11.87) = 12 (rather than the obsolete placeholder 42).
    """
    wavelengths = start_nm + np.arange(n_bands) * sampling_nm
    idx = int(np.argmin(np.abs(wavelengths - target_nm)))
    return idx


def select_reference_band(
    cube: np.ndarray,
    target_wavelength_nm: float = 1000.0,
    wavelengths_nm: Optional[np.ndarray] = None,
    start_nm: float = 800.0,
    sampling_nm: float = 16.85,
) -> np.ndarray:
    """
    Selects a single 2D spatial slice from a 3D hyperspectral cube closest to the target wavelength.
    """
    n_bands = cube.shape[-1]
    if wavelengths_nm is not None and len(wavelengths_nm) == n_bands:
        idx = int(np.argmin(np.abs(wavelengths_nm - target_wavelength_nm)))
    else:
        idx = get_band_index_for_wavelength(target_wavelength_nm, start_nm, sampling_nm, n_bands)

    return cube[:, :, idx]
