"""IIRS hyperspectral cube loader for Chandrayaan-2 lunar spectral data."""

import os
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


DEFAULT_WAVELENGTH_START_NM = 800.0
DEFAULT_SAMPLING_NM = 16.85
DEFAULT_N_BANDS = 256


def get_default_wavelengths(n_bands: int = DEFAULT_N_BANDS, start_nm: float = DEFAULT_WAVELENGTH_START_NM, sampling_nm: float = DEFAULT_SAMPLING_NM) -> np.ndarray:
    """Generate default Chandrayaan-2 IIRS nominal wavelengths."""
    return np.linspace(start_nm, start_nm + (n_bands - 1) * sampling_nm, n_bands, dtype=np.float32)


def parse_envi_header(hdr_path: str) -> Dict[str, Any]:
    """Parse standard ENVI .hdr header file."""
    metadata: Dict[str, Any] = {}
    with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_bracket = False
    current_key = ""
    bracket_content = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue

        if in_bracket:
            bracket_content.append(line)
            if "}" in line:
                in_bracket = False
                val = " ".join(bracket_content).replace("{", "").replace("}", "").strip()
                metadata[current_key] = val
            continue

        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip().lower()
            val = parts[1].strip()

            if "{" in val and "}" not in val:
                in_bracket = True
                current_key = key
                bracket_content = [val]
            else:
                metadata[key] = val.replace("{", "").replace("}", "").strip()

    return metadata


def load_iirs_cube(path: str) -> Dict[str, Any]:
    """
    Load an IIRS hyperspectral cube from ENVI (.hdr/.img), GeoTIFF, or NumPy format.

    Parameters:
        path: Path to the spectral cube file.

    Returns:
        dict containing:
            - cube: float32 ndarray of shape (H, W, B)
            - wavelengths_nm: 1D float32 ndarray of length B
            - bad_band_mask: 1D boolean ndarray of length B (True = bad band)
            - provenance: dict with source ('measured'|'simulated'), processing_level,
                          thermal_corrected (bool|None), notes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spectral cube file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    base_no_ext = os.path.splitext(path)[0]

    cube_arr: Optional[np.ndarray] = None
    wavelengths: Optional[np.ndarray] = None
    thermal_corrected: Optional[bool] = None
    processing_level = "Level-2"
    source = "measured"
    notes = ""

    if ext in [".npy", ".npz"]:
        if ext == ".npz":
            data = np.load(path)
            cube_arr = data["cube"] if "cube" in data else data[data.files[0]]
            if "wavelengths" in data:
                wavelengths = data["wavelengths"].astype(np.float32)
            elif "wavelengths_nm" in data:
                wavelengths = data["wavelengths_nm"].astype(np.float32)
            if "provenance" in data:
                prov = data["provenance"].item() if hasattr(data["provenance"], "item") else dict(data["provenance"])
                source = prov.get("source", source)
                thermal_corrected = prov.get("thermal_corrected", thermal_corrected)
                notes = prov.get("notes", notes)
        else:
            cube_arr = np.load(path)

    elif ext in [".img", ".dat", ".raw"] or os.path.exists(base_no_ext + ".hdr") or ext == ".hdr":
        # ENVI format
        hdr_path = path if ext == ".hdr" else base_no_ext + ".hdr"
        img_path = base_no_ext + ".img" if ext == ".hdr" else path
        if not os.path.exists(img_path) and os.path.exists(base_no_ext + ".dat"):
            img_path = base_no_ext + ".dat"

        meta = parse_envi_header(hdr_path)
        samples = int(meta.get("samples", 512))
        lines = int(meta.get("lines", 512))
        bands = int(meta.get("bands", DEFAULT_N_BANDS))
        data_type = int(meta.get("data type", 4))  # 4 = float32, 2 = int16, 12 = uint16
        interleave = meta.get("interleave", "bsq").lower()

        dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}
        np_dtype = dtype_map.get(data_type, np.float32)

        if os.path.exists(img_path):
            raw_bytes = np.fromfile(img_path, dtype=np_dtype)
            if interleave == "bil":
                cube_arr = raw_bytes.reshape((lines, bands, samples)).transpose(0, 2, 1)
            elif interleave == "bip":
                cube_arr = raw_bytes.reshape((lines, samples, bands))
            else:  # bsq
                cube_arr = raw_bytes.reshape((bands, lines, samples)).transpose(1, 2, 0)
        else:
            raise FileNotFoundError(f"ENVI image file corresponding to {hdr_path} not found at {img_path}")

        # Extract wavelengths from ENVI header
        if "wavelength" in meta:
            w_str = meta["wavelength"].replace(",", " ").split()
            try:
                wavelengths = np.array([float(x) for x in w_str], dtype=np.float32)
                # Check if wavelengths are in micrometers (e.g. 0.8 to 3.0)
                if np.max(wavelengths) < 10.0:
                    wavelengths = wavelengths * 1000.0  # convert to nm
            except ValueError:
                pass

        if "thermal correction" in meta:
            thermal_corrected = meta["thermal correction"].lower() in ["true", "yes", "applied", "1"]

    elif ext in [".tif", ".tiff"]:
        # GeoTIFF loader via rasterio if available, else tifffile or imageio
        try:
            import rasterio
            with rasterio.open(path) as src:
                # rasterio reads as (bands, H, W)
                raw_tif = src.read()
                if raw_tif.ndim == 3:
                    cube_arr = raw_tif.transpose(1, 2, 0)
                elif raw_tif.ndim == 2:
                    cube_arr = raw_tif[:, :, np.newaxis]
                if src.tags().get("WAVELENGTHS"):
                    w_str = src.tags()["WAVELENGTHS"].split(",")
                    wavelengths = np.array([float(x) for x in w_str], dtype=np.float32)
                if src.tags().get("THERMAL_CORRECTED"):
                    thermal_corrected = src.tags()["THERMAL_CORRECTED"].lower() in ["true", "1"]
        except ImportError:
            try:
                import tifffile
                raw_tif = tifffile.imread(path)
                if raw_tif.ndim == 3:
                    cube_arr = raw_tif if raw_tif.shape[2] > raw_tif.shape[0] else raw_tif.transpose(1, 2, 0)
                elif raw_tif.ndim == 2:
                    cube_arr = raw_tif[:, :, np.newaxis]
            except ImportError:
                import cv2
                raw_tif = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if raw_tif.ndim == 2:
                    cube_arr = raw_tif[:, :, np.newaxis]
                else:
                    cube_arr = raw_tif
    else:
        raise ValueError(f"Unsupported spectral cube format: {ext}")

    if cube_arr is None:
        raise RuntimeError(f"Could not load spectral array from {path}")

    # Ensure shape is 3D (H, W, B)
    if cube_arr.ndim == 2:
        cube_arr = cube_arr[:, :, np.newaxis]

    h, w, b = cube_arr.shape
    cube_arr = cube_arr.astype(np.float32)

    # Replace fill values (-9999, -32768, etc.) and non-finite values with NaN
    cube_arr[cube_arr <= -9998.0] = np.nan
    cube_arr[~np.isfinite(cube_arr)] = np.nan

    # Generate or validate wavelengths
    if wavelengths is None or len(wavelengths) != b:
        wavelengths = get_default_wavelengths(n_bands=b)

    # Compute bad band mask (bands with >50% NaN pixels or all-zero)
    nan_ratio = np.mean(np.isnan(cube_arr), axis=(0, 1))
    std_per_band = np.nanstd(cube_arr, axis=(0, 1))
    bad_band_mask = (nan_ratio > 0.5) | (std_per_band < 1e-7)

    return {
        "cube": cube_arr,
        "wavelengths_nm": wavelengths,
        "bad_band_mask": bad_band_mask,
        "provenance": {
            "source": source,
            "processing_level": processing_level,
            "thermal_corrected": thermal_corrected,
            "notes": notes,
        },
    }
