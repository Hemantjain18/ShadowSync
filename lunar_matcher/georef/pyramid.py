"""Scale bridging and area-weighted image pyramid generation for multi-sensor lunar imagery."""

import os
from typing import Dict, Any, Tuple, Union, Optional, List
import numpy as np
import cv2
import rasterio
from rasterio.enums import Resampling


# Standard Chandrayaan-2 sensor ground sample distances (meters/pixel)
SENSOR_GSDS = {
    "OHRC": 0.25,
    "TMC2": 5.0,
    "TMC": 5.0,
    "IIRS": 80.0,
    "LROC_NAC": 0.5,
    "LROC_WAC": 100.0,
}


def pair_planner(
    sensor_a_gsd: Union[float, str],
    sensor_b_gsd: Union[float, str],
    max_direct_ratio: float = 25.0,
) -> Dict[str, Any]:
    """
    Decides whether to match two sensors directly or route through an intermediate pivot sensor.
    
    Encodes the scale-bridging strategy:
    - OHRC (0.25m) -> TMC (5m): ~20x gap (Direct match viable with pyramid)
    - TMC (5m) -> IIRS (80m): ~16x gap (Direct match viable with pyramid)
    - OHRC (0.25m) -> IIRS (80m): ~320x gap (Direct matching IMPOSSIBLE -> Route through TMC as pivot)

    Parameters:
        sensor_a_gsd: GSD in meters or sensor name (e.g. 'OHRC' or 0.25)
        sensor_b_gsd: GSD in meters or sensor name (e.g. 'IIRS' or 80.0)
        max_direct_ratio: Maximum allowable resolution ratio for direct matching (default: 25.0)

    Returns:
        Dict detailing strategy ('direct' | 'pivot'), pivot sensor, scale ratio, and step execution chain.
    """
    # Resolve names if passed as strings
    name_a = str(sensor_a_gsd).upper()
    name_b = str(sensor_b_gsd).upper()

    gsd_a = SENSOR_GSDS.get(name_a, float(sensor_a_gsd) if not isinstance(sensor_a_gsd, str) or sensor_a_gsd.replace('.', '', 1).isdigit() else None)
    gsd_b = SENSOR_GSDS.get(name_b, float(sensor_b_gsd) if not isinstance(sensor_b_gsd, str) or sensor_b_gsd.replace('.', '', 1).isdigit() else None)

    if gsd_a is None or gsd_b is None:
        raise ValueError(f"Unknown sensor or invalid GSD provided: {sensor_a_gsd}, {sensor_b_gsd}")

    min_gsd = min(gsd_a, gsd_b)
    max_gsd = max(gsd_a, gsd_b)
    ratio = max_gsd / max(min_gsd, 1e-6)

    # If the sensors are OHRC and IIRS or ratio exceeds threshold
    is_ohrc_iirs = (name_a in ["OHRC", "0.25"] and name_b in ["IIRS", "80.0"]) or \
                    (name_b in ["OHRC", "0.25"] and name_a in ["IIRS", "80.0"])

    if is_ohrc_iirs or ratio > max_direct_ratio:
        # Route through TMC-2 (5m) as geometric pivot
        return {
            "strategy": "pivot",
            "direct_match_allowed": False,
            "scale_ratio": round(ratio, 2),
            "pivot_sensor": "TMC2",
            "pivot_gsd_m": 5.0,
            "chain_steps": [
                {
                    "step": 1,
                    "sensor_pair": (name_a if gsd_a < gsd_b else "TMC2", "TMC2" if gsd_a < gsd_b else name_b),
                    "scale_ratio": round(5.0 / min_gsd, 2),
                    "description": f"Register fine {name_a if gsd_a < gsd_b else 'TMC2'} to intermediate TMC-2 (5.0m)"
                },
                {
                    "step": 2,
                    "sensor_pair": ("TMC2", name_b if gsd_a < gsd_b else name_a),
                    "scale_ratio": round(max_gsd / 5.0, 2),
                    "description": f"Register intermediate TMC-2 (5.0m) to coarse {name_b if gsd_a < gsd_b else name_a}"
                }
            ],
            "rationale": (
                f"Direct scale gap of {ratio:.1f}x between {name_a} ({min_gsd}m) and {name_b} ({max_gsd}m) "
                "exceeds safe feature matching limit (25x). Routed through TMC-2 (5.0m) as geometric pivot."
            )
        }

    return {
        "strategy": "direct",
        "direct_match_allowed": True,
        "scale_ratio": round(ratio, 2),
        "pivot_sensor": None,
        "chain_steps": [
            {
                "step": 1,
                "sensor_pair": (name_a, name_b),
                "scale_ratio": round(ratio, 2),
                "description": f"Direct matching between {name_a} and {name_b} with pyramid scale normalization"
            }
        ],
        "rationale": f"Scale gap of {ratio:.1f}x is within direct matching threshold (<= {max_direct_ratio}x)."
    }


def build_pyramid(
    raster: Union[np.ndarray, str],
    target_gsd: float,
    current_gsd: Optional[float] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Downsamples a high-resolution raster to a target ground sample distance (GSD)
    using area-weighted resampling (cv2.INTER_AREA or rasterio average),
    preserving radiometric integrity and avoiding high-frequency aliasing.

    Parameters:
        raster: 2D or 3D numpy array, or path to GeoTIFF
        target_gsd: Target ground sample distance in meters
        current_gsd: Current ground sample distance in meters (required if array passed)
        output_path: Optional path to save downsampled GeoTIFF if input was a GeoTIFF

    Returns:
        Resampled numpy array matching the target GSD.
    """
    if isinstance(raster, str):
        # Open GeoTIFF
        with rasterio.open(raster) as src:
            # Estimate GSD from pixel resolution
            x_res = abs(src.transform.a)
            # If coordinates are in meters
            if current_gsd is None:
                current_gsd = x_res
            
            arr = src.read(1)
            src_profile = src.profile.copy()
            src_transform = src.transform

        scale_factor = current_gsd / target_gsd
        if scale_factor >= 1.0:
            # Already coarser than or equal to target
            return arr

        new_width = max(1, int(round(arr.shape[1] * scale_factor)))
        new_height = max(1, int(round(arr.shape[0] * scale_factor)))

        # Area-weighted downsampling
        downsampled = cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            new_transform = src_transform * src_transform.scale(
                (arr.shape[1] / new_width),
                (arr.shape[0] / new_height)
            )
            src_profile.update({
                "width": new_width,
                "height": new_height,
                "transform": new_transform,
            })
            with rasterio.open(output_path, "w", **src_profile) as dst:
                dst.write(downsampled, 1)

        return downsampled

    elif isinstance(raster, np.ndarray):
        if current_gsd is None:
            raise ValueError("current_gsd must be specified when passing a numpy array to build_pyramid")

        scale_factor = current_gsd / target_gsd
        if scale_factor >= 1.0:
            return raster

        orig_h, orig_w = raster.shape[:2]
        new_w = max(1, int(round(orig_w * scale_factor)))
        new_h = max(1, int(round(orig_h * scale_factor)))

        if raster.ndim == 2:
            return cv2.resize(raster, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # Multichannel / hyperspectral cube
            channels = raster.shape[2]
            resampled_cube = np.zeros((new_h, new_w, channels), dtype=raster.dtype)
            for c in range(channels):
                resampled_cube[:, :, c] = cv2.resize(raster[:, :, c], (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resampled_cube
    else:
        raise TypeError(f"Unsupported raster type: {type(raster)}")
