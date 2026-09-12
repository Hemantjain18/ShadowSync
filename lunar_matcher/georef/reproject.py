"""Reproject raw lunar orbital rasters (OHRC, TMC-2, IIRS) to common lunar CRS."""

import os
import json
import yaml
from typing import Dict, Any, Optional, Tuple
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

from lunar_matcher.ingest.metadata import parse_issdc_metadata, MalformedMetadataError


DEFAULT_MOON_CRS = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
DEFAULT_MOON_LONGLAT_CRS = "+proj=longlat +a=1737400 +b=1737400 +no_defs"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_resampling_method(method_name: str) -> Resampling:
    mapping = {
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "nearest": Resampling.nearest,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "area": Resampling.average,
    }
    return mapping.get(method_name.lower(), Resampling.bilinear)


def reproject_raster(
    raster_path: str,
    metadata_path: str,
    output_path: str,
    config_path: str = "config.yaml",
    target_crs_override: Optional[str] = None,
    resampling_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reprojects an input raster using its PDS/ISSDC sidecar metadata to a common Moon CRS.

    Parameters:
        raster_path: Path to input raster (GeoTIFF, raw binary with sidecar, or img)
        metadata_path: Path to PDS/ISSDC .lbl or .xml sidecar
        output_path: Path to destination GeoTIFF
        config_path: Path to pipeline config.yaml
        target_crs_override: Optional CRS string to override config
        resampling_override: Optional resampling method (e.g. 'bilinear', 'cubic')

    Returns:
        Dict summarizing original CRS, target CRS, resampling method, and output bounds.
        Also writes a sidecar JSON adjacent to output_path.
    """
    cfg = load_config(config_path)
    georef_cfg = cfg.get("georef", {})
    
    target_crs_str = target_crs_override or georef_cfg.get("target_crs", DEFAULT_MOON_CRS)
    resampling_str = resampling_override or georef_cfg.get("resampling_method", "bilinear")
    resampling_enum = _get_resampling_method(resampling_str)

    # 1. Parse metadata sidecar
    meta = parse_issdc_metadata(metadata_path)
    corners = meta.get("corners")
    if not corners:
        raise MalformedMetadataError(f"Missing corner coordinates in metadata sidecar: {metadata_path}")

    # Validate all 4 corners exist
    required_corners = ["ul_lat", "ul_lon", "lr_lat", "lr_lon"]
    for field in required_corners:
        if field not in corners or corners[field] is None:
            raise MalformedMetadataError(f"Malformed metadata in {metadata_path}: missing required field '{field}'")

    ul_lon = float(corners["ul_lon"])
    ul_lat = float(corners["ul_lat"])
    lr_lon = float(corners["lr_lon"])
    lr_lat = float(corners["lr_lat"])

    west = min(ul_lon, lr_lon)
    east = max(ul_lon, lr_lon)
    south = min(ul_lat, lr_lat)
    north = max(ul_lat, lr_lat)

    # 2. Open source raster
    if not os.path.exists(raster_path):
        raise FileNotFoundError(f"Input raster not found: {raster_path}")

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_count = src.count
        src_dtype = src.dtypes[0]
        src_nodata = src.nodata

        # If source raster lacks embedded CRS or transform, synthesize from ISSDC metadata corners
        if src_crs is None:
            src_crs = CRS.from_string(DEFAULT_MOON_LONGLAT_CRS)
            src_transform = from_bounds(west, south, east, north, src.width, src.height)

        dst_crs = CRS.from_string(target_crs_str)

        # 3. Calculate destination transform and shape in target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height,
            left=west, bottom=south, right=east, top=north
        )

        dst_profile = src.profile.copy()
        dst_profile.update({
            "driver": "GTiff",
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "count": src_count,
            "dtype": src_dtype,
            "nodata": src_nodata,
        })

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with rasterio.open(output_path, "w", **dst_profile) as dst:
            for band_idx in range(1, src_count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling_enum,
                )

    # 4. Write provenance sidecar JSON
    sidecar_data = {
        "source_raster": raster_path,
        "source_metadata": metadata_path,
        "sensor": meta.get("sensor", "UNKNOWN"),
        "original_crs": str(src_crs),
        "target_crs": str(dst_crs),
        "resampling_method": resampling_str,
        "source_selenographic_corners": corners,
        "solar_angles": {
            "elevation_deg": meta.get("sun_elevation_deg"),
            "azimuth_deg": meta.get("sun_azimuth_deg"),
            "incidence_deg": meta.get("solar_incidence_angle_deg"),
        },
        "target_dimensions": {
            "width": dst_width,
            "height": dst_height,
            "bands": src_count,
        },
        "target_bounds": {
            "left": dst_transform.c,
            "top": dst_transform.f,
            "pixel_size_x": dst_transform.a,
            "pixel_size_y": dst_transform.e,
        }
    }

    json_sidecar_path = f"{output_path}.json"
    with open(json_sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    return sidecar_data
