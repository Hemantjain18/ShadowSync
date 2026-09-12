"""Unit tests for georef/reproject.py with synthetic rasters and ISSDC metadata."""

import os
import json
import tempfile
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from lunar_matcher.georef.reproject import reproject_raster
from lunar_matcher.ingest.metadata import parse_issdc_metadata, MalformedMetadataError


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def create_synthetic_raster(filepath: str, width: int = 64, height: int = 64):
    """Create a synthetic single-band GeoTIFF with crater-like features."""
    arr = np.zeros((height, width), dtype=np.uint8)
    # Background lunar regolith
    arr[:] = 120
    # Synthetic crater in center
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
    crater_rim = (dist_from_center >= 12) & (dist_from_center <= 16)
    crater_floor = dist_from_center < 12
    arr[crater_rim] = 220
    arr[crater_floor] = 40

    transform = from_origin(20.0, -10.0, 0.0001, 0.0001)
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=arr.dtype,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)


def create_valid_lbl_metadata(filepath: str):
    content = """PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
INSTRUMENT_NAME = "OHRC"
INSTRUMENT_ID = "OHRC"
PRODUCT_ID = "ch2_ohr_ncp_20200812T142512"
UPPER_LEFT_LATITUDE = -9.95
UPPER_LEFT_LONGITUDE = 20.01
LOWER_RIGHT_LATITUDE = -10.05
LOWER_RIGHT_LONGITUDE = 20.15
MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"
SOLAR_INCIDENCE_ANGLE = 45.2
SUN_AZIMUTH = 135.0
SUN_ELEVATION = 44.8
END
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def create_malformed_lbl_metadata(filepath: str):
    content = """PDS_VERSION_ID = PDS3
INSTRUMENT_NAME = "OHRC"
PRODUCT_ID = "ch2_broken"
# Missing coordinates completely!
SOLAR_INCIDENCE_ANGLE = 45.0
END
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def test_metadata_parsing(temp_dir):
    lbl_path = os.path.join(temp_dir, "meta.lbl")
    create_valid_lbl_metadata(lbl_path)
    
    meta = parse_issdc_metadata(lbl_path)
    assert meta["sensor"] == "OHRC"
    assert meta["corners"]["ul_lat"] == -9.95
    assert meta["corners"]["lr_lon"] == 20.15
    assert meta["solar_incidence_angle_deg"] == 45.2
    assert meta["sun_azimuth_deg"] == 135.0


def test_malformed_metadata_raises_error(temp_dir):
    lbl_path = os.path.join(temp_dir, "broken.lbl")
    create_malformed_lbl_metadata(lbl_path)
    
    with pytest.raises(MalformedMetadataError) as exc_info:
        parse_issdc_metadata(lbl_path)
    assert "Missing required coordinate fields" in str(exc_info.value)


def test_reproject_raster_end_to_end(temp_dir):
    raster_path = os.path.join(temp_dir, "synthetic_ohrc.tif")
    lbl_path = os.path.join(temp_dir, "synthetic_ohrc.lbl")
    out_path = os.path.join(temp_dir, "reprojected_ohrc.tif")
    
    create_synthetic_raster(raster_path)
    create_valid_lbl_metadata(lbl_path)
    
    result = reproject_raster(
        raster_path=raster_path,
        metadata_path=lbl_path,
        output_path=out_path,
    )
    
    assert os.path.exists(out_path)
    assert os.path.exists(f"{out_path}.json")
    
    # Check JSON sidecar contents
    with open(f"{out_path}.json", "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    
    assert "target_crs" in sidecar
    assert "resampling_method" in sidecar
    assert sidecar["resampling_method"] == "bilinear"
    assert sidecar["sensor"] == "OHRC"
    assert sidecar["target_dimensions"]["width"] > 0
    assert sidecar["target_dimensions"]["height"] > 0
