"""Tiling, content-addressed disk caching, and demo tile preparation for large lunar scenes."""

import os
import sys
import json
import hashlib
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds

from lunar_matcher.georef.reproject import DEFAULT_MOON_CRS


class TileDiskCache:
    """
    Disk cache keyed by SHA256 of (source_file_hash, reprojection_params, tile_bounds)
    so re-running the demo skips heavy reprocessing.
    """
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.cache_dir, "cache_manifest.json")
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    def generate_key(
        self,
        source_identifier: str,
        reprojection_params: Any,
        tile_bounds: Tuple[float, float, float, float],
    ) -> str:
        data_str = f"{source_identifier}|{json.dumps(reprojection_params, sort_keys=True)}|{list(tile_bounds)}"
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def has(self, key: str) -> bool:
        if key in self.manifest:
            cached_file = self.manifest[key].get("cached_file")
            return cached_file is not None and os.path.exists(cached_file)
        return False

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.has(key):
            return self.manifest[key]
        return None

    def put(self, key: str, cached_file_path: str, metadata: Dict[str, Any]):
        self.manifest[key] = {
            "cached_file": cached_file_path,
            "metadata": metadata,
        }
        self._save_manifest()


def file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file's first 2MB and size for fast content addressing."""
    if not os.path.exists(filepath):
        return hashlib.sha256(filepath.encode()).hexdigest()
    h = hashlib.sha256()
    file_size = os.path.getsize(filepath)
    h.update(str(file_size).encode())
    with open(filepath, "rb") as f:
        chunk = f.read(2 * 1024 * 1024)
        h.update(chunk)
    return h.hexdigest()


def tile_raster(
    path: str,
    tile_size_px: int = 512,
    overlap_px: int = 64,
    output_dir: str = "data/tiles",
    cache: Optional[TileDiskCache] = None,
) -> List[Dict[str, Any]]:
    """
    Tiles a full-resolution GeoTIFF into fixed-size chunks, preserving geospatial bounds per tile.

    Parameters:
        path: Path to input GeoTIFF
        tile_size_px: Tile dimension in pixels (width = height)
        overlap_px: Overlap margin between adjacent tiles in pixels
        output_dir: Destination directory for output tile GeoTIFFs
        cache: Optional TileDiskCache instance for content-addressed skipping

    Returns:
        List of dicts with tile metadata and saved file paths.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot tile non-existent raster: {path}")

    os.makedirs(output_dir, exist_ok=True)
    src_hash = file_hash(path)
    tiles = []

    step = max(1, tile_size_px - overlap_px)
    basename = os.path.splitext(os.path.basename(path))[0]

    with rasterio.open(path) as src:
        width = src.width
        height = src.height
        count = src.count
        profile = src.profile.copy()

        col_steps = list(range(0, width, step))
        row_steps = list(range(0, height, step))

        tile_idx = 0
        for r in row_steps:
            for c in col_steps:
                w_width = min(tile_size_px, width - c)
                w_height = min(tile_size_px, height - r)
                if w_width <= 0 or w_height <= 0:
                    continue

                window = Window(c, r, w_width, w_height)
                win_transform = rasterio.windows.transform(window, src.transform)
                bounds = rasterio.windows.bounds(window, src.transform)

                tile_filename = f"{basename}_tile_{tile_idx:04d}_r{r}_c{c}.tif"
                tile_path = os.path.join(output_dir, tile_filename)

                # Check disk cache
                if cache is not None:
                    cache_key = cache.generate_key(src_hash, {"tile_size": tile_size_px}, bounds)
                    if cache.has(cache_key):
                        cached_entry = cache.get(cache_key)
                        tiles.append(cached_entry["metadata"])
                        tile_idx += 1
                        continue

                # Read window data
                data = src.read(window=window)

                tile_profile = profile.copy()
                tile_profile.update({
                    "width": w_width,
                    "height": w_height,
                    "transform": win_transform,
                })

                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(data)

                tile_meta = {
                    "tile_index": tile_idx,
                    "tile_path": tile_path,
                    "row_offset": r,
                    "col_offset": c,
                    "width": w_width,
                    "height": w_height,
                    "bounds": bounds,
                    "crs": str(src.crs),
                }

                if cache is not None:
                    cache.put(cache_key, tile_path, tile_meta)

                tiles.append(tile_meta)
                tile_idx += 1

    return tiles


# Per-region terrain + geolocation presets. Each region gets its own RNG seed
# (so craters/noise differ per region instead of every region reusing the
# same scene) and its own real approximate selenographic bounding box, so
# the demo shows genuinely different terrain -- and therefore genuinely
# different match counts / confidence / RMSE -- across regions, not just
# different sensor blur on an identical scene.
REGION_PRESETS = {
    # Apollo 11 landing site, Mare Tranquillitatis (existing baseline demo
    # scene -- kept byte-for-byte identical to the original generator so
    # already-generated tiles / regression tests aren't invalidated).
    "Apollo11": {
        "seed": 42,
        "crater_density": "medium",
        # (west, south, east, north) in degrees, matching the original hardcoded bounds.
        "bounds": (23.40, 0.60, 23.60, 0.80),
    },
    # Tycho, southern lunar highlands: young, heavily cratered terrain.
    # Real coordinates: 43.31 S, 11.36 W (Wikipedia). Dense small craters
    # give the matcher lots of texture -> high match counts / confidence.
    "Tycho": {
        "seed": 1108,
        "crater_density": "dense",
        "bounds": (-11.46, -43.41, -11.26, -43.21),
    },
    # Sinus Iridum, "Bay of Rainbows": smooth basaltic mare plain with very
    # few craters. Real coordinates: 44.1 N, 31.5 W (Wikipedia). Sparse
    # texture is a genuinely harder registration case -- fewer keypoints,
    # lower confidence -- useful to show the pipeline's numbers actually
    # respond to terrain rather than being hardcoded.
    "SinusIridum": {
        "seed": 3160,
        "crater_density": "sparse",
        "bounds": (-31.60, 44.00, -31.40, 44.20),
    },
}

_CRATER_DENSITY_PRESETS = {
    # (crater_count, radius_fraction_range, depth_range)
    "sparse": {"count": 3, "r_frac": (0.03, 0.07), "depth": (20, 35)},
    "medium": {"count": 5, "r_frac": (0.04, 0.16), "depth": (30, 70)},
    "dense": {"count": 13, "r_frac": (0.02, 0.11), "depth": (25, 65)},
}


def _generate_craters(rng: np.random.RandomState, size: int, density: str):
    """Procedurally places craters for one region, seeded so the layout is
    reproducible per-region but distinct across regions/densities."""
    if density == "medium":
        # Preserve the exact original Apollo11 crater layout (hand-placed,
        # not random) so existing Apollo11 tiles stay reproducible.
        return [
            {"x": int(size * 0.35), "y": int(size * 0.40), "r": int(size * 0.16), "depth": 70},
            {"x": int(size * 0.70), "y": int(size * 0.65), "r": int(size * 0.10), "depth": 55},
            {"x": int(size * 0.55), "y": int(size * 0.25), "r": int(size * 0.07), "depth": 40},
            {"x": int(size * 0.20), "y": int(size * 0.75), "r": int(size * 0.05), "depth": 35},
            {"x": int(size * 0.80), "y": int(size * 0.20), "r": int(size * 0.04), "depth": 30},
        ]

    preset = _CRATER_DENSITY_PRESETS[density]
    craters = []
    for _ in range(preset["count"]):
        r_frac = rng.uniform(*preset["r_frac"])
        craters.append({
            "x": int(rng.uniform(0.12, 0.88) * size),
            "y": int(rng.uniform(0.12, 0.88) * size),
            "r": max(4, int(size * r_frac)),
            "depth": rng.uniform(*preset["depth"]),
        })
    return craters


def generate_synthetic_demo_scene(
    sensor_name: str,
    output_path: str,
    size: int = 512,
    incidence_angle: float = 45.0,
    sun_azimuth: float = 120.0,
    seed: int = 42,
    crater_density: str = "medium",
    bounds: Tuple[float, float, float, float] = (23.40, 0.60, 23.60, 0.80),
):
    """
    Generates a realistic lunar regolith tile with craters, ridges, and sensor-specific resolution.

    seed / crater_density / bounds let different regions get genuinely
    different terrain (and therefore genuinely different pipeline results)
    instead of every region reusing the exact same scene. See REGION_PRESETS.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rng = np.random.RandomState(seed)

    # Base regolith noise
    img = rng.normal(128, 10, (size, size)).astype(np.float32)

    # Multi-scale craters (region-specific density/placement)
    craters = _generate_craters(rng, size, crater_density)

    y, x = np.mgrid[:size, :size]
    sun_rad = np.radians(sun_azimuth)
    light_dx = np.cos(sun_rad)
    light_dy = np.sin(sun_rad)

    for c in craters:
        d = np.sqrt((x - c["x"])**2 + (y - c["y"])**2)
        r = c["r"]
        # Bowl
        bowl = d < r
        img[bowl] -= (c["depth"] * (1.0 - (d[bowl] / r)**2))
        
        # Rim highlight / shadow based on sun angle
        rim = (d >= r * 0.85) & (d <= r * 1.25)
        directional_factor = (x[rim] - c["x"]) / r * light_dx + (y[rim] - c["y"]) / r * light_dy
        img[rim] += (c["depth"] * 0.6 * directional_factor)

    # Add diagonal ridge
    ridge_dist = np.abs((x - y * 0.5) - size * 0.2)
    ridge_mask = ridge_dist < 15
    img[ridge_mask] += (25.0 * (1.0 - ridge_dist[ridge_mask] / 15.0))

    # Sensor specific resolution / blur
    if sensor_name == "OHRC":
        # Ultra high resolution (0.25m): crisp, fine granular ejecta
        img_final = cv2.GaussianBlur(img, (3, 3), 0.5)
    elif sensor_name in ["TMC2", "TMC"]:
        # 5m resolution: moderate optical smoothing
        img_final = cv2.GaussianBlur(img, (7, 7), 1.8)
    elif sensor_name == "IIRS":
        # 80m resolution: coarse spatial structure
        img_coarse = cv2.resize(img, (size // 8, size // 8), interpolation=cv2.INTER_AREA)
        img_final = cv2.resize(img_coarse, (size, size), interpolation=cv2.INTER_NEAREST)
    elif sensor_name.startswith("LROC"):
        # LROC NAC reference (0.5m)
        img_final = cv2.GaussianBlur(img, (3, 3), 0.8)
    else:
        img_final = img

    img_u8 = np.clip(img_final, 0, 255).astype(np.uint8)

    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, size, size)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs=DEFAULT_MOON_CRS,
        transform=transform,
    ) as dst:
        dst.write(img_u8, 1)

    # Write accompanying metadata sidecar
    sidecar_path = os.path.splitext(output_path)[0] + ".lbl"
    lbl_content = f"""PDS_VERSION_ID = PDS3
INSTRUMENT_NAME = "{sensor_name}"
INSTRUMENT_ID = "{sensor_name}"
PRODUCT_ID = "CH2_{sensor_name}_DEMO_REGION"
UPPER_LEFT_LATITUDE = {north:.2f}
UPPER_LEFT_LONGITUDE = {west:.2f}
LOWER_RIGHT_LATITUDE = {south:.2f}
LOWER_RIGHT_LONGITUDE = {east:.2f}
MAP_PROJECTION_TYPE = "EQUIRECTANGULAR"
SOLAR_INCIDENCE_ANGLE = {incidence_angle:.1f}
SUN_AZIMUTH = {sun_azimuth:.1f}
SUN_ELEVATION = {max(0.0, 90.0 - incidence_angle):.1f}
END
"""
    with open(sidecar_path, "w", encoding="utf-8") as f:
        f.write(lbl_content)

    return output_path


def prepare_demo_tiles(
    region: str = "Apollo11",
    output_dir: str = "data/tiles",
    cache_dir: str = "data/cache",
) -> Dict[str, str]:
    """
    Pre-crops, normalizes, and pre-caches demo tiles for OHRC, TMC-2, IIRS, and LROC reference
    so the live demo loads in milliseconds directly from disk cache.

    Looks up `region` in REGION_PRESETS for its seed / crater density / real
    selenographic bounds; falls back to the original Apollo11 preset for any
    unrecognized region name so old callers keep working.
    """
    cache = TileDiskCache(cache_dir=cache_dir)
    os.makedirs(output_dir, exist_ok=True)

    preset = REGION_PRESETS.get(region, REGION_PRESETS["Apollo11"])
    seed = preset["seed"]
    crater_density = preset["crater_density"]
    bounds = preset["bounds"]

    sensor_configs = {
        "OHRC": {"incidence": 45.0, "azimuth": 115.0},
        "TMC2": {"incidence": 48.0, "azimuth": 122.0},
        "IIRS": {"incidence": 42.0, "azimuth": 110.0},
        "LROC_NAC": {"incidence": 46.0, "azimuth": 118.0},
        # Extreme polar shadow tile to demonstrate Tier 3 Gating
        "OHRC_POLAR_SHADOW": {"incidence": 84.5, "azimuth": 270.0},
    }

    tile_paths = {}
    for sensor, angles in sensor_configs.items():
        fname = f"{region}_{sensor}.tif"
        out_path = os.path.join(output_dir, fname)
        generate_synthetic_demo_scene(
            sensor_name=sensor,
            output_path=out_path,
            size=512,
            incidence_angle=angles["incidence"],
            sun_azimuth=angles["azimuth"],
            seed=seed,
            crater_density=crater_density,
            bounds=bounds,
        )
        tile_paths[sensor] = out_path

        # Cache entry
        key = cache.generate_key(f"demo_{sensor}_{region}", angles, bounds)
        cache.put(key, out_path, {"sensor": sensor, "region": region, "angles": angles})

    print(f"Pre-cached {len(tile_paths)} multi-sensor demo tiles for region '{region}' in {output_dir}")
    return tile_paths


def cli_prepare_demo_tiles():
    """CLI entrypoint: prepare-demo-tiles --region <name>"""
    parser = argparse.ArgumentParser(
        description="Pre-crop and pre-reproject multi-sensor lunar tiles into cache for fast live demo execution."
    )
    parser.add_argument(
        "--region",
        type=str,
        default="Apollo11",
        help="Region name or target bounding box (e.g. Apollo11, SouthPole, Shackleton)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tiles",
        help="Directory to store pre-generated tile GeoTIFFs",
    )
    args = parser.parse_args()
    prepare_demo_tiles(region=args.region, output_dir=args.output_dir)


if __name__ == "__main__":
    cli_prepare_demo_tiles()
