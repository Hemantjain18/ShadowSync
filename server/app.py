"""
Lunar Matcher backend bridge and spectral mineralogy server.
"""
import os
import sys
import base64
import json
import yaml
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    Flask = None

try:
    import cv2
except ImportError:
    cv2 = None

from lunar_matcher.spectral.cube_loader import load_iirs_cube
from lunar_matcher.spectral.simulate import build_simulated_cube
from lunar_matcher.spectral.analysis import run_mineralogy_pipeline
from lunar_matcher.spectral.library import MINERAL_LIBRARY

CONFIG_PATH = os.path.join(ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

TILE_DIR = os.path.join(ROOT, "data", "tiles")
RAW_DIR = os.path.join(ROOT, "data", "raw")

COLOR_PALETTE = {
    0: (59, 130, 246),   # Low-Ca Pyroxene (Blue, BGR: 246, 130, 59)
    1: (16, 185, 129),  # High-Ca Pyroxene (Green)
    2: (234, 179, 8),   # Olivine (Amber/Yellow)
    3: (168, 85, 247),  # Plagioclase (Purple)
    4: (236, 72, 153),  # Spinel (Pink)
    5: (100, 116, 139), # Featureless/Dark (Slate)
    6: (71, 85, 105),   # Unclassified
}

LEGEND_HEX = {
    "Low-Ca Pyroxene (Orthopyroxene)": "#3b82f6",
    "High-Ca Pyroxene (Clinopyroxene)": "#10b981",
    "Olivine-rich (Dunite/Troctolite)": "#eab308",
    "Plagioclase-rich (Anorthosite)": "#a855f7",
    "Spinel-like (Mg-Al Spinel)": "#ec4899",
    "Featureless / Dark (Ilmenite-rich)": "#64748b",
    "Unclassified Mixture": "#475569",
}


def load_iirs_spectral(region: str):
    """Try data/raw for real cube, fall back to simulated cube from tile."""
    raw_candidates = [
        os.path.join(RAW_DIR, f"{region}_IIRS.npz"),
        os.path.join(RAW_DIR, f"{region}_IIRS.hdr"),
        os.path.join(RAW_DIR, f"{region}_IIRS.tif"),
    ]
    for c in raw_candidates:
        if os.path.exists(c):
            try:
                return load_iirs_cube(c)
            except Exception:
                pass

    if CONFIG.get("spectral", {}).get("simulated_fallback", True):
        prefix = "Apollo11"
        if region == "tycho":
            prefix = "Tycho"
        elif region == "sinusiridum":
            prefix = "SinusIridum"
        tile_path = os.path.join(TILE_DIR, f"{prefix}_IIRS.tif")
        if not os.path.exists(tile_path):
            tile_path = os.path.join(TILE_DIR, "Apollo11_IIRS.tif")
        return build_simulated_cube(tile_path=tile_path, seed=42)

    raise FileNotFoundError(f"No IIRS spectral cube available for region {region}")


def generate_class_map_rgb(class_map: np.ndarray) -> np.ndarray:
    """Generate RGB image (H, W, 3) from uint8 class map."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c_id, (r, g, b) in COLOR_PALETTE.items():
        mask = class_map == c_id
        rgb[mask] = [r, g, b]
    return rgb


def encode_png_b64(rgb_arr: np.ndarray) -> str:
    """Encode RGB or Grayscale array to PNG base64 data URL."""
    if cv2 is None:
        return ""
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR) if rgb_arr.ndim == 3 else rgb_arr
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def sanitize_for_json(obj):
    """Recursively convert NumPy objects and NaNs to standard JSON-safe types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def create_app():
    if Flask is None:
        return None
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/mineralogy")
    def mineralogy_endpoint():
        region = request.args.get("region", "apollo11")
        try:
            cube_data = load_iirs_spectral(region)
            illum_conf = 0.25 if region == "southpole" else 1.0
            report = run_mineralogy_pipeline(cube_data, illumination_confidence=illum_conf)

            from lunar_matcher.spectral.analysis import classify_pixels, preprocess_cube
            clean_cube, clean_w = preprocess_cube(cube_data["cube"], cube_data["wavelengths_nm"])
            class_map, _, _ = classify_pixels(clean_cube, clean_w)
            class_rgb = generate_class_map_rgb(class_map)
            class_map_png = encode_png_b64(class_rgb)

            resp = {
                "region": region,
                "mineralogy": report,
                "classMapImage": class_map_png,
                "legend": LEGEND_HEX,
            }
            return jsonify(sanitize_for_json(resp))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    flask_app = create_app()
    if flask_app:
        flask_app.run(host="0.0.0.0", port=5001, debug=True)
