"""
Lunar Matcher backend bridge.

Runs the real lunar_matcher pipeline (illumination normalization, feature
benchmark, RANSAC fitting, RMSE / LROC validation) against the pre-cached
demo tiles in data/tiles/, and serves the results as JSON so the React
frontend can display genuine numbers instead of hardcoded ones.

Run with:  python server/app.py
Listens on http://localhost:5001
"""
import os
import sys
import base64
import traceback

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Make the repo root importable so `import lunar_matcher` works regardless
# of the working directory this script is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lunar_matcher.ingest.metadata import parse_issdc_metadata, MalformedMetadataError
from lunar_matcher.matching.illumination import normalize_and_score
from lunar_matcher.matching.benchmark import run_benchmark
from lunar_matcher.fitting.ransac import fit_transform
from lunar_matcher.validate.rmse import (
    compute_control_point_rmse,
    lroc_cross_validation,
    generate_validation_report,
)
from lunar_matcher.georef.pyramid import pair_planner

# In production (Render, etc.) Flask also serves the built React app from
# dist/, so the whole thing is one deployable web service. Locally, Vite
# serves the frontend on :3000 and proxies /api to this Flask process
# instead, so dist/ won't exist yet -- that's fine, these routes just won't
# be hit in dev.
DIST_DIR = os.path.join(ROOT, "dist")
app = Flask(__name__, static_folder=DIST_DIR, static_url_path="")
CORS(app)

TILE_DIR = os.path.join(ROOT, "data", "tiles")
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

INCIDENCE_CUTOFF = CONFIG["illumination"]["incidence_angle_cutoff_deg"]
RANSAC_MODEL = CONFIG["fitting"]["default_model"]
RANSAC_THRESH = CONFIG["fitting"]["ransac_reproj_threshold_px"]
MIN_INLIER_COUNT = CONFIG["fitting"]["min_inlier_count"]
MIN_INLIER_RATIO = CONFIG["fitting"]["min_inlier_ratio"]
MAX_RMSE = CONFIG["validation"]["max_acceptable_rmse_px"]
LROC_THRESH = CONFIG["validation"]["lroc_cross_val_threshold_px"]


# Maps the lowercase region id the frontend sends (e.g. "tycho") to the
# tile filename prefix used in data/tiles/ (e.g. "Tycho_OHRC.tif"). Add an
# entry here whenever a new region is generated via prepare-demo-tiles.
REGION_TILE_PREFIX = {
    "apollo11": "Apollo11",
    "tycho": "Tycho",
    "sinusiridum": "SinusIridum",
}


def tile_base_name(sensor: str, region: str) -> str:
    """Resolve which pre-cached demo tile to load for a given sensor + region."""
    if region == "southpole" and sensor == "OHRC":
        # South Pole PSR gating demo reuses the Apollo11 scene with a
        # dedicated extreme-shadow OHRC tile -- it's a lighting-gate demo,
        # not a distinct piece of terrain.
        return "Apollo11_OHRC_POLAR_SHADOW"
    prefix = REGION_TILE_PREFIX.get(region, "Apollo11")
    return f"{prefix}_{sensor}"


def lroc_reference_region(region: str) -> str:
    """Which region's LROC_NAC tile to use as the independent cross-
    validation reference. South Pole has no tile of its own (it's a
    metadata-only gating demo on the Apollo11 scene), so it borrows
    Apollo11's -- everything else validates against its own region."""
    return "apollo11" if region == "southpole" else region


def load_tile(sensor: str, region: str):
    """Loads a tile's pixel array + parsed ISSDC metadata. Applies the
    South-Pole gating-demo angle override for sensors that have no dedicated
    polar-shadow tile of their own (TMC2 / IIRS)."""
    base = tile_base_name(sensor, region)
    img_path = os.path.join(TILE_DIR, f"{base}.tif")
    lbl_path = os.path.join(TILE_DIR, f"{base}.lbl")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read tile image: {img_path}")

    meta = parse_issdc_metadata(lbl_path)

    if region == "southpole" and sensor != "OHRC":
        # No dedicated polar-shadow tile exists for TMC2/IIRS in this demo
        # dataset yet -- reuse the sensor's normal pixels but apply the real
        # grazing-sun metadata so Tier-3 gating is evaluated honestly.
        meta = dict(meta)
        meta["solar_incidence_angle_deg"] = 84.5
        meta["sun_azimuth_deg"] = 270.0
        meta["sun_elevation_deg"] = 5.5

    return img, meta


def encode_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def run_leg(img_a, img_b, methods=None):
    """Runs the full benchmark + RANSAC fit for one sensor pair leg.
    Returns (benchmark_records, best_method_name, fit_result, raw_results)."""
    df, raw = run_benchmark(img_a, img_b, methods=methods)
    if df.empty:
        return [], None, None, raw

    records = df.to_dict(orient="records")
    best_name = records[0]["Method"]
    best_raw = raw[best_name]

    fit_res = fit_transform(
        best_raw["pts_a"],
        best_raw["pts_b"],
        model=RANSAC_MODEL,
        ransac_reproj_threshold=RANSAC_THRESH,
        min_inlier_count=MIN_INLIER_COUNT,
        min_inlier_ratio=MIN_INLIER_RATIO,
    )
    return records, best_name, fit_res, raw


def leg_inlier_rmse(fit_res, raw_best):
    """Self-consistency RMSE on the RANSAC inlier tie points for one leg."""
    if fit_res is None or not fit_res["success"]:
        return None
    mask = fit_res["inlier_mask"]
    pts_a_in = raw_best["pts_a"][mask]
    pts_b_in = raw_best["pts_b"][mask]
    return compute_control_point_rmse(pts_a_in, pts_b_in, fit_res["transform"])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/run-pipeline")
def run_pipeline():
    region = request.args.get("region", "apollo11")
    sensor_a = request.args.get("sensorA", "OHRC")
    sensor_b = request.args.get("sensorB", "TMC2")

    try:
        img_a_raw, meta_a = load_tile(sensor_a, region)
        img_b_raw, meta_b = load_tile(sensor_b, region)
    except (FileNotFoundError, MalformedMetadataError) as e:
        return jsonify({"error": str(e)}), 404

    # --- Illumination normalization + Tier-3 gating (real, per-tile) ---
    norm_a, conf_a, flag_a = normalize_and_score(
        img_a_raw, sun_angle_metadata=meta_a, incidence_cutoff_deg=INCIDENCE_CUTOFF
    )
    norm_b, conf_b, flag_b = normalize_and_score(
        img_b_raw, sun_angle_metadata=meta_b, incidence_cutoff_deg=INCIDENCE_CUTOFF
    )

    response = {
        "region": region,
        "sensorA": sensor_a,
        "sensorB": sensor_b,
        "tileImageA": encode_png(img_a_raw),
        "tileImageB": encode_png(img_b_raw),
        "incidenceA": meta_a.get("solar_incidence_angle_deg"),
        "incidenceB": meta_b.get("solar_incidence_angle_deg"),
        "illuminationConfidenceA": conf_a,
        "illuminationConfidenceB": conf_b,
    }

    gate_flag = flag_a or flag_b
    if gate_flag:
        response["gated"] = True
        response["gateReason"] = gate_flag
        return jsonify(response)

    response["gated"] = False

    # --- Scale-bridging strategy (real decision logic from georef/pyramid.py) ---
    plan = pair_planner(sensor_a, sensor_b, max_direct_ratio=CONFIG["scale_bridging"]["max_direct_scale_ratio"])
    response["scaleBridging"] = plan

    if plan["strategy"] == "pivot":
        # Real 2-step chain through TMC2, as decisions.md #1 specifies.
        pivot_img, pivot_meta = load_tile("TMC2", region)
        norm_pivot, conf_pivot, flag_pivot = normalize_and_score(
            pivot_img, sun_angle_metadata=pivot_meta, incidence_cutoff_deg=INCIDENCE_CUTOFF
        )
        if flag_pivot:
            response["gated"] = True
            response["gateReason"] = flag_pivot
            return jsonify(response)

        bench1, best1, fit1, raw1 = run_leg(norm_a, norm_pivot)
        bench2, best2, fit2, raw2 = run_leg(norm_pivot, norm_b)

        # A quick *direct* SIFT-only attempt between A and B, purely to show
        # why the pivot is necessary (decisions.md #1's rationale, made concrete).
        direct_df, direct_raw = run_benchmark(norm_a, norm_b, methods=["SIFT"])
        direct_matches = int(direct_df.iloc[0]["Match Count"]) if not direct_df.empty else 0

        chain_ok = fit1 is not None and fit1["success"] and fit2 is not None and fit2["success"]
        response["pivotChain"] = {
            "pivotSensor": "TMC2",
            "leg1": {"sensorPair": [sensor_a, "TMC2"], "benchmark": bench1, "bestMethod": best1, "fit": _fit_public(fit1)},
            "leg2": {"sensorPair": ["TMC2", sensor_b], "benchmark": bench2, "bestMethod": best2, "fit": _fit_public(fit2)},
            "directAttemptMatchCount": direct_matches,
            "chainSucceeded": chain_ok,
        }

        if not chain_ok:
            response["registrationSucceeded"] = False
            return jsonify(response)

        composed_h = fit2["transform"] @ fit1["transform"]
        response["composedTransform"] = composed_h.tolist()
        response["benchmark"] = bench2
        response["bestMethod"] = best2
        response["fit"] = _fit_public(fit2)

        rmse1 = leg_inlier_rmse(fit1, raw1[best1])
        rmse2 = leg_inlier_rmse(fit2, raw2[best2])
        response["chainLegInlierRmse"] = {"leg1": rmse1, "leg2": rmse2}

        # Independent LROC cross-validation, run against the reference tile.
        lroc_img, lroc_meta = load_tile("LROC_NAC", lroc_reference_region(region))
        norm_lroc, _, _ = normalize_and_score(lroc_img, sun_angle_metadata=lroc_meta)
        lroc_res = lroc_cross_validation(composed_h, norm_a, norm_lroc, threshold_px=LROC_THRESH)
        response["lrocValidation"] = lroc_res

        cp_stub = rmse2 or {"rmse_px": 999.0, "point_count": 0, "max_residual_px": 0.0}
        response["validationReport"] = generate_validation_report(
            cp_stub, lroc_res, max_acceptable_rmse=MAX_RMSE, lroc_threshold=LROC_THRESH
        )
        response["registrationSucceeded"] = True
        return jsonify(response)

    # --- Direct match path ---
    bench, best, fit_res, raw = run_leg(norm_a, norm_b)
    response["benchmark"] = bench
    response["bestMethod"] = best
    response["fit"] = _fit_public(fit_res)

    if fit_res is None or not fit_res["success"]:
        response["registrationSucceeded"] = False
        return jsonify(response)

    response["registrationSucceeded"] = True

    cp_res = leg_inlier_rmse(fit_res, raw[best])
    response["controlPointRmse"] = cp_res

    lroc_img, lroc_meta = load_tile("LROC_NAC", lroc_reference_region(region))
    norm_lroc, _, _ = normalize_and_score(lroc_img, sun_angle_metadata=lroc_meta)
    lroc_res = lroc_cross_validation(fit_res["transform"], norm_a, norm_lroc, threshold_px=LROC_THRESH)
    response["lrocValidation"] = lroc_res

    response["validationReport"] = generate_validation_report(
        cp_res, lroc_res, max_acceptable_rmse=MAX_RMSE, lroc_threshold=LROC_THRESH
    )
    return jsonify(response)


def _fit_public(fit_res):
    """Strips numpy arrays out of a fit_transform() result so it's JSON-safe."""
    if fit_res is None:
        return None
    return {
        "success": fit_res["success"],
        "model": fit_res["model"],
        "inlierCount": fit_res["inlier_count"],
        "inlierRatio": fit_res["inlier_ratio"],
        "totalMatches": fit_res["total_matches"],
        "status": fit_res["status"],
        "errorMessage": fit_res["error_message"],
    }


@app.errorhandler(Exception)
def handle_error(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# --- Serve the built React app (production only) ---
# Any path that isn't /api/... and matches a real file in dist/ (JS, CSS,
# images) is served as a static asset. Everything else falls through to
# index.html so React Router-style client-side paths still work.
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
