# Lunar Matcher: Multi-Sensor Lunar Image Registration Pipeline

A robust, multi-sensor registration, scale-bridging, and cross-validation pipeline designed for Chandrayaan-2 planetary data products (ISSDC / PDS):
- **OHRC**: Orbiter High Resolution Camera (0.25 m/pixel panchromatic)
- **TMC-2**: Terrain Mapping Camera 2 (5 m/pixel stereo panchromatic)
- **IIRS**: Imaging Infra-Red Spectrometer (80 m/pixel hyperspectral, ~256 bands)
- **LROC Reference**: Lunar Reconnaissance Orbiter Camera (NAC/WAC) independent ground reference

---

## 1. Pipeline Stages

```
Raw ISSDC Products (OHRC / TMC-2 / IIRS) + Metadata (.lbl/.xml)
        │
        ▼
[Stage 1] Ingestion & Tiling (`lunar_matcher/ingest/`)
        - PDS / ISSDC sidecar metadata extraction (selenographic corners, sun elevation/azimuth)
        - Tile-based windowing with coordinate bounding preservation
        - SHA256 content-addressed tile caching (`data/cache/`)
        │
        ▼
[Stage 2] Coordinate Reprojection & Scale Bridging (`lunar_matcher/georef/`)
        - Reproject to common IAU Moon 2000 Equirectangular CRS (LROC standard) via rasterio.warp
        - Export GeoTIFF + JSON provenance sidecar
        - Area-weighted pyramid downsampling (`build_pyramid`)
        - Automated Scale-Bridging Planner: OHRC (0.25m) ──[20x]──> TMC-2 (5m) ──[16x]──> IIRS (80m)
          (Direct 320x OHRC↔IIRS registration is rejected; TMC-2 serves as geometric pivot)
        │
        ▼
[Stage 3] Illumination Normalization & Gating (`lunar_matcher/matching/illumination.py`)
        - Tier 1: Local adaptive CLAHE + inter-sensor histogram matching
        - Tier 2: Solar azimuth/elevation shadow-direction compensation filter
        - Tier 3: Incidence angle confidence gating (polar / shadowed regions > 80° flagged prior to matching)
        │
        ▼
[Stage 4] Modality & Band Reduction (`lunar_matcher/matching/band_select.py`)
        - 256-band IIRS cube reduction via 1st Principal Component (PCA) or selected VNIR band
        - Solved geometric transform broadcast across all cube channels simultaneously
        │
        ▼
[Stage 5] Pluggable Feature Benchmark (`lunar_matcher/matching/benchmark.py`)
        - Classical detectors: SIFT, AKAZE
        - Cross-modal stub: RIFT2 (radiation-invariant feature transform)
        - Deep learning: SuperPoint + LightGlue pipeline
        - Quantitative comparative metrics: match count, mean confidence, latency
        │
        ▼
[Stage 6] Geometric Transform & Outlier Rejection (`lunar_matcher/fitting/ransac.py`)
        - RANSAC Homography / Affine partial fitting with strict inlier ratio gates (> 15%)
        │
        ▼
[Stage 7] Accuracy & Cross-Validation (`lunar_matcher/validate/rmse.py`)
        - Manual control-point (tie point) RMSE calculation
        - Third-party independent LROC reference cross-validation
        │
        ▼
[Stage 8] Live Interactive Dashboard (`lunar_matcher/dashboard/` & Web UI)
        - Interactive sensor pair comparison, live pipeline execution, overlay opacity blend,
          confidence score indicators, and RMSE validation scorecards.

---

## 2. Directory Structure

```
lunar_matcher/
  ├── config.yaml          # Pipeline hyperparameters & CRS configs
  ├── pyproject.toml       # Python package metadata & dependencies
  ├── README.md            # Architecture & usage guide
  ├── decisions.md         # Technical decisions & framework trade-offs
  ├── lunar_matcher/
  │   ├── ingest/          # PDS/ISSDC metadata parsing, tiling, caching
  │   │   ├── metadata.py
  │   │   └── tiling.py
  │   ├── georef/          # Coordinate reprojection & scale bridging
  │   │   ├── reproject.py
  │   │   └── pyramid.py
  │   ├── matching/        # Illumination, band selection & matching benchmark
  │   │   ├── illumination.py
  │   │   ├── band_select.py
  │   │   └── benchmark.py
  │   ├── fitting/         # RANSAC homography/affine estimation
  │   │   └── ransac.py
  │   ├── validate/        # Manual RMSE & LROC cross-validation
  │   │   └── rmse.py
  │   └── dashboard/       # Streamlit demo dashboard
  │       └── app.py
  ├── data/
  │   ├── raw/             # Stored raw ISSDC PDS raster products
  │   ├── cache/           # Disk cache for reprojected/tiled rasters
  │   └── tiles/           # Pre-cut demo tiles
  └── tests/               # Unit & synthetic verification tests
      ├── test_reproject.py
      ├── test_band_select.py
      └── test_pipeline.py
```

---

## 3. Quickstart

### Installation
```bash
pip install -e .
# Or install with deep matcher extras:
pip install -e ".[deep]"
```

### Pre-caching Demo Tiles
```bash
prepare-demo-tiles --region "Apollo11"
```

### Running Streamlit Dashboard
```bash
streamlit run lunar_matcher/dashboard/app.py
```

### Running Test Suite
```bash
pytest tests/
```

---

## 4. Live Web Demo (React ↔ Python bridge)

The React UI in `src/App.tsx` previously rendered hardcoded numbers and static SVGs — it never called the
Python pipeline. It now talks to a small Flask bridge (`server/app.py`) that runs the *real*
`lunar_matcher` pipeline (illumination normalization, SIFT/AKAZE/RIFT2/LightGlue benchmark, RANSAC fitting,
control-point + LROC cross-validation) against the pre-cached tiles in `data/tiles/`, and returns the
genuine results as JSON.

### First-time setup
```bash
pip install -e .                       # installs the lunar_matcher package
pip install -r server/requirements.txt # installs Flask + flask-cors
npm install                            # installs concurrently (used to run both servers)
```

### Run it
```bash
npm run dev
```
This starts the Vite dev server (port 3000) and the Flask backend (port 5001) together, with `/api/*`
requests proxied from Vite to Flask (see `vite.config.ts`). Open http://localhost:3000.

To run them separately (e.g. for debugging the backend):
```bash
npm run dev:api   # Flask on :5001
npm run dev:web   # Vite on :3000
```

### What's real vs. what's still simplified
- **Real**: tile images, illumination confidence, Tier-3 gating decision + message, scale-bridging
  strategy (`pair_planner`), the full SIFT/AKAZE/RIFT2/LightGlue(-fallback) benchmark, RANSAC fit,
  inlier-tie-point RMSE, and independent LROC NAC cross-validation — all computed live by the Python
  package on each "Run Registration" click.
- **Simplified for the demo dataset**: the South Pole gating scenario reuses the normal Apollo-11 TMC-2/IIRS
  pixel tiles with an overridden solar-angle metadata (84.5°) when no dedicated polar-shadow tile exists for
  that sensor (only OHRC has one, `Apollo11_OHRC_POLAR_SHADOW.tif`) — the *gating computation itself* is
  genuine, just the source pixels for those two sensors are reused rather than distinct polar captures.
- The "Aligned Registration Overlay" panel blends the two real source tiles client-side for visual reference;
  it does not apply the solved homography pixel-by-pixel (that would need warping in the browser or an
  extra backend endpoint returning a warped PNG — flag if you want that added next).
