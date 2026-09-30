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
        - 256-band IIRS cube reduction via 1st Principal Component (PCA) or selected VNIR band (Band 12 ~1000nm)
        - Solved geometric transform broadcast across all cube channels simultaneously
        │
        ▼
[Stage 5] IIRS Hyperspectral Mineralogy & Absorption Analysis (`lunar_matcher/spectral/`)
        - Diagnostic absorption band parameters (1 µm, 1.25 µm, 2 µm, 3 µm)
        - Continuum removal & polynomial sub-band center derivation
        - Rule-based mineral classification (pyroxenes, olivine, anorthosite, spinel, mature mare)
        - Band Area Ratio (BAR) calculation and thermal emission correction gating
        │
        ▼
[Stage 6] Pluggable Feature Benchmark (`lunar_matcher/matching/benchmark.py`)
        - Classical detectors: SIFT, AKAZE
        - Cross-modal stub: RIFT2 (radiation-invariant feature transform)
        - Deep learning: SuperPoint + LightGlue pipeline
        - Quantitative comparative metrics: match count, mean confidence, latency
        │
        ▼
[Stage 7] Geometric Transform & Outlier Rejection (`lunar_matcher/fitting/ransac.py`)
        - RANSAC Homography / Affine partial fitting with strict inlier ratio gates (> 15%)
        │
        ▼
[Stage 8] Accuracy & Cross-Validation (`lunar_matcher/validate/rmse.py`)
        - Manual control-point (tie point) RMSE calculation
        - Third-party independent LROC reference cross-validation
        │
        ▼
[Stage 9] Live Interactive Dashboard & Mineralogy Viewer (`lunar_matcher/dashboard/` & Web UI)
        - Interactive sensor pair comparison, live pipeline execution, overlay opacity blend,
          spectral reflectance & continuum-removed curves, mineral classification map, and validation scorecards.

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

### API Endpoints
- `GET /api/health`: Health status.
- `GET /api/mineralogy?region=<region>`: Run diagnostic mineralogy analysis on IIRS cube without requiring image registration.
- `GET /api/run-pipeline?region=<region>&sensorA=<sensor>&sensorB=<sensor>`: Full registration pipeline, returning tie-points, benchmark, scale bridging, validation report, and if IIRS is selected, the mineralogical classification map and spectral curves.

### Real vs. Simulated Spectral Data
- **GeoTIFF Source Tiles**: The pre-cached files in `data/tiles/` are single-band 8-bit GeoTIFFs (512×512) representing calibrated surface albedo.
- **Simulated IIRS Cubes**: In demo mode when no real 256-band Level-2 ENVI or multi-band GeoTIFF is present in `data/raw/`, `lunar_matcher.spectral.simulate.build_simulated_cube` synthesizes a 256-channel spectral cube (800–5100 nm, 16.85 nm spacing) by modulating endmember crystal-field absorption profiles (low/high-Ca pyroxenes, olivine, plagioclase, spinel, ilmenite) with the spatial albedo.
- **Provenance Transparency**: Every returned mineralogy payload explicitly specifies `provenance.source = "simulated"` and notes the synthetic origin.
- **Thermal Emission Gating**: The 3 µm hydration band index is strictly gated. Since daytime lunar thermal emission overwhelms solar reflectance beyond 2.5 µm, hydration analysis returns `unavailable` unless explicit Level-2 thermal emission modeling (`provenance.thermal_corrected = True`) is confirmed.
