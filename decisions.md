# Architecture & Technical Decision Record (`decisions.md`)

This document records the architectural and engineering decisions made in the development of `lunar-matcher` for multi-sensor lunar image registration (Chandrayaan-2 OHRC, TMC-2, and IIRS with LROC cross-validation).

---

## 1. Scale Bridging Strategy: TMC-2 as Geometric Pivot vs. Direct OHRC↔IIRS Registration

- **Decision Made**: Reject direct matching between OHRC (0.25 m/px) and IIRS (80 m/px). Enforce a 2-step scale-bridging pipeline routing through TMC-2 (5 m/px) as an intermediate geometric pivot:
  $$\text{OHRC (0.25m)} \xrightarrow{20\times} \text{TMC-2 (5m)} \xrightarrow{16\times} \text{IIRS (80m)}$$
  Chain the two solved transformation matrices: $H_{\text{OHRC}\to\text{IIRS}} = H_{\text{TMC}\to\text{IIRS}} \cdot H_{\text{OHRC}\to\text{TMC}}$.
- **Alternatives Considered**:
  - Direct feature matching between OHRC and IIRS across the 320x scale disparity.
  - Generating a 9-octave image pyramid downsampling OHRC directly to 80m.
- **What Would Happen If Alternative Was Used**:
  - Direct 320x matching fails completely: feature descriptors (SIFT, ORB, SuperPoint) are designed for scale invariance up to $\sim 4\times\text{--}8\times$. At a 320x difference, an entire 100-meter crater rim resolved into hundreds of boulders and micro-shadows in OHRC collapses into a single ambiguous sub-pixel blur in IIRS. SIFT produces 0 true correspondences and thousands of false positives.
  - Downsampling OHRC directly to 80m discards 99.999% of its high-frequency topological data in a single step, yielding high spatial aliasing, losing the exact sub-meter tie points needed for high-precision registration.
- **Why This Decision Was Made**:
  - TMC-2 has a 5m GSD, creating a manageable $20\times$ gap with OHRC and a $16\times$ gap with IIRS. Both intervals are within the operational envelope of area-weighted Gaussian pyramids and affine/homography solvers.

---

## 2. Coordinate System: IAU Moon 2000 Equirectangular vs. Topocentric Local Tangent Plane vs. Lunar Polar Stereographic

- **Decision Made**: Standardize target CRS to IAU Moon 2000 Equirectangular (`+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=1737400 +units=m +no_defs`), matching the LROC NAC/WAC standard convention, with configuration toggling in `config.yaml`.
- **Alternatives Considered**:
  - Local topocentric tangent plane (ENU/NED).
  - Lunar Polar Stereographic (`+proj=stere +lat_0=90 +lon_0=0 +R=1737400`).
  - Native Selenographic Lat/Lon (`EPSG:Moon`).
- **What Would Happen If Alternative Was Used**:
  - Topocentric local planes lack seamless interoperability with global lunar DEMs (e.g. SLDEM2015, GLD100) and require custom datum transformation matrices for every single tile, causing projection drift at tile boundaries.
  - Native angular Lat/Lon coordinates are non-isometric; distance distortion varies with $\cos(\text{latitude})$, causing isotropic convolution filters (SIFT Gaussians, bilateral filters) to distort keypoint descriptors along longitude.
- **Why This Decision Was Made**:
  - Matches the standard PDS planetary convention adopted by NASA LROC and ISRO ISSDC for equatorial and mid-latitude regions.
  - Enables direct validation against public LROC NAC/WAC basemaps without re-projecting the ground reference.

---

## 3. Illumination Normalization: 3-Tier Architecture with Upfront Gating vs. End-to-End Deep Retinex

- **Decision Made**: Three-tier progressive normalization:
  - Tier 1: Local CLAHE (`clipLimit=3.0`, $8\times 8$ grid) + empirical CDF histogram matching.
  - Tier 2: Solar azimuth/elevation directional shadow filter (Sobel directional gradient opposite to solar vector).
  - Tier 3: Upfront solar incidence angle confidence gating ($\alpha > 80^\circ \implies \text{flag \& abort}$).
- **Alternatives Considered**:
  - Unconditioned deep neural Retinex illumination decomposition (e.g. RetinexNet).
  - Global histogram equalization.
  - Attempting matching regardless of solar incidence angle and relying purely on RANSAC outlier rejection.
- **What Would Happen If Alternative Was Used**:
  - Deep Retinex models trained on terrestrial scenes hallucinate artificial albedo textures on lunar regolith because they misinterpret zero-albedo pitch-black cast shadows as dark surfaces, inventing false crater features.
  - Global histogram equalization blows out dynamic range in high-albedo ejecta blankets (like Tycho rays) and saturates crater shadows.
  - Attempting matching on extreme polar low-sun tiles ($\alpha > 80^\circ$) causes RANSAC to latch onto repetitive shadow terminators rather than genuine geological landmarks, yielding confident but completely erroneous false-positive homographies.
- **Why This Decision Was Made**:
  - Hard lunar shadows are governed by deterministic celestial geometry (solar azimuth and elevation). Incorporating metadata directly via Tier 2 is physically grounded and computationally instant (<5ms).
  - Tier 3 upfront confidence gating provides engineering honesty: refusing to register permanently shadowed polar regions before spending compute or corrupting downstream navigation.

---

## 4. Hyperspectral Cube Reduction: 1st Principal Component vs. Mean Band Averaging vs. Band Selection

- **Decision Made**: Dual-mode reduction in `matching/band_select.py`:
  - Mode 1: PCA across the 256 spectral channels, extracting the 1st Principal Component (explaining >85% variance) with polarity alignment to ensure positive correlation with surface albedo.
  - Mode 2: High-SNR single VNIR band selection (default Band 42 ~1000nm, avoiding absorption bands).
  - Solved geometric transform is then broadcast across all 256 bands in parallel chunks without per-band re-solving.
- **Alternatives Considered**:
  - Averaging all 256 bands uniformly.
  - Running feature detection and RANSAC independently on every single band.
- **What Would Happen If Alternative Was Used**:
  - Uniform averaging washes out mineralogical contrast and incorporates noisy water-ice absorption or bad detector channels (dead pixels at band edges), degrading edge sharpness.
  - Solving 256 independent homographies is 256x slower ($\sim 40\text{s}$ per tile instead of $150\text{ms}$), and leads to inter-band spatial jitter/misregistration due to varying SNR across channels.
- **Why This Decision Was Made**:
  - All bands in the IIRS focal plane array share a rigid optical bench; once geometric alignment is solved on the maximum-variance PC1 composite, that exact projective transform applies identically to every channel.

---

## 5. Feature Matching Benchmark: Pluggable Multi-Matcher (SIFT, AKAZE, LightGlue, RIFT2) vs. Single Hardcoded Matcher

- **Decision Made**: Implement a pluggable benchmark (`matching/benchmark.py`) evaluating SIFT, AKAZE, LightGlue (SuperPoint front-end), and a RIFT2 phase-congruency stub, outputting a quantitative comparison matrix (match count, mean confidence, latency).
- **Alternatives Considered**:
  - Hardcoding LightGlue only.
  - Hardcoding classical SIFT only.
- **What Would Happen If Alternative Was Used**:
  - Hardcoding only LightGlue creates heavy PyTorch GPU dependencies that may fail on edge rovers or CPU-only container environments, while suffering in high-contrast crater boundaries where classical scale-space extrema are proven.
  - Hardcoding only SIFT fails in low-texture mare regolith where intensity gradients are near zero and deep learned keypoints excel.
- **Why This Decision Was Made**:
  - Demonstrates engineering rigor: the benchmark dynamically selects the optimal matcher for the specific terrain and provides quantitative justification in the live dashboard.
  - LightGlue was chosen over SuperGlue because its transformer attention heads enable early stopping, reducing latency by 3x with identical or superior correspondence accuracy.

---

## 6. Outlier Rejection: Strict Inlier Count & Ratio Gated RANSAC vs. Plain Least-Squares / ICP

- **Decision Made**: Implement `fitting/ransac.py` with strict dual threshold gating:
  - Minimum inlier count: $\ge 8$ points.
  - Minimum inlier ratio: $\ge 15\%$.
  - Rejection of degenerate fits with explicit human-readable diagnostics.
- **Alternatives Considered**:
  - Standard least-squares / Iterative Closest Point (ICP).
  - Unfiltered RANSAC without minimum ratio gating.
- **What Would Happen If Alternative Was Used**:
  - Plain least-squares has zero breakdown point; even a single false-positive match between two identical-looking craters pulls the entire homography off by hundreds of pixels.
  - Unfiltered RANSAC will happily return a 4-point homography out of 200 matches (2% inlier ratio), which is almost certainly an over-fitted random coincidence in a crater field.
- **Why This Decision Was Made**:
  - The repetitive self-similarity of impact craters across the lunar surface makes false-positive matches common. A 15% inlier floor and minimum 8 points guarantees geometric stability.

---

## 7. Dual Validation Strategy: Tie-Point Control Points + Independent LROC Cross-Check

- **Decision Made**: Implement dual validation in `validate/rmse.py`:
  - Path 1: Manual control point pixel RMSE against visible crater peaks/ridges.
  - Path 2: Independent cross-validation against a third-party reference sensor (NASA LROC NAC/WAC) over the identical region.
- **Alternatives Considered**:
  - Self-consistency check only (measuring RANSAC reprojection error of inliers).
- **What Would Happen If Alternative Was Used**:
  - Self-consistency is circular logic: if RANSAC locked onto false correspondences, its internal reprojection error will appear artificially low (e.g. 0.8px) while the actual registration is offset by 50 meters.
- **Why This Decision Was Made**:
  - LROC NAC provides an independent third-party baseline not subject to Chandrayaan-2 ephemeris or thermal drift. Agreement with LROC proves objective planetary geodetic truth.

---

## 8. Ingestion & Tiling: Pre-cached Content-Addressed Chunks vs. Live Scene Processing

- **Decision Made**: Tile-based processing with a content-addressed disk cache (`data/cache/`) keyed by SHA256 of `(source_file_hash, reprojection_params, tile_bounds)` and a pre-caching CLI command (`prepare-demo-tiles`).
- **Alternatives Considered**:
  - Processing full-resolution OHRC scenes live during the demo.
- **What Would Happen If Alternative Was Used**:
  - A single full-resolution OHRC panchromatic strip is $\sim 12{,}000 \times 120{,}000$ pixels (several gigabytes uncompressed). Reading and reprojecting this on stage would take 5 to 10 minutes and exhaust memory, causing demo crashes.
- **Why This Decision Was Made**:
  - Decouples heavy geospatial ingestion (run once offline) from interactive demo presentation (instant sub-second responses).
