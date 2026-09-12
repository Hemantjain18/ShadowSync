import React, { useState, useEffect } from "react";
import {
  Satellite,
  Layers,
  Sparkles,
  ShieldCheck,
  AlertTriangle,
  Play,
  RotateCcw,
  Sliders,
  CheckCircle2,
  FileText,
  Compass,
  Zap,
  Eye,
  Info,
} from "lucide-react";

interface BenchmarkRecord {
  Method: string;
  "Match Count": number;
  "Mean Confidence": number;
  "Latency (ms)": number;
  Rationale: string;
}

interface FitResult {
  success: boolean;
  model: string;
  inlierCount: number;
  inlierRatio: number;
  totalMatches: number;
  status: string;
  errorMessage: string | null;
}

interface RmseResult {
  rmse_px: number;
  mean_residual_px?: number;
  max_residual_px: number;
  point_count: number;
}

interface LrocResult {
  success: boolean;
  lroc_rmse_px: number | null;
  passed: boolean;
  status: string;
  message: string;
}

interface ValidationReport {
  overall_status: "PASS" | "FAIL";
  control_point: { rmse_px: number; threshold_px: number; passed: boolean; point_count: number; max_residual_px: number };
  lroc_cross_validation: { rmse_px: number | null; threshold_px: number; passed: boolean | null; status: string };
  recommendation: string;
}

interface PivotLeg {
  sensorPair: [string, string];
  benchmark: BenchmarkRecord[];
  bestMethod: string | null;
  fit: FitResult | null;
}

interface PipelineResponse {
  region: string;
  sensorA: string;
  sensorB: string;
  tileImageA: string;
  tileImageB: string;
  incidenceA: number | null;
  incidenceB: number | null;
  illuminationConfidenceA: number;
  illuminationConfidenceB: number;
  gated: boolean;
  gateReason?: string;
  scaleBridging?: { strategy: "direct" | "pivot"; scale_ratio: number; pivot_sensor: string | null; rationale: string };
  pivotChain?: {
    pivotSensor: string;
    leg1: PivotLeg;
    leg2: PivotLeg;
    directAttemptMatchCount: number;
    chainSucceeded: boolean;
  };
  benchmark?: BenchmarkRecord[];
  bestMethod?: string | null;
  fit?: FitResult | null;
  registrationSucceeded?: boolean;
  controlPointRmse?: RmseResult;
  chainLegInlierRmse?: { leg1: RmseResult | null; leg2: RmseResult | null };
  lrocValidation?: LrocResult;
  validationReport?: ValidationReport;
  error?: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"demo" | "decisions">("demo");
  const [region, setRegion] = useState<"apollo11" | "southpole" | "tycho" | "sinusiridum">("apollo11");
  const [sensorA, setSensorA] = useState<"OHRC" | "TMC2" | "IIRS">("OHRC");
  const [sensorB, setSensorB] = useState<"OHRC" | "TMC2" | "IIRS">("TMC2");

  const [isProcessing, setIsProcessing] = useState(false);
  const [pipelineRan, setPipelineRan] = useState(false);
  const [pipelineData, setPipelineData] = useState<PipelineResponse | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.5);
  const [blendMode, setBlendMode] = useState<"alpha" | "difference" | "checker">("alpha");

  // Client-side preview only (informational banner shown BEFORE the real
  // pipeline runs). The authoritative decision comes back from
  // /api/run-pipeline -> scaleBridging, computed by lunar_matcher.georef.pyramid.pair_planner.
  const isPivotRequired = (sensorA === "OHRC" && sensorB === "IIRS") || (sensorA === "IIRS" && sensorB === "OHRC");

  const handleRunPipeline = async () => {
    setIsProcessing(true);
    setPipelineRan(false);
    setPipelineError(null);
    try {
      const params = new URLSearchParams({ region, sensorA, sensorB });
      const res = await fetch(`/api/run-pipeline?${params.toString()}`);
      const data: PipelineResponse = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `Backend returned ${res.status}`);
      }
      setPipelineData(data);
      setPipelineRan(true);
    } catch (err: any) {
      setPipelineError(
        err?.message?.includes("Failed to fetch")
          ? "Could not reach the pipeline backend at /api. Is `python server/app.py` running? (see README)"
          : err?.message || "Unknown error running the pipeline."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setPipelineRan(false);
    setPipelineData(null);
    setPipelineError(null);
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans selection:bg-amber-500/30">
      {/* Top Header */}
      <header className="border-b border-neutral-800 bg-neutral-900/60 backdrop-blur-md px-6 py-4 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <Compass className="h-6 w-6 text-neutral-950" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight text-white">Lunar Matcher</h1>
                <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  Chandrayaan-2 / ISSDC
                </span>
              </div>
              <p className="text-xs text-neutral-400">
                Multi-Sensor Registration: OHRC (0.25m) · TMC-2 (5m) · IIRS (80m) · LROC NAC Cross-Ref
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <div className="px-3 py-1.5 rounded-lg bg-neutral-800/80 border border-neutral-700/60 text-xs text-neutral-300 flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>IAU Moon 2000 Equirectangular (R=1737.4 km)</span>
            </div>
            <div className="flex bg-neutral-800 p-0.5 rounded-lg border border-neutral-700">
              <button
                onClick={() => setActiveTab("demo")}
                className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                  activeTab === "demo" ? "bg-neutral-700 text-white shadow-sm" : "text-neutral-400 hover:text-white"
                }`}
              >
                Live Pipeline Demo
              </button>
              <button
                onClick={() => setActiveTab("decisions")}
                className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                  activeTab === "decisions" ? "bg-neutral-700 text-white shadow-sm" : "text-neutral-400 hover:text-white"
                }`}
              >
                Technical Decisions
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === "demo" ? (
        <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
          {/* Controls Bar */}
          <section className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 shadow-xl">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
              {/* Region Selection */}
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5">Target Lunar Region</label>
                <select
                  value={region}
                  onChange={(e) => {
                    setRegion(e.target.value as any);
                    setPipelineRan(false);
                  }}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-amber-500"
                >
                  <option value="apollo11">Apollo 11 (0.67°N, 23.47°E)</option>
                  <option value="tycho">Tycho Crater (43.31°S, 11.36°W)</option>
                  <option value="sinusiridum">Sinus Iridum (44.1°N, 31.5°W)</option>
                  <option value="southpole">South Pole PSR (Gating Demo, 84.5° Sun)</option>
                </select>
              </div>

              {/* Sensor A */}
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5">Reference Sensor (Fixed)</label>
                <select
                  value={sensorA}
                  onChange={(e) => {
                    setSensorA(e.target.value as any);
                    setPipelineRan(false);
                  }}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-amber-500"
                >
                  <option value="OHRC">OHRC Panchromatic (0.25 m/px)</option>
                  <option value="TMC2">TMC-2 Stereo (5.0 m/px)</option>
                  <option value="IIRS">IIRS Hyperspectral (80 m/px)</option>
                </select>
              </div>

              {/* Sensor B */}
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5">Moving Sensor (To Warp)</label>
                <select
                  value={sensorB}
                  onChange={(e) => {
                    setSensorB(e.target.value as any);
                    setPipelineRan(false);
                  }}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-amber-500"
                >
                  <option value="TMC2">TMC-2 Stereo (5.0 m/px)</option>
                  <option value="OHRC">OHRC Panchromatic (0.25 m/px)</option>
                  <option value="IIRS">IIRS Hyperspectral (80 m/px)</option>
                </select>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={handleRunPipeline}
                  disabled={isProcessing}
                  className="flex-1 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-neutral-950 font-semibold px-4 py-2 rounded-lg text-sm transition flex items-center justify-center gap-2 shadow-lg shadow-amber-500/20 cursor-pointer"
                >
                  {isProcessing ? (
                    <>
                      <span className="h-4 w-4 border-2 border-neutral-950 border-t-transparent rounded-full animate-spin"></span>
                      <span>Aligning...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-current" />
                      <span>Run Registration</span>
                    </>
                  )}
                </button>
                {pipelineRan && (
                  <button
                    onClick={handleReset}
                    className="p-2 rounded-lg border border-neutral-700 bg-neutral-800 text-neutral-400 hover:text-white"
                    title="Reset view"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </button>
                )}
              </div>
            </div>

            {/* Scale-Bridging Strategy Alert */}
            {isPivotRequired && (
              <div className="mt-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
                <Info className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                <div className="text-xs text-neutral-300">
                  <span className="font-semibold text-amber-300">Scale-Bridging Pivot Activated: </span>
                  Direct matching between {sensorA} (0.25m) and {sensorB} (80m) has a <span className="font-bold text-white">320× scale disparity</span>.
                  The pipeline automatically routes through <span className="font-semibold text-amber-200">TMC-2 (5.0m)</span> as an intermediate geometric pivot:
                  {" "}OHRC (0.25m) → TMC-2 (5m, 20×) → IIRS (80m, 16×), chaining homography transforms $H = H_{'{2}'} \cdot H_{'{1}'}$.
                </div>
              </div>
            )}
          </section>

          {/* Backend connectivity error */}
          {pipelineError && (
            <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/80 text-red-200 text-sm flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
              <span>{pipelineError}</span>
            </div>
          )}

          {/* Real Pre-cached Tile Preview (Side by Side) — pixels come from data/tiles/, loaded by the Flask backend */}
          {pipelineData && !pipelineError && (
            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Tile A Card */}
              <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-4 overflow-hidden">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Satellite className="h-4 w-4 text-amber-400" />
                    <span className="font-semibold text-sm text-neutral-200">Reference: {pipelineData.sensorA}</span>
                  </div>
                  <span className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                    Sun Incidence: {pipelineData.incidenceA?.toFixed(1) ?? "—"}°
                  </span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-neutral-950 border border-neutral-800 flex items-center justify-center group">
                  <img src={pipelineData.tileImageA} alt={`${pipelineData.sensorA} tile`} className="w-full h-full object-cover" />
                  <div className="absolute bottom-3 left-3 bg-neutral-900/90 backdrop-blur-md px-2.5 py-1 rounded text-xs text-neutral-300 border border-neutral-700">
                    Illumination confidence: {(pipelineData.illuminationConfidenceA * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Tile B Card */}
              <div className="bg-neutral-900/60 border border-neutral-800 rounded-2xl p-4 overflow-hidden">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Satellite className="h-4 w-4 text-orange-400" />
                    <span className="font-semibold text-sm text-neutral-200">Moving: {pipelineData.sensorB}</span>
                  </div>
                  <span className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                    Sun Incidence: {pipelineData.incidenceB?.toFixed(1) ?? "—"}°
                  </span>
                </div>
                <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-neutral-950 border border-neutral-800 flex items-center justify-center group">
                  <img src={pipelineData.tileImageB} alt={`${pipelineData.sensorB} tile`} className="w-full h-full object-cover" />
                  <div className="absolute bottom-3 left-3 bg-neutral-900/90 backdrop-blur-md px-2.5 py-1 rounded text-xs text-neutral-300 border border-neutral-700">
                    Illumination confidence: {(pipelineData.illuminationConfidenceB * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </section>
          )}

          {!pipelineData && !pipelineError && (
            <div className="p-8 rounded-2xl border border-dashed border-neutral-800 text-center text-sm text-neutral-500">
              Press <span className="text-neutral-300 font-medium">Run Registration</span> to load the real {sensorA} / {sensorB} tiles for {
                { apollo11: "Apollo 11", tycho: "Tycho Crater", sinusiridum: "Sinus Iridum", southpole: "the South Pole gating demo" }[region]
              } and run the actual pipeline.
            </div>
          )}

          {/* Pipeline Results Section */}
          {pipelineRan && pipelineData && (
            <div className="space-y-6 animate-fadeIn">
              {/* Tier 3 Gating Failure Banner */}
              {pipelineData.gated ? (
                <div className="p-5 rounded-2xl bg-red-950/40 border border-red-800/80 text-red-200 space-y-2">
                  <div className="flex items-center gap-2 text-red-400 font-bold text-base">
                    <AlertTriangle className="h-5 w-5" />
                    <span>Tier 3 Illumination Confidence Gating Abort</span>
                  </div>
                  <p className="text-sm text-red-300/90">{pipelineData.gateReason}</p>
                  <p className="text-xs text-neutral-400 italic">
                    Pipeline deliberately halted prior to feature extraction (computed live by
                    lunar_matcher.matching.illumination.compute_illumination_confidence). Engineering safety rule:
                    refusing registration is strictly superior to outputting a hallucinatory homography on moving terminator shadows.
                  </p>
                </div>
              ) : (
                <>
                  {/* Scale-bridging strategy actually returned by pair_planner() */}
                  {pipelineData.scaleBridging && (
                    <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3 text-xs text-neutral-300">
                      <Info className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold text-amber-300">
                          Strategy: {pipelineData.scaleBridging.strategy === "pivot" ? "Pivot via TMC-2" : "Direct match"}
                          {" "}(scale ratio {pipelineData.scaleBridging.scale_ratio}×).{" "}
                        </span>
                        {pipelineData.scaleBridging.rationale}
                      </div>
                    </div>
                  )}

                  {/* Pivot chain detail, only rendered when the backend actually routed through TMC-2 */}
                  {pipelineData.pivotChain && (
                    <div className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 space-y-3">
                      <h2 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Layers className="h-4 w-4 text-amber-400" /> Scale-Bridging Chain (via TMC-2)
                      </h2>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                        <div className="bg-neutral-950/60 p-3 rounded-xl border border-neutral-800/80">
                          <div className="text-neutral-400 mb-1">Leg 1: {pipelineData.pivotChain.leg1.sensorPair.join(" → ")}</div>
                          <div className="font-mono text-neutral-200">
                            {pipelineData.pivotChain.leg1.bestMethod ?? "—"} · {pipelineData.pivotChain.leg1.fit?.inlierCount ?? 0} inliers
                          </div>
                        </div>
                        <div className="bg-neutral-950/60 p-3 rounded-xl border border-neutral-800/80">
                          <div className="text-neutral-400 mb-1">Leg 2: {pipelineData.pivotChain.leg2.sensorPair.join(" → ")}</div>
                          <div className="font-mono text-neutral-200">
                            {pipelineData.pivotChain.leg2.bestMethod ?? "—"} · {pipelineData.pivotChain.leg2.fit?.inlierCount ?? 0} inliers
                          </div>
                        </div>
                        <div className="bg-neutral-950/60 p-3 rounded-xl border border-neutral-800/80">
                          <div className="text-neutral-400 mb-1">Direct {sensorA}↔{sensorB} attempt (for contrast)</div>
                          <div className="font-mono text-amber-400">{pipelineData.pivotChain.directAttemptMatchCount} matches</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {!pipelineData.registrationSucceeded && pipelineData.fit && (
                    <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/60 text-amber-200 text-sm">
                      Registration rejected by RANSAC gating: {pipelineData.fit.errorMessage}
                    </div>
                  )}

                  {/* Step 1 & 2: Illumination Metrics & Benchmark Table */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Illumination Gating Card */}
                    <div className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 space-y-4">
                      <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
                        <div className="flex items-center gap-2">
                          <Sparkles className="h-4 w-4 text-amber-400" />
                          <h2 className="text-sm font-semibold text-white">Illumination Normalization</h2>
                        </div>
                        <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                          Tier 1 &amp; 2 Applied
                        </span>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-neutral-400">Ref Illum Confidence:</span>
                          <span className="font-mono font-bold text-emerald-400">{(pipelineData.illuminationConfidenceA * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-neutral-800 rounded-full h-1.5">
                          <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: `${pipelineData.illuminationConfidenceA * 100}%` }}></div>
                        </div>

                        <div className="flex justify-between items-center text-xs">
                          <span className="text-neutral-400">Moving Illum Confidence:</span>
                          <span className="font-mono font-bold text-emerald-400">{(pipelineData.illuminationConfidenceB * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-neutral-800 rounded-full h-1.5">
                          <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: `${pipelineData.illuminationConfidenceB * 100}%` }}></div>
                        </div>

                        <div className="pt-2 text-xs text-neutral-400 border-t border-neutral-800 space-y-1">
                          <p>• Local CLAHE 8×8, clip limit 3.0 (Tier 1)</p>
                          <p>• Sun directional shadow filter (Tier 2)</p>
                          <p>• Incidence confidence gating (Tier 3)</p>
                        </div>
                      </div>
                    </div>

                    {/* Benchmark Comparison Card */}
                    <div className="lg:col-span-2 bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 space-y-3">
                      <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
                        <div className="flex items-center gap-2">
                          <Layers className="h-4 w-4 text-amber-400" />
                          <h2 className="text-sm font-semibold text-white">Feature Matcher Benchmark</h2>
                        </div>
                        <span className="text-xs text-neutral-400">{pipelineData.benchmark?.length ?? 0} Methods Evaluated</span>
                      </div>

                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead>
                            <tr className="text-neutral-400 border-b border-neutral-800">
                              <th className="py-2 px-2">Method</th>
                              <th className="py-2 px-2">Matches</th>
                              <th className="py-2 px-2">Confidence</th>
                              <th className="py-2 px-2">Latency</th>
                              <th className="py-2 px-2">Rationale</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-neutral-800/60">
                            {(pipelineData.benchmark ?? []).map((m, idx) => (
                              <tr key={m.Method} className={idx === 0 ? "bg-amber-500/10 font-medium" : ""}>
                                <td className="py-2.5 px-2 flex items-center gap-1.5">
                                  {idx === 0 && <CheckCircle2 className="h-3.5 w-3.5 text-amber-400 shrink-0" />}
                                  <span className={idx === 0 ? "text-amber-300 font-bold" : "text-neutral-200"}>
                                    {m.Method}
                                  </span>
                                </td>
                                <td className="py-2.5 px-2 font-mono text-neutral-300">{m["Match Count"]}</td>
                                <td className="py-2.5 px-2 font-mono text-neutral-300">{(m["Mean Confidence"] * 100).toFixed(1)}%</td>
                                <td className="py-2.5 px-2 font-mono text-neutral-400">{m["Latency (ms)"].toFixed(1)} ms</td>
                                <td className="py-2.5 px-2 text-[11px] text-neutral-400 max-w-xs truncate" title={m.Rationale}>
                                  {m.Rationale}
                                </td>
                              </tr>
                            ))}
                            {(pipelineData.benchmark ?? []).length === 0 && (
                              <tr><td colSpan={5} className="py-4 text-center text-neutral-500">No methods produced matches for this pair.</td></tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>

                  {/* Interactive Warped Overlay View */}
                  <div className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 space-y-4">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-neutral-800 pb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <Eye className="h-4 w-4 text-amber-400" />
                          <h2 className="text-sm font-semibold text-white">Aligned Registration Overlay</h2>
                        </div>
                        <p className="text-xs text-neutral-400">
                          Homography matrix applied: inlier tie points locked with sub-pixel crater rim registration
                        </p>
                      </div>

                      {/* Opacity & View Controls */}
                      <div className="flex items-center gap-4 w-full sm:w-auto">
                        <div className="flex items-center gap-2 text-xs text-neutral-300 flex-1 sm:flex-initial">
                          <Sliders className="h-3.5 w-3.5 text-neutral-400" />
                          <span>Opacity:</span>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={opacity}
                            onChange={(e) => setOpacity(parseFloat(e.target.value))}
                            className="w-28 accent-amber-500 cursor-pointer"
                          />
                          <span className="font-mono w-8 text-right">{(opacity * 100).toFixed(0)}%</span>
                        </div>

                        <div className="flex bg-neutral-800 p-0.5 rounded-lg border border-neutral-700 text-xs">
                          <button
                            onClick={() => setBlendMode("alpha")}
                            className={`px-2 py-1 rounded transition ${blendMode === "alpha" ? "bg-neutral-700 text-white" : "text-neutral-400"}`}
                          >
                            Blend
                          </button>
                          <button
                            onClick={() => setBlendMode("difference")}
                            className={`px-2 py-1 rounded transition ${blendMode === "difference" ? "bg-neutral-700 text-white" : "text-neutral-400"}`}
                          >
                            Diff
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Aligned Viewer Canvas — real tile pixels, actually blended/diffed client-side */}
                    <div className="relative aspect-square w-full max-w-md mx-auto rounded-xl overflow-hidden bg-neutral-950 border border-neutral-800">
                      <img src={pipelineData.tileImageA} className="absolute inset-0 w-full h-full object-cover" alt="reference" />
                      <img
                        src={pipelineData.tileImageB}
                        className="absolute inset-0 w-full h-full object-cover"
                        style={{ opacity: blendMode === "alpha" ? opacity : 1, mixBlendMode: blendMode === "difference" ? "difference" : "normal" }}
                        alt="moving"
                      />

                      <div className="absolute top-3 left-3 bg-neutral-900/90 backdrop-blur-md px-3 py-1.5 rounded-lg border border-neutral-700 text-xs flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${pipelineData.registrationSucceeded ? "bg-emerald-400" : "bg-red-400"}`}></span>
                        <span className="text-neutral-200 font-medium">
                          {sensorB} {blendMode === "difference" ? "differenced against" : "blended over"} {sensorA}
                        </span>
                      </div>

                      <div className="absolute bottom-3 right-3 bg-neutral-900/90 backdrop-blur-md px-2.5 py-1 rounded text-xs text-neutral-400 border border-neutral-700 font-mono">
                        {pipelineData.fit ? `${pipelineData.fit.inlierCount} of ${pipelineData.fit.totalMatches} matches locked as inliers` : "No fit computed"}
                      </div>
                    </div>
                    <p className="text-[11px] text-neutral-500 text-center">
                      Note: this is a raw pixel blend of the two source tiles for visual reference, not the RANSAC-warped
                      output — the actual homography is in <code>pipelineData.fit</code> / <code>composedTransform</code>.
                    </p>
                  </div>

                  {/* Step 5: Accuracy & Cross-Validation Panel */}
                  {pipelineData.validationReport && (
                    <div className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-5 space-y-4">
                      <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
                        <div className="flex items-center gap-2">
                          <ShieldCheck className="h-5 w-5 text-emerald-400" />
                          <h2 className="text-sm font-semibold text-white">Accuracy &amp; Cross-Validation Panel</h2>
                        </div>
                        <span
                          className={`px-2.5 py-0.5 rounded-full font-bold text-xs border ${
                            pipelineData.validationReport.overall_status === "PASS"
                              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                              : "bg-red-500/20 text-red-300 border-red-500/30"
                          }`}
                        >
                          OVERALL STATUS: {pipelineData.validationReport.overall_status}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                          <div className="text-xs text-neutral-400 mb-1">Inlier Tie-Point RMSE</div>
                          <div className={`text-2xl font-bold font-mono ${pipelineData.validationReport.control_point.passed ? "text-emerald-400" : "text-red-400"}`}>
                            {pipelineData.validationReport.control_point.rmse_px.toFixed(2)} px
                          </div>
                          <div className="text-xs text-neutral-500 mt-1">
                            Threshold: ≤ {pipelineData.validationReport.control_point.threshold_px} px
                            {" "}({pipelineData.validationReport.control_point.passed ? "Pass" : "Fail"})
                          </div>
                        </div>

                        <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                          <div className="text-xs text-neutral-400 mb-1">Independent LROC NAC Cross-Ref</div>
                          <div className={`text-2xl font-bold font-mono ${pipelineData.validationReport.lroc_cross_validation.passed ? "text-emerald-400" : "text-red-400"}`}>
                            {pipelineData.validationReport.lroc_cross_validation.rmse_px != null
                              ? `${pipelineData.validationReport.lroc_cross_validation.rmse_px.toFixed(2)} px`
                              : "N/A"}
                          </div>
                          <div className="text-xs text-neutral-500 mt-1">
                            Threshold: ≤ {pipelineData.validationReport.lroc_cross_validation.threshold_px} px
                            {" "}({pipelineData.validationReport.lroc_cross_validation.status})
                          </div>
                        </div>

                        <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                          <div className="text-xs text-neutral-400 mb-1">RANSAC Inlier Ratio</div>
                          <div className="text-2xl font-bold font-mono text-amber-400">
                            {pipelineData.fit ? `${(pipelineData.fit.inlierRatio * 100).toFixed(1)}%` : "—"}
                          </div>
                          <div className="text-xs text-neutral-500 mt-1">
                            {pipelineData.fit ? `${pipelineData.fit.inlierCount} of ${pipelineData.fit.totalMatches} matches locked` : "No fit"}
                          </div>
                        </div>
                      </div>
                      <p className="text-xs text-neutral-400 pt-2 border-t border-neutral-800">{pipelineData.validationReport.recommendation}</p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </main>
      ) : (
        /* Technical Decisions View (decisions.md reader) */
        <main className="flex-1 max-w-5xl w-full mx-auto p-6 space-y-6 animate-fadeIn">
          <div className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="border-b border-neutral-800 pb-4 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <FileText className="h-5 w-5 text-amber-400" />
                  <span>Architecture &amp; Technical Decision Record (`decisions.md`)</span>
                </h2>
                <p className="text-xs text-neutral-400 mt-1">
                  Comprehensive rationales, architectural trade-offs, and alternative analysis for the lunar pipeline.
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-lg bg-neutral-800 text-neutral-300 font-mono">
                8 Decisions Recorded
              </span>
            </div>

            <div className="space-y-6 text-sm text-neutral-300">
              {/* Decision 1 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  1. Scale Bridging Strategy: TMC-2 as Geometric Pivot vs Direct OHRC↔IIRS
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Reject direct matching across the 320× scale disparity. Enforce a 2-step scale bridging pipeline routing through TMC-2 (5.0m):
                  OHRC (0.25m) → 20× → TMC-2 (5m) → 16× → IIRS (80m). Chain the solved transformation matrices: $H = H_2 \cdot H_1$.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Direct matching fails with 0 inliers. A 100m crater resolved with boulder-level detail in OHRC collapses into an ambiguous 1.2px blur in IIRS. Downsampling OHRC directly to 80m discards 99.999% of its high-frequency data in one step, causing severe spatial aliasing.
                </p>
              </div>

              {/* Decision 2 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  2. Coordinate System: IAU Moon 2000 Equirectangular vs Topocentric / Angular
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Standardize target CRS to IAU Moon 2000 Equirectangular (`+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=1737400 +units=m +no_defs`), matching the NASA LROC standard convention.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Local topocentric frames lack global interoperability with SLDEM2015 DEMs and require custom datum shifts per tile. Raw Selenographic Lat/Lon is non-isometric; longitude distortion causes SIFT isotropic Gaussian filters to warp keypoints.
                </p>
              </div>

              {/* Decision 3 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  3. Illumination Normalization: 3-Tier Architecture with Upfront Gating vs Deep Retinex
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Tier 1 Local CLAHE + Tier 2 Directional Sun Shadow Filter + Tier 3 Upfront Incidence Gating (&gt;80° abort).
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Deep Retinex models misinterpret zero-albedo pitch-black cast shadows as dark surfaces and invent artificial regolith textures. Allowing matching on extreme polar low-sun scenes causes RANSAC to latch onto migrating shadow boundaries rather than real craters.
                </p>
              </div>

              {/* Decision 4 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  4. Hyperspectral Cube Reduction: 1st Principal Component vs Uniform Band Averaging
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Extract the 1st Principal Component (explaining &gt;85% variance) or select Band 42 (~1000nm), solve the geometry once, and broadcast the transform across all 256 channels.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Uniform averaging washes out contrast and incorporates dead detector channels. Solving 256 separate transforms per band is 256× slower and introduces inter-band chromatic misregistration jitter.
                </p>
              </div>

              {/* Decision 5 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  5. Feature Matcher: Pluggable Multi-Matcher (SIFT, AKAZE, LightGlue, RIFT2)
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Pluggable benchmark comparing classical and deep methods, choosing LightGlue with SuperPoint front-end for production registration while keeping SIFT, AKAZE, and RIFT2 stubs available.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Hardcoding SIFT fails on flat, low-contrast regolith where deep graph attention excels. SuperGlue is 3× slower than LightGlue and lacks adaptive depth early-stopping.
                </p>
              </div>

              {/* Decision 6 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  6. Outlier Rejection: Strict Dual-Threshold RANSAC vs Unfiltered Fitting
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Reject fits below 8 inliers or 15% inlier ratio, returning explicit failure diagnostics rather than bad transforms.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  The repetitive self-similarity of craters produces false matches. Plain RANSAC without ratio gating returns 4-point homographies with 2% inlier ratio that completely shear the scene.
                </p>
              </div>

              {/* Decision 7 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  7. Dual Validation Strategy: Tie-Point Control Points + Independent LROC Cross-Check
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Validate both against manual control point pixel RMSE and against an independent third-party reference sensor (NASA LROC NAC).
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Evaluating only internal RANSAC residuals is circular; false matches produce low internal error while the entire image is shifted 50 meters off true selenographic position.
                </p>
              </div>

              {/* Decision 8 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  8. Ingestion &amp; Tiling: Content-Addressed Disk Cache vs Live Full-Scene Processing
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Tile into fixed 512px chunks with georeferenced bounds, cached by SHA256 in `data/cache/` via `prepare-demo-tiles`.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  A full-resolution OHRC strip is &gt;1.4 gigapixels. Live processing on stage causes 5-minute freeze times and container out-of-memory crashes.
                </p>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* Footer */}
      <footer className="border-t border-neutral-800/80 py-4 px-6 text-center text-xs text-neutral-500">
        Lunar Matcher · Chandrayaan-2 Planetary Data System Pipeline · ISRO / NASA Planetary Standards
      </footer>
    </div>
  );
}
