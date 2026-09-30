import React, { useState } from "react";
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
  Eye,
  Info,
  Download,
  Activity,
  FileSpreadsheet,
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

interface MineralClass {
  id: number;
  key: string;
  name: string;
  coverage_pct: number;
  mean_band1_nm: number | null;
  mean_band2_nm: number | null;
  mean_depth: number;
  confidence: number;
  color_hex: string;
}

interface DiagnosticWindow {
  name: string;
  start_nm: number;
  end_nm: number;
  color: string;
}

interface SpectrumData {
  wavelengths_nm: number[];
  reflectance: number[];
  continuum_removed: number[];
  continuum: number[];
}

interface MineralogyReport {
  source: "measured" | "simulated";
  dominant_class: string;
  dominant_coverage_pct: number;
  classes: MineralClass[];
  indices: {
    band1_1um: { center_nm: number | null; depth: number; area: number; valid: boolean };
    band2_2um: { center_nm: number | null; depth: number; area: number; valid: boolean };
    band_area_ratio: number | null;
    plagioclase_1250nm: { center_nm: number | null; depth: number; area: number; valid: boolean };
    hydration_3um: { depth: number | null; center_nm: number | null; valid: boolean; status: string; reason?: string };
  };
  mean_spectrum: SpectrumData;
  diagnostic_windows: DiagnosticWindow[];
  quality: {
    thermal_corrected: boolean | null;
    bad_band_count: number;
    warnings: string[];
  };
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
  mineralogy?: MineralogyReport;
  classMapImage?: string;
  classMapWarped?: string;
  legend?: Record<string, string>;
  plainLanguageSummary?: string;
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

  // Spectral UI state
  const [spectrumMode, setSpectrumMode] = useState<"reflectance" | "continuum_removed">("reflectance");
  const [classMapDisplayMode, setClassMapDisplayMode] = useState<"map" | "overlay" | "warped">("map");

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
          ? "Could not reach the registration pipeline backend at /api. Please check that the server is active."
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

  const downloadJsonReport = () => {
    if (!pipelineData?.mineralogy) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(pipelineData.mineralogy, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `iirs_mineralogy_${region}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const downloadCsvReport = () => {
    if (!pipelineData?.mineralogy) return;
    const m = pipelineData.mineralogy;
    let csv = "Mineral Class,Coverage (%),Mean Band 1 (nm),Mean Band 2 (nm),Mean Depth,Confidence (%)\n";
    m.classes.forEach((c) => {
      csv += `"${c.name}",${c.coverage_pct},${c.mean_band1_nm ?? "N/A"},${c.mean_band2_nm ?? "N/A"},${c.mean_depth},${(c.confidence * 100).toFixed(1)}\n`;
    });
    csv += "\nSpectral Index,Value,Unit/Status\n";
    csv += `Band 1 Center,${m.indices.band1_1um.center_nm ?? "N/A"},nm\n`;
    csv += `Band 1 Depth,${(m.indices.band1_1um.depth * 100).toFixed(2)},%\n`;
    csv += `Band 1 Area,${m.indices.band1_1um.area},nm\n`;
    csv += `Band 2 Center,${m.indices.band2_2um.center_nm ?? "N/A"},nm\n`;
    csv += `Band 2 Depth,${(m.indices.band2_2um.depth * 100).toFixed(2)},%\n`;
    csv += `Band 2 Area,${m.indices.band2_2um.area},nm\n`;
    csv += `Band Area Ratio (BAR),${m.indices.band_area_ratio ?? "N/A"},unitless\n`;
    csv += `1250nm Plagioclase Depth,${(m.indices.plagioclase_1250nm.depth * 100).toFixed(2)},%\n`;
    csv += `3um Hydration Index,${m.indices.hydration_3um.depth ?? "N/A"},${m.indices.hydration_3um.status}\n`;

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `iirs_mineralogy_${region}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // Helper to render SVG spectrum path
  const renderSpectrumSvg = (specData: SpectrumData) => {
    const { wavelengths_nm, reflectance, continuum_removed, continuum } = specData;
    const wMin = 800;
    const wMax = 3200;
    const svgWidth = 600;
    const svgHeight = 220;
    const padX = 45;
    const padY = 25;
    const plotW = svgWidth - padX * 2;
    const plotH = svgHeight - padY * 2;

    const scaleX = (w: number) => padX + ((w - wMin) / (wMax - wMin)) * plotW;

    // Y scale depends on view mode
    let yMin = 0.0;
    let yMax = 0.45;
    if (spectrumMode === "continuum_removed") {
      yMin = 0.70;
      yMax = 1.05;
    }
    const scaleY = (val: number) => padY + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

    const toPath = (vals: number[]) => {
      return vals
        .map((v, i) => {
          const x = scaleX(wavelengths_nm[i]);
          const y = scaleY(v);
          return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(" ");
    };

    const refPath = toPath(reflectance);
    const contPath = toPath(continuum);
    const crPath = toPath(continuum_removed);

    // Diagnostic windows for background shading
    const windows = [
      { name: "1 µm Mafic", start: 820, end: 1320, color: "rgba(59, 130, 246, 0.12)" },
      { name: "1.25 µm Plag", start: 1200, end: 1350, color: "rgba(168, 85, 247, 0.12)" },
      { name: "2 µm Pyroxene", start: 1600, end: 2400, color: "rgba(16, 185, 129, 0.12)" },
      { name: "3 µm H₂O/OH", start: 2700, end: 3100, color: "rgba(236, 72, 153, 0.12)" },
    ];

    const xTicks = [800, 1000, 1500, 2000, 2500, 3000];

    return (
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto select-none overflow-visible">
        {/* Shaded Diagnostic Windows */}
        {windows.map((win) => {
          const x1 = Math.max(padX, scaleX(win.start));
          const x2 = Math.min(padX + plotW, scaleX(win.end));
          const w = Math.max(0, x2 - x1);
          return (
            <g key={win.name}>
              <rect x={x1} y={padY} width={w} height={plotH} fill={win.color} />
              <text x={x1 + 4} y={padY + 12} fontSize="9" fill="#94a3b8" opacity="0.8">
                {win.name}
              </text>
            </g>
          );
        })}

        {/* Grid lines */}
        {xTicks.map((tick) => (
          <g key={tick}>
            <line x1={scaleX(tick)} y1={padY} x2={scaleX(tick)} y2={padY + plotH} stroke="#334155" strokeDasharray="3 3" opacity="0.5" />
            <text x={scaleX(tick)} y={padY + plotH + 15} fontSize="10" fill="#94a3b8" textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}

        {/* Y-axis Ticks */}
        {spectrumMode === "reflectance" ? (
          [0.1, 0.2, 0.3, 0.4].map((yVal) => (
            <g key={yVal}>
              <line x1={padX} y1={scaleY(yVal)} x2={padX + plotW} y2={scaleY(yVal)} stroke="#334155" strokeDasharray="3 3" opacity="0.4" />
              <text x={padX - 8} y={scaleY(yVal) + 4} fontSize="9" fill="#94a3b8" textAnchor="end">
                {yVal.toFixed(2)}
              </text>
            </g>
          ))
        ) : (
          [0.75, 0.85, 0.95, 1.0].map((yVal) => (
            <g key={yVal}>
              <line x1={padX} y1={scaleY(yVal)} x2={padX + plotW} y2={scaleY(yVal)} stroke="#334155" strokeDasharray="3 3" opacity="0.4" />
              <text x={padX - 8} y={scaleY(yVal) + 4} fontSize="9" fill="#94a3b8" textAnchor="end">
                {yVal.toFixed(2)}
              </text>
            </g>
          ))
        )}

        {/* Axis borders */}
        <line x1={padX} y1={padY} x2={padX} y2={padY + plotH} stroke="#475569" />
        <line x1={padX} y1={padY + plotH} x2={padX + plotW} y2={padY + plotH} stroke="#475569" />

        {/* Curves */}
        {spectrumMode === "reflectance" ? (
          <>
            <path d={contPath} fill="none" stroke="#64748b" strokeWidth="1.5" strokeDasharray="4 4" />
            <path d={refPath} fill="none" stroke="#f59e0b" strokeWidth="2.5" />
          </>
        ) : (
          <>
            <line x1={padX} y1={scaleY(1.0)} x2={padX + plotW} y2={scaleY(1.0)} stroke="#64748b" strokeDasharray="4 4" strokeWidth="1.5" />
            <path d={crPath} fill="none" stroke="#10b981" strokeWidth="2.5" />
          </>
        )}

        {/* Axis Labels */}
        <text x={padX + plotW / 2} y={svgHeight - 2} fontSize="10" fill="#cbd5e1" textAnchor="middle" fontWeight="bold">
          Wavelength λ (nm)
        </text>
        <text
          x={-padY - plotH / 2}
          y={12}
          fontSize="10"
          fill="#cbd5e1"
          textAnchor="middle"
          fontWeight="bold"
          transform="rotate(-90)"
        >
          {spectrumMode === "reflectance" ? "Reflectance Factor" : "Continuum Removed (R / R_cont)"}
        </text>
      </svg>
    );
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
                Multi-Sensor Registration &amp; IIRS Spectral Mineralogy · OHRC (0.25m) · TMC-2 (5m) · IIRS (80m)
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
                  <option value="apollo11">Apollo 11 (0.67°N, 23.47°E) · Mare Basalt</option>
                  <option value="tycho">Tycho Crater (43.31°S, 11.36°W) · Anorthosite / Highlands</option>
                  <option value="sinusiridum">Sinus Iridum (44.1°N, 31.5°W) · Imbrium Mare Flow</option>
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
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-current" />
                      <span>Run Pipeline</span>
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
                  {" "}OHRC (0.25m) → TMC-2 (5m, 20×) → IIRS (80m, 16×), chaining homography transforms $H = H_2 \cdot H_1$.
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

          {/* Pre-cached Tile Preview */}
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
              Press <span className="text-neutral-300 font-medium">Run Pipeline</span> to load real tiles for {
                { apollo11: "Apollo 11", tycho: "Tycho Crater", sinusiridum: "Sinus Iridum", southpole: "the South Pole gating demo" }[region]
              } and evaluate multi-sensor registration &amp; IIRS spectral mineralogy.
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
                    Pipeline registration halted prior to feature extraction (computed live by
                    lunar_matcher.matching.illumination.compute_illumination_confidence). Engineering safety rule:
                    refusing registration on extreme grazing shadows prevents hallucinated correspondences.
                  </p>
                </div>
              ) : (
                <>
                  {/* Scale-bridging strategy */}
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

                  {/* Pivot chain detail */}
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

                  {/* Feature Matcher Benchmark Table */}
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
                    </div>
                  </div>

                  {/* Accuracy & Cross-Validation Panel */}
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

              {/* ------------------------------------------------------------- */}
              {/* Mineralogy Report Panel (IIRS) — rendered even if gated!     */}
              {/* ------------------------------------------------------------- */}
              {pipelineData.mineralogy && (
                <section className="bg-neutral-900/80 border border-neutral-800 rounded-2xl p-6 space-y-6 shadow-xl animate-fadeIn">
                  {/* Panel Header */}
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-700 flex items-center justify-center text-neutral-950">
                        <Activity className="h-5 w-5" />
                      </div>
                      <div>
                        <h2 className="text-base font-bold text-white flex items-center gap-2">
                          <span>Mineralogy Report (Chandrayaan-2 IIRS)</span>
                        </h2>
                        <p className="text-xs text-neutral-400">
                          Diagnostic Crystal Field Absorption Analysis · 256 Spectral Channels (800–5100 nm)
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Provenance Badge */}
                      {pipelineData.mineralogy.source === "measured" ? (
                        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                          MEASURED IIRS CUBE
                        </span>
                      ) : (
                        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          SIMULATED SPECTRA - demo tile has no spectral data
                        </span>
                      )}

                      {/* Download Buttons */}
                      <button
                        onClick={downloadJsonReport}
                        className="p-1.5 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-300 hover:text-white hover:bg-neutral-700 transition"
                        title="Download Report JSON"
                      >
                        <Download className="h-4 w-4" />
                      </button>
                      <button
                        onClick={downloadCsvReport}
                        className="p-1.5 rounded-lg bg-neutral-800 border border-neutral-700 text-neutral-300 hover:text-white hover:bg-neutral-700 transition"
                        title="Download Data CSV"
                      >
                        <FileSpreadsheet className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* AI Summary Banner (if generated) */}
                  {pipelineData.plainLanguageSummary && (
                    <div className="p-4 rounded-xl bg-gradient-to-r from-emerald-950/40 to-neutral-900 border border-emerald-800/50 flex items-start gap-3">
                      <Sparkles className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
                      <div className="space-y-1">
                        <div className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
                          AI Geochemical Synopsis
                        </div>
                        <p className="text-xs text-neutral-200 leading-relaxed">
                          {pipelineData.plainLanguageSummary}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Summary Cards Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                      <div className="text-xs text-neutral-400 mb-1">Dominant Mineral Lithology</div>
                      <div className="text-base font-bold text-emerald-400 truncate" title={pipelineData.mineralogy.dominant_class}>
                        {pipelineData.mineralogy.dominant_class}
                      </div>
                      <div className="text-xs text-neutral-500 mt-1">
                        Spatial coverage: <span className="font-mono text-neutral-300 font-bold">{pipelineData.mineralogy.dominant_coverage_pct}%</span>
                      </div>
                    </div>

                    <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                      <div className="text-xs text-neutral-400 mb-1">1 µm Mafic Band Center</div>
                      <div className="text-xl font-bold font-mono text-blue-400">
                        {pipelineData.mineralogy.indices.band1_1um.center_nm?.toFixed(1) ?? "—"} nm
                      </div>
                      <div className="text-xs text-neutral-500 mt-1">
                        Absorption depth: {(pipelineData.mineralogy.indices.band1_1um.depth * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                      <div className="text-xs text-neutral-400 mb-1">2 µm Pyroxene / BAR</div>
                      <div className="text-xl font-bold font-mono text-emerald-400">
                        {pipelineData.mineralogy.indices.band2_2um.center_nm?.toFixed(1) ?? "—"} nm
                      </div>
                      <div className="text-xs text-neutral-500 mt-1">
                        Band Area Ratio: <span className="font-mono text-neutral-300">{pipelineData.mineralogy.indices.band_area_ratio ?? "N/A"}</span>
                      </div>
                    </div>

                    <div className="bg-neutral-950/60 p-4 rounded-xl border border-neutral-800/80">
                      <div className="text-xs text-neutral-400 mb-1">3 µm Hydration Index</div>
                      <div className="text-sm font-bold font-mono text-amber-400/90 pt-1">
                        {pipelineData.mineralogy.indices.hydration_3um.depth != null
                          ? `${(pipelineData.mineralogy.indices.hydration_3um.depth * 100).toFixed(2)}%`
                          : "n/a - needs thermal correction"}
                      </div>
                      <div className="text-[11px] text-neutral-500 mt-1 truncate" title={pipelineData.mineralogy.indices.hydration_3um.reason}>
                        Thermal model unverified
                      </div>
                    </div>
                  </div>

                  {/* Spectral Curve Chart & Classification Map */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                    {/* Inline-SVG Spectrum Chart */}
                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded-xl p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-xs font-bold text-neutral-200">IIRS Mean Reflectance Spectrum</h3>
                          <p className="text-[11px] text-neutral-400">Diagnostic crystal field absorption windows shaded</p>
                        </div>
                        <div className="flex bg-neutral-800 p-0.5 rounded-lg border border-neutral-700 text-xs">
                          <button
                            onClick={() => setSpectrumMode("reflectance")}
                            className={`px-2 py-0.5 rounded transition ${spectrumMode === "reflectance" ? "bg-neutral-700 text-white font-medium" : "text-neutral-400"}`}
                          >
                            Reflectance
                          </button>
                          <button
                            onClick={() => setSpectrumMode("continuum_removed")}
                            className={`px-2 py-0.5 rounded transition ${spectrumMode === "continuum_removed" ? "bg-neutral-700 text-white font-medium" : "text-neutral-400"}`}
                          >
                            Cont-Removed
                          </button>
                        </div>
                      </div>

                      {/* SVG Spectrum Renderer */}
                      <div className="w-full bg-neutral-950 rounded-lg p-2 border border-neutral-800/50">
                        {renderSpectrumSvg(pipelineData.mineralogy.mean_spectrum)}
                      </div>

                      {/* Chart Legend */}
                      <div className="flex flex-wrap items-center gap-4 text-[11px] text-neutral-400 pt-1">
                        {spectrumMode === "reflectance" ? (
                          <>
                            <div className="flex items-center gap-1.5">
                              <span className="w-3 h-0.5 bg-amber-500"></span>
                              <span>Mean Reflectance R(λ)</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                              <span className="w-3 h-0.5 border-t border-dashed border-slate-400"></span>
                              <span>Linear Continuum Baseline</span>
                            </div>
                          </>
                        ) : (
                          <div className="flex items-center gap-1.5">
                            <span className="w-3 h-0.5 bg-emerald-500"></span>
                            <span>Continuum-Removed Ratio R / R_cont</span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Thematic Mineral Map Viewer */}
                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded-xl p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-xs font-bold text-neutral-200">Thematic Mineral Classification Map</h3>
                          <p className="text-[11px] text-neutral-400">Pixel classification derived from absorption band parameters</p>
                        </div>
                        <div className="flex bg-neutral-800 p-0.5 rounded-lg border border-neutral-700 text-xs">
                          <button
                            onClick={() => setClassMapDisplayMode("map")}
                            className={`px-2 py-0.5 rounded transition ${classMapDisplayMode === "map" ? "bg-neutral-700 text-white font-medium" : "text-neutral-400"}`}
                          >
                            Class Map
                          </button>
                          <button
                            onClick={() => setClassMapDisplayMode("overlay")}
                            className={`px-2 py-0.5 rounded transition ${classMapDisplayMode === "overlay" ? "bg-neutral-700 text-white font-medium" : "text-neutral-400"}`}
                          >
                            Overlay
                          </button>
                          {pipelineData.classMapWarped && (
                            <button
                              onClick={() => setClassMapDisplayMode("warped")}
                              className={`px-2 py-0.5 rounded transition ${classMapDisplayMode === "warped" ? "bg-neutral-700 text-white font-medium" : "text-neutral-400"}`}
                            >
                              Warped
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Map Canvas */}
                      <div className="relative aspect-square w-full max-w-sm mx-auto rounded-lg overflow-hidden bg-neutral-950 border border-neutral-800">
                        {/* Background reference tile if in overlay or warped mode */}
                        {(classMapDisplayMode === "overlay" || classMapDisplayMode === "warped") && (
                          <img src={pipelineData.tileImageA} className="absolute inset-0 w-full h-full object-cover" alt="reference" />
                        )}

                        {/* Class map image */}
                        {pipelineData.classMapImage && (
                          <img
                            src={classMapDisplayMode === "warped" && pipelineData.classMapWarped ? pipelineData.classMapWarped : pipelineData.classMapImage}
                            className="absolute inset-0 w-full h-full object-cover"
                            style={{
                              opacity: classMapDisplayMode === "map" ? 1.0 : opacity,
                              imageRendering: "pixelated",
                            }}
                            alt="mineralogy class map"
                          />
                        )}

                        <div className="absolute top-2 left-2 bg-neutral-900/90 backdrop-blur-md px-2 py-1 rounded text-[11px] text-neutral-300 border border-neutral-700">
                          {classMapDisplayMode === "map"
                            ? "Raw Spectral Classes"
                            : classMapDisplayMode === "warped"
                            ? `Registered to ${pipelineData.sensorA} frame`
                            : `Overlaid on ${pipelineData.sensorA} (${(opacity * 100).toFixed(0)}%)`}
                        </div>
                      </div>

                      {/* Legend Chips */}
                      <div className="flex flex-wrap gap-2 text-[11px] pt-1">
                        {pipelineData.mineralogy.classes.map((cls) => (
                          <div key={cls.id} className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-neutral-900 border border-neutral-800">
                            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: cls.color_hex }}></span>
                            <span className="text-neutral-300">{cls.name.split(" ")[0]}</span>
                            <span className="text-neutral-500 font-mono">({cls.coverage_pct}%)</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Coverage Breakdown Bars & Diagnostic Index Table */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Coverage Breakdown */}
                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded-xl p-4 space-y-3">
                      <h3 className="text-xs font-bold text-neutral-200">Mineral Phase Distribution</h3>

                      {/* Multi-segment stacked bar */}
                      <div className="w-full h-3 rounded-full overflow-hidden flex bg-neutral-800">
                        {pipelineData.mineralogy.classes.map((cls) => (
                          <div
                            key={cls.id}
                            style={{ width: `${cls.coverage_pct}%`, backgroundColor: cls.color_hex }}
                            title={`${cls.name}: ${cls.coverage_pct}%`}
                          ></div>
                        ))}
                      </div>

                      {/* Detailed Class List */}
                      <div className="divide-y divide-neutral-800/60 text-xs">
                        {pipelineData.mineralogy.classes.map((cls) => (
                          <div key={cls.id} className="py-2 flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: cls.color_hex }}></span>
                              <span className="text-neutral-200 font-medium">{cls.name}</span>
                            </div>
                            <div className="flex items-center gap-4 font-mono text-neutral-400">
                              <span>Depth: {(cls.mean_depth * 100).toFixed(1)}%</span>
                              <span className="w-14 text-right text-neutral-200 font-bold">{cls.coverage_pct}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Diagnostic Absorption Indices Table */}
                    <div className="bg-neutral-950/60 border border-neutral-800/80 rounded-xl p-4 space-y-3">
                      <h3 className="text-xs font-bold text-neutral-200">Absorption Band Indices &amp; Parameters</h3>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead>
                            <tr className="text-neutral-400 border-b border-neutral-800">
                              <th className="py-1.5 px-2">Diagnostic Feature</th>
                              <th className="py-1.5 px-2">Center</th>
                              <th className="py-1.5 px-2">Depth</th>
                              <th className="py-1.5 px-2">Area / Ratio</th>
                              <th className="py-1.5 px-2">Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-neutral-800/60 font-mono text-neutral-300">
                            <tr>
                              <td className="py-2 px-2 font-sans font-medium text-neutral-200">1 µm Mafic Silicate</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.band1_1um.center_nm ?? "—"} nm</td>
                              <td className="py-2 px-2">{(pipelineData.mineralogy.indices.band1_1um.depth * 100).toFixed(1)}%</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.band1_1um.area} nm</td>
                              <td className="py-2 px-2 text-emerald-400">Valid</td>
                            </tr>
                            <tr>
                              <td className="py-2 px-2 font-sans font-medium text-neutral-200">2 µm Pyroxene / Spinel</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.band2_2um.center_nm ?? "—"} nm</td>
                              <td className="py-2 px-2">{(pipelineData.mineralogy.indices.band2_2um.depth * 100).toFixed(1)}%</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.band2_2um.area} nm</td>
                              <td className="py-2 px-2 text-emerald-400">Valid</td>
                            </tr>
                            <tr>
                              <td className="py-2 px-2 font-sans font-medium text-neutral-200">Band Area Ratio (BAR)</td>
                              <td className="py-2 px-2">—</td>
                              <td className="py-2 px-2">—</td>
                              <td className="py-2 px-2 text-amber-400 font-bold">{pipelineData.mineralogy.indices.band_area_ratio ?? "N/A"}</td>
                              <td className="py-2 px-2 text-neutral-400">Area2/Area1</td>
                            </tr>
                            <tr>
                              <td className="py-2 px-2 font-sans font-medium text-neutral-200">1.25 µm Plagioclase</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.plagioclase_1250nm.center_nm ?? "—"} nm</td>
                              <td className="py-2 px-2">{(pipelineData.mineralogy.indices.plagioclase_1250nm.depth * 100).toFixed(1)}%</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.plagioclase_1250nm.area} nm</td>
                              <td className="py-2 px-2 text-neutral-300">
                                {pipelineData.mineralogy.indices.plagioclase_1250nm.valid ? "Fe²⁺ detected" : "Trace"}
                              </td>
                            </tr>
                            <tr>
                              <td className="py-2 px-2 font-sans font-medium text-neutral-200">3 µm Hydration (H₂O/OH)</td>
                              <td className="py-2 px-2">{pipelineData.mineralogy.indices.hydration_3um.center_nm ?? "—"}</td>
                              <td className="py-2 px-2">
                                {pipelineData.mineralogy.indices.hydration_3um.depth != null
                                  ? `${(pipelineData.mineralogy.indices.hydration_3um.depth * 100).toFixed(1)}%`
                                  : "—"}
                              </td>
                              <td className="py-2 px-2">—</td>
                              <td className="py-2 px-2 text-amber-400">
                                {pipelineData.mineralogy.indices.hydration_3um.status}
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>

                  {/* Caveats Box */}
                  {pipelineData.mineralogy.quality.warnings.length > 0 && (
                    <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-800/40 text-amber-300/90 text-xs space-y-1.5">
                      <div className="font-semibold text-amber-200 flex items-center gap-1.5">
                        <AlertTriangle className="h-4 w-4" />
                        <span>Quality &amp; Processing Caveats:</span>
                      </div>
                      <ul className="list-disc list-inside space-y-1 text-neutral-300/80">
                        {pipelineData.mineralogy.quality.warnings.map((warn, i) => (
                          <li key={i}>{warn}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </section>
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
                9 Decisions Recorded
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
                  4. Hyperspectral Cube Reduction: 1st Principal Component vs Nearest ~1000 nm Band Selection
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Extract the 1st Principal Component (explaining &gt;85% variance) or dynamically compute the reference band closest to 1000 nm: at 16.85 nm sampling starting from 800 nm, 1000 nm corresponds to Band 12 (round((1000 - 800) / 16.85) = 12), rather than a static index. Solve geometry once and broadcast across all 256 channels.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Uniform averaging washes out contrast and incorporates dead detector channels. Solving 256 separate transforms per band is 256× slower and introduces inter-band chromatic misregistration jitter. Hardcoding index 42 sampled at ~1507 nm instead of 1000 nm.
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

              {/* Decision 9 */}
              <div className="p-4 rounded-xl bg-neutral-950/60 border border-neutral-800/80 space-y-2">
                <h3 className="text-amber-400 font-semibold text-base">
                  9. Mineral Characterisation from IIRS: Diagnostic Band-Parameter Analysis vs Black-Box Classifier
                </h3>
                <p>
                  <strong className="text-white">Decision Made: </strong>
                  Physically grounded absorption band parameter analysis (`lunar_matcher/spectral/analysis.py`):
                  straight-line shoulder continuum removal across standard diagnostic windows (1 µm, 1.25 µm, 2 µm, 3 µm), polynomial sub-band center derivation, absorption depth, and Band Area Ratio (BAR = Area 2µm / Area 1µm). Rule-based mineral classification is verified against laboratory reference libraries (RELAB / USGS) and gated strictly against thermal emission artifacts.
                </p>
                <p className="text-xs text-neutral-400">
                  <strong className="text-red-400">What would happen if alternative was used: </strong>
                  Deep black-box neural networks lack geochemical explainability; slight photometric variations, sensor vignetting, or uncorrected thermal emission tails (&gt;2.5 µm) cause them to hallucinate exotic minerals or misclassify high-Ca pyroxene as low-Ca pyroxene without diagnostic traceability. Unsupervised clustering groups pixels by surface albedo rather than crystal field absorption features.
                </p>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* Footer */}
      <footer className="border-t border-neutral-800/80 py-4 px-6 text-center text-xs text-neutral-500">
        Lunar Matcher · Chandrayaan-2 Planetary Data System Pipeline &amp; IIRS Spectral Mineralogy · ISRO / NASA Planetary Standards
      </footer>
    </div>
  );
}
