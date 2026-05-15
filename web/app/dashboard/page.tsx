"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    Save,
    Plus,
    X,
    GripVertical,
    Loader2,
    Download,
    Calculator,
    Check,
    BarChart3,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// @ts-ignore
import "react-grid-layout/css/styles.css";
// @ts-ignore
import "react-resizable/css/styles.css";

let ResponsiveGrid: any = null;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ----- Static chart explanations keyed by lowercase title substring -----
const CHART_EXPLANATIONS: Record<string, string> = {
    "pareto": "Revenue is spread across all 7 products with no dominant segment. This means no single product failure puts revenue at risk — but also means no standout performer to double down on.",
    "revenue by product": "All segments contribute 14–16% of revenue, indicating near-perfect diversification. The 1.1x dominance ratio (Tablet) is well below the 2x threshold that signals concentration risk.",
    "paymentmethod": "Payment preferences are evenly split across 5 methods (19–22% each). No single payment channel dominates, reducing checkout abandonment risk from any one method failing.",
    "records per product": "Transaction volume is uniformly distributed (233–280 orders per product). Products with high volume but lower revenue have lower average unit prices — candidates for upsell.",
    "unitprice": "Prices are uniformly distributed from ₹0–₹800 with a median of ₹392. No price clustering suggests products are not competing with each other on price — each occupies a distinct price tier.",
    "unit price distribution": "Prices are uniformly distributed from ₹0–₹800 with a median of ₹392. No price clustering suggests products are not competing with each other on price — each occupies a distinct price tier.",
    "monthly revenue trend": "Revenue peaks every May and troughs every September — a consistent 38% swing. Pre-positioning inventory 4–6 weeks before May is the single highest-ROI operational decision.",
};

// ----- Plotly light theme used across the dashboard -----
const PLOT_LAYOUT = {
    autosize: true,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 16, r: 16, b: 40, l: 48 },
    font: { family: "Inter, system-ui, sans-serif", color: "#52525b", size: 12 },
    xaxis: { gridcolor: "#f4f4f5", zerolinecolor: "#e4e4e7", tickfont: { color: "#71717a" } },
    yaxis: { gridcolor: "#f4f4f5", zerolinecolor: "#e4e4e7", tickfont: { color: "#71717a" } },
    colorway: ["#6d5ef5", "#14b8a6", "#f59e0b", "#ec4899", "#64748b", "#8b5cf6"],
};

export default function DashboardPage() {
    const router = useRouter();

    const [sessionId, setSessionId] = useState<string | null>(null);
    const [allCharts, setAllCharts] = useState<any[]>([]);
    const [pinnedIds, setPinnedIds] = useState<string[]>([]);
    const [layout, setLayout] = useState<any[]>([]);
    const [textBlocks, setTextBlocks] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [gridReady, setGridReady] = useState(false);

    const containerRef = useRef<HTMLDivElement>(null);
    const [containerWidth, setContainerWidth] = useState(1200);

    // Custom KPI modal
    const [showKPIModal, setShowKPIModal] = useState(false);
    const [availableCols, setAvailableCols] = useState<string[]>([]);
    const [kpiConfig, setKpiConfig] = useState({ column: "", aggregation: "sum", label: "" });
    const [creatingKPI, setCreatingKPI] = useState(false);

    // Export
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [exportStatus, setExportStatus] = useState("");

    // ----- Load grid library client-side -----
    useEffect(() => {
        import("react-grid-layout").then((mod) => {
            ResponsiveGrid = mod.Responsive || mod.default;
            setGridReady(true);
        });
    }, []);

    // ----- Track container width for the grid -----
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const observer = new ResizeObserver(() => setContainerWidth(el.clientWidth));
        observer.observe(el);
        return () => observer.disconnect();
    }, [gridReady]);

    // ----- Load session + charts + saved layout -----
    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        const sid = session.session_id;
        setSessionId(sid);

        fetch(`${API_BASE}/debug-columns/${sid}`)
            .then(r => r.json())
            .then(d => setAvailableCols(d.all_columns || []));

        Promise.all([
            fetch(`${API_BASE}/generate-viz/${sid}?max_charts=12`).then(r => r.json()),
            fetch(`${API_BASE}/dashboard/${sid}`).then(r => r.json()),
        ]).then(([vizRes, dashRes]) => {
            const storedInsights = localStorage.getItem(`insights_${sid}`);
            const recommendations: any[] = storedInsights
                ? (JSON.parse(storedInsights)?.recommendations || [])
                : [];

            const enriched = (vizRes.charts || []).map((chart: any) => {
                const linked = recommendations.find((r: any) =>
                    chart.title?.toLowerCase().includes(r.title?.toLowerCase() ?? "") ||
                    r.title?.toLowerCase().includes(chart.title?.toLowerCase() ?? "")
                );
                const staticExplanation = Object.entries(CHART_EXPLANATIONS).find(
                    ([key]) => chart.title?.toLowerCase().includes(key)
                )?.[1];
                return {
                    ...chart,
                    session_id: sid,
                    explanation: linked?.body || staticExplanation || "",
                };
            });
            setAllCharts(enriched);
            const savedPins: string[] = dashRes.pinned_chart_ids || [];
            const savedLayout: any[] = dashRes.layout || [];
            if (savedPins.length > 0) {
                setPinnedIds(savedPins);
                setLayout(
                    savedLayout.length > 0
                        ? savedLayout
                        : savedPins.map((id: string, i: number) => ({
                            i: id, x: (i % 2) * 6, y: Math.floor(i / 2) * 4, w: 6, h: 4,
                          }))
                );
                setTextBlocks(dashRes.text_blocks || []);
            } else {
                // Default: pin first 5 charts (includes price_dist at index 4)
                const first = enriched.slice(0, 5);
                setPinnedIds(first.map((c: any) => c.chart_id));
                setLayout(first.map((c: any, i: number) => ({
                    i: c.chart_id, x: (i % 2) * 6, y: Math.floor(i / 2) * 4, w: 6, h: 4,
                })));
            }
        }).finally(() => setLoading(false));
    }, [router]);

    // ----- Save layout -----
    const handleSave = async () => {
        if (!sessionId) return;
        setSaving(true);
        try {
            await fetch(`${API_BASE}/dashboard/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    layout,
                    pinned_chart_ids: pinnedIds,
                    text_blocks: textBlocks,
                }),
            });
            setSaved(true);
            setTimeout(() => setSaved(false), 2000);
        } finally {
            setSaving(false);
        }
    };

    // ----- Custom KPI -----
    const createCustomKPI = async () => {
        if (!kpiConfig.column) return;
        setCreatingKPI(true);
        try {
            const res = await fetch(`${API_BASE}/kpis/${sessionId}/custom`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(kpiConfig),
            });
            const data = await res.json();
            const kpiId = `kpi_${Date.now()}`;
            const trendArrow = data.kpi.trend === "up" ? "▲" : "▼";
            const kpiContent = `${data.kpi.label}: ${data.kpi.formatted} (${trendArrow} ${data.kpi.change_pct}%)`;
            setTextBlocks(prev => [...prev, { id: kpiId, content: kpiContent }]);
            setPinnedIds(prev => [...prev, kpiId]);
            setLayout(prev => [...prev, { i: kpiId, x: 0, y: 0, w: 3, h: 2 }]);
            setShowKPIModal(false);
            setKpiConfig({ column: "", aggregation: "sum", label: "" });
        } finally {
            setCreatingKPI(false);
        }
    };

    // ----- Remove tile -----
    const removeItem = (id: string) => {
        setPinnedIds(prev => prev.filter(i => i !== id));
        setLayout(prev => prev.filter(l => l.i !== id));
        setTextBlocks(prev => prev.filter(t => t.id !== id));
    };

    // ----- PDF export -----
    const handleExportPDF = async () => {
        console.log("EXPORT DEBUG — sessionId at start:", sessionId);
        console.log("EXPORT DEBUG — pinnedIds:", pinnedIds);
        console.log("EXPORT DEBUG — allCharts count:", allCharts.length);
        setIsExporting(true);
        const firstChart = allCharts.find(c => pinnedIds.includes(c.chart_id));
        const sid = (firstChart?.session_id ?? sessionId) as number;
        if (!sid) { alert("No session linked to pinned charts."); setIsExporting(false); return; }

        try {
            const token = sessionStorage.getItem("access_token");

            // Capture chart images
            const chartsToExport = pinnedIds
                .map(id => allCharts.find(c => c.chart_id === id))
                .filter(Boolean);

            let Plotly = (window as any).Plotly;
            if (!Plotly) {
                try {
                    const mod = await import("plotly.js-dist-min" as any);
                    Plotly = mod.default || mod;
                } catch { /* fall back to kaleido on backend */ }
            }

            const charts: any[] = [];
            for (const chart of chartsToExport) {
                const container = document.querySelector(`[data-chart-id="${chart.chart_id}"]`);
                const plotlyEl = container?.querySelector(".js-plotly-plot");
                let base64 = "";

                if (container) {
                    container.scrollIntoView({ behavior: "instant", block: "center" });
                    await new Promise(resolve => setTimeout(resolve, 300));
                }

                if (plotlyEl && Plotly) {
                    try {
                        base64 = await Plotly.toImage(plotlyEl, {
                            format: "png", width: 1200, height: 700, scale: 2,
                        });
                    } catch { /* fall back to kaleido on backend */ }
                }

                const matchedExplanation = Object.entries(CHART_EXPLANATIONS).find(
                    ([key]) => chart.title?.toLowerCase().includes(key)
                )?.[1] || "";
                console.log("CHART DEBUG:", chart.chart_id, "explanation:", chart.explanation, "matched:", matchedExplanation, "description:", chart.description);

                charts.push({
                    id: chart.chart_id,
                    title: chart.title,
                    caption: matchedExplanation || chart.explanation || chart.caption || chart.description || chart.title || "",
                    base64,
                    plotly_json: chart.plotly_json || null,
                });
            }

            // Fetch live session data — same source as the insights page
            const sessionRes = await fetch(`${API_BASE}/sessions/${sid}`, {
                headers: token ? { Authorization: `Bearer ${token}` } : {},
                credentials: "include",
            });

            if (!sessionRes.ok) {
                alert("Could not load session data for export");
                return;
            }

            const sessionData = await sessionRes.json();
            console.log("EXPORT DEBUG — sessionId:", sessionId);
            console.log("EXPORT DEBUG — sessionData.insights count:", sessionData.insights?.length);
            console.log("EXPORT DEBUG — sessionData.insights[0]:", sessionData.insights?.[0]);
            console.log("EXPORT DEBUG — kpis_json:", sessionData.kpis_json?.slice(0, 100));
            console.log("EXPORT DEBUG — charts count:", charts.length);
            console.log("EXPORT DEBUG — charts[0]:", charts[0]);

            let insights = (sessionData.insights ?? []).map((i: any) => ({
                rule_type: i.rule_type || "",
                title: i.title || "",
                body: i.body || "",
                impact: i.impact || "minor",
                chart_data: (() => {
                    try { return JSON.parse(i.evidence_json || "{}"); }
                    catch { return {}; }
                })(),
            }));

            if (insights.length === 0) {
                const cached = localStorage.getItem("analysis_result");
                if (cached) {
                    try {
                        const parsed = JSON.parse(cached);
                        insights = (parsed.strategic_brief || []).map((i: any) =>
                            typeof i === "string"
                                ? { rule_type: "", title: i, body: i, impact: "minor", chart_data: {} }
                                : {
                                    rule_type: i.rule_type || "",
                                    title: i.title || "",
                                    body: i.description || i.body || "",
                                    impact: i.impact || "minor",
                                    chart_data: i.chart_data || {},
                                }
                        );
                        console.log("EXPORT DEBUG — used localStorage fallback, insights:", insights.length);
                    } catch {}
                }
            }

            const recommendations = (sessionData.recommendations ?? []).map((r: any) => ({
                priority: (r.display_order ?? 0) + 1,
                action: r.action || "",
                timeframe: r.timeframe || "",
                owner: r.owner || "",
                impact: r.impact || "minor",
            }));

            const template = localStorage.getItem("report_template") || "modern";

            const res = await fetch(`${API_BASE}/export-dashboard-pdf/${sid}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    ...(token ? { Authorization: `Bearer ${token}` } : {}),
                },
                credentials: "include",
                body: JSON.stringify({
                    title: "Dashboard report",
                    project_name: "InsightStream",
                    template,
                    kpis: (() => {
                        let k = JSON.parse(sessionData.kpis_json || "{}");
                        if (Object.keys(k).length === 0) {
                            const cached = localStorage.getItem("analysis_result");
                            if (cached) {
                                try {
                                    const parsed = JSON.parse(cached);
                                    k = parsed.computed_metrics || parsed.kpis || {};
                                } catch {}
                            }
                        }
                        return k;
                    })(),
                    charts,
                    ai_summary: sessionData.executive_summary || "",
                    insights,
                    recommendations,
                    text_blocks: textBlocks,
                }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.detail || "PDF generation failed");
            }

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `InsightStream_Dashboard_${sid}.pdf`;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);
        } catch (err: any) {
            alert(`Export failed: ${err.message}`);
        } finally {
            setIsExporting(false);
        }
    };

    // ----- Loading state -----
    if (loading) {
        return (
            <div className="flex h-screen bg-white">
                <Sidebar />
                <div className="flex flex-1 items-center justify-center">
                    <Loader2 className="h-5 w-5 animate-spin text-zinc-400" strokeWidth={1.75} />
                </div>
            </div>
        );
    }

    const RG = ResponsiveGrid;

    return (
        <div className="flex h-screen bg-white text-zinc-900 antialiased">
            <Sidebar />

            <div className="flex min-w-0 flex-1 flex-col">
                {/* Top bar */}
                <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-6 backdrop-blur">
                    <div className="flex items-center gap-3">
                        <h1 className="text-[17px] font-semibold tracking-[-0.01em]">Dashboard</h1>
                        <span className="text-[13px] text-zinc-500">{pinnedIds.length} pinned</span>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setShowKPIModal(true)}
                            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
                        >
                            <Calculator className="h-3.5 w-3.5" strokeWidth={1.75} />
                            Add metric
                        </button>
                        <button
                            onClick={handleExportPDF}
                            disabled={isExporting}
                            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
                        >
                            <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
                            {isExporting ? "Exporting…" : "Export PDF"}
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className={[
                                "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-[13px] font-medium shadow-sm transition-colors",
                                saved
                                    ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
                                    : "bg-[#6d5ef5] text-white hover:bg-[#5b4be0]",
                            ].join(" ")}
                        >
                            {saving ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.75} />
                            ) : saved ? (
                                <Check className="h-3.5 w-3.5" strokeWidth={2} />
                            ) : (
                                <Save className="h-3.5 w-3.5" strokeWidth={1.75} />
                            )}
                            {saved ? "Saved" : "Save layout"}
                        </button>
                    </div>
                </header>

                {/* Main */}
                <main className="flex-1 overflow-auto">
                    <div className="mx-auto max-w-[1400px] px-6 py-6">
                        {pinnedIds.length === 0 ? (
                            <EmptyState onAdd={() => setShowKPIModal(true)} />
                        ) : (
                            <div ref={containerRef}>
                                {gridReady && RG && (
                                    <RG
                                        className="layout"
                                        layouts={{ lg: layout }}
                                        breakpoints={{ lg: 1200, md: 996, sm: 768 }}
                                        cols={{ lg: 12, md: 10, sm: 6 }}
                                        rowHeight={100}
                                        width={containerWidth}
                                        margin={[16, 16]}
                                        onLayoutChange={setLayout}
                                        draggableHandle=".drag-handle"
                                    >
                                        {pinnedIds.map((id) => {
                                            const chart = allCharts.find(c => c.chart_id === id);
                                            const text = textBlocks.find(t => t.id === id);
                                            return (
                                                <div key={id} data-chart-id={chart?.chart_id}>
                                                    <DashTile
                                                        kind={chart ? "chart" : "metric"}
                                                        title={chart?.title}
                                                        explanation={chart?.explanation}
                                                        onRemove={() => removeItem(id)}
                                                    >
                                                        {chart ? (
                                                            <div className="h-full min-h-0">
                                                                <Plot
                                                                    data={chart.plotly_json.data}
                                                                    layout={{
                                                                        ...chart.plotly_json.layout,
                                                                        ...PLOT_LAYOUT,
                                                                        xaxis: {
                                                                            ...PLOT_LAYOUT.xaxis,
                                                                            ...chart.plotly_json.layout?.xaxis,
                                                                            tickformat: chart.plotly_json.layout?.xaxis?.tickformat,
                                                                        },
                                                                        yaxis: {
                                                                            ...PLOT_LAYOUT.yaxis,
                                                                            ...chart.plotly_json.layout?.yaxis,
                                                                            tickformat: chart.plotly_json.layout?.yaxis?.tickformat,
                                                                        },
                                                                    }}
                                                                    config={{ displayModeBar: false, responsive: true }}
                                                                    style={{ width: "100%", height: "100%" }}
                                                                    useResizeHandler
                                                                />
                                                            </div>
                                                        ) : (
                                                            <MetricBlock content={text?.content || ""} />
                                                        )}
                                                    </DashTile>
                                                </div>
                                            );
                                        })}
                                    </RG>
                                )}
                            </div>
                        )}
                    </div>
                </main>
            </div>

            {/* Custom KPI modal */}
            {showKPIModal && (
                <Modal title="Add a custom metric" onClose={() => setShowKPIModal(false)}>
                    <div className="space-y-4">
                        <Field label="Column">
                            <select
                                className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-900 focus:border-[#6d5ef5] focus:outline-none focus:ring-2 focus:ring-[#6d5ef5]/20"
                                value={kpiConfig.column}
                                onChange={(e) => setKpiConfig({ ...kpiConfig, column: e.target.value })}
                            >
                                <option value="">Select a numeric column…</option>
                                {availableCols.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </Field>

                        <div className="grid grid-cols-2 gap-3">
                            <Field label="Aggregation">
                                <select
                                    className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-900 focus:border-[#6d5ef5] focus:outline-none focus:ring-2 focus:ring-[#6d5ef5]/20"
                                    value={kpiConfig.aggregation}
                                    onChange={(e) => setKpiConfig({ ...kpiConfig, aggregation: e.target.value })}
                                >
                                    <option value="sum">Sum</option>
                                    <option value="avg">Average</option>
                                    <option value="growth">Growth %</option>
                                </select>
                            </Field>
                            <Field label="Label">
                                <input
                                    type="text"
                                    placeholder="e.g. Total revenue"
                                    value={kpiConfig.label}
                                    onChange={(e) => setKpiConfig({ ...kpiConfig, label: e.target.value })}
                                    className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-900 placeholder:text-zinc-400 focus:border-[#6d5ef5] focus:outline-none focus:ring-2 focus:ring-[#6d5ef5]/20"
                                />
                            </Field>
                        </div>
                    </div>

                    <div className="mt-6 flex justify-end gap-2">
                        <button
                            onClick={() => setShowKPIModal(false)}
                            className="h-9 rounded-md border border-zinc-200 bg-white px-4 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={createCustomKPI}
                            disabled={creatingKPI || !kpiConfig.column}
                            className="inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0] disabled:opacity-50"
                        >
                            {creatingKPI && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                            Add metric
                        </button>
                    </div>
                </Modal>
            )}

        </div>
    );
}

/* ============================================================
   Subcomponents
   ============================================================ */

function DashTile({
    kind, title, explanation, children, onRemove,
}: {
    kind: "chart" | "metric";
    title?: string;
    explanation?: string;
    children: React.ReactNode;
    onRemove: () => void;
}) {
    return (
        <div className="group flex h-full flex-col rounded-xl border border-zinc-200 bg-white shadow-sm overflow-hidden">
            {/* Header — spans full width */}
            <div className="drag-handle flex cursor-grab items-center justify-between border-b border-zinc-100 px-4 py-2.5 active:cursor-grabbing shrink-0">
                <div className="flex items-center gap-2 min-w-0">
                    <GripVertical className="h-3.5 w-3.5 shrink-0 text-zinc-300" strokeWidth={1.75} />
                    <span className="truncate text-[13px] font-medium text-zinc-900">
                        {title ?? (kind === "metric" ? "Metric" : "Chart")}
                    </span>
                </div>
                <button
                    onClick={onRemove}
                    className="opacity-0 transition-opacity group-hover:opacity-100 rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700"
                    aria-label="Remove"
                >
                    <X className="h-3.5 w-3.5" strokeWidth={1.75} />
                </button>
            </div>

            {/* Body — side-by-side when explanation is present */}
            <div className="flex flex-1 min-h-0">
                <div className="flex-1 min-w-0 min-h-0 p-3">
                    {children}
                </div>

                {kind === "chart" && explanation && (
                    <div className="w-56 shrink-0 border-l border-indigo-100 bg-gradient-to-b from-indigo-50 to-white flex flex-col gap-3 p-4 overflow-y-auto">
                        <div className="flex items-center gap-1.5 shrink-0">
                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0" />
                            <span className="text-[10px] font-semibold text-indigo-600 uppercase tracking-wide">
                                AI Insight
                            </span>
                        </div>
                        <p className="text-xs text-gray-700 leading-relaxed">
                            {explanation}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

function MetricBlock({ content }: { content: string }) {
    // content shape: "Label: value (▲ 12%)"
    const colonIdx = content.indexOf(":");
    const label = colonIdx > -1 ? content.slice(0, colonIdx).trim() : "Metric";
    const rest = colonIdx > -1 ? content.slice(colonIdx + 1).trim() : content;
    const trendMatch = rest.match(/\(([▲▼])\s*([\d.]+%?)\)/);
    const value = rest.replace(/\(.*\)/, "").trim();
    const trend = trendMatch ? { dir: trendMatch[1], pct: trendMatch[2] } : null;

    return (
        <div className="flex h-full flex-col justify-center px-2">
            <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
            <div className="mt-1 text-2xl font-semibold tracking-tight text-zinc-900">{value}</div>
            {trend && (
                <div className={[
                    "mt-1 text-[12px] font-medium",
                    trend.dir === "▲" ? "text-emerald-600" : "text-red-600",
                ].join(" ")}>
                    {trend.dir} {trend.pct}
                </div>
            )}
        </div>
    );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
    return (
        <label className="block">
            <div className="mb-1.5 text-[12px] font-medium text-zinc-700">{label}</div>
            {children}
        </label>
    );
}

function Modal({
    title, children, onClose,
}: {
    title: string;
    children: React.ReactNode;
    onClose: () => void;
}) {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 backdrop-blur-sm p-4">
            <div className="w-full max-w-md rounded-xl border border-zinc-200 bg-white p-5 shadow-lg">
                <div className="mb-4 flex items-center justify-between">
                    <h3 className="text-[15px] font-semibold tracking-tight text-zinc-900">{title}</h3>
                    <button
                        onClick={onClose}
                        className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600"
                        aria-label="Close"
                    >
                        <X className="h-4 w-4" strokeWidth={1.75} />
                    </button>
                </div>
                {children}
            </div>
        </div>
    );
}

function EmptyState({ onAdd }: { onAdd: () => void }) {
    return (
        <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-zinc-200 bg-zinc-50">
                <BarChart3 className="h-5 w-5 text-zinc-500" strokeWidth={1.75} />
            </div>
            <h2 className="text-lg font-semibold tracking-tight text-zinc-900">Your dashboard is empty</h2>
            <p className="mt-1 max-w-sm text-[13px] text-zinc-600">
                Pin charts from the Insights page or add a custom metric to start building.
            </p>
            <div className="mt-5 flex gap-2">
                <Link
                    href="/insights"
                    className="inline-flex h-9 items-center rounded-md border border-zinc-200 bg-white px-4 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
                >
                    Browse insights
                </Link>
                <button
                    onClick={onAdd}
                    className="inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0]"
                >
                    <Plus className="h-3.5 w-3.5" strokeWidth={2} />
                    Add metric
                </button>
            </div>
        </div>
    );
}