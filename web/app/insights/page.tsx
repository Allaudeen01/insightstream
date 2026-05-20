// web/app/insights/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    Loader2,
    Download,
    Sparkles,
    Share2,
    Check,
    ChevronRight,
    TrendingUp,
    Lightbulb,
    AlertCircle,
    LayoutDashboard,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import { apiFetch } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ----- Plotly light theme (shared with /dashboard) -----
const PLOT_LAYOUT = {
    autosize: true,
    paper_bgcolor: "transparent",
    plot_bgcolor: "#fafafa",
    margin: { t: 12, r: 16, b: 36, l: 44 },
    font: { family: "Inter, system-ui, sans-serif", color: "#3f3f46", size: 12 },
    xaxis: {
        gridcolor: "#e4e4e7",
        zerolinecolor: "#d4d4d8",
        tickfont: { color: "#52525b" },
        linecolor: "#d4d4d8"
    },
    yaxis: {
        gridcolor: "#e4e4e7",
        zerolinecolor: "#d4d4d8",
        tickfont: { color: "#52525b" },
        linecolor: "#d4d4d8"
    },
    colorway: ["#6d5ef5", "#0ea5e9", "#f59e0b", "#ec4899", "#10b981", "#8b5cf6"],
};

// ----- Confidence based on row count (plain copy) -----
function getConfidenceTier(rowCount: number) {
    if (rowCount < 100) return { label: "Low confidence", tone: "text-red-700", dot: "bg-red-500" };
    if (rowCount < 500) return { label: "Medium confidence", tone: "text-amber-700", dot: "bg-amber-500" };
    return { label: "High confidence", tone: "text-emerald-700", dot: "bg-emerald-500" };
}

// ----- Types -----
interface InsightData {
    title: string;
    description: string;
    chart_type: string;
    impact: "High" | "Medium" | "Low";
    decision_implication: string;
    recommendation?: string;
}
interface RecommendationCard {
    priority: number;
    action: string;
    timeframe: string;
    owner: string;
    linked_insight?: string;
    impact?: string;
}
interface InsightsData {
    session_id: string;
    executive_summary: string;
    strategic_brief: InsightData[];
    recommendations: RecommendationCard[];
    computed_metrics?: Record<string, any>;
}

export default function InsightsPage() {
    const router = useRouter();

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [data, setData] = useState<InsightsData | null>(null);
    const [vizData, setVizData] = useState<any>(null);
    const [activeTab, setActiveTab] = useState<"insights" | "charts" | "chat">("charts");
    const [copied, setCopied] = useState(false);

    // Branding inputs (used by export)
    const [projectName, setProjectName] = useState("");
    const [reportTitle, setReportTitle] = useState("InsightStream analysis report");
    const [sessionFilename, setSessionFilename] = useState("");

    // Export state
    const [pinnedCharts, setPinnedCharts] = useState<string[]>([]);
    const [showExportModal, setShowExportModal] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [exportStatus, setExportStatus] = useState("");

    // ----- Load session + data -----
    useEffect(() => {
        // Prefer ?session= URL param, fall back to localStorage
        const urlParams = new URLSearchParams(window.location.search);
        const urlSid = urlParams.get("session");

        const stored = localStorage.getItem("analysis_session");
        const cached = stored ? JSON.parse(stored) : null;

        const sid = urlSid || cached?.session_id;
        if (!sid) { router.push("/upload"); return; }

        const name = (cached?.filename || cached?.original_filename || String(sid)).split(".")[0];
        setProjectName(name);
        setSessionFilename(cached?.filename || cached?.original_filename || "");
        fetchData(String(sid), cached && String(cached.session_id) === String(sid) ? cached : null);
    }, [router]);

    const fetchData = async (sid: string, cachedSession: any) => {
        try {
            setLoading(true);
            setError(null);

            // ── Insight data ─────────────────────────────────────────────────
            if (cachedSession) {
                // Full engine output is already in localStorage — use it directly.
                setData(cachedSession);
                localStorage.setItem(`insights_${sid}`, JSON.stringify(cachedSession));
            } else {
                // Direct URL navigation — fetch from DB and map to display shape.
                const res = await apiFetch(`/sessions/${sid}`);
                if (!res.ok) {
                    if (res.status === 401) { router.push("/login"); return; }
                    if (res.status === 404) {
                        localStorage.removeItem("analysis_session");
                        setError("session_expired");
                        return;
                    }
                    throw new Error(`Session fetch failed: ${res.status}`);
                }
                const sessionData = await res.json();
                const mapped: InsightsData = {
                    session_id: String(sessionData.id),
                    executive_summary: "",
                    strategic_brief: (sessionData.insights ?? []).map((i: any) => ({
                        title: i.title,
                        description: i.body,
                        chart_type: i.chart_type || "bar",
                        impact: i.impact as "High" | "Medium" | "Low",
                        decision_implication: "",
                        rule_type: i.rule_type || "",
                        chart_data: (() => { try { return JSON.parse(i.evidence_json || "{}"); } catch { return {}; } })(),
                    })),
                    recommendations: (sessionData.recommendations ?? []).map((r: any) => ({
                        priority: (r.display_order ?? 0) + 1,
                        action: r.action,
                        timeframe: r.timeframe,
                        owner: r.owner,
                        impact: r.impact,
                    })),
                    computed_metrics: sessionData.kpis_json ? JSON.parse(sessionData.kpis_json) : {},
                };
                setData(mapped);
            }

            // ── Charts (generated from the uploaded file on disk) ────────────
            const vizRes = await apiFetch(`/generate-viz/${sid}?max_charts=12`);
            if (vizRes.ok) {
                setVizData(await vizRes.json());
            }
            // Charts failing is non-fatal — insights still display

            // ── Seed pinned state from dashboard backend ─────────────────────
            const dashRes = await apiFetch(`/dashboard/${sid}`);
            if (dashRes.ok) {
                const dashData = await dashRes.json();
                setPinnedCharts(dashData.pinned_chart_ids || []);
            }
        } catch (err: any) {
            console.error("Insights fetch error:", err);
            if (err instanceof TypeError && err.message === "Failed to fetch") {
                setError("backend_unreachable");
            } else {
                setError(err.message || "unknown_error");
            }
        } finally {
            setLoading(false);
        }
    };

    // ----- Pin chart to dashboard -----
    const togglePin = async (chartId: string) => {
        const sid = data?.session_id;
        if (!sid) return;
        const alreadyPinned = pinnedCharts.includes(chartId);
        const next = alreadyPinned
            ? pinnedCharts.filter(id => id !== chartId)
            : [...pinnedCharts, chartId];
        setPinnedCharts(next);
        try {
            await fetch(`${API_BASE}/dashboard/${sid}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ pinned_chart_ids: next, layout: [], text_blocks: [] }),
            });
        } catch {
            setPinnedCharts(pinnedCharts);
        }
    };

    // ----- Share link -----
    const copyShareLink = () => {
        const sid = data?.session_id;
        if (!sid) return;
        const link = `${window.location.origin}/share/dashboard/${sid}`;
        navigator.clipboard.writeText(link);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    // ----- PDF export (logic preserved) -----
    const handleProfessionalExport = async () => {
        if (!data?.session_id) {
            alert("Analysis session not loaded yet. Please wait.");
            return;
        }
        setIsExporting(true);
        setExportProgress(5);
        setExportStatus("Preparing report…");

        try {
            const sid = data.session_id;
            const chartAssets: any[] = [];
            const chartsToExport = vizData?.charts || [];

            setExportProgress(15);
            setExportStatus("Loading chart engine…");

            // Switch to charts tab so Plotly elements are mounted in the DOM.
            // Without this, Plotly.toImage returns "Element not found" and charts
            // fall back to the server-side renderer.
            setActiveTab("charts");
            await new Promise(resolve => setTimeout(resolve, 700));

            let Plotly = (window as any).Plotly;
            if (!Plotly) {
                try {
                    const mod = await import("plotly.js-dist-min" as any);
                    Plotly = mod.default || mod;
                } catch {/* per-chart error */ }
            }

            for (let i = 0; i < chartsToExport.length; i++) {
                const chart = chartsToExport[i];
                setExportStatus(`Capturing ${chart.title}…`);

                const container = document.querySelector(`[data-chart-id="${chart.chart_id}"]`);
                const plotlyEl = container?.querySelector(".js-plotly-plot");
                let image_base64 = "";
                let error = "";

                // Scroll chart into view before capturing to ensure it's fully rendered
                if (container) {
                    container.scrollIntoView({ behavior: "instant", block: "center" });
                    // Wait for render after scroll
                    await new Promise(resolve => setTimeout(resolve, 300));
                }

                if (plotlyEl && Plotly) {
                    try {
                        image_base64 = await Plotly.toImage(plotlyEl, {
                            format: "png", width: 1200, height: 700, scale: 2,
                        });
                    } catch (e: any) { error = e.message; }
                } else {
                    error = !plotlyEl ? "Element not found" : "Plotly missing";
                }

                chartAssets.push({
                    id: chart.chart_id,
                    title: chart.title,
                    image_base64,
                    error,
                    insight: chart.description || "",
                    // Included so the backend can render via kaleido when base64 capture fails
                    plotly_json: chart.plotly_json || null,
                });

                setExportProgress(20 + Math.floor(((i + 1) / chartsToExport.length) * 50));
            }

            setExportStatus("Building PDF…");
            setExportProgress(80);

            const template = localStorage.getItem("report_template") || "modern";
            const res = await fetch(`${API_BASE}/export-dashboard-pdf/${sid}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    title: reportTitle,
                    project_name: projectName,
                    template,
                    kpis: data.computed_metrics || {},
                    charts: chartAssets,
                    ai_summary: data.executive_summary || "",
                    insights: (data.strategic_brief || []).map((i: any) =>
                        typeof i === "string" ? { title: i, body: i, rule_type: "", chart_data: {} } : {
                            rule_type: i.rule_type || "",
                            title: i.title || "",
                            body: i.description || i.body || "",
                            impact: i.impact || "minor",
                            chart_data: i.chart_data || {},
                        }
                    ).filter((i: any) => i.title || i.body),
                    recommendations: data.recommendations || [],
                    text_blocks: [],
                }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                const detail = errData.detail;
                const msg = Array.isArray(detail)
                    ? detail.map((e: any) => e.msg ?? JSON.stringify(e)).join("; ")
                    : typeof detail === "string"
                        ? detail
                        : "Report generation failed";
                throw new Error(msg);
            }

            setExportStatus("Downloading…");
            setExportProgress(95);

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `InsightStream_${projectName.replace(/\s+/g, "_")}_Report.pdf`;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);

            setExportProgress(100);
            setTimeout(() => {
                setShowExportModal(false);
                setIsExporting(false);
                setExportProgress(0);
                setExportStatus("");
            }, 600);
        } catch (err: any) {
            alert(`Export failed: ${err.message}`);
            setIsExporting(false);
            setExportStatus("");
        }
    };

    // ----- Loading -----
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

    // ----- Error states -----
    if (error) {
        const isExpired = error === "session_expired";
        const isUnreachable = error === "backend_unreachable";
        const isTimeout = error === "timeout";
        return (
            <div className="flex h-screen bg-white">
                <Sidebar />
                <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-50">
                        <AlertCircle className="h-6 w-6 text-red-500" strokeWidth={1.75} />
                    </div>
                    <div>
                        <h2 className="text-[17px] font-semibold text-zinc-900">
                            {isExpired ? "Analysis session expired" :
                             isUnreachable ? "Backend unreachable" :
                             isTimeout ? "Analysis taking longer than expected" :
                             "Something went wrong"}
                        </h2>
                        <p className="mt-1.5 max-w-sm text-[13px] text-zinc-500">
                            {isExpired
                                ? "The server was restarted and your previous session was cleared. Please re-upload your file to run a fresh analysis."
                                : isUnreachable
                                ? `Could not connect to ${API_BASE}. Make sure the Python backend is running (python main.py).`
                                : isTimeout
                                ? "Your dataset is taking longer to analyze than expected. Try uploading a smaller file or use the 'Analyze' button on the dashboard for background processing."
                                : error}
                        </p>
                    </div>
                    <button
                        onClick={() => router.push("/upload")}
                        className="mt-2 inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0]"
                    >
                        Upload new file
                    </button>
                </div>
            </div>
        );
    }

    const charts = vizData?.charts || [];
    const totalRecords = data?.computed_metrics?.Records?.value || 0;
    const tier = getConfidenceTier(totalRecords);

    return (
        <div className="flex h-screen bg-white text-zinc-900 antialiased">
            <Sidebar />

            <div className="flex min-w-0 flex-1 flex-col">
                {/* Top bar */}
                <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-4 md:px-6 backdrop-blur">
                    <div className="flex items-center gap-3 min-w-0 pl-8 md:pl-0">
                        <h1 className="truncate text-[17px] font-semibold tracking-[-0.01em]">
                            {projectName}
                        </h1>
                        <span className="text-zinc-300">/</span>
                        <span className="text-[13px] text-zinc-500">Insights</span>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={copyShareLink}
                            className="hidden sm:inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
                        >
                            {copied
                                ? <Check className="h-3.5 w-3.5 text-emerald-600" strokeWidth={2} />
                                : <Share2 className="h-3.5 w-3.5" strokeWidth={1.75} />}
                            {copied ? "Copied" : "Share"}
                        </button>
                        <button
                            onClick={() => setShowExportModal(true)}
                            className="inline-flex h-8 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-3 text-[13px] font-medium text-white hover:bg-[#5b4be0]"
                        >
                            <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
                            Export PDF
                        </button>
                    </div>
                </header>

                {/* Body */}
                <main className="flex-1 overflow-auto">
                    <div className="mx-auto max-w-[1200px] px-4 md:px-6 py-4 md:py-6">
                        {/* Executive summary */}
                        {data?.executive_summary && (
                            <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
                                <div className="flex items-center gap-2 text-[12px] font-medium text-[#6d5ef5]">
                                    <Sparkles className="h-3.5 w-3.5" strokeWidth={1.75} />
                                    AI summary
                                </div>
                                <p className="mt-3 text-[15px] leading-relaxed text-zinc-800">
                                    {data.executive_summary}
                                </p>
                                <div className="mt-4 inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-[12px] font-medium text-zinc-700">
                                    <span className={`h-1.5 w-1.5 rounded-full ${tier.dot}`} />
                                    <span className={tier.tone}>{tier.label}</span>
                                    <span className="text-zinc-400">·</span>
                                    <span className="text-zinc-500">
                                        Based on {totalRecords.toLocaleString("en-IN")} records
                                    </span>
                                </div>
                            </section>
                        )}

                        {/* Tabs */}
                        <div className="mt-6 flex h-9 items-center gap-1 border-b border-zinc-200">
                            <TabBtn active={activeTab === "charts"} onClick={() => setActiveTab("charts")}>
                                Visualizations
                                <Pill>{charts.length}</Pill>
                            </TabBtn>
                            <TabBtn active={activeTab === "insights"} onClick={() => setActiveTab("insights")}>
                                Insights
                                <Pill>{data?.strategic_brief?.length ?? 0}</Pill>
                            </TabBtn>
                            <TabBtn active={activeTab === "chat"} onClick={() => setActiveTab("chat")}>
                                Chat
                            </TabBtn>
                        </div>

                        {/* Tab content */}
                        <div className="pt-6">
                            {activeTab === "chat" && (
                                data?.session_id ? (
                                    <div className="h-[600px]">
                                        <ChatPanel
                                            sessionId={Number(data.session_id)}
                                            filename={sessionFilename || projectName || "your dataset"}
                                        />
                                    </div>
                                ) : (
                                    <div className="flex items-center justify-center h-64 text-zinc-400 text-sm">
                                        Select an analysis from the sidebar to start chatting
                                    </div>
                                )
                            )}
                            {activeTab === "charts" ? (
                                charts.length === 0 ? (
                                    <EmptyState
                                        icon={TrendingUp}
                                        title="No charts generated"
                                        body="We couldn't generate any visualizations from this dataset. Try uploading a file with at least one numeric column."
                                    />
                                ) : (
                                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                                        {charts.map((chart: any, i: number) => {
                                            const isLast = i === charts.length - 1;
                                            const isOdd = charts.length % 2 === 1;
                                            const span = isLast && isOdd ? "lg:col-span-2" : "";
                                            return (
                                                <div
                                                    key={chart.chart_id}
                                                    data-chart-id={chart.chart_id}
                                                    className={`rounded-xl border border-zinc-200 bg-white shadow-sm ${span}`}
                                                >
                                                    <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
                                                        <h3 className="truncate text-[14px] font-medium text-zinc-900">
                                                            {chart.title}
                                                        </h3>
                                                        <button
                                                            onClick={() => togglePin(chart.chart_id)}
                                                            title={pinnedCharts.includes(chart.chart_id) ? "Unpin from dashboard" : "Pin to dashboard"}
                                                            className={[
                                                                "ml-3 shrink-0 rounded-md p-1.5 transition-colors",
                                                                pinnedCharts.includes(chart.chart_id)
                                                                    ? "bg-indigo-100 text-indigo-600"
                                                                    : "text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700",
                                                            ].join(" ")}
                                                        >
                                                            <LayoutDashboard className="h-3.5 w-3.5" strokeWidth={1.75} />
                                                        </button>
                                                    </div>
                                                    <div className="p-3">
                                                        <div className="h-[340px]">
                                                            <Plot
                                                                data={chart.plotly_json.data.map((trace: any) => ({
                                                                    ...trace,
                                                                    marker: {
                                                                        ...trace.marker,
                                                                        color: trace.marker?.color ?? "#6d5ef5",
                                                                        opacity: 1,
                                                                    },
                                                                }))}
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
                                                        {chart.description && (
                                                            <p className="mt-2 px-1 text-[13px] leading-relaxed text-zinc-600">
                                                                {chart.description}
                                                            </p>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )
                            ) : (
                                (data?.strategic_brief?.length ?? 0) === 0 ? (
                                    <EmptyState
                                        icon={Lightbulb}
                                        title="No insights yet"
                                        body="Insights appear once the analysis finishes processing your data."
                                    />
                                ) : (
                                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                                        {data!.strategic_brief.map((ins, i) => (
                                            <InsightCard
                                                key={i}
                                                title={ins.title}
                                                impact={ins.impact}
                                                description={ins.description}
                                                recommendation={ins.recommendation || ins.decision_implication}
                                            />
                                        ))}
                                    </div>
                                )
                            )}

                            {/* Recommendations strip — insights tab only */}
                            {activeTab === "insights" && (data?.recommendations?.length ?? 0) > 0 && (
                                <section className="mt-8">
                                    <h2 className="mb-3 text-[14px] font-semibold tracking-[-0.01em] text-zinc-900">
                                        Recommended actions
                                    </h2>
                                    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
                                        {data!.recommendations.map((rec, i) => (
                                            <div
                                                key={i}
                                                className={[
                                                    "flex items-start gap-4 px-5 py-4",
                                                    i > 0 && "border-t border-zinc-100",
                                                ].filter(Boolean).join(" ")}
                                            >
                                                <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-100 text-[12px] font-semibold text-zinc-700">
                                                    {rec.priority || i + 1}
                                                </div>
                                                <div className="min-w-0 flex-1">
                                                    <div className="text-[14px] font-medium text-zinc-900">
                                                        {rec.action}
                                                    </div>
                                                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-zinc-500">
                                                        {rec.owner && <span>Owner: {rec.owner}</span>}
                                                        {rec.timeframe && <span>Timeframe: {rec.timeframe}</span>}
                                                        {rec.impact && <span>Impact: {rec.impact}</span>}
                                                    </div>
                                                </div>
                                                <ChevronRight className="mt-1 h-4 w-4 text-zinc-300" strokeWidth={1.75} />
                                            </div>
                                        ))}
                                    </div>
                                </section>
                            )}
                        </div>
                    </div>
                </main>
            </div>

            {/* Export modal */}
            {showExportModal && (
                <Modal title="Export report" onClose={() => !isExporting && setShowExportModal(false)}>
                    <div className="space-y-4">
                        <Field label="Project name">
                            <input
                                type="text"
                                value={projectName}
                                onChange={(e) => setProjectName(e.target.value)}
                                disabled={isExporting}
                                className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-900 focus:border-[#6d5ef5] focus:outline-none focus:ring-2 focus:ring-[#6d5ef5]/20 disabled:opacity-50"
                            />
                        </Field>
                        <Field label="Report title">
                            <input
                                type="text"
                                value={reportTitle}
                                onChange={(e) => setReportTitle(e.target.value)}
                                disabled={isExporting}
                                className="w-full rounded-md border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-900 focus:border-[#6d5ef5] focus:outline-none focus:ring-2 focus:ring-[#6d5ef5]/20 disabled:opacity-50"
                            />
                        </Field>
                    </div>

                    {isExporting ? (
                        <div className="mt-5">
                            <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-100">
                                <div
                                    className="h-full bg-[#6d5ef5] transition-all duration-500"
                                    style={{ width: `${exportProgress}%` }}
                                />
                            </div>
                            <div className="mt-2 flex justify-between text-[12px] text-zinc-500">
                                <span>{exportStatus || "Working…"}</span>
                                <span className="tabular-nums">{exportProgress}%</span>
                            </div>
                        </div>
                    ) : (
                        <div className="mt-6 flex justify-end gap-2">
                            <button
                                onClick={() => setShowExportModal(false)}
                                className="h-9 rounded-md border border-zinc-200 bg-white px-4 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleProfessionalExport}
                                className="inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0]"
                            >
                                <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
                                Generate PDF
                            </button>
                        </div>
                    )}
                </Modal>
            )}
        </div>
    );
}

/* ============================================================
   Subcomponents
   ============================================================ */

function TabBtn({
    active, onClick, children,
}: {
    active: boolean;
    onClick: () => void;
    children: React.ReactNode;
}) {
    return (
        <button
            onClick={onClick}
            className={[
                "relative inline-flex h-9 items-center gap-1.5 px-3 text-[13px] font-medium transition-colors",
                active ? "text-zinc-900" : "text-zinc-500 hover:text-zinc-900",
            ].join(" ")}
        >
            {children}
            {active && (
                <span className="absolute inset-x-2 -bottom-px h-[2px] rounded-full bg-[#6d5ef5]" />
            )}
        </button>
    );
}

function Pill({ children }: { children: React.ReactNode }) {
    return (
        <span className="inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-zinc-100 px-1.5 text-[11px] font-medium text-zinc-600">
            {children}
        </span>
    );
}

function InsightCard({
    title, impact, description, recommendation,
}: {
    title: string;
    impact: "High" | "Medium" | "Low";
    description: string;
    recommendation?: string;
}) {
    const impactStyle: Record<string, string> = {
        High: "bg-red-50 text-red-700 border-red-200",
        Medium: "bg-amber-50 text-amber-700 border-amber-200",
        Low: "bg-sky-50 text-sky-700 border-sky-200",
    };
    return (
        <article className="flex flex-col rounded-xl border border-zinc-200 bg-white p-5 shadow-sm transition-colors hover:border-zinc-300">
            <header className="mb-3 flex items-start justify-between gap-3">
                <h3 className="text-[15px] font-semibold leading-snug tracking-[-0.01em] text-zinc-900">
                    {title}
                </h3>
                <span
                    className={[
                        "shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-medium",
                        impactStyle[impact] ?? "bg-zinc-50 text-zinc-700 border-zinc-200",
                    ].join(" ")}
                >
                    {impact} impact
                </span>
            </header>

            <p className="text-[13px] leading-relaxed text-zinc-600">{description}</p>

            {recommendation && (
                <div className="mt-4 flex gap-2.5 rounded-lg bg-[#f1efff] px-3.5 py-3">
                    <Lightbulb className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#6d5ef5]" strokeWidth={1.75} />
                    <p className="text-[13px] leading-relaxed text-zinc-800">{recommendation}</p>
                </div>
            )}
        </article>
    );
}

function EmptyState({
    icon: Icon, title, body,
}: {
    icon: typeof TrendingUp;
    title: string;
    body: string;
}) {
    return (
        <div className="flex min-h-[40vh] flex-col items-center justify-center text-center">
            <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl border border-zinc-200 bg-zinc-50">
                <Icon className="h-5 w-5 text-zinc-500" strokeWidth={1.75} />
            </div>
            <h2 className="text-[15px] font-semibold tracking-tight text-zinc-900">{title}</h2>
            <p className="mt-1 max-w-sm text-[13px] text-zinc-600">{body}</p>
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
                        <AlertCircle className="hidden" />
                        <span className="block h-4 w-4 leading-none">×</span>
                    </button>
                </div>
                {children}
            </div>
        </div>
    );
}