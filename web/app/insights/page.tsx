"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    ArrowLeft,
    ArrowRight,
    Lightbulb,
    Target,
    Loader2,
    BarChart3,
    MessageSquare,
    CheckCircle,
    TrendingUp,
    TrendingDown,
    Minus,
    Download,
    X,
    Send,
    RefreshCw,
    Pin,
    LayoutDashboard,
    Sparkles,
    FileText,
    Plus,
    ImageUp
} from "lucide-react";

// Dynamic import for Plotly (no SSR)
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface InsightCard {
    title: string;
    description: string;
    chart_type: string;
    chart_data?: { labels: string[]; values: number[] };
    importance: string;
}

interface InsightsData {
    session_id: string;
    executive_summary: string;
    insights: InsightCard[];
    recommendations: string[];
}

interface ChartData {
    chart_id: string;
    chart_type: string;
    title: string;
    description: string;
    plotly_json: {
        data: Plotly.Data[];
        layout: Partial<Plotly.Layout>;
    };
    columns_used: string[];
    priority_score?: number;
    insight_reason?: string;
    interest_level?: "high" | "recommended" | "standard";
}

interface VizResponse {
    session_id: string;
    charts: ChartData[];
    total_generated: number;
}

interface ChatMsg {
    role: "user" | "assistant";
    content: string;
}

interface SessionColumnInfo {
    name: string;
    dtype: string;
}

interface KPIItem {
    label: string;
    column: string;
    value: number;
    formatted: string;
    avg: number;
    min: number;
    max: number;
    count: number;
    change_pct?: number;
    trend?: "up" | "down" | "flat";
    best_category?: { name: string; value: number };
    worst_category?: { name: string; value: number };
}

export default function InsightsPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<InsightsData | null>(null);
    const [vizData, setVizData] = useState<VizResponse | null>(null);
    const [loadingViz, setLoadingViz] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"insights" | "charts">("charts");
    const [isExporting, setIsExporting] = useState(false);

    // Chat State
    const [chatOpen, setChatOpen] = useState(false);
    const [messages, setMessages] = useState<ChatMsg[]>([
        { role: "assistant", content: "Hi! I can help you refine these charts. Try saying 'Show only bar charts' or 'Group by Region'." }
    ]);
    const [input, setInput] = useState("");
    const [chatLoading, setChatLoading] = useState(false);

    // Dashboard pinning
    const [pinnedCharts, setPinnedCharts] = useState<Set<string>>(new Set());

    // KPI state
    const [kpis, setKpis] = useState<KPIItem[]>([]);
    const [customKpiOpen, setCustomKpiOpen] = useState(false);
    const [customKpiLabel, setCustomKpiLabel] = useState("");
    const [customKpiColumn, setCustomKpiColumn] = useState("");
    const [customKpiAggregation, setCustomKpiAggregation] = useState<"sum" | "avg" | "growth">("sum");
    const [numericColumns, setNumericColumns] = useState<string[]>([]);

    // Report branding state
    const [reportLogo, setReportLogo] = useState<string | null>(null);

    // Chart explanation state
    const [explaining, setExplaining] = useState<string | null>(null);
    const [explanation, setExplanation] = useState<{
        pattern: string; importance: string; business_reason: string; risk_or_opportunity: string;
    } | null>(null);
    const [explainOpen, setExplainOpen] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }

        const session = JSON.parse(stored);
        fetchInsights(session.session_id);
        fetchVisualizations(session.session_id);
        fetchKpis(session.session_id);

        const numericDtypes = ["Float64", "Float32", "Int64", "Int32", "Int16", "Int8", "UInt64", "UInt32", "UInt16", "UInt8", "Decimal"];
        const sessionCols: SessionColumnInfo[] = Array.isArray(session.columns) ? session.columns : [];
        const detectedNumeric = sessionCols
            .filter((col) => typeof col?.dtype === "string" && numericDtypes.some((dt) => col.dtype.includes(dt)))
            .map((col) => col.name);
        setNumericColumns(detectedNumeric);

        // Load pinned charts from localStorage
        const pinned = localStorage.getItem(`pinned_${session.session_id}`);
        if (pinned) setPinnedCharts(new Set(JSON.parse(pinned)));

        const logo = localStorage.getItem(`report_logo_${session.session_id}`);
        if (logo) setReportLogo(logo);
    }, [router]);

    const togglePin = (chartId: string) => {
        setPinnedCharts(prev => {
            const next = new Set(prev);
            if (next.has(chartId)) next.delete(chartId);
            else next.add(chartId);
            const stored = localStorage.getItem("analysis_session");
            if (stored) {
                const session = JSON.parse(stored);
                localStorage.setItem(`pinned_${session.session_id}`, JSON.stringify([...next]));
            }
            return next;
        });
    };

    const fetchInsights = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/insights/${sessionId}`);
            if (!response.ok) throw new Error("Failed to fetch insights");
            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const fetchVisualizations = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/generate-viz/${sessionId}?max_charts=8`);
            if (!response.ok) throw new Error("Failed to fetch visualizations");
            const result = await response.json();
            setVizData(result);
        } catch (err) {
            console.error("Viz error:", err);
        } finally {
            setLoadingViz(false);
        }
    };

    const fetchKpis = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/kpis/${sessionId}`);
            if (response.ok) {
                const result = await response.json();
                setKpis(result.kpis || []);
            }
        } catch (err) {
            console.error("KPI fetch error:", err);
        }
    };


    const handleCreateCustomKpi = async () => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored || !customKpiColumn || !customKpiLabel.trim()) return;
        const session = JSON.parse(stored);

        try {
            const response = await fetch(`${API_BASE}/kpis/${session.session_id}/custom`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    column: customKpiColumn,
                    aggregation: customKpiAggregation,
                    label: customKpiLabel.trim(),
                }),
            });
            if (!response.ok) throw new Error("Failed to create custom KPI");
            const result = await response.json();
            if (result.kpi) setKpis((prev) => [result.kpi, ...prev]);
            setCustomKpiOpen(false);
            setCustomKpiLabel("");
            setCustomKpiColumn("");
            setCustomKpiAggregation("sum");
        } catch (err) {
            setError("Could not create custom KPI");
        }
    };

    const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            const content = String(reader.result || "");
            setReportLogo(content);
            const stored = localStorage.getItem("analysis_session");
            if (stored) {
                const session = JSON.parse(stored);
                localStorage.setItem(`report_logo_${session.session_id}`, content);
            }
        };
        reader.readAsDataURL(file);
    };

    const explainChart = async (chart: ChartData) => {
        setExplaining(chart.chart_id);
        setExplanation(null);
        setExplainOpen(true);
        try {
            const stored = localStorage.getItem("analysis_session");
            if (!stored) return;
            const session = JSON.parse(stored);
            const response = await fetch(`${API_BASE}/explain-chart/${session.session_id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    chart_id: chart.chart_id,
                    chart_type: chart.chart_type,
                    chart_title: chart.title,
                    columns_used: chart.columns_used,
                    data_summary: chart.insight_reason || chart.description,
                }),
            });
            if (response.ok) {
                const result = await response.json();
                setExplanation(result);
            }
        } catch (err) {
            console.error("Explain error:", err);
        } finally {
            setExplaining(null);
        }
    };

    const handleExport = async () => {
        setIsExporting(true);
        try {
            const stored = localStorage.getItem("analysis_session");
            if (!stored) return;
            const session = JSON.parse(stored);

            const response = await fetch(`${API_BASE}/export-excel/${session.session_id}`);
            if (!response.ok) throw new Error("Export failed");

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `InsightStream_Report_${session.session_id.slice(0, 8)}.xlsx`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            setError("Failed to export Excel report");
        } finally {
            setIsExporting(false);
        }
    };

    const handleChatSubmit = async (e?: React.FormEvent) => {
        if (e) e.preventDefault();
        if (!input.trim() || chatLoading) return;

        const userMsg = input.trim();
        setInput("");
        setMessages(prev => [...prev, { role: "user", content: userMsg }]);
        setChatLoading(true);

        try {
            const stored = localStorage.getItem("analysis_session");
            if (!stored) return;
            const session = JSON.parse(stored);

            const response = await fetch(`${API_BASE}/chat/${session.session_id}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: userMsg })
            });

            if (!response.ok) throw new Error("Chat failed");
            const data = await response.json();

            setMessages(prev => [...prev, { role: "assistant", content: data.answer }]);

            // Update charts if refinements were made
            if (data.chart_data && data.chart_data.charts) {
                setVizData(data.chart_data);
                setActiveTab("charts"); // Switch to charts tab to show changes
            }
        } catch (err) {
            setMessages(prev => [...prev, { role: "assistant", content: "Sorry, I encountered an error processing your request." }]);
        } finally {
            setChatLoading(false);
        }
    };

    const resetView = () => {
        const stored = localStorage.getItem("analysis_session");
        if (stored) {
            const session = JSON.parse(stored);
            fetchVisualizations(session.session_id);
            setMessages(prev => [...prev, { role: "assistant", content: "I've reset the view to the default analysis." }]);
        }
    };

    const getImportanceColor = (importance: string) => {
        switch (importance) {
            case "high": return "border-red-500/30 bg-red-500/5";
            case "medium": return "border-yellow-500/30 bg-yellow-500/5";
            default: return "border-blue-500/30 bg-blue-500/5";
        }
    };

    const renderMiniChart = (insight: InsightCard) => {
        if (!insight.chart_data || insight.chart_type === "none") return null;

        const { labels, values } = insight.chart_data;
        const maxVal = Math.max(...values);

        if (insight.chart_type === "bar") {
            return (
                <div className="mt-4 flex items-end gap-2 h-20">
                    {values.map((val, i) => (
                        <div key={i} className="flex-1 flex flex-col items-center gap-1">
                            <div
                                className="w-full bg-indigo-500 rounded-t"
                                style={{ height: `${(val / maxVal) * 100}%`, minHeight: '4px' }}
                            />
                            <span className="text-xs text-slate-500 truncate w-full text-center">{labels[i]?.slice(0, 8)}</span>
                        </div>
                    ))}
                </div>
            );
        }

        return null;
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-500 mx-auto mb-4" />
                    <p className="text-slate-400">Generating insights...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/eda" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <Lightbulb className="w-5 h-5" />
                        </div>
                        <span className="font-bold text-lg tracking-tight">Insights Engine</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Link
                            href="/dashboard"
                            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-white/10 rounded-lg text-sm font-medium transition-colors"
                        >
                            <LayoutDashboard className="w-4 h-4" />
                            Dashboard
                            {pinnedCharts.size > 0 && (
                                <span className="ml-1 px-1.5 py-0.5 text-xs bg-indigo-500 rounded-full">{pinnedCharts.size}</span>
                            )}
                        </Link>
                        <button
                            onClick={() => {
                                const stored = localStorage.getItem("analysis_session");
                                if (stored) {
                                    const session = JSON.parse(stored);
                                    const params = new URLSearchParams({
                                        project_name: session.project_name || session.filename || "InsightStream Project",
                                        report_title: "InsightStream Report",
                                    });
                                    if (reportLogo) params.set("logo_data", reportLogo);
                                    window.open(`${API_BASE}/export-report/${session.session_id}?${params.toString()}`, '_blank');
                                }
                            }}
                            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-white/10 rounded-lg text-sm font-medium transition-colors"
                        >
                            <FileText className="w-4 h-4" />
                            Export Report
                        </button>
                        <label className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 border border-white/10 rounded-lg text-sm font-medium transition-colors cursor-pointer">
                            <ImageUp className="w-4 h-4" />
                            Logo
                            <input type="file" accept="image/*" className="hidden" onChange={handleLogoUpload} />
                        </label>
                        <button
                            onClick={handleExport}
                            disabled={isExporting}
                            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                            Export Excel
                        </button>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-10 max-w-6xl">
                {error && (
                    <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                        {error}
                    </div>
                )}

                {data && (
                    <>
                        {/* Executive Summary */}
                        <div className="mb-8 p-6 rounded-2xl bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
                            <h2 className="text-sm font-medium text-indigo-400 uppercase tracking-wider mb-2">Executive Summary</h2>
                            <p className="text-lg text-white leading-relaxed">{data.executive_summary}</p>
                        </div>

                        {/* KPI Cards */}
                        <div className="mb-4">
                            <button
                                onClick={() => setCustomKpiOpen(true)}
                                className="inline-flex items-center gap-2 px-3 py-2 bg-slate-900 border border-white/10 hover:border-indigo-500/30 rounded-lg text-sm text-slate-200"
                            >
                                <Plus className="w-4 h-4" />
                                Add Custom KPI
                            </button>
                        </div>
                        {kpis.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4 mb-8">
                                {kpis.map((kpi, idx) => (
                                    <div
                                        key={`${kpi.label}-${kpi.column}-${idx}`}
                                        className="p-4 rounded-2xl bg-slate-900 border border-white/10 hover:border-indigo-500/20 transition-all duration-200 hover:shadow-lg hover:shadow-indigo-500/5"
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-xs text-slate-500 font-medium uppercase tracking-wider truncate">{kpi.label}</span>
                                            {kpi.trend === "up" && <TrendingUp className="w-4 h-4 text-emerald-400" />}
                                            {kpi.trend === "down" && <TrendingDown className="w-4 h-4 text-red-400" />}
                                            {kpi.trend === "flat" && <Minus className="w-4 h-4 text-slate-500" />}
                                        </div>
                                        <div className="text-2xl font-bold text-white mb-1">{kpi.formatted}</div>
                                        {kpi.change_pct !== undefined && kpi.change_pct !== 0 && (
                                            <div className={`text-xs font-medium ${kpi.change_pct > 0 ? "text-emerald-400" : "text-red-400"}`}>
                                                {kpi.change_pct > 0 ? "↑" : "↓"} {Math.abs(kpi.change_pct)}%
                                            </div>
                                        )}
                                        {kpi.best_category && (
                                            <div className="mt-2 text-xs text-slate-500">
                                                Best: <span className="text-slate-300">{kpi.best_category.name}</span>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Tab Selector */}
                        <div className="flex gap-2 mb-6">
                            <button
                                onClick={() => setActiveTab("charts")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === "charts"
                                    ? "bg-indigo-600 text-white"
                                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                                    }`}
                            >
                                <TrendingUp className="w-4 h-4" />
                                Advanced Charts
                                {vizData && <span className="text-xs opacity-70">({vizData.charts.length})</span>}
                            </button>
                            <button
                                onClick={() => setActiveTab("insights")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === "insights"
                                    ? "bg-indigo-600 text-white"
                                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                                    }`}
                            >
                                <BarChart3 className="w-4 h-4" />
                                Key Insights
                                <span className="text-xs opacity-70">({data.insights.length})</span>
                            </button>
                        </div>

                        {/* Advanced Charts Tab */}
                        {activeTab === "charts" && (
                            <div className="mb-10">
                                {loadingViz ? (
                                    <div className="flex items-center justify-center py-20">
                                        <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
                                        <span className="ml-3 text-slate-400">Generating visualizations...</span>
                                    </div>
                                ) : vizData && vizData.charts.length > 0 ? (
                                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                        {vizData.charts.map((chart) => (
                                            <div
                                                key={chart.chart_id}
                                                className="p-4 rounded-2xl bg-slate-900 border border-white/10"
                                            >
                                                <div className="mb-2">
                                                    <div className="flex items-start justify-between gap-2">
                                                        <h3 className="font-semibold text-white">{chart.title}</h3>
                                                        {chart.interest_level && chart.interest_level !== "standard" && (
                                                            <span
                                                                aria-label={`${chart.interest_level === "high" ? "High Insight" : "Smart Pick"} chart – ${chart.insight_reason || "Prioritized visualization"}`}
                                                                className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap flex items-center gap-1 cursor-default
                                                                    transition-all duration-150 ease-out hover:scale-105
                                                                    ${chart.interest_level === "high"
                                                                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:shadow-[0_0_8px_rgba(16,185,129,0.3)]"
                                                                        : "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:shadow-[0_0_8px_rgba(245,158,11,0.3)]"
                                                                    }`}>
                                                                {chart.interest_level === "high" ? "✨ High Insight" : "💡 Smart Pick"}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="text-sm text-slate-400">{chart.description}</p>
                                                    {chart.insight_reason && (
                                                        <div className="mt-2 p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 animate-fade-in">
                                                            <p className="text-xs text-slate-500 font-medium mb-1">Why prioritized:</p>
                                                            <ul className="text-xs text-slate-400 space-y-0.5">
                                                                {chart.insight_reason.split(" • ").map((reason, i) => (
                                                                    <li key={i} className="flex items-start gap-1.5">
                                                                        <span className="text-indigo-400 mt-0.5">•</span>
                                                                        <span>{reason}</span>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="rounded-xl overflow-hidden bg-slate-800">
                                                    <Plot
                                                        data={chart.plotly_json.data}
                                                        layout={{
                                                            ...chart.plotly_json.layout,
                                                            autosize: true,
                                                            height: 350,
                                                            margin: { l: 50, r: 30, t: 40, b: 50 },
                                                            paper_bgcolor: 'transparent',
                                                            plot_bgcolor: 'rgba(30,41,59,0.5)',
                                                            font: { color: '#94a3b8' }
                                                        }}
                                                        config={{
                                                            displayModeBar: true,
                                                            responsive: true,
                                                            displaylogo: false
                                                        }}
                                                        style={{ width: '100%' }}
                                                    />
                                                </div>
                                                <div className="mt-2 flex items-center justify-between">
                                                    <div className="flex flex-wrap gap-1">
                                                        {chart.columns_used.map((col, idx) => (
                                                            <span key={`${col}-${idx}`} className="text-xs px-2 py-0.5 bg-slate-700 rounded text-slate-300">
                                                                {col}
                                                            </span>
                                                        ))}
                                                    </div>
                                                    <div className="flex items-center gap-1.5">
                                                        <button
                                                            onClick={() => explainChart(chart)}
                                                            disabled={explaining === chart.chart_id}
                                                            className="flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg transition-all bg-slate-800 text-slate-500 hover:text-amber-400 border border-transparent hover:border-amber-500/20 disabled:opacity-50"
                                                        >
                                                            {explaining === chart.chart_id ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3" />}
                                                            Explain
                                                        </button>
                                                        <button
                                                            onClick={() => togglePin(chart.chart_id)}
                                                            className={`flex items-center gap-1 text-xs px-2.5 py-1 rounded-lg transition-all ${pinnedCharts.has(chart.chart_id)
                                                                ? "bg-indigo-500/20 text-indigo-400 border border-indigo-500/30"
                                                                : "bg-slate-800 text-slate-500 hover:text-indigo-400 border border-transparent hover:border-indigo-500/20"
                                                                }`}
                                                        >
                                                            <Pin className="w-3 h-3" />
                                                            {pinnedCharts.has(chart.chart_id) ? "Pinned" : "Pin"}
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="p-8 rounded-2xl bg-slate-900 border border-white/10 text-center">
                                        <TrendingUp className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                                        <p className="text-slate-400">No visualizations available for this dataset.</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Insights Tab */}
                        {activeTab === "insights" && (
                            <div className="mb-10">
                                {data.insights.length === 0 ? (
                                    <div className="p-8 rounded-2xl bg-slate-900 border border-white/10 text-center">
                                        <Lightbulb className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                                        <p className="text-slate-400">No significant insights detected. Try uploading a larger dataset.</p>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {data.insights.map((insight, i) => (
                                            <div
                                                key={i}
                                                className={`p-5 rounded-xl border ${getImportanceColor(insight.importance)}`}
                                            >
                                                <div className="flex items-start justify-between mb-2">
                                                    <h3 className="font-semibold text-white">{insight.title}</h3>
                                                    <span className={`text-xs px-2 py-0.5 rounded-full ${insight.importance === "high"
                                                        ? "bg-red-500/20 text-red-400"
                                                        : insight.importance === "medium"
                                                            ? "bg-yellow-500/20 text-yellow-400"
                                                            : "bg-blue-500/20 text-blue-400"
                                                        }`}>
                                                        {insight.importance}
                                                    </span>
                                                </div>
                                                <p className="text-slate-300 text-sm leading-relaxed">{insight.description}</p>
                                                {renderMiniChart(insight)}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Recommendations */}
                        {data.recommendations.length > 0 && (
                            <div className="mb-10">
                                <div className="flex items-center gap-2 mb-6">
                                    <Target className="w-5 h-5 text-green-400" />
                                    <h2 className="text-xl font-semibold">Recommendations</h2>
                                </div>
                                <div className="space-y-3">
                                    {data.recommendations.map((rec, i) => (
                                        <div
                                            key={i}
                                            className="p-4 rounded-xl bg-green-500/5 border border-green-500/20 flex items-start gap-3"
                                        >
                                            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                                            <p className="text-white">{rec}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Navigation */}
                        <div className="flex justify-between">
                            <button
                                onClick={() => router.push("/eda")}
                                className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 font-medium transition-colors"
                            >
                                Back to EDA
                            </button>
                            <button
                                onClick={() => router.push("/chat")}
                                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors"
                            >
                                <MessageSquare className="w-4 h-4" />
                                Chat with Data
                                <ArrowRight className="w-4 h-4" />
                            </button>
                        </div>
                    </>
                )}
            </main>
            {customKpiOpen && (
                <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
                    <div className="w-full max-w-md bg-slate-900 border border-white/10 rounded-2xl p-6">
                        <h3 className="text-lg font-semibold mb-4">Create Custom KPI</h3>
                        <div className="space-y-3">
                            <input
                                placeholder="KPI label (e.g. Gross Margin)"
                                value={customKpiLabel}
                                onChange={(e) => setCustomKpiLabel(e.target.value)}
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
                            />
                            <select
                                value={customKpiColumn}
                                onChange={(e) => setCustomKpiColumn(e.target.value)}
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
                            >
                                <option value="">Pick numeric column</option>
                                {numericColumns.map((column) => (
                                    <option key={column} value={column}>{column}</option>
                                ))}
                            </select>
                            <select
                                value={customKpiAggregation}
                                onChange={(e) => setCustomKpiAggregation(e.target.value as "sum" | "avg" | "growth")}
                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
                            >
                                <option value="sum">Sum</option>
                                <option value="avg">Average</option>
                                <option value="growth">Growth</option>
                            </select>
                        </div>
                        <div className="flex justify-end gap-2 mt-5">
                            <button onClick={() => setCustomKpiOpen(false)} className="px-4 py-2 text-slate-300">Cancel</button>
                            <button
                                onClick={handleCreateCustomKpi}
                                disabled={!customKpiColumn || !customKpiLabel.trim()}
                                className="px-4 py-2 bg-indigo-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                Create
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Chat Floating Action Button */}
            <button
                onClick={() => setChatOpen(!chatOpen)}
                className={`fixed bottom-6 right-6 p-4 rounded-full shadow-2xl transition-all duration-300 z-50 ${chatOpen ? "bg-red-500 rotate-90" : "bg-indigo-600 hover:bg-indigo-500"
                    }`}
            >
                {chatOpen ? <X className="w-6 h-6" /> : <MessageSquare className="w-6 h-6" />}
            </button>

            {/* Chat Window */}
            <div className={`fixed bottom-24 right-6 w-96 max-h-[600px] bg-slate-900 border border-white/10 rounded-2xl shadow-2xl z-40 flex flex-col transition-all duration-300 transform origin-bottom-right ${chatOpen ? "scale-100 opacity-100" : "scale-90 opacity-0 pointer-events-none"
                }`} style={{ height: 'calc(100vh - 150px)' }}>
                {/* Chat Header */}
                <div className="p-4 border-b border-white/10 flex items-center justify-between bg-slate-950/50 rounded-t-2xl">
                    <div className="flex items-center gap-2">
                        <MessageSquare className="w-5 h-5 text-indigo-400" />
                        <span className="font-semibold">Chart Assistant</span>
                    </div>
                    <button onClick={resetView} className="p-1.5 hover:bg-white/10 rounded-lg text-xs flex items-center gap-1 text-slate-400 hover:text-white transition-colors">
                        <RefreshCw className="w-3.5 h-3.5" />
                        Reset
                    </button>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                            <div className={`max-w-[85%] p-3 rounded-2xl text-sm leading-relaxed ${msg.role === "user"
                                ? "bg-indigo-600 text-white rounded-br-none"
                                : "bg-slate-800 text-slate-200 rounded-bl-none border border-white/5"
                                }`}>
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {chatLoading && (
                        <div className="flex justify-start">
                            <div className="bg-slate-800 p-3 rounded-2xl rounded-bl-none border border-white/5">
                                <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                            </div>
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <form onSubmit={handleChatSubmit} className="p-4 border-t border-white/10 bg-slate-950/50 rounded-b-2xl">
                    <div className="relative">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type a request (e.g., 'Group by Region')..."
                            className="w-full bg-slate-800 border border-white/10 rounded-xl pl-4 pr-12 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-slate-500"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || chatLoading}
                            className="absolute right-2 top-2 p-1.5 bg-indigo-600 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-indigo-500 transition-colors"
                        >
                            <Send className="w-4 h-4" />
                        </button>
                    </div>
                </form>
            </div>

            {/* AI Explanation Panel */}
            {explainOpen && (
                <div className="fixed inset-0 z-50 flex items-end justify-center">
                    <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setExplainOpen(false)} />
                    <div className="relative w-full max-w-2xl bg-slate-900 border border-white/10 rounded-t-2xl p-6 animate-slide-up">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Sparkles className="w-5 h-5 text-amber-400" />
                                <h3 className="font-semibold text-white">AI Chart Explanation</h3>
                            </div>
                            <button onClick={() => setExplainOpen(false)} className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        {!explanation ? (
                            <div className="flex items-center justify-center py-12">
                                <Loader2 className="w-6 h-6 animate-spin text-indigo-500 mr-3" />
                                <span className="text-slate-400">Analyzing chart with AI...</span>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                {explanation.pattern && (
                                    <div className="p-3 rounded-xl bg-slate-800/50 border border-white/5">
                                        <div className="text-xs text-indigo-400 font-medium mb-1">📊 Pattern</div>
                                        <p className="text-sm text-slate-300">{explanation.pattern}</p>
                                    </div>
                                )}
                                {explanation.importance && (
                                    <div className="p-3 rounded-xl bg-slate-800/50 border border-white/5">
                                        <div className="text-xs text-emerald-400 font-medium mb-1">⭐ Importance</div>
                                        <p className="text-sm text-slate-300">{explanation.importance}</p>
                                    </div>
                                )}
                                {explanation.business_reason && (
                                    <div className="p-3 rounded-xl bg-slate-800/50 border border-white/5">
                                        <div className="text-xs text-amber-400 font-medium mb-1">💡 Business Reason</div>
                                        <p className="text-sm text-slate-300">{explanation.business_reason}</p>
                                    </div>
                                )}
                                {explanation.risk_or_opportunity && (
                                    <div className="p-3 rounded-xl bg-slate-800/50 border border-white/5">
                                        <div className="text-xs text-rose-400 font-medium mb-1">🎯 Risk / Opportunity</div>
                                        <p className="text-sm text-slate-300">{explanation.risk_or_opportunity}</p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
