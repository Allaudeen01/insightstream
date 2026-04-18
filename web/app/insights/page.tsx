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
    FileText
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
    // New smart engine fields
    impact?: string;           // "high" | "medium" | "low"
    recommendation?: string;   // Actionable next step
}

interface InsightsData {
    session_id: string;
    executive_summary: string;
    insights: InsightCard[];
    recommendations: string[];
    // New smart engine fields
    warnings?: string[];
    computed_metrics?: Record<string, {
        name: string;
        value: number;
        formatted: string;
        description: string;
    }>;
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

    // Chart explanation state
    const [explaining, setExplaining] = useState<string | null>(null);
    const [explanation, setExplanation] = useState<{
        pattern: string; importance: string; business_reason: string; risk_or_opportunity: string;
    } | null>(null);
    const [explainOpen, setExplainOpen] = useState(false);

    // ═══════════ PERFORMANCE: Background analysis + progress ═══════════
    const [analysisProgress, setAnalysisProgress] = useState(0);
    const [analysisStage, setAnalysisStage] = useState<string>("");

    // sessionStorage cache key helper
    const cacheKey = (sessionId: string, kind: string) => `is_cache_${sessionId}_${kind}`;

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }

        const session = JSON.parse(stored);
        const sid = session.session_id;

        // ── Instant render from sessionStorage if available ──────
        const cachedInsights = sessionStorage.getItem(cacheKey(sid, "insights"));
        const cachedKpis     = sessionStorage.getItem(cacheKey(sid, "kpis"));
        const cachedViz      = sessionStorage.getItem(cacheKey(sid, "viz"));

        if (cachedInsights) {
            try { setData(JSON.parse(cachedInsights)); setLoading(false); } catch { /* ignore */ }
        }
        if (cachedKpis) {
            try { setKpis(JSON.parse(cachedKpis)); } catch { /* ignore */ }
        }
        if (cachedViz) {
            try { setVizData(JSON.parse(cachedViz)); setLoadingViz(false); } catch { /* ignore */ }
        }

        // ── Fire background analysis + parallel fetches ─────────
        startBackgroundAnalysis(sid);

        // Load pinned charts from localStorage
        const pinned = localStorage.getItem(`pinned_${sid}`);
        if (pinned) setPinnedCharts(new Set(JSON.parse(pinned)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [router]);

    // ── Background analysis: POST /analyze then poll /analyze-status ──
    const startBackgroundAnalysis = async (sessionId: string) => {
        try {
            // 1. Kick off background analysis
            setAnalysisStage("Starting analysis...");
            const analyzeRes = await fetch(`${API_BASE}/analyze/${sessionId}`, { method: "POST" });
            const analyzeData = await analyzeRes.json();

            if (analyzeData.status === "done") {
                // Already cached on server — just fetch results in parallel
                setAnalysisProgress(100);
                setAnalysisStage("");
                fetchAllParallel(sessionId);
                return;
            }

            // 2. Poll for progress
            setAnalysisStage("Classifying columns...");
            const pollInterval = setInterval(async () => {
                try {
                    const statusRes = await fetch(`${API_BASE}/analyze-status/${sessionId}`);
                    const status = await statusRes.json();

                    setAnalysisProgress(status.progress || 0);

                    // Map progress to stage labels
                    if (status.progress < 15) setAnalysisStage("Reading dataset...");
                    else if (status.progress < 30) setAnalysisStage("Classifying columns...");
                    else if (status.progress < 50) setAnalysisStage("Computing business metrics...");
                    else if (status.progress < 65) setAnalysisStage("Evaluating business rules...");
                    else if (status.progress < 80) setAnalysisStage("Generating narratives...");
                    else if (status.progress < 95) setAnalysisStage("Building visualizations...");
                    else setAnalysisStage("Finalizing...");

                    if (status.status === "done") {
                        clearInterval(pollInterval);
                        setAnalysisStage("");
                        fetchAllParallel(sessionId);
                    } else if (status.status === "error") {
                        clearInterval(pollInterval);
                        setAnalysisStage("");
                        setError(status.error || "Analysis failed");
                        setLoading(false);
                        setLoadingViz(false);
                    }
                } catch {
                    clearInterval(pollInterval);
                }
            }, 750);

            // Safety: clear after 60s max
            setTimeout(() => clearInterval(pollInterval), 60_000);

        } catch {
            // Fallback: if /analyze fails, fetch synchronously
            fetchAllParallel(sessionId);
        }
    };

    // ── Parallel fetch: all 3 endpoints at once via Promise.allSettled ──
    const fetchAllParallel = async (sessionId: string) => {
        const [insightsRes, kpisRes, vizRes] = await Promise.allSettled([
            fetch(`${API_BASE}/insights/${sessionId}`),
            fetch(`${API_BASE}/kpis/${sessionId}`),
            fetch(`${API_BASE}/generate-viz/${sessionId}?max_charts=8`),
        ]);

        // Insights
        if (insightsRes.status === "fulfilled" && insightsRes.value.ok) {
            const result = await insightsRes.value.json();
            setData(result);
            sessionStorage.setItem(cacheKey(sessionId, "insights"), JSON.stringify(result));
        } else {
            setError("Failed to fetch insights");
        }
        setLoading(false);

        // KPIs
        if (kpisRes.status === "fulfilled" && kpisRes.value.ok) {
            const result = await kpisRes.value.json();
            setKpis(result.kpis || []);
            sessionStorage.setItem(cacheKey(sessionId, "kpis"), JSON.stringify(result.kpis || []));
        }

        // Visualizations
        if (vizRes.status === "fulfilled" && vizRes.value.ok) {
            const result = await vizRes.value.json();
            setVizData(result);
            sessionStorage.setItem(cacheKey(sessionId, "viz"), JSON.stringify(result));
        }
        setLoadingViz(false);
    };


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

    const explainChart = async (chart: any) => {
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
        const level = importance === "high" || importance === "medium" || importance === "low"
            ? importance : "medium";
        switch (level) {
            case "high": return "border-red-500/30 bg-red-500/5";
            case "medium": return "border-yellow-500/30 bg-yellow-500/5";
            default: return "border-blue-500/30 bg-blue-500/5";
        }
    };

    const getImpactBadge = (impact: string) => {
        switch (impact) {
            case "high":   return { label: "High Impact",   cls: "bg-red-500/20 text-red-400 border border-red-500/30" };
            case "medium": return { label: "Medium Impact", cls: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30" };
            default:       return { label: "Low Impact",    cls: "bg-blue-500/20 text-blue-400 border border-blue-500/30" };
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
                            <span className="text-xs text-slate-500 overflow-visible text-center" style={{ textOverflow: 'unset' }}>{labels[i]}</span>
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
                <div className="w-full max-w-md px-6">
                    {/* Animated brain icon */}
                    <div className="flex justify-center mb-6">
                        <div className="relative">
                            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 animate-pulse">
                                <Sparkles className="w-8 h-8 text-white" />
                            </div>
                            <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full animate-ping" />
                        </div>
                    </div>

                    {/* Stage label */}
                    <p className="text-center text-slate-300 font-medium mb-4">
                        {analysisStage || "Preparing analysis..."}
                    </p>

                    {/* Progress bar */}
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden mb-2">
                        <div
                            className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${Math.max(analysisProgress, 5)}%` }}
                        />
                    </div>

                    {/* Progress percentage */}
                    <p className="text-center text-xs text-slate-500">
                        {analysisProgress}% complete
                    </p>

                    {/* Pipeline steps */}
                    <div className="mt-6 space-y-2">
                        {[
                            { label: "Reading dataset", threshold: 10 },
                            { label: "Classifying columns", threshold: 25 },
                            { label: "Computing metrics", threshold: 45 },
                            { label: "Evaluating business rules", threshold: 60 },
                            { label: "Generating narratives", threshold: 75 },
                            { label: "Building charts", threshold: 85 },
                        ].map((step) => (
                            <div key={step.label} className="flex items-center gap-3">
                                {analysisProgress >= step.threshold ? (
                                    <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                ) : analysisProgress >= step.threshold - 15 ? (
                                    <Loader2 className="w-4 h-4 text-indigo-400 animate-spin flex-shrink-0" />
                                ) : (
                                    <div className="w-4 h-4 rounded-full border border-slate-700 flex-shrink-0" />
                                )}
                                <span className={`text-sm ${
                                    analysisProgress >= step.threshold
                                        ? "text-slate-300"
                                        : analysisProgress >= step.threshold - 15
                                            ? "text-indigo-300"
                                            : "text-slate-600"
                                }`}>
                                    {step.label}
                                </span>
                            </div>
                        ))}
                    </div>
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
                                    window.open(`${API_BASE}/export-report/${session.session_id}`, '_blank');
                                }
                            }}
                            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-white/10 rounded-lg text-sm font-medium transition-colors"
                        >
                            <FileText className="w-4 h-4" />
                            Export Report
                        </button>
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
                        {/* Warnings Banner */}
                        {data.warnings && data.warnings.length > 0 && (
                            <div className="mb-6 p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 space-y-1">
                                {data.warnings.map((w, i) => (
                                    <p key={i} className="text-sm text-amber-300">{w}</p>
                                ))}
                            </div>
                        )}

                        {/* Executive Summary */}
                        <div className="mb-8 p-6 rounded-2xl bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
                            <h2 className="text-sm font-medium text-indigo-400 uppercase tracking-wider mb-2">Executive Summary</h2>
                            <p className="text-lg text-white leading-relaxed">{data.executive_summary}</p>
                        </div>

                        {/* Computed Business Metrics */}
                        {data.computed_metrics && Object.keys(data.computed_metrics).length > 0 && (
                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
                                {Object.entries(data.computed_metrics).map(([key, metric]) => (
                                    <div key={key} className="p-4 rounded-2xl bg-gradient-to-br from-indigo-900/40 to-purple-900/20 border border-indigo-500/20 hover:border-indigo-400/40 transition-all">
                                        <p className="text-xs text-indigo-300 font-medium uppercase tracking-wider mb-1 truncate">{metric.name}</p>
                                        <p className="text-2xl font-bold text-white">{metric.formatted}</p>
                                        <p className="text-xs text-slate-500 mt-1 leading-snug">{metric.description}</p>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* KPI Cards */}
                        {kpis.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4 mb-8">
                                {kpis.map((kpi) => (
                                    <div
                                        key={kpi.column}
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
                                        {data.insights.map((insight, i) => {
                                            const level = insight.impact || insight.importance || "medium";
                                            const badge = getImpactBadge(level);
                                            return (
                                                <div
                                                    key={i}
                                                    className={`p-5 rounded-xl border ${getImportanceColor(level)} flex flex-col gap-3`}
                                                >
                                                    {/* Header */}
                                                    <div className="flex items-start justify-between gap-2">
                                                        <h3 className="font-semibold text-white leading-snug">{insight.title}</h3>
                                                        <span className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap flex-shrink-0 ${badge.cls}`}>
                                                            {badge.label}
                                                        </span>
                                                    </div>

                                                    {/* Description */}
                                                    <p className="text-slate-300 text-sm leading-relaxed">{insight.description}</p>

                                                    {/* Mini chart */}
                                                    {renderMiniChart(insight)}

                                                    {/* Recommendation */}
                                                    {insight.recommendation && (
                                                        <div className="pt-3 border-t border-white/10">
                                                            <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">→ Action</p>
                                                            <p className="text-sm text-slate-300 leading-relaxed">{insight.recommendation}</p>
                                                        </div>
                                                    )}
                                                </div>
                                            );
                                        })}
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
