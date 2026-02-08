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
    TrendingUp
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

export default function InsightsPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<InsightsData | null>(null);
    const [vizData, setVizData] = useState<VizResponse | null>(null);
    const [loadingViz, setLoadingViz] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"insights" | "charts">("charts");

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }

        const session = JSON.parse(stored);
        fetchInsights(session.session_id);
        fetchVisualizations(session.session_id);
    }, [router]);

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
                    <div className="w-20" />
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
                                                            <span className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap flex items-center gap-1 ${chart.interest_level === "high"
                                                                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                                                                : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                                                                }`}>
                                                                {chart.interest_level === "high" ? "✨ High Insight" : "💡 Smart Pick"}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="text-sm text-slate-400">{chart.description}</p>
                                                    {chart.insight_reason && (
                                                        <div className="mt-2 p-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
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
                                                <div className="mt-2 flex flex-wrap gap-1">
                                                    {chart.columns_used.map((col) => (
                                                        <span key={col} className="text-xs px-2 py-0.5 bg-slate-700 rounded text-slate-300">
                                                            {col}
                                                        </span>
                                                    ))}
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
        </div>
    );
}
