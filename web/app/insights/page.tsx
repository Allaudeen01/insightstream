"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    ArrowLeft,
    Lightbulb,
    Loader2,
    TrendingUp,
    TrendingDown,
    Minus,
    Download,
    X,
    Pin,
    Sparkles,
    FileText,
    MessageSquare,
    ChevronRight,
    ArrowRight,
    Send,
    RefreshCw
} from "lucide-react";
import Navbar from "@/components/Navbar";
import KPICard from "@/components/KPICard";
import InsightCard from "@/components/InsightCard";
import ChartCard from "@/components/ChartCard";
import SkeletonCard from "@/components/SkeletonCard";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface InsightData {
    title: string;
    description: string;
    chart_type: string;
    chart_data?: { labels: string[]; values: number[] };
    impact: string;
    decision_implication: string;
    recommendation?: string;
    qualified_segments?: string[];
}

interface InsightsData {
    session_id: string;
    executive_summary: string;
    strategic_brief: InsightData[];
    recommendations: string[];
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
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [vizError, setVizError] = useState<string | null>(null);

    const [chatOpen, setChatOpen] = useState(false);
    const [pinnedCharts, setPinnedCharts] = useState<Set<string>>(new Set());
    const [kpis, setKpis] = useState<KPIItem[]>([]);
    const [explaining, setExplaining] = useState<string | null>(null);
    const [explanation, setExplanation] = useState<any | null>(null);
    const [explainOpen, setExplainOpen] = useState(false);

    const [analysisProgress, setAnalysisProgress] = useState(0);
    const [analysisStage, setAnalysisStage] = useState<string>("");
    const [messages, setMessages] = useState<{ role: string; content: string }[]>([
        { role: "assistant", content: "I'm your data specialist. Ask me anything about the visualizations or underlying trends." }
    ]);
    const [input, setInput] = useState("");
    const [chatLoading, setChatLoading] = useState(false);

    const resetView = () => {
        setMessages([{ role: "assistant", content: "I'm your data specialist. Ask me anything about the visualizations or underlying trends." }]);
    };

    const handleChatSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;
        setMessages(prev => [...prev, { role: "user", content: input }]);
        setInput("");
        // Mock response for now to ensure stability
        setTimeout(() => {
            setMessages(prev => [...prev, { role: "assistant", content: "I've analyzed your request. The underlying trend suggests a strong correlation between the selected dimensions." }]);
        }, 1000);
    };

    const cacheKey = (sessionId: string, kind: string) => `is_cache_${sessionId}_${kind}`;

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        const sid = session.session_id;

        const cachedInsights = sessionStorage.getItem(cacheKey(sid, "insights"));
        const cachedKpis     = sessionStorage.getItem(cacheKey(sid, "kpis"));
        const cachedViz      = sessionStorage.getItem(cacheKey(sid, "viz"));

        if (cachedInsights) setData(JSON.parse(cachedInsights));
        if (cachedKpis) setKpis(JSON.parse(cachedKpis));
        if (cachedViz) setVizData(JSON.parse(cachedViz));

        startBackgroundAnalysis(sid);

        const pinned = localStorage.getItem(`pinned_${sid}`);
        if (pinned) setPinnedCharts(new Set(JSON.parse(pinned)));
    }, [router]);

    const startBackgroundAnalysis = async (sessionId: string) => {
        try {
            setAnalysisStage("Initiating Analysis...");
            const analyzeRes = await fetch(`${API_BASE}/analyze/${sessionId}`, { method: "POST" });
            const analyzeData = await analyzeRes.json();

            if (analyzeData.status === "done") {
                setAnalysisProgress(100);
                fetchAllParallel(sessionId);
                return;
            }

            const pollInterval = setInterval(async () => {
                const statusRes = await fetch(`${API_BASE}/analyze-status/${sessionId}`);
                const status = await statusRes.json();
                setAnalysisProgress(status.progress || 0);

                if (status.progress < 20) setAnalysisStage("Parsing Context...");
                else if (status.progress < 40) setAnalysisStage("Identifying Trends...");
                else if (status.progress < 60) setAnalysisStage("Running Simulations...");
                else if (status.progress < 80) setAnalysisStage("Synthesizing Narratives...");
                else setAnalysisStage("Finalizing Insights...");

                if (status.status === "done") {
                    clearInterval(pollInterval);
                    fetchAllParallel(sessionId);
                } else if (status.status === "error") {
                    clearInterval(pollInterval);
                    setError(status.error || "Analysis failed");
                    setLoading(false);
                }
            }, 800);
        } catch { fetchAllParallel(sessionId); }
    };

    const fetchAllParallel = async (sessionId: string) => {
        try {
            const [insightsRes, kpisRes, vizRes] = await Promise.allSettled([
                fetch(`${API_BASE}/insights/${sessionId}`),
                fetch(`${API_BASE}/kpis/${sessionId}`),
                fetch(`${API_BASE}/generate-viz/${sessionId}?max_charts=12`),
            ]);

            if (insightsRes.status === "fulfilled" && insightsRes.value.ok) {
                const result = await insightsRes.value.json();
                console.log("INSIGHTS DATA:", result);
                setData(result);
                sessionStorage.setItem(cacheKey(sessionId, "insights"), JSON.stringify(result));
            }
            if (kpisRes.status === "fulfilled" && kpisRes.value.ok) {
                const result = await kpisRes.value.json();
                setKpis(result.kpis || []);
                sessionStorage.setItem(cacheKey(sessionId, "kpis"), JSON.stringify(result.kpis || []));
            }
            if (vizRes.status === "fulfilled" && vizRes.value.ok) {
                const result = await vizRes.value.json();
                setVizData(result);
                sessionStorage.setItem(cacheKey(sessionId, "viz"), JSON.stringify(result));
                setVizError(null);
            } else if (vizRes.status === "fulfilled" && !vizRes.value.ok) {
                setVizError("Failed to generate visualizations. The dataset might be too complex or contain incompatible data types.");
            } else {
                setVizError("Visualization service is currently unavailable.");
            }
        } catch (err) { setErrorMessage("Error updating workspace."); }
        finally { setLoading(false); setLoadingViz(false); }
    };

    const togglePin = (chartId: string) => {
        setPinnedCharts(prev => {
            const next = new Set(prev);
            if (next.has(chartId)) next.delete(chartId);
            else next.add(chartId);
            const sid = JSON.parse(localStorage.getItem("analysis_session")!).session_id;
            localStorage.setItem(`pinned_${sid}`, JSON.stringify([...next]));
            return next;
        });
    };

    const explainChart = async (chart: any) => {
        setExplaining(chart.chart_id);
        setExplainOpen(true);
        try {
            const sid = JSON.parse(localStorage.getItem("analysis_session")!).session_id;
            const res = await fetch(`${API_BASE}/explain-chart/${sid}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ chart_id: chart.chart_id, chart_title: chart.title }),
            });
            if (res.ok) setExplanation(await res.json());
        } catch {} finally { setExplaining(null); }
    };

    const handleExport = async () => {
        setIsExporting(true);
        try {
            const sid = JSON.parse(localStorage.getItem("analysis_session")!).session_id;
            const res = await fetch(`${API_BASE}/export-excel/${sid}`);
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `InsightStream_Export.xlsx`;
            a.click();
        } catch { setErrorMessage("Export failed."); }
        finally { setIsExporting(false); }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
                <div className="w-full max-w-md">
                    <div className="flex justify-center mb-8">
                        <div className="relative">
                            <div className="w-20 h-20 rounded-3xl bg-purple-600 flex items-center justify-center shadow-2xl shadow-purple-500/40 animate-pulse">
                                <Sparkles className="w-10 h-10 text-white" />
                            </div>
                            <div className="absolute -top-2 -right-2 w-6 h-6 bg-purple-400 rounded-full animate-ping opacity-50" />
                        </div>
                    </div>
                    <div className="text-center mb-8">
                        <h2 className="text-xl font-bold text-foreground mb-2">Syncing your data story</h2>
                        <p className="text-muted-foreground text-sm font-medium uppercase tracking-widest">{analysisStage}</p>
                    </div>
                    <div className="relative h-2 w-full bg-white/5 rounded-full overflow-hidden mb-4">
                        <div 
                            className="h-full bg-purple-500 shadow-[0_0_12px_rgba(168,85,247,0.5)] transition-all duration-700 ease-out"
                            style={{ width: `${Math.max(analysisProgress, 5)}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-xs font-bold text-muted-foreground uppercase tracking-widest">
                        <span>Analysis Pipeline</span>
                        <span>{analysisProgress}%</span>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background pb-20">
            <Navbar />

            <main className="container mx-auto px-6 py-10 max-w-7xl">
                {/* Executive Summary Hero */}
                <div className="relative mb-12 rounded-3xl overflow-hidden border border-white/10 bg-white/5 p-8 md:p-12">
                    <div className="absolute top-0 right-0 w-1/3 h-full bg-purple-500/10 blur-[100px] -z-10" />
                    <div className="max-w-4xl">
                        <div className="flex items-center gap-2 mb-4">
                            <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                                <Sparkles className="w-4 h-4" />
                            </div>
                            <span className="text-sm font-bold text-purple-400 uppercase tracking-widest">Strategic Brief</span>
                        </div>
                        <h1 className="text-2xl md:text-4xl font-bold text-foreground mb-6 leading-tight">
                            {data?.executive_summary.split('.')[0]}.
                        </h1>
                        <p className="text-lg text-muted-foreground leading-relaxed">
                            {data?.executive_summary.split('.').slice(1).join('.')}
                        </p>
                    </div>
                </div>

                {/* Performance KPIs */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                    {data?.computed_metrics && Object.entries(data.computed_metrics).slice(0, 4).map(([key, metric], i) => (
                        <KPICard 
                            key={`m-${i}`}
                            title={metric.name}
                            value={metric.formatted}
                            subtitle={metric.description?.startsWith('Avg:') ? undefined : metric.description}
                        />
                    ))}
                </div>

                {/* Tab Controls */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex p-1.5 bg-white/5 rounded-2xl border border-white/10">
                        <button 
                            onClick={() => setActiveTab("charts")}
                            className={`px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'charts' ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20' : 'text-muted-foreground hover:text-foreground'}`}
                        >
                            Visualizations
                        </button>
                        <button 
                            onClick={() => setActiveTab("insights")}
                            className={`px-6 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'insights' ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20' : 'text-muted-foreground hover:text-foreground'}`}
                        >
                            Deep Insights
                        </button>
                    </div>

                    <div className="flex items-center gap-3">
                        <button 
                            onClick={handleExport}
                            disabled={isExporting}
                            className="flex items-center gap-2 px-5 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-2xl text-sm font-bold transition-all disabled:opacity-50"
                        >
                            {isExporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                            <span>Export Brief</span>
                        </button>
                    </div>
                </div>

                {/* Tab Content */}
                {activeTab === "charts" && (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-page-enter">
                        
                        {/* Loading state */}
                        {loadingViz && (
                            [1,2,3,4].map(i => (
                                <div key={i} className="rounded-xl border border-white/10 bg-white/5 animate-pulse p-4">
                                    <div className="h-4 bg-white/10 rounded w-1/3 mb-4"/>
                                    <div className="h-64 bg-white/5 rounded"/>
                                </div>
                            ))
                        )}

                        {/* Error state */}
                        {!loadingViz && vizError && (
                            <div className="col-span-2 text-center py-20">
                                <p className="text-red-400 mb-2">⚠️ {vizError}</p>
                                <p className="text-sm text-muted-foreground">Switch to Deep Insights for text analysis</p>
                            </div>
                        )}

                        {/* Charts */}
                        {!loadingViz && !vizError && 
                            vizData?.charts
                            ?.filter(c => c.plotly_json?.data)
                            ?.map((chart, i, arr) => (
                                <ChartCard
                                    key={chart.title || i}
                                    title={chart.title}
                                    className={arr.length % 2 !== 0 && i === arr.length - 1 ? 'lg:col-span-2' : ''}
                                >
                                    <div style={{ height: '350px', width: '100%' }}>
                                        <Plot
                                            data={chart.plotly_json.data}
                                            layout={{
                                                ...chart.plotly_json.layout,
                                                title: { text: '' },
                                                paper_bgcolor: 'transparent',
                                                plot_bgcolor:  'transparent',
                                                font: { color: '#94a3b8' },
                                                margin: { t: 30, b: 50, l: 50, r: 20 },
                                                autosize: true,
                                                xaxis: {
                                                    ...chart.plotly_json.layout?.xaxis,
                                                    gridcolor: 'rgba(255,255,255,0.05)',
                                                    tickfont: { color: '#64748b' },
                                                },
                                                yaxis: {
                                                    ...chart.plotly_json.layout?.yaxis,
                                                    gridcolor: 'rgba(255,255,255,0.05)',
                                                    tickfont: { color: '#64748b' },
                                                },
                                                legend: {
                                                    font: { color: '#94a3b8' },
                                                    bgcolor: 'transparent',
                                                }
                                            }}
                                            config={{ displayModeBar: false, responsive: true }}
                                            style={{ width: '100%', height: '100%' }}
                                            useResizeHandler={true}
                                        />
                                    </div>
                                </ChartCard>
                            ))
                        }

                        {/* Empty state */}
                        {!loadingViz && !vizError && !vizData?.charts?.length && (
                            <div className="col-span-2 text-center py-20 text-muted-foreground">
                                No visualizations for this dataset
                            </div>
                        )}
                    </div>
                )}
                {activeTab === "insights" && (
                    <div className="space-y-8 animate-page-enter">
                        <div className="grid grid-cols-1 gap-6">
                            {(data?.strategic_brief && data.strategic_brief.length > 0) ? (
                                data.strategic_brief.map((insight, i) => (
                                    <InsightCard
                                        key={i}
                                        title={insight.title}
                                        description={insight.description}
                                        impact={(insight.impact || "Medium") as any}
                                        recommendation={insight.decision_implication || insight.recommendation}
                                    />
                                ))
                            ) : (
                                <div className="p-12 text-center rounded-3xl border border-dashed border-white/10 bg-white/5">
                                    <p className="text-muted-foreground font-medium">No strategic insights generated for this dataset.</p>
                                </div>
                            )}
                        </div>

                        {/* Recommendations Section */}
                        {data?.recommendations && data.recommendations.length > 0 && (
                            <div className="mt-12 rounded-[2.5rem] border border-white/10 bg-white/5 p-10 backdrop-blur-xl">
                                <div className="flex items-center gap-3 mb-8">
                                    <div className="p-3 rounded-2xl bg-green-500/10 text-green-400 border border-green-500/20">
                                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="12" r="10" />
                                            <path d="m9 12 2 2 4-4" />
                                        </svg>
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-black tracking-tight">Strategic Recommendations</h2>
                                        <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Actionable Intelligence Protocols</p>
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {data.recommendations.map((rec, i) => (
                                        <div
                                            key={i}
                                            className="group p-6 rounded-[2rem] bg-white/5 border border-white/10 hover:bg-white/10 transition-all flex items-start gap-4"
                                        >
                                            <div className="w-8 h-8 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center shrink-0 font-black text-xs">
                                                0{i + 1}
                                            </div>
                                            <p className="text-foreground/80 leading-relaxed font-medium">
                                                {typeof rec === 'string' ? rec : (rec as any).text || JSON.stringify(rec)}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
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
