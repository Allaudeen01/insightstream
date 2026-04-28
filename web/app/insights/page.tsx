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
    RefreshCw,
    Share2,
    Check
} from "lucide-react";
import Navbar from "@/components/Navbar";
import KPICard from "@/components/KPICard";
import InsightCard from "@/components/InsightCard";
import ChartCard from "@/components/ChartCard";
import SkeletonCard from "@/components/SkeletonCard";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

interface InsightData {
    title: string;
    description: string;
    chart_type: string;
    impact: string;
    decision_implication: string;
    recommendation?: string;
}

interface InsightsData {
    session_id: string;
    executive_summary: string;
    strategic_brief: InsightData[];
    recommendations: string[];
    computed_metrics?: Record<string, any>;
}

export default function InsightsPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<InsightsData | null>(null);
    const [vizData, setVizData] = useState<any>(null);
    const [loadingViz, setLoadingViz] = useState(true);
    const [activeTab, setActiveTab] = useState<"insights" | "charts">("charts");
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);
    const [showExportModal, setShowExportModal] = useState(false);
    const [copied, setCopied] = useState(false);

    // Branding state
    const [projectName, setProjectName] = useState("");
    const [reportTitle, setReportTitle] = useState("InsightStream Analysis Report");

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        setProjectName(session.filename?.split('.')[0] || "Project");
        fetchData(session.session_id);
    }, [router]);

    const fetchData = async (sid: string) => {
        try {
            setLoading(true);
            setLoadingViz(true);
            
            const [insightsRes, vizRes] = await Promise.all([
                fetch(`${API_BASE}/insights/${sid}`),
                fetch(`${API_BASE}/generate-viz/${sid}?max_charts=12`),
            ]);

            if (!insightsRes.ok) throw new Error(`Insights fetch failed: ${insightsRes.status}`);
            if (!vizRes.ok) throw new Error(`Viz fetch failed: ${vizRes.status}`);

            setData(await insightsRes.json());
            setVizData(await vizRes.json());
        } catch (error) {
            console.error('Fetch error:', error);
            // Optionally set error state to show in UI
        } finally {
            setLoading(false);
            setLoadingViz(false);
        }
    };

    const [exportStatus, setExportStatus] = useState("");

    const handleProfessionalExport = async () => {
        if (!data?.session_id) {
            alert("Analysis session not fully loaded. Please refresh or wait.");
            return;
        }
        setIsExporting(true);
        setExportProgress(5);
        setExportStatus("Initializing engine...");
        
        try {
            const sid = data.session_id;
            const chartAssets: any[] = [];
            const chartsToExport = vizData?.charts || [];
            
            setExportProgress(15);
            setExportStatus("Resolving visual dependencies...");
            
            // 1. Resolve Plotly Library
            let Plotly = (window as any).Plotly;
            if (!Plotly) {
                try {
                    const mod = await import('plotly.js-dist-min' as any);
                    Plotly = mod.default || mod;
                } catch (e) {
                    console.warn("Plotly dynamic resolve failed");
                }
            }

            // 2. Capture Charts
            for (let i = 0; i < chartsToExport.length; i++) {
                const chart = chartsToExport[i];
                setExportStatus(`Capturing ${chart.title}...`);
                
                console.log('CAPTURE', i, 'looking for', chart.chart_id,
                            'container:', !!document.querySelector(`[data-chart-id="${chart.chart_id}"]`),
                            'plotly child:', !!document.querySelector(`[data-chart-id="${chart.chart_id}"] .js-plotly-plot`),
                            'all chart_ids in DOM:', Array.from(document.querySelectorAll('[data-chart-id]')).map(el => el.getAttribute('data-chart-id')));
                const container = document.querySelector(`[data-chart-id="${chart.chart_id}"]`);
                const plotlyEl = container?.querySelector('.js-plotly-plot');
                
                let image_base64 = "";
                let error = "";

                if (plotlyEl && Plotly) {
                    try {
                        image_base64 = await Plotly.toImage(plotlyEl, {
                            format: 'png',
                            width: 1200,
                            height: 700,
                            scale: 2
                        });
                    } catch (e: any) { 
                        console.error(`Capture failed for ${chart.chart_id}:`, e);
                        error = e.message; 
                    }
                } else { 
                    error = !plotlyEl ? "Element not found" : "Plotly library missing";
                    console.warn(`Skipping capture for ${chart.chart_id}: ${error}`);
                }

                chartAssets.push({
                    id: chart.chart_id,
                    title: chart.title,
                    image_base64,
                    error,
                    insight: chart.description || "Narrative segment."
                });

                setExportProgress(20 + Math.floor(((i + 1) / chartsToExport.length) * 50));
            }

            // 3. Prepare Payload & Submit
            setExportStatus("Assembling professional report...");
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
                    insights: data.recommendations || [],
                    text_blocks: []
                })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Backend assembly failed");
            }

            setExportStatus("Ready for download!");
            setExportProgress(95);
            
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `InsightStream_${projectName.replace(/\s+/g, '_')}_Report.pdf`;
            document.body.appendChild(a);
            a.click();
            
            // Cleanup
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
            }, 800);

        } catch (err: any) {
            console.error("Export Critical Failure:", err);
            alert(`Report Export Failed: ${err.message}`);
            setIsExporting(false);
            setExportStatus("");
        }
    };

    const copyShareLink = () => {
        const sid = data?.session_id;
        const link = `${window.location.origin}/share/dashboard/${sid}`;
        navigator.clipboard.writeText(link);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    if (loading) return <div className="min-h-screen bg-background flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-indigo-500" /></div>;

    return (
        <div className="min-h-screen bg-background pb-20">
            <Navbar />
            <main className="container mx-auto px-6 py-10 max-w-7xl">
                <div className="flex items-center justify-between mb-8">
                    <h1 className="text-3xl font-black tracking-tight">{projectName} <span className="text-muted-foreground font-normal">/ Insights</span></h1>
                    <div className="flex items-center gap-3">
                        <button 
                            onClick={copyShareLink}
                            className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-xs font-bold hover:bg-white/10 transition-all"
                        >
                            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Share2 className="w-3.5 h-3.5" />}
                            {copied ? "Link Copied" : "Share Dashboard"}
                        </button>
                        <button 
                            onClick={() => setShowExportModal(true)}
                            className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-bold shadow-lg shadow-indigo-500/20 transition-all"
                        >
                            <Download className="w-4 h-4" />
                            Export Professional Report
                        </button>
                    </div>
                </div>

                <div className="flex p-1.5 bg-white/5 rounded-2xl border border-white/10 w-fit mb-12">
                    <button onClick={() => setActiveTab("charts")} className={`px-8 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'charts' ? 'bg-indigo-600 text-white' : 'text-muted-foreground hover:text-white'}`}>Visualizations</button>
                    <button onClick={() => setActiveTab("insights")} className={`px-8 py-2.5 rounded-xl text-sm font-bold transition-all ${activeTab === 'insights' ? 'bg-indigo-600 text-white' : 'text-muted-foreground hover:text-white'}`}>Strategic Insights</button>
                </div>

                {activeTab === "charts" ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-page-enter">
                        {vizData?.charts?.map((chart: any, i: number) => (
                            <div key={i} data-chart-id={chart.chart_id}>
                                <ChartCard title={chart.title}>
                                    <div className="h-[350px]">
                                    <Plot
                                        data={chart.plotly_json.data}
                                        layout={{ ...chart.plotly_json.layout, autosize: true, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#94a3b8' } }}
                                        config={{ displayModeBar: false, responsive: true }}
                                        style={{ width: '100%', height: '100%' }}
                                        useResizeHandler
                                    />
                                </div>
                            </ChartCard>
                        </div>
                        ))}
                    </div>
                ) : (
                    <div className="space-y-6 animate-page-enter">
                        {data?.strategic_brief.map((insight, i) => (
                            <InsightCard key={i} {...insight} impact={insight.impact as any} recommendation={insight.decision_implication} />
                        ))}
                    </div>
                )}
            </main>

            {/* Professional Export Modal */}
            {showExportModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-6">
                    <div className="bg-slate-900 border border-white/10 rounded-[2.5rem] p-10 w-full max-w-xl animate-scale-up">
                        <div className="flex items-center justify-between mb-8">
                            <div className="flex items-center gap-3">
                                <div className="p-3 rounded-2xl bg-indigo-500/20 text-indigo-400"><FileText className="w-6 h-6" /></div>
                                <h2 className="text-2xl font-black">Export Settings</h2>
                            </div>
                            <button onClick={() => setShowExportModal(false)}><X className="w-6 h-6 text-muted-foreground" /></button>
                        </div>

                        <div className="space-y-6 mb-10">
                            <div>
                                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Project Name</label>
                                <input 
                                    type="text" 
                                    value={projectName} 
                                    onChange={(e) => setProjectName(e.target.value)}
                                    className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:border-indigo-500 focus:outline-none transition-all"
                                />
                            </div>
                            <div>
                                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Report Title</label>
                                <input 
                                    type="text" 
                                    value={reportTitle} 
                                    onChange={(e) => setReportTitle(e.target.value)}
                                    className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:border-indigo-500 focus:outline-none transition-all"
                                />
                            </div>
                        </div>

                        {isExporting ? (
                            <div className="space-y-4">
                                <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                                    <div className="h-full bg-indigo-500 transition-all duration-500" style={{ width: `${exportProgress}%` }} />
                                </div>
                                <div className="flex justify-between items-center">
                                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-400 animate-pulse">{exportStatus || "Generating PDF Intelligence..."}</p>
                                    <p className="text-[10px] font-bold text-muted-foreground">{exportProgress}%</p>
                                </div>
                            </div>
                        ) : (
                            <button 
                                onClick={handleProfessionalExport}
                                className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-black uppercase tracking-widest shadow-xl shadow-indigo-500/20 transition-all"
                            >
                                Generate PDF Report
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
