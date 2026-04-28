"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    ArrowLeft,
    Save,
    LayoutDashboard,
    Pin,
    Type,
    Check,
    GripVertical,
    X,
    Plus,
    Loader2,
    Share2,
    Calculator,
    Target
} from "lucide-react";
import Navbar from "@/components/Navbar";
import ChartCard from "@/components/ChartCard";
import SkeletonCard from "@/components/SkeletonCard";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// @ts-ignore
import "react-grid-layout/css/styles.css";
// @ts-ignore
import "react-resizable/css/styles.css";

let ResponsiveGrid: any = null;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

    // Custom KPI Modal State
    const [showKPIModal, setShowKPIModal] = useState(false);
    const [availableCols, setAvailableCols] = useState<string[]>([]);
    const [kpiConfig, setKpiConfig] = useState({ column: "", aggregation: "sum", label: "" });
    const [creatingKPI, setCreatingKPI] = useState(false);

    // Export State
    const [isExporting, setIsExporting] = useState(false);
    const [exportProgress, setExportProgress] = useState(0);

    useEffect(() => {
        import("react-grid-layout").then((mod) => {
            ResponsiveGrid = mod.Responsive || mod.default;
            setGridReady(true);
        });
    }, []);

    const [exportStatus, setExportStatus] = useState("");

    const handleProfessionalExport = async () => {
        if (!sessionId) {
            alert("Session not initialized.");
            return;
        }
        setIsExporting(true);
        setExportProgress(5);
        setExportStatus("Initializing report engine...");
        
        try {
            const chartAssets: any[] = [];
            const chartsToExport = pinnedIds.map(id => allCharts.find(c => c.chart_id === id)).filter(Boolean);
            
            setExportProgress(15);
            setExportStatus("Resolving visuals...");

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
            for (let i = 0; i < (chartsToExport as any[]).length; i++) {
                const chart = chartsToExport[i];
                setExportStatus(`Capturing ${chart.title}...`);
                
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
                        console.error(`Capture failed: ${chart.chart_id}`, e);
                        error = e.message;
                    }
                } else {
                    error = !plotlyEl ? "DOM not found" : "Plotly missing";
                }

                chartAssets.push({
                    id: chart.chart_id,
                    title: chart.title,
                    image_base64,
                    error,
                    insight: chart.description || "Segment visualization."
                });

                setExportProgress(20 + Math.floor(((i + 1) / (chartsToExport as any[]).length) * 50));
            }

            // 3. Narrative & Context
            const kpiData: any = {};
            textBlocks.forEach(t => {
                if (t.content.includes(":")) {
                    const [k, v] = t.content.split(":");
                    kpiData[k.trim()] = v.trim();
                }
            });

            const storedInsights = localStorage.getItem(`insights_${sessionId}`);
            const insightsData = storedInsights ? JSON.parse(storedInsights) : null;

            setExportStatus("Building multi-page PDF...");
            setExportProgress(80);
            
            const template = localStorage.getItem("report_template") || "modern";
            const res = await fetch(`${API_BASE}/export-dashboard-pdf/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    title: "Executive Strategic Intelligence",
                    project_name: "InsightStream BI",
                    template,
                    kpis: kpiData,
                    charts: chartAssets,
                    ai_summary: insightsData?.executive_summary || "",
                    insights: insightsData?.recommendations || [],
                    text_blocks: textBlocks
                })
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "PDF generation failed");
            }

            setExportStatus("Downloading report...");
            setExportProgress(95);

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `InsightStream_Dashboard_Report_${sessionId.slice(0,8)}.pdf`;
            document.body.appendChild(a);
            a.click();
            
            setTimeout(() => {
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            }, 100);

            setExportProgress(100);
            setTimeout(() => {
                // @ts-ignore
                if (typeof setShowExportModal !== 'undefined') setShowExportModal(false);
                setIsExporting(false);
                setExportProgress(0);
                setExportStatus("");
            }, 800);

        } catch (err: any) {
            console.error("Export failure:", err);
            alert(`Export Failed: ${err.message}`);
            setIsExporting(false);
            setExportStatus("");
        }
    };

    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const observer = new ResizeObserver(() => setContainerWidth(el.clientWidth));
        observer.observe(el);
        return () => observer.disconnect();
    }, [gridReady]);

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
            setAllCharts(vizRes.charts || []);
            if (dashRes.layout && dashRes.layout.length > 0) {
                setLayout(dashRes.layout);
                setPinnedIds(dashRes.pinned_chart_ids || []);
                setTextBlocks(dashRes.text_blocks || []);
            }
        }).finally(() => setLoading(false));
    }, [router]);

    const handleSave = async () => {
        if (!sessionId) return;
        setSaving(true);
        try {
            await fetch(`${API_BASE}/dashboard/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ layout, pinned_chart_ids: pinnedIds, text_blocks: textBlocks }),
            });
            setSaved(true);
            setTimeout(() => setSaved(false), 2000);
        } finally { setSaving(false); }
    };

    const createCustomKPI = async () => {
        if (!kpiConfig.column) return;
        setCreatingKPI(true);
        try {
            const res = await fetch(`${API_BASE}/kpis/${sessionId}/custom`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(kpiConfig)
            });
            const data = await res.json();
            const kpiId = `kpi_${Date.now()}`;
            const kpiContent = `${data.kpi.label}: ${data.kpi.formatted} (${data.kpi.trend === 'up' ? '▲' : '▼'} ${data.kpi.change_pct}%)`;
            setTextBlocks(prev => [...prev, { id: kpiId, content: kpiContent }]);
            setPinnedIds(prev => [...prev, kpiId]);
            setLayout(prev => [...prev, { i: kpiId, x: 0, y: 0, w: 3, h: 2 }]);
            setShowKPIModal(false);
        } finally { setCreatingKPI(false); }
    };

    const removeItem = (id: string) => {
        setPinnedIds(prev => prev.filter(i => i !== id));
        setLayout(prev => prev.filter(l => l.i !== id));
        setTextBlocks(prev => prev.filter(t => t.id !== id));
    };

    if (loading) return <div className="min-h-screen bg-background flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-indigo-500" /></div>;

    const RG = ResponsiveGrid;

    return (
        <div className="min-h-screen bg-background">
            <Navbar />
            <main className="container mx-auto px-6 py-8">
                <div className="flex items-center justify-between mb-12">
                    <div className="flex items-center gap-4">
                        <Link href="/insights" className="p-2 bg-white/5 rounded-xl hover:bg-white/10 transition-all"><ArrowLeft className="w-5 h-5" /></Link>
                        <h1 className="text-2xl font-black tracking-tight">Executive Workspace</h1>
                    </div>
                    <div className="flex items-center gap-3">
                        <button onClick={handleProfessionalExport} disabled={isExporting} className="flex items-center gap-2 px-5 py-2.5 bg-white/5 border border-white/10 rounded-xl text-sm font-bold hover:bg-white/10 transition-all group">
                            <Target className={`w-4 h-4 text-purple-400 ${isExporting ? 'animate-ping' : 'group-hover:scale-110 transition-transform'}`} />
                            {isExporting ? "Capturing..." : "Professional Export"}
                        </button>
                        <button onClick={() => setShowKPIModal(true)} className="flex items-center gap-2 px-5 py-2.5 bg-white/5 border border-white/10 rounded-xl text-sm font-bold hover:bg-white/10 transition-all">
                            <Calculator className="w-4 h-4 text-emerald-400" /> Custom KPI
                        </button>
                        <button onClick={handleSave} disabled={saving} className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold shadow-lg transition-all ${saved ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-500/20"}`}>
                            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                            {saved ? "Synchronized" : "Secure Perspective"}
                        </button>
                    </div>
                </div>

                <div ref={containerRef}>
                    {gridReady && RG && (
                        <RG
                            className="layout"
                            layouts={{ lg: layout }}
                            breakpoints={{ lg: 1200, md: 996, sm: 768 }}
                            cols={{ lg: 12, md: 10, sm: 6 }}
                            rowHeight={100}
                            width={containerWidth}
                            onLayoutChange={setLayout}
                            draggableHandle=".drag-handle"
                        >
                            {pinnedIds.map((id) => {
                                const chart = allCharts.find(c => c.chart_id === id);
                                const text = textBlocks.find(t => t.id === id);
                                return (
                                    <div key={id} className="group" data-chart-id={chart?.chart_id}>
                                        <div className="h-full rounded-[2rem] bg-slate-900 border border-white/10 p-5 flex flex-col relative">
                                            <div className="flex items-center justify-between mb-4">
                                                <div className="flex items-center gap-2 drag-handle cursor-grab active:cursor-grabbing">
                                                    <GripVertical className="w-4 h-4 text-white/20" />
                                                    <span className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">{chart ? "Visual" : "Metric"}</span>
                                                </div>
                                                <button onClick={() => removeItem(id)} className="p-1.5 hover:bg-red-500/10 rounded-lg text-white/20 hover:text-red-400 transition-all"><X className="w-4 h-4" /></button>
                                            </div>
                                            {chart ? (
                                                <div className="flex-1 min-h-0">
                                                    <Plot
                                                        data={chart.plotly_json.data}
                                                        layout={{ ...chart.plotly_json.layout, autosize: true, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: 'rgba(255,255,255,0.5)', size: 10 } }}
                                                        config={{ displayModeBar: false, responsive: true }}
                                                        style={{ width: '100%', height: '100%' }}
                                                        useResizeHandler
                                                    />
                                                </div>
                                            ) : (
                                                <div className="flex-1 flex flex-col justify-center text-center">
                                                    <div className="p-4 rounded-2xl bg-white/5 border border-white/5 font-bold text-lg">{text?.content}</div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </RG>
                    )}
                </div>
            </main>

            {/* Custom KPI Modal */}
            {showKPIModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-6">
                    <div className="bg-slate-900 border border-white/10 rounded-[2.5rem] p-10 w-full max-w-lg">
                        <h2 className="text-2xl font-black mb-8 flex items-center gap-3"><Target className="w-6 h-6 text-emerald-400" /> Forge Custom Metric</h2>
                        <div className="space-y-6 mb-10">
                            <div>
                                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Source Dimension</label>
                                <select 
                                    className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:outline-none"
                                    value={kpiConfig.column}
                                    onChange={(e) => setKpiConfig({ ...kpiConfig, column: e.target.value })}
                                >
                                    <option value="">Select numeric column</option>
                                    {availableCols.map(c => <option key={c} value={c}>{c}</option>)}
                                </select>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Aggregation</label>
                                    <select 
                                        className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:outline-none"
                                        value={kpiConfig.aggregation}
                                        onChange={(e) => setKpiConfig({ ...kpiConfig, aggregation: e.target.value })}
                                    >
                                        <option value="sum">Summation</option>
                                        <option value="avg">Mean Average</option>
                                        <option value="growth">Growth %</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Metric Label</label>
                                    <input 
                                        className="w-full bg-white/5 border border-white/10 rounded-2xl px-5 py-4 focus:outline-none"
                                        placeholder="e.g. Total Revenue"
                                        value={kpiConfig.label}
                                        onChange={(e) => setKpiConfig({ ...kpiConfig, label: e.target.value })}
                                    />
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-3">
                            <button onClick={() => setShowKPIModal(false)} className="flex-1 py-4 bg-white/5 hover:bg-white/10 rounded-2xl font-bold transition-all">Abort</button>
                            <button onClick={createCustomKPI} disabled={creatingKPI || !kpiConfig.column} className="flex-[2] py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl font-black uppercase tracking-widest shadow-xl shadow-emerald-500/20 disabled:opacity-50">
                                {creatingKPI ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : "Deploy Metric"}
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {/* Export Progress Modal */}
            {isExporting && (
                <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/90 backdrop-blur-xl p-6">
                    <div className="bg-slate-900 border border-white/10 rounded-[3rem] p-12 w-full max-w-xl text-center shadow-2xl shadow-purple-500/10">
                        <div className="w-24 h-24 bg-indigo-500/20 rounded-full flex items-center justify-center mx-auto mb-8 animate-pulse">
                            <Target className="w-12 h-12 text-indigo-400" />
                        </div>
                        <h2 className="text-3xl font-black mb-4">Generating Intelligence</h2>
                        <p className="text-muted-foreground mb-10 text-sm font-medium leading-relaxed">
                            We're capturing high-fidelity snapshots of your workspace<br/> 
                            to build a pixel-perfect professional report.
                        </p>
                        
                        <div className="space-y-4">
                            <div className="h-3 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
                                <div 
                                    className="h-full bg-gradient-to-r from-indigo-600 to-purple-600 transition-all duration-500 ease-out shadow-[0_0_20px_rgba(99,102,241,0.5)]" 
                                    style={{ width: `${exportProgress}%` }} 
                                />
                            </div>
                            <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-[0.2em] text-indigo-400 px-1">
                                <span>{exportStatus || 'Generating Intelligence...'}</span>
                                <span>{exportProgress}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
