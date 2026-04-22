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
    Loader2
} from "lucide-react";
import Navbar from "@/components/Navbar";
import ChartCard from "@/components/ChartCard";
import SkeletonCard from "@/components/SkeletonCard";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// @ts-ignore
import "react-grid-layout/css/styles.css";
// @ts-ignore
import "react-resizable/css/styles.css";

// We import react-grid-layout client-side only
let ResponsiveGrid: any = null;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
    insight_reason?: string;
}

interface TextBlock {
    id: string;
    content: string;
}

interface LayoutItem {
    i: string;
    x: number;
    y: number;
    w: number;
    h: number;
}

export default function DashboardPage() {
    const router = useRouter();
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [allCharts, setAllCharts] = useState<ChartData[]>([]);
    const [pinnedIds, setPinnedIds] = useState<string[]>([]);
    const [layout, setLayout] = useState<LayoutItem[]>([]);
    const [textBlocks, setTextBlocks] = useState<TextBlock[]>([]);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [saved, setSaved] = useState(false);
    const [editingText, setEditingText] = useState<string | null>(null);
    const [editContent, setEditContent] = useState("");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [gridReady, setGridReady] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const [containerWidth, setContainerWidth] = useState(1200);

    useEffect(() => {
        // Load react-grid-layout on client side
        import("react-grid-layout").then((mod) => {
            ResponsiveGrid = mod.Responsive || mod.default;
            setGridReady(true);
        }).catch((err) => {
            console.error("Failed to load react-grid-layout:", err);
        });
    }, []);

    // Measure container width
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;
        const observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                setContainerWidth(entry.contentRect.width);
            }
        });
        observer.observe(el);
        setContainerWidth(el.clientWidth);
        return () => observer.disconnect();
    }, [gridReady]);

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        const sid = session.session_id;
        setSessionId(sid);

        const pinned = localStorage.getItem(`pinned_${sid}`);
        const pinnedList: string[] = pinned ? JSON.parse(pinned) : [];
        setPinnedIds(pinnedList);

        Promise.all([
            fetch(`${API_BASE}/generate-viz/${sid}?max_charts=12`).then(r => r.json()),
            fetch(`${API_BASE}/dashboard/${sid}`).then(r => r.json()),
        ]).then(([vizRes, dashRes]) => {
            setAllCharts(vizRes.charts || []);
            if (dashRes.layout && dashRes.layout.length > 0) {
                setLayout(dashRes.layout);
                setPinnedIds(dashRes.pinned_chart_ids || pinnedList);
                setTextBlocks(dashRes.text_blocks || []);
            } else {
                const defaultLayout = pinnedList.map((id, i) => ({
                    i: id,
                    x: (i % 2) * 6,
                    y: Math.floor(i / 2) * 4,
                    w: 6,
                    h: 4,
                }));
                setLayout(defaultLayout);
            }
        }).catch(console.error).finally(() => setLoading(false));
    }, [router]);

    const pinnedCharts = allCharts.filter(c => pinnedIds.includes(c.chart_id));

    const handleLayoutChange = useCallback((newLayout: LayoutItem[]) => {
        setLayout(newLayout);
        setSaved(false);
    }, []);

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
        } catch (err) { console.error("Save failed:", err); }
        finally { setSaving(false); }
    };

    const addTextBlock = () => {
        const id = `text_${Date.now()}`;
        setTextBlocks(prev => [...prev, { id, content: "Click to edit this text block..." }]);
        setPinnedIds(prev => [...prev, id]);
        setLayout(prev => [...prev, { i: id, x: 0, y: Infinity, w: 6, h: 2 }]);
        setSaved(false);
    };

    const removeItem = (itemId: string) => {
        setPinnedIds(prev => prev.filter(id => id !== itemId));
        setLayout(prev => prev.filter(l => l.i !== itemId));
        setTextBlocks(prev => prev.filter(t => t.id !== itemId));
        setSaved(false);
    };

    const saveTextEdit = (blockId: string) => {
        setTextBlocks(prev => prev.map(t => t.id === blockId ? { ...t, content: editContent } : t));
        setEditingText(null);
        setSaved(false);
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background">
                <Navbar />
                <div className="container mx-auto px-6 py-12">
                    <div className="flex items-center justify-between mb-8">
                        <div className="h-8 bg-white/5 rounded w-48 animate-pulse" />
                        <div className="h-10 bg-white/5 rounded w-32 animate-pulse" />
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[1, 2, 3, 4, 5, 6].map(i => <SkeletonCard key={i} />)}
                    </div>
                </div>
            </div>
        );
    }

    const renderGrid = () => {
        if (!gridReady || !ResponsiveGrid) return null;
        const RG = ResponsiveGrid;
        return (
            <div ref={containerRef}>
                <RG
                    className="layout"
                    layouts={{ lg: layout }}
                    breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
                    cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
                    rowHeight={100}
                    width={containerWidth}
                    isDraggable
                    isResizable
                    onLayoutChange={handleLayoutChange}
                    draggableHandle=".drag-handle"
                >
                    {pinnedIds.map((itemId) => {
                        const chart = pinnedCharts.find(c => c.chart_id === itemId);
                        const textBlock = textBlocks.find(t => t.id === itemId);

                        if (chart) {
                            return (
                                <div key={itemId} className="flex flex-col group">
                                    <ChartCard>
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="flex items-center gap-2 min-w-0">
                                                <GripVertical className="drag-handle w-4 h-4 text-muted-foreground/30 cursor-grab flex-shrink-0 hover:text-purple-400 transition-colors" />
                                                <h3 className="text-sm font-semibold text-foreground truncate">{chart.title}</h3>
                                            </div>
                                            <button onClick={() => removeItem(itemId)} className="p-1.5 hover:bg-red-500/10 rounded-lg text-muted-foreground hover:text-red-400 transition-all">
                                                <X className="w-4 h-4" />
                                            </button>
                                        </div>
                                        <div className="flex-1 min-h-0">
                                            <Plot
                                                data={chart.plotly_json.data}
                                                layout={{
                                                    ...chart.plotly_json.layout,
                                                    autosize: true,
                                                    margin: { l: 40, r: 20, t: 20, b: 40 },
                                                    paper_bgcolor: 'transparent',
                                                    plot_bgcolor: 'transparent',
                                                    font: { color: 'rgba(255,255,255,0.5)', size: 10 },
                                                    xaxis: { ...chart.plotly_json.layout.xaxis, gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                                                    yaxis: { ...chart.plotly_json.layout.yaxis, gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' }
                                                }}
                                                config={{ displayModeBar: false, responsive: true, displaylogo: false }}
                                                style={{ width: '100%', height: '100%' }}
                                                useResizeHandler
                                            />
                                        </div>
                                    </ChartCard>
                                </div>
                            );
                        }

                        if (textBlock) {
                            return (
                                <div key={itemId} className="flex flex-col group">
                                    <div className="rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm p-6 hover:border-white/20 transition-all flex flex-col h-full">
                                        <div className="flex items-center justify-between mb-4">
                                            <div className="flex items-center gap-2">
                                                <GripVertical className="drag-handle w-4 h-4 text-muted-foreground/30 cursor-grab hover:text-purple-400" />
                                                <Type className="w-4 h-4 text-purple-400" />
                                                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Note</span>
                                            </div>
                                            <button onClick={() => removeItem(itemId)} className="p-1.5 hover:bg-red-500/10 rounded-lg text-muted-foreground hover:text-red-400 transition-all">
                                                <X className="w-4 h-4" />
                                            </button>
                                        </div>
                                        <div className="flex-1 overflow-auto">
                                            {editingText === itemId ? (
                                                <div className="h-full flex flex-col gap-3">
                                                    <textarea
                                                        autoFocus
                                                        value={editContent}
                                                        onChange={(e) => setEditContent(e.target.value)}
                                                        className="flex-1 bg-white/5 border border-purple-500/30 rounded-xl p-4 text-sm text-foreground resize-none focus:outline-none focus:ring-2 focus:ring-purple-500/20"
                                                    />
                                                    <div className="flex gap-2">
                                                        <button onClick={() => saveTextEdit(itemId)} className="px-4 py-2 text-xs font-medium bg-purple-600 rounded-lg text-white hover:bg-purple-500 transition-colors">Save</button>
                                                        <button onClick={() => setEditingText(null)} className="px-4 py-2 text-xs font-medium bg-white/5 rounded-lg text-muted-foreground hover:bg-white/10 transition-colors">Cancel</button>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div
                                                    onClick={() => { setEditingText(itemId); setEditContent(textBlock.content); }}
                                                    className="h-full text-sm text-muted-foreground cursor-text whitespace-pre-wrap hover:text-foreground transition-colors p-2 rounded-lg hover:bg-white/5"
                                                >
                                                    {textBlock.content}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            );
                        }

                        return (
                            <div key={itemId} className="rounded-xl border border-white/10 bg-white/5 p-4 flex items-center justify-center">
                                <p className="text-muted-foreground text-sm">Item not found</p>
                            </div>
                        );
                    })}
                </RG>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-background">
            <Navbar />

            <main className="container mx-auto px-6 py-8">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                    <div className="flex items-center gap-4">
                        <Link href="/insights" className="p-2 hover:bg-white/5 rounded-lg text-muted-foreground hover:text-foreground transition-all">
                            <ArrowLeft className="w-5 h-5" />
                        </Link>
                        <div>
                            <div className="flex items-center gap-3">
                                <h1 className="text-2xl font-bold text-foreground tracking-tight">Executive Dashboard</h1>
                                {saved && (
                                    <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded-full border border-emerald-400/20 animate-fade-in">
                                        <Check className="w-3 h-3" />
                                        Changes Saved
                                    </span>
                                )}
                            </div>
                            <p className="text-sm text-muted-foreground">Customize your data workspace</p>
                        </div>
                    </div>
                    
                    <div className="flex items-center gap-3">
                        <button onClick={addTextBlock} className="flex items-center gap-2 px-4 py-2.5 bg-white/5 hover:bg-white/8 border border-white/10 rounded-xl text-sm font-medium transition-all group">
                            <Plus className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
                            Add Insight Note
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                                saved 
                                ? "bg-emerald-600/20 text-emerald-400 border border-emerald-600/30" 
                                : "bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/20"
                            } disabled:opacity-50`}
                        >
                            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                            {saved ? "Layout Secured" : "Save Perspective"}
                        </button>
                    </div>
                </div>

                {errorMessage && (
                    <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-8 flex items-center justify-between text-red-400 text-sm animate-fade-in">
                        <div className="flex items-center gap-3">
                            <span className="flex items-center justify-center w-6 h-6 rounded-full bg-red-500/20 text-xs font-bold">!</span>
                            {errorMessage}
                        </div>
                        <button onClick={() => setErrorMessage(null)} className="p-1 hover:bg-red-500/10 rounded-lg">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                )}

                {pinnedIds.length === 0 ? (
                    <div className="text-center py-32 rounded-3xl border border-dashed border-white/10 bg-white/[0.02]">
                        <div className="w-20 h-20 bg-white/5 rounded-3xl flex items-center justify-center mx-auto mb-6">
                            <Pin className="w-10 h-10 text-muted-foreground/20" />
                        </div>
                        <h2 className="text-xl font-bold mb-2">Workspace is Empty</h2>
                        <p className="text-muted-foreground mb-8 max-w-sm mx-auto">Discover key findings in the Insights engine and pin them here to build your narrative.</p>
                        <Link href="/insights" className="inline-flex items-center gap-2 px-8 py-3.5 bg-purple-600 hover:bg-purple-500 rounded-xl font-bold transition-all shadow-xl shadow-purple-600/25">
                            Discover Insights
                        </Link>
                    </div>
                ) : (
                    renderGrid()
                )}
            </main>
        </div>
    );
}

