"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import {
    ArrowLeft,
    Save,
    Loader2,
    LayoutDashboard,
    Pin,
    Type,
    Check,
    GripVertical,
    X
} from "lucide-react";

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
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-500 mx-auto mb-4" />
                    <p className="text-slate-400">Loading dashboard...</p>
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
                                <div key={itemId} className="rounded-2xl bg-slate-900 border border-white/10 overflow-hidden flex flex-col">
                                    <div className="flex items-center justify-between px-3 py-2 border-b border-white/5">
                                        <div className="flex items-center gap-2 min-w-0">
                                            <GripVertical className="drag-handle w-4 h-4 text-slate-600 cursor-grab flex-shrink-0" />
                                            <h3 className="text-sm font-semibold text-white truncate">{chart.title}</h3>
                                        </div>
                                        <button onClick={() => removeItem(itemId)} className="p-1 hover:bg-red-500/10 rounded text-slate-500 hover:text-red-400 transition-colors flex-shrink-0">
                                            <X className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                    <div className="flex-1 p-2">
                                        <Plot
                                            data={chart.plotly_json.data}
                                            layout={{
                                                ...chart.plotly_json.layout,
                                                autosize: true,
                                                margin: { l: 40, r: 20, t: 30, b: 40 },
                                                paper_bgcolor: 'transparent',
                                                plot_bgcolor: 'rgba(30,41,59,0.5)',
                                                font: { color: '#94a3b8', size: 10 }
                                            }}
                                            config={{ displayModeBar: false, responsive: true, displaylogo: false }}
                                            style={{ width: '100%', height: '100%' }}
                                            useResizeHandler
                                        />
                                    </div>
                                </div>
                            );
                        }

                        if (textBlock) {
                            return (
                                <div key={itemId} className="rounded-2xl bg-slate-900 border border-white/10 overflow-hidden flex flex-col">
                                    <div className="flex items-center justify-between px-3 py-2 border-b border-white/5">
                                        <div className="flex items-center gap-2">
                                            <GripVertical className="drag-handle w-4 h-4 text-slate-600 cursor-grab" />
                                            <Type className="w-3.5 h-3.5 text-slate-500" />
                                            <span className="text-xs text-slate-500">Text Block</span>
                                        </div>
                                        <button onClick={() => removeItem(itemId)} className="p-1 hover:bg-red-500/10 rounded text-slate-500 hover:text-red-400 transition-colors">
                                            <X className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                    <div className="flex-1 p-4">
                                        {editingText === itemId ? (
                                            <div className="h-full flex flex-col gap-2">
                                                <textarea
                                                    autoFocus
                                                    value={editContent}
                                                    onChange={(e) => setEditContent(e.target.value)}
                                                    className="flex-1 bg-slate-800 border border-indigo-500/30 rounded-lg p-3 text-sm text-white resize-none focus:outline-none"
                                                />
                                                <div className="flex gap-2">
                                                    <button onClick={() => saveTextEdit(itemId)} className="px-3 py-1 text-xs bg-indigo-600 rounded text-white hover:bg-indigo-500">Save</button>
                                                    <button onClick={() => setEditingText(null)} className="px-3 py-1 text-xs bg-slate-700 rounded text-slate-300 hover:bg-slate-600">Cancel</button>
                                                </div>
                                            </div>
                                        ) : (
                                            <div
                                                onClick={() => { setEditingText(itemId); setEditContent(textBlock.content); }}
                                                className="h-full text-sm text-slate-300 cursor-text whitespace-pre-wrap hover:bg-slate-800/50 rounded-lg p-2 transition-colors"
                                            >
                                                {textBlock.content}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        }

                        return (
                            <div key={itemId} className="rounded-2xl bg-slate-900 border border-white/10 p-4 flex items-center justify-center">
                                <p className="text-slate-500 text-sm">Chart not found</p>
                            </div>
                        );
                    })}
                </RG>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-50">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/insights" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                            <ArrowLeft className="w-5 h-5" />
                            <span className="font-medium">Back</span>
                        </Link>
                        <div className="flex items-center gap-2">
                            <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                                <LayoutDashboard className="w-5 h-5" />
                            </div>
                            <span className="font-bold text-lg tracking-tight">Dashboard</span>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={addTextBlock} className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 border border-white/10 rounded-lg text-sm transition-colors">
                            <Type className="w-4 h-4" />
                            Add Text
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${saved ? "bg-emerald-600 text-white" : "bg-indigo-600 hover:bg-indigo-500 text-white"
                                } disabled:opacity-50`}
                        >
                            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> :
                                saved ? <Check className="w-4 h-4" /> :
                                    <Save className="w-4 h-4" />}
                            {saved ? "Saved!" : "Save Layout"}
                        </button>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-6">
                {pinnedIds.length === 0 ? (
                    <div className="text-center py-20">
                        <Pin className="w-16 h-16 text-slate-700 mx-auto mb-4" />
                        <h2 className="text-xl font-semibold mb-2">No charts pinned</h2>
                        <p className="text-slate-400 mb-6">Go to the Insights page and pin charts to build your dashboard.</p>
                        <Link href="/insights" className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-medium transition-all">
                            <ArrowLeft className="w-4 h-4" />
                            Back to Insights
                        </Link>
                    </div>
                ) : (
                    renderGrid()
                )}
            </main>
        </div>
    );
}
