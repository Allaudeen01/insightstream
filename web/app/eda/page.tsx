"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
    ArrowLeft,
    ArrowRight,
    BarChart3,
    PieChart,
    TrendingUp,
    Loader2,
    Lightbulb,
    Download,
    CheckCircle2,
    Search,
    Grid,
    Target
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ColumnStats {
    column: string;
    dtype: string;
    role?: string;
    mean?: number;
    median?: number;
    std?: number;
    min_val?: number;
    max_val?: number;
    unique_count: number;
    top_values: Record<string, unknown>[];
}

interface EDAData {
    session_id: string;
    numeric_columns: string[];
    categorical_columns: string[];
    date_columns: string[];
    identifier_columns: string[];
    binary_columns: string[];
    column_stats: ColumnStats[];
    correlation_matrix: Record<string, Record<string, number>>;
    insights: string[];
    warnings: string[];
}

export default function EDAPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [edaData, setEdaData] = useState<EDAData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"overview" | "correlation" | "insights">("overview");

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        fetchEDA(session.session_id);
    }, [router]);

    const fetchEDA = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/eda/${sessionId}`);
            if (!response.ok) throw new Error("Failed to fetch dimensions");
            const data = await response.json();
            setEdaData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Network protocol error");
        } finally {
            setLoading(false);
        }
    };

    const getCorrelationColor = (value: number) => {
        if (value >= 0.7) return "bg-green-500/40 text-green-300 border-green-500/20";
        if (value >= 0.4) return "bg-green-500/20 text-green-400 border-green-500/10";
        if (value >= 0) return "bg-white/5 text-muted-foreground border-white/5";
        if (value >= -0.4) return "bg-red-500/20 text-red-400 border-red-500/10";
        return "bg-red-500/40 text-red-300 border-red-500/20";
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center">
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin" />
                    <Search className="w-6 h-6 text-purple-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <p className="mt-6 text-muted-foreground font-medium animate-pulse">Scanning Data Dimensions...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            <Navbar />

            <main className="container mx-auto px-6 py-10 max-w-7xl">
                {error && (
                    <div className="mb-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 flex items-center gap-3">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
                        {error}
                    </div>
                )}

                {edaData && (
                    <>
                        {/* Dimensional Summary */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                            {[
                                { label: 'Numerical', icon: BarChart3, count: edaData.numeric_columns.length, color: 'text-blue-400', bg: 'bg-blue-400/10' },
                                { label: 'Categorical', icon: PieChart, count: edaData.categorical_columns.length, color: 'text-purple-400', bg: 'bg-purple-400/10' },
                                { label: 'Temporal', icon: TrendingUp, count: edaData.date_columns.length, color: 'text-emerald-400', bg: 'bg-emerald-400/10' },
                                { label: 'Identifiers', icon: Target, count: edaData.identifier_columns?.length || 0, color: 'text-muted-foreground', bg: 'bg-white/5', sub: 'Excluded' },
                            ].map((stat, i) => (
                                <div key={i} className="p-6 rounded-2xl bg-white/5 border border-white/10 flex items-center gap-5 hover:border-white/20 transition-all">
                                    <div className={`p-4 rounded-xl ${stat.bg}`}>
                                        <stat.icon className={`w-6 h-6 ${stat.color}`} />
                                    </div>
                                    <div>
                                        <p className="text-xs font-bold text-muted-foreground uppercase tracking-widest mb-1">{stat.label}</p>
                                        <div className="flex items-baseline gap-2">
                                            <p className="text-3xl font-bold text-foreground">{stat.count}</p>
                                            {stat.sub && <span className="text-[10px] text-red-400/80 font-bold uppercase">{stat.sub}</span>}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Page Header */}
                        <div className="flex items-end justify-between mb-8 border-b border-white/10 pb-6">
                            <div>
                                <h1 className="text-3xl font-bold text-foreground tracking-tight">Data Explorer</h1>
                                <p className="text-muted-foreground mt-1">Deep structural analysis of your workspace</p>
                            </div>
                            
                            <div className="flex p-1.5 bg-white/5 rounded-2xl border border-white/10">
                                {(["overview", "correlation", "insights"] as const).map(tab => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`px-5 py-2 rounded-xl text-sm font-bold transition-all ${activeTab === tab
                                            ? "bg-purple-600 text-white shadow-lg shadow-purple-500/20"
                                            : "text-muted-foreground hover:text-foreground"
                                        }`}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Overview Table */}
                        {activeTab === "overview" && (
                            <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
                                <div className="overflow-x-auto">
                                    <table className="w-full text-left">
                                        <thead>
                                            <tr className="bg-white/5 border-b border-white/10">
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest">Dimension</th>
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest">Logical Role</th>
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest">DType</th>
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest">Centrality</th>
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest">Variance</th>
                                                <th className="py-4 px-6 text-xs font-bold text-muted-foreground uppercase tracking-widest text-right">Cardinality</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-white/5">
                                            {edaData.column_stats.map((stat, i) => (
                                                <tr key={i} className={`hover:bg-white/5 transition-colors ${stat.role === "identifier" ? "opacity-40 grayscale" : ""}`}>
                                                    <td className="py-4 px-6 font-semibold text-foreground">
                                                        {stat.column}
                                                    </td>
                                                    <td className="py-4 px-6">
                                                        <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-wider border ${
                                                            stat.role === "identifier" ? "bg-red-500/10 text-red-300 border-red-500/20" :
                                                            stat.role === "numerical" ? "bg-blue-500/10 text-blue-300 border-blue-500/20" :
                                                            stat.role === "categorical" ? "bg-purple-500/10 text-purple-300 border-purple-500/20" :
                                                            stat.role === "temporal" ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20" :
                                                            "bg-white/5 text-muted-foreground border-white/10"
                                                        }`}>{stat.role || "unknown"}</span>
                                                    </td>
                                                    <td className="py-4 px-6 text-xs text-muted-foreground font-medium">{stat.dtype}</td>
                                                    <td className="py-4 px-6 text-sm text-foreground">{stat.mean ? stat.mean.toFixed(2) : "-"}</td>
                                                    <td className="py-4 px-6 text-sm text-foreground">{stat.std ? stat.std.toFixed(2) : "-"}</td>
                                                    <td className="py-4 px-6 text-sm text-foreground text-right font-mono">{stat.unique_count}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {/* Correlation Heatmap */}
                        {activeTab === "correlation" && (
                            <div className="rounded-2xl border border-white/10 bg-white/5 p-8">
                                <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                                    <Grid className="w-5 h-5 text-purple-400" />
                                    Multivariate Dependencies
                                </h2>
                                {edaData.numeric_columns.length < 2 ? (
                                    <div className="py-20 text-center">
                                        <p className="text-muted-foreground">Insufficient numerical dimensions for correlation analysis.</p>
                                    </div>
                                ) : (
                                    <div className="overflow-x-auto pb-4">
                                        <table className="border-separate border-spacing-1">
                                            <thead>
                                                <tr>
                                                    <th className="p-2"></th>
                                                    {edaData.numeric_columns.map(col => (
                                                        <th key={col} className="p-3 text-[10px] font-bold text-muted-foreground uppercase tracking-widest text-center vertical-text-header">
                                                            <div className="max-w-[100px] truncate">{col}</div>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {edaData.numeric_columns.map(row => (
                                                    <tr key={row}>
                                                        <td className="p-3 text-[10px] font-bold text-muted-foreground uppercase tracking-widest text-right whitespace-nowrap">{row}</td>
                                                        {edaData.numeric_columns.map(col => {
                                                            const val = edaData.correlation_matrix[row]?.[col] ?? 0;
                                                            return (
                                                                <td key={col} className="p-0.5">
                                                                    <div
                                                                        className={`w-14 h-14 rounded-lg flex items-center justify-center text-xs font-bold border transition-all hover:scale-105 cursor-default ${getCorrelationColor(val)}`}
                                                                        title={`${row} vs ${col}: ${val.toFixed(4)}`}
                                                                    >
                                                                        {val.toFixed(2)}
                                                                    </div>
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Distributional Insights */}
                        {activeTab === "insights" && (
                            <div className="space-y-6">
                                {edaData.warnings.length > 0 && (
                                    <div className="p-5 rounded-2xl bg-amber-500/10 border border-amber-500/20">
                                        <div className="flex items-center gap-2 mb-3">
                                            <Target className="w-4 h-4 text-amber-400" />
                                            <span className="text-xs font-bold text-amber-400 uppercase tracking-widest">Quality Alerts</span>
                                        </div>
                                        <div className="space-y-2">
                                            {edaData.warnings.map((w, i) => (
                                                <p key={i} className="text-sm text-amber-200/80 leading-relaxed">• {w}</p>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {edaData.insights.length === 0 ? (
                                    <div className="py-32 text-center rounded-3xl border border-dashed border-white/10 bg-white/5">
                                        <Lightbulb className="w-12 h-12 text-muted-foreground/20 mx-auto mb-4" />
                                        <p className="text-muted-foreground">Synthesizing patterns... Try adding more dimensions.</p>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {edaData.insights.map((insight, i) => (
                                            <div
                                                key={i}
                                                className="p-6 rounded-2xl bg-white/5 border border-white/10 flex items-start gap-4 hover:border-purple-500/30 transition-all"
                                            >
                                                <div className="p-3 rounded-xl bg-purple-500/20 shrink-0">
                                                    <Lightbulb className="w-5 h-5 text-purple-400" />
                                                </div>
                                                <p className="text-foreground leading-relaxed pt-1">{insight}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Persistent Controls */}
                        <div className="flex items-center justify-between mt-16 pt-8 border-t border-white/10">
                            <button
                                onClick={() => router.push("/health-check")}
                                className="px-8 py-3 rounded-2xl bg-white/5 hover:bg-white/10 text-foreground font-bold transition-all border border-white/10"
                            >
                                Re-evaluate Health
                            </button>
                            <button
                                onClick={() => router.push("/insights")}
                                className="flex items-center gap-2 px-10 py-3 rounded-2xl bg-purple-600 hover:bg-purple-500 text-white font-bold transition-all shadow-xl shadow-purple-500/25"
                            >
                                Synthetic Insights
                                <ArrowRight className="w-5 h-5 ml-2" />
                            </button>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
}
