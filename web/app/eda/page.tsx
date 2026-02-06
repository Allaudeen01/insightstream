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
    Download
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ColumnStats {
    column: string;
    dtype: string;
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
    column_stats: ColumnStats[];
    correlation_matrix: Record<string, Record<string, number>>;
    insights: string[];
}

export default function EDAPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [edaData, setEdaData] = useState<EDAData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"overview" | "correlation" | "insights">("overview");

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }

        const session = JSON.parse(stored);
        fetchEDA(session.session_id);
    }, [router]);

    const fetchEDA = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/eda/${sessionId}`);
            if (!response.ok) throw new Error("Failed to fetch EDA");
            const data = await response.json();
            setEdaData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const getCorrelationColor = (value: number) => {
        if (value >= 0.7) return "bg-green-500";
        if (value >= 0.4) return "bg-green-400/60";
        if (value >= 0) return "bg-slate-700";
        if (value >= -0.4) return "bg-red-400/60";
        return "bg-red-500";
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/health-check" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <span className="font-bold text-lg">V</span>
                        </div>
                        <span className="font-bold text-lg tracking-tight">Auto EDA</span>
                    </div>
                    <div className="w-20" />
                </div>
            </header>

            <main className="container mx-auto px-4 py-10">
                {error && (
                    <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                        {error}
                    </div>
                )}

                {edaData && (
                    <>
                        {/* Column Type Summary */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                            <div className="p-6 rounded-2xl bg-slate-900 border border-white/10 flex items-center gap-4">
                                <div className="p-3 rounded-xl bg-blue-500/20">
                                    <BarChart3 className="w-6 h-6 text-blue-400" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Numeric Columns</p>
                                    <p className="text-2xl font-bold">{edaData.numeric_columns.length}</p>
                                </div>
                            </div>
                            <div className="p-6 rounded-2xl bg-slate-900 border border-white/10 flex items-center gap-4">
                                <div className="p-3 rounded-xl bg-purple-500/20">
                                    <PieChart className="w-6 h-6 text-purple-400" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Categorical Columns</p>
                                    <p className="text-2xl font-bold">{edaData.categorical_columns.length}</p>
                                </div>
                            </div>
                            <div className="p-6 rounded-2xl bg-slate-900 border border-white/10 flex items-center gap-4">
                                <div className="p-3 rounded-xl bg-green-500/20">
                                    <TrendingUp className="w-6 h-6 text-green-400" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-sm">Date Columns</p>
                                    <p className="text-2xl font-bold">{edaData.date_columns.length}</p>
                                </div>
                            </div>
                        </div>

                        {/* Tabs */}
                        <div className="flex gap-2 mb-6">
                            {(["overview", "correlation", "insights"] as const).map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === tab
                                        ? "bg-indigo-600 text-white"
                                        : "bg-white/5 text-slate-400 hover:bg-white/10"
                                        }`}
                                >
                                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                </button>
                            ))}
                        </div>

                        {/* Tab Content */}
                        {activeTab === "overview" && (
                            <div className="space-y-4">
                                <h2 className="text-xl font-semibold mb-4">Column Statistics</h2>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-left">
                                        <thead>
                                            <tr className="border-b border-white/10">
                                                <th className="py-3 px-4 text-slate-400 font-medium">Column</th>
                                                <th className="py-3 px-4 text-slate-400 font-medium">Type</th>
                                                <th className="py-3 px-4 text-slate-400 font-medium">Mean</th>
                                                <th className="py-3 px-4 text-slate-400 font-medium">Median</th>
                                                <th className="py-3 px-4 text-slate-400 font-medium">Std Dev</th>
                                                <th className="py-3 px-4 text-slate-400 font-medium">Unique</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {edaData.column_stats.map((stat, i) => (
                                                <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                                                    <td className="py-3 px-4 font-medium">{stat.column}</td>
                                                    <td className="py-3 px-4">
                                                        <span className="text-xs px-2 py-1 rounded bg-white/10">{stat.dtype}</span>
                                                    </td>
                                                    <td className="py-3 px-4 text-slate-300">{stat.mean ?? "-"}</td>
                                                    <td className="py-3 px-4 text-slate-300">{stat.median ?? "-"}</td>
                                                    <td className="py-3 px-4 text-slate-300">{stat.std ?? "-"}</td>
                                                    <td className="py-3 px-4 text-slate-300">{stat.unique_count}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        )}

                        {activeTab === "correlation" && (
                            <div>
                                <h2 className="text-xl font-semibold mb-4">Correlation Matrix</h2>
                                {edaData.numeric_columns.length < 2 ? (
                                    <p className="text-slate-400">Need at least 2 numeric columns for correlation analysis.</p>
                                ) : (
                                    <div className="overflow-x-auto">
                                        <table className="text-sm">
                                            <thead>
                                                <tr>
                                                    <th className="p-2"></th>
                                                    {edaData.numeric_columns.map(col => (
                                                        <th key={col} className="p-2 text-slate-400 font-medium text-xs truncate max-w-[80px]">
                                                            {col}
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {edaData.numeric_columns.map(row => (
                                                    <tr key={row}>
                                                        <td className="p-2 text-slate-400 font-medium text-xs truncate max-w-[80px]">{row}</td>
                                                        {edaData.numeric_columns.map(col => {
                                                            const val = edaData.correlation_matrix[row]?.[col] ?? 0;
                                                            return (
                                                                <td key={col} className="p-1">
                                                                    <div
                                                                        className={`w-12 h-12 rounded flex items-center justify-center text-xs font-medium ${getCorrelationColor(val)}`}
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

                        {activeTab === "insights" && (
                            <div>
                                <h2 className="text-xl font-semibold mb-4">Auto-Generated Insights</h2>
                                {edaData.insights.length === 0 ? (
                                    <div className="p-8 rounded-2xl bg-slate-900 border border-white/10 text-center">
                                        <Lightbulb className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                                        <p className="text-slate-400">No significant insights detected in this dataset.</p>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {edaData.insights.map((insight, i) => (
                                            <div
                                                key={i}
                                                className="p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-start gap-3"
                                            >
                                                <Lightbulb className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                                                <p className="text-white">{insight}</p>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Navigation */}
                        <div className="flex justify-between mt-10">
                            <button
                                onClick={() => router.push("/health-check")}
                                className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 font-medium transition-colors"
                            >
                                Back to Health Check
                            </button>
                            <button
                                onClick={() => router.push("/insights")}
                                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors"
                            >
                                View Insights
                                <ArrowRight className="w-4 h-4" />
                            </button>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
}
