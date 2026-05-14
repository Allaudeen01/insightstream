"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
    ArrowRight,
    BarChart3,
    PieChart,
    TrendingUp,
    Lightbulb,
    Search,
    Grid,
    Target,
    AlertTriangle
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import { apiFetch } from "@/lib/api";

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
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [edaData, setEdaData] = useState<EDAData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<"overview" | "correlation" | "insights">("overview");

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const sid = urlParams.get("session");
        if (!sid) { router.push("/upload"); return; }
        setSessionId(sid);
        fetchEDA(sid);
    }, [router]);

    const fetchEDA = async (sid: string) => {
        try {
            const response = await apiFetch(`/eda/${sid}`);
            if (!response.ok) throw new Error("Failed to fetch dimensions");
            const data = await response.json();
            setEdaData(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Couldn't load the data. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const getCorrelationColor = (value: number) => {
        // Light-theme heatmap: green for positive, red for negative, neutral near zero.
        if (value >= 0.7) return "bg-emerald-200 text-emerald-900 border-emerald-300";
        if (value >= 0.4) return "bg-emerald-100 text-emerald-800 border-emerald-200";
        if (value >= 0) return "bg-slate-50 text-slate-600 border-slate-200";
        if (value >= -0.4) return "bg-rose-100 text-rose-800 border-rose-200";
        return "bg-rose-200 text-rose-900 border-rose-300";
    };

    const roleBadgeClass = (role?: string) => {
        switch (role) {
            case "identifier":
                return "bg-slate-100 text-slate-600 border-slate-200";
            case "numerical":
                return "bg-blue-50 text-blue-700 border-blue-200";
            case "categorical":
                return "bg-indigo-50 text-indigo-700 border-indigo-200";
            case "temporal":
                return "bg-emerald-50 text-emerald-700 border-emerald-200";
            default:
                return "bg-slate-50 text-slate-600 border-slate-200";
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-50 flex">
                <Sidebar />
                <div className="flex-1 flex flex-col items-center justify-center">
                    <div className="relative">
                        <div className="w-12 h-12 border-2 border-slate-200 border-t-indigo-600 rounded-full animate-spin" />
                        <Search className="w-4 h-4 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                    </div>
                    <p className="mt-5 text-sm text-slate-500">Loading your data…</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-50 flex">
            <Sidebar />

            <main className="flex-1 px-8 py-10">
                <div className="max-w-6xl mx-auto">
                    {/* Page header */}
                    <header className="mb-8">
                        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Explore data</h1>
                        <p className="text-sm text-slate-500 mt-1">
                            A summary of your columns, how they relate, and things worth knowing.
                        </p>
                    </header>

                    {error && (
                        <div className="mb-8 p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-700 text-sm flex items-start gap-3">
                            <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {edaData && (
                        <>
                            {/* Summary cards */}
                            <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
                                {[
                                    { label: "Numeric", icon: BarChart3, count: edaData.numeric_columns.length },
                                    { label: "Categorical", icon: PieChart, count: edaData.categorical_columns.length },
                                    { label: "Dates", icon: TrendingUp, count: edaData.date_columns.length },
                                    {
                                        label: "Identifiers",
                                        icon: Target,
                                        count: edaData.identifier_columns?.length || 0,
                                        sub: "Excluded from analysis"
                                    },
                                ].map((stat, i) => (
                                    <div
                                        key={i}
                                        className="p-5 rounded-lg bg-white border border-slate-200 flex items-start gap-4"
                                    >
                                        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
                                            <stat.icon className="w-5 h-5 text-slate-600" />
                                        </div>
                                        <div>
                                            <p className="text-xs text-slate-500 mb-1">{stat.label}</p>
                                            <p className="text-2xl font-semibold text-slate-900 leading-none">{stat.count}</p>
                                            {stat.sub && (
                                                <p className="text-[11px] text-slate-400 mt-1.5">{stat.sub}</p>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </section>

                            {/* Tabs */}
                            <div className="flex items-center gap-1 mb-6 border-b border-slate-200">
                                {([
                                    { id: "overview", label: "Overview" },
                                    { id: "correlation", label: "Correlation" },
                                    { id: "insights", label: "Insights" },
                                ] as const).map(tab => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`px-4 py-2.5 text-sm font-medium -mb-px border-b-2 transition-colors ${activeTab === tab.id
                                                ? "border-indigo-600 text-indigo-700"
                                                : "border-transparent text-slate-500 hover:text-slate-800"
                                            }`}
                                    >
                                        {tab.label}
                                    </button>
                                ))}
                            </div>

                            {/* Overview table */}
                            {activeTab === "overview" && (
                                <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-left text-sm">
                                            <thead>
                                                <tr className="bg-slate-50 border-b border-slate-200">
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500">Column</th>
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500">Role</th>
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500">Type</th>
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500">Mean</th>
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500">Std. dev</th>
                                                    <th className="py-3 px-5 text-xs font-medium text-slate-500 text-right">Unique values</th>
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-100">
                                                {edaData.column_stats.map((stat, i) => (
                                                    <tr
                                                        key={i}
                                                        className={`hover:bg-slate-50/60 transition-colors ${stat.role === "identifier" ? "text-slate-400" : ""
                                                            }`}
                                                    >
                                                        <td className="py-3 px-5 font-medium text-slate-900">
                                                            {stat.column}
                                                        </td>
                                                        <td className="py-3 px-5">
                                                            <span
                                                                className={`text-xs px-2 py-0.5 rounded-full font-medium border ${roleBadgeClass(stat.role)}`}
                                                            >
                                                                {stat.role || "unknown"}
                                                            </span>
                                                        </td>
                                                        <td className="py-3 px-5 text-slate-500 font-mono text-xs">{stat.dtype}</td>
                                                        <td className="py-3 px-5 text-slate-700">
                                                            {stat.mean !== undefined && stat.mean !== null ? stat.mean.toFixed(2) : "—"}
                                                        </td>
                                                        <td className="py-3 px-5 text-slate-700">
                                                            {stat.std !== undefined && stat.std !== null ? stat.std.toFixed(2) : "—"}
                                                        </td>
                                                        <td className="py-3 px-5 text-slate-700 text-right font-mono">
                                                            {stat.unique_count}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            )}

                            {/* Correlation heatmap */}
                            {activeTab === "correlation" && (
                                <div className="rounded-lg border border-slate-200 bg-white p-6">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Grid className="w-4 h-4 text-slate-500" />
                                        <h2 className="text-base font-semibold text-slate-900">How columns relate</h2>
                                    </div>
                                    <p className="text-sm text-slate-500 mb-6">
                                        Values close to 1 or −1 mean two columns move together. Values near 0 mean they don&apos;t.
                                    </p>
                                    {edaData.numeric_columns.length < 2 ? (
                                        <div className="py-16 text-center text-sm text-slate-500">
                                            You need at least two numeric columns to see correlations.
                                        </div>
                                    ) : (
                                        <div className="overflow-x-auto pb-2">
                                            <table className="border-separate border-spacing-1">
                                                <thead>
                                                    <tr>
                                                        <th className="p-2"></th>
                                                        {edaData.numeric_columns.map(col => (
                                                            <th
                                                                key={col}
                                                                className="p-2 text-xs font-medium text-slate-500 text-center"
                                                            >
                                                                <div className="max-w-[100px] truncate">{col}</div>
                                                            </th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {edaData.numeric_columns.map(row => (
                                                        <tr key={row}>
                                                            <td className="p-2 text-xs font-medium text-slate-500 text-right whitespace-nowrap">
                                                                {row}
                                                            </td>
                                                            {edaData.numeric_columns.map(col => {
                                                                const val = edaData.correlation_matrix[row]?.[col] ?? 0;
                                                                return (
                                                                    <td key={col} className="p-0.5">
                                                                        <div
                                                                            className={`w-14 h-14 rounded-md flex items-center justify-center text-xs font-medium border cursor-default ${getCorrelationColor(val)}`}
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

                            {/* Insights */}
                            {activeTab === "insights" && (
                                <div className="space-y-4">
                                    {edaData.warnings.length > 0 && (
                                        <div className="p-5 rounded-lg bg-amber-50 border border-amber-200">
                                            <div className="flex items-center gap-2 mb-2">
                                                <AlertTriangle className="w-4 h-4 text-amber-600" />
                                                <span className="text-sm font-semibold text-amber-800">Things to check</span>
                                            </div>
                                            <ul className="space-y-1.5">
                                                {edaData.warnings.map((w, i) => (
                                                    <li key={i} className="text-sm text-amber-800/90 leading-relaxed">
                                                        • {w}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}

                                    {edaData.insights.length === 0 ? (
                                        <div className="py-24 text-center rounded-lg border border-dashed border-slate-200 bg-white">
                                            <Lightbulb className="w-8 h-8 text-slate-300 mx-auto mb-3" />
                                            <p className="text-sm text-slate-500">
                                                No insights yet. Try adding more columns to your data.
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                            {edaData.insights.map((insight, i) => (
                                                <div
                                                    key={i}
                                                    className="p-5 rounded-lg bg-white border border-slate-200 flex items-start gap-3"
                                                >
                                                    <div className="p-2 rounded-md bg-indigo-50 border border-indigo-100 shrink-0">
                                                        <Lightbulb className="w-4 h-4 text-indigo-600" />
                                                    </div>
                                                    <p className="text-sm text-slate-700 leading-relaxed pt-0.5">
                                                        {insight}
                                                    </p>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Footer actions */}
                            <div className="flex items-center justify-between mt-12 pt-6 border-t border-slate-200">
                                <button
                                    onClick={() => router.push(`/health-check?session=${sessionId}`)}
                                    className="px-5 py-2.5 rounded-md bg-white hover:bg-slate-50 text-slate-700 text-sm font-medium border border-slate-200 transition-colors"
                                >
                                    Back to health check
                                </button>
                                <button
                                    onClick={() => router.push(`/insights?session=${sessionId}`)}
                                    className="flex items-center gap-2 px-5 py-2.5 rounded-md bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors"
                                >
                                    See insights
                                    <ArrowRight className="w-4 h-4" />
                                </button>
                            </div>
                        </>
                    )}
                </div>
            </main>
        </div>
    );
}