"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
    ArrowLeft,
    ArrowRight,
    Brain,
    Loader2,
    Trophy,
    Target,
    Zap,
    BarChart
} from "lucide-react";

const API_BASE = "http://localhost:8000";

interface ModelResult {
    model_name: string;
    model_type: string;
    accuracy_or_r2: number;
    secondary_metric: number;
    feature_importance: Record<string, number>;
}

interface ModelingData {
    session_id: string;
    target_column: string;
    goal: string;
    best_model: string;
    models: ModelResult[];
    prediction_sample: Record<string, unknown>[];
}

interface SessionData {
    session_id: string;
    columns: { name: string; dtype: string }[];
}

export default function ModelingPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [columns, setColumns] = useState<{ name: string; dtype: string }[]>([]);
    const [data, setData] = useState<ModelingData | null>(null);
    const [error, setError] = useState<string | null>(null);

    // Form state
    const [targetColumn, setTargetColumn] = useState("");
    const [goal, setGoal] = useState<"predict" | "classify">("predict");

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }
        const session = JSON.parse(stored);
        setSessionId(session.session_id);
        setColumns(session.columns || []);
    }, [router]);

    const numericColumns = columns.filter(c =>
        ["Int64", "Int32", "Float64", "Float32"].some(t => c.dtype.includes(t))
    );

    const handleTrain = async () => {
        if (!sessionId || !targetColumn) return;

        setLoading(true);
        setError(null);
        setData(null);

        try {
            const response = await fetch(`${API_BASE}/model/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    target_column: targetColumn,
                    goal: goal
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Training failed");
            }

            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const getMetricLabel = (modelType: string) => {
        return modelType === "classification" ? "Accuracy" : "R² Score";
    };

    const getSecondaryLabel = (modelType: string) => {
        return modelType === "classification" ? "F1 Score" : "MSE";
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/chat" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <Brain className="w-5 h-5" />
                        </div>
                        <span className="font-bold text-lg tracking-tight">Smart Modeling</span>
                    </div>
                    <div className="w-20" />
                </div>
            </header>

            <main className="container mx-auto px-4 py-10 max-w-4xl">
                {/* Configuration */}
                {!data && (
                    <div className="p-6 rounded-2xl bg-slate-900 border border-white/10 mb-8">
                        <h2 className="text-xl font-semibold mb-6">Configure Model</h2>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2">Target Column (What to predict)</label>
                                <select
                                    value={targetColumn}
                                    onChange={(e) => setTargetColumn(e.target.value)}
                                    className="w-full bg-slate-800 border border-white/10 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                >
                                    <option value="">Select column...</option>
                                    {numericColumns.map(col => (
                                        <option key={col.name} value={col.name}>{col.name}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-slate-400 mb-2">Goal</label>
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setGoal("predict")}
                                        className={`flex-1 px-4 py-3 rounded-lg border transition-colors ${goal === "predict"
                                                ? "bg-indigo-600 border-indigo-500 text-white"
                                                : "bg-slate-800 border-white/10 text-slate-400 hover:bg-slate-700"
                                            }`}
                                    >
                                        <Target className="w-5 h-5 mx-auto mb-1" />
                                        Predict (Regression)
                                    </button>
                                    <button
                                        onClick={() => setGoal("classify")}
                                        className={`flex-1 px-4 py-3 rounded-lg border transition-colors ${goal === "classify"
                                                ? "bg-indigo-600 border-indigo-500 text-white"
                                                : "bg-slate-800 border-white/10 text-slate-400 hover:bg-slate-700"
                                            }`}
                                    >
                                        <Zap className="w-5 h-5 mx-auto mb-1" />
                                        Classify
                                    </button>
                                </div>
                            </div>
                        </div>

                        {error && (
                            <div className="mt-4 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                                {error}
                            </div>
                        )}

                        <button
                            onClick={handleTrain}
                            disabled={!targetColumn || loading}
                            className="mt-6 w-full px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Training Models...
                                </>
                            ) : (
                                <>
                                    <Brain className="w-5 h-5" />
                                    Auto Build Models
                                </>
                            )}
                        </button>
                    </div>
                )}

                {/* Results */}
                {data && (
                    <>
                        {/* Best Model Banner */}
                        <div className="p-6 rounded-2xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 mb-8">
                            <div className="flex items-center gap-3 mb-2">
                                <Trophy className="w-6 h-6 text-yellow-400" />
                                <h2 className="text-xl font-semibold">Best Model: {data.best_model}</h2>
                            </div>
                            <p className="text-slate-400">
                                Trained to {data.goal === "classify" ? "classify" : "predict"} <span className="text-white font-medium">{data.target_column}</span>
                            </p>
                        </div>

                        {/* Model Comparison */}
                        <div className="mb-8">
                            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <BarChart className="w-5 h-5 text-indigo-400" />
                                Model Comparison
                            </h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {data.models.map((model, i) => (
                                    <div
                                        key={i}
                                        className={`p-5 rounded-xl border ${model.model_name === data.best_model
                                                ? "bg-green-500/10 border-green-500/30"
                                                : "bg-slate-900 border-white/10"
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-3">
                                            <h4 className="font-semibold">{model.model_name}</h4>
                                            {model.model_name === data.best_model && (
                                                <Trophy className="w-4 h-4 text-yellow-400" />
                                            )}
                                        </div>
                                        <div className="space-y-2">
                                            <div className="flex justify-between">
                                                <span className="text-slate-400 text-sm">{getMetricLabel(model.model_type)}</span>
                                                <span className="font-mono">{(model.accuracy_or_r2 * 100).toFixed(1)}%</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-slate-400 text-sm">{getSecondaryLabel(model.model_type)}</span>
                                                <span className="font-mono">{model.secondary_metric.toFixed(4)}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Feature Importance */}
                        {data.models[0]?.feature_importance && Object.keys(data.models[0].feature_importance).length > 0 && (
                            <div className="mb-8 p-6 rounded-2xl bg-slate-900 border border-white/10">
                                <h3 className="text-lg font-semibold mb-4">Feature Importance (Best Model)</h3>
                                <div className="space-y-3">
                                    {Object.entries(
                                        data.models.find(m => m.model_name === data.best_model)?.feature_importance || {}
                                    )
                                        .sort(([, a], [, b]) => b - a)
                                        .slice(0, 5)
                                        .map(([feature, importance]) => {
                                            const maxImportance = Math.max(...Object.values(
                                                data.models.find(m => m.model_name === data.best_model)?.feature_importance || {}
                                            ));
                                            const pct = (importance / maxImportance) * 100;
                                            return (
                                                <div key={feature} className="flex items-center gap-3">
                                                    <div className="w-32 truncate text-sm text-slate-400">{feature}</div>
                                                    <div className="flex-1 h-4 bg-slate-800 rounded-full overflow-hidden">
                                                        <div
                                                            className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 rounded-full"
                                                            style={{ width: `${pct}%` }}
                                                        />
                                                    </div>
                                                    <div className="w-16 text-sm text-right">{(importance * 100).toFixed(1)}%</div>
                                                </div>
                                            );
                                        })}
                                </div>
                            </div>
                        )}

                        {/* Navigation */}
                        <div className="flex justify-between">
                            <button
                                onClick={() => setData(null)}
                                className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 font-medium transition-colors"
                            >
                                Train New Model
                            </button>
                            <button
                                onClick={() => router.push("/report")}
                                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors"
                            >
                                Generate Report
                                <ArrowRight className="w-4 h-4" />
                            </button>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
}
