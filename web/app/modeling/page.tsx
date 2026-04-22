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
    BarChart,
    ChevronRight,
    Sparkles,
    Activity,
    Cpu
} from "lucide-react";
import Navbar from "@/components/Navbar";
import KPICard from "@/components/KPICard";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export default function ModelingPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [columns, setColumns] = useState<{ name: string; dtype: string }[]>([]);
    const [data, setData] = useState<ModelingData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

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
                body: JSON.stringify({ target_column: targetColumn, goal })
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
        <div className="min-h-screen bg-background pb-20">
            <Navbar />

            <main className="container mx-auto px-6 py-10 max-w-6xl">
                {/* Hero / Config Header */}
                <div className="relative mb-12 rounded-3xl overflow-hidden border border-white/10 bg-white/5 p-8 md:p-12">
                    <div className="absolute top-0 right-0 w-1/2 h-full bg-indigo-500/10 blur-[120px] -z-10" />
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-8">
                        <div className="max-w-2xl">
                            <div className="flex items-center gap-2 mb-4">
                                <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400">
                                    <Cpu className="w-4 h-4" />
                                </div>
                                <span className="text-sm font-bold text-indigo-400 uppercase tracking-widest">Predictive Studio</span>
                            </div>
                            <h1 className="text-3xl md:text-5xl font-black text-foreground mb-4 tracking-tight">
                                Autonomous <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">Modeling Engine</span>
                            </h1>
                            <p className="text-lg text-muted-foreground leading-relaxed">
                                Transform raw signals into actionable foresight. Select your target variable and let the engine architecture determine the optimal algorithmic path.
                            </p>
                        </div>
                        
                        {!data && (
                            <div className="w-full md:w-80 shrink-0 p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md">
                                <div className="space-y-4">
                                    <div>
                                        <label className="block text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2">Target Variable</label>
                                        <select
                                            value={targetColumn}
                                            onChange={(e) => setTargetColumn(e.target.value)}
                                            className="w-full bg-background border border-white/10 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-indigo-500 transition-all"
                                        >
                                            <option value="">Select column...</option>
                                            {numericColumns.map(col => (
                                                <option key={col.name} value={col.name}>{col.name}</option>
                                            ))}
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2">Analysis Type</label>
                                        <div className="grid grid-cols-2 gap-2">
                                            <button
                                                onClick={() => setGoal("predict")}
                                                className={`px-3 py-2.5 rounded-lg border text-[10px] font-bold uppercase transition-all ${goal === 'predict' ? 'bg-indigo-600 border-indigo-400 shadow-lg shadow-indigo-500/20' : 'bg-white/5 border-white/10 text-muted-foreground'}`}
                                            >
                                                Regression
                                            </button>
                                            <button
                                                onClick={() => setGoal("classify")}
                                                className={`px-3 py-2.5 rounded-lg border text-[10px] font-bold uppercase transition-all ${goal === 'classify' ? 'bg-indigo-600 border-indigo-400 shadow-lg shadow-indigo-500/20' : 'bg-white/5 border-white/10 text-muted-foreground'}`}
                                            >
                                                Classifier
                                            </button>
                                        </div>
                                    </div>
                                    <button
                                        onClick={handleTrain}
                                        disabled={!targetColumn || loading}
                                        className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-xl font-black text-sm uppercase tracking-tighter transition-all flex items-center justify-center gap-2"
                                    >
                                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
                                        {loading ? 'Synthesizing...' : 'Build Model Architecture'}
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {error && (
                    <div className="mb-12 p-6 rounded-2xl bg-red-500/5 border border-red-500/20 text-red-400 animate-page-enter">
                        <div className="flex items-center gap-3">
                            <Activity className="w-5 h-5" />
                            <span className="font-semibold">{error}</span>
                        </div>
                    </div>
                )}

                {/* Results Section */}
                {data && (
                    <div className="animate-page-enter">
                        <div className="flex items-center justify-between mb-8">
                            <div>
                                <h2 className="text-2xl font-black tracking-tight">Deployment Ready Architecture</h2>
                                <p className="text-sm text-muted-foreground">Comparative performance metrics across top candidates</p>
                            </div>
                            <button
                                onClick={() => setData(null)}
                                className="px-5 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-xs font-bold transition-all"
                            >
                                Reset Configuration
                            </button>
                        </div>

                        {/* Best Model Banner */}
                        <div className="relative mb-12 p-8 rounded-3xl bg-green-500/5 border border-green-500/20 overflow-hidden">
                            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/10 blur-[60px]" />
                            <div className="flex flex-col md:flex-row items-center gap-8">
                                <div className="p-6 rounded-2xl bg-green-500/20 text-green-400 border border-green-500/20 shadow-xl shadow-green-500/10">
                                    <Trophy className="w-12 h-12" />
                                </div>
                                <div className="flex-1 text-center md:text-left">
                                    <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
                                        <Sparkles className="w-4 h-4 text-emerald-400" />
                                        <span className="text-[10px] font-black uppercase tracking-widest text-emerald-400">Leaderboard Champion</span>
                                    </div>
                                    <h3 className="text-4xl font-black mb-2 tracking-tighter">{data.best_model}</h3>
                                    <p className="text-muted-foreground font-medium">
                                        Achieved the highest generalized accuracy for targeting <span className="text-foreground font-bold">{data.target_column}</span>.
                                    </p>
                                </div>
                                <div className="grid grid-cols-2 gap-4 w-full md:w-auto">
                                    <KPICard 
                                        title={getMetricLabel(data.models.find(m => m.model_name === data.best_model)?.model_type || "")}
                                        value={`${((data.models.find(m => m.model_name === data.best_model)?.accuracy_or_r2 || 0) * 100).toFixed(1)}%`}
                                    />
                                    <KPICard 
                                        title={getSecondaryLabel(data.models.find(m => m.model_name === data.best_model)?.model_type || "")}
                                        value={(data.models.find(m => m.model_name === data.best_model)?.secondary_metric || 0).toFixed(4)}
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Analysis Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                            {/* Feature Importance */}
                            {data.models[0]?.feature_importance && Object.keys(data.models[0].feature_importance).length > 0 && (
                                <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl">
                                    <div className="flex items-center gap-2 mb-8">
                                        <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                                            <Target className="w-4 h-4" />
                                        </div>
                                        <h3 className="text-xl font-black italic uppercase tracking-tighter">Relative Influence</h3>
                                    </div>
                                    <div className="space-y-6">
                                        {Object.entries(
                                            data.models.find(m => m.model_name === data.best_model)?.feature_importance || {}
                                        )
                                            .sort(([, a], [, b]) => b - a)
                                            .slice(0, 6)
                                            .map(([feature, importance]) => {
                                                const maxImportance = Math.max(...Object.values(
                                                    data.models.find(m => m.model_name === data.best_model)?.feature_importance || {}
                                                ));
                                                const pct = (importance / maxImportance) * 100;
                                                return (
                                                    <div key={feature} className="space-y-2">
                                                        <div className="flex justify-between text-xs font-bold font-mono">
                                                            <span className="text-muted-foreground uppercase">{feature}</span>
                                                            <span className="text-purple-400">{(importance * 100).toFixed(1)}%</span>
                                                        </div>
                                                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full bg-gradient-to-r from-purple-500 to-indigo-500 rounded-full transition-all duration-1000"
                                                                style={{ width: `${pct}%` }}
                                                            />
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                    </div>
                                </div>
                            )}

                            {/* Candidate Leaderboard */}
                            <div className="p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl">
                                <div className="flex items-center gap-2 mb-8">
                                    <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400">
                                        <BarChart className="w-4 h-4" />
                                    </div>
                                    <h3 className="text-xl font-black italic uppercase tracking-tighter">Candidate Pool</h3>
                                </div>
                                <div className="space-y-4">
                                    {data.models.map((model, i) => (
                                        <div
                                            key={i}
                                            className={`p-4 rounded-2xl border transition-all ${model.model_name === data.best_model ? 'bg-green-500/10 border-green-500/20 ring-1 ring-green-500/30' : 'bg-white/5 border-white/10 hover:bg-white/10'}`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <h4 className="font-bold text-sm">{model.model_name}</h4>
                                                    <p className="text-[10px] text-muted-foreground uppercase tracking-widest">{model.model_type}</p>
                                                </div>
                                                <div className="text-right">
                                                    <div className="text-sm font-black text-foreground">{(model.accuracy_or_r2 * 100).toFixed(1)}%</div>
                                                    <div className="text-[10px] text-muted-foreground font-mono">{model.secondary_metric.toFixed(4)}</div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Navigation Actions */}
                        <div className="flex flex-col md:flex-row justify-between items-center gap-6 p-10 rounded-[3rem] bg-indigo-600/10 border border-indigo-500/20">
                            <div>
                                <h4 className="text-xl font-black tracking-tight">Synthesize the Knowledge</h4>
                                <p className="text-sm text-indigo-300 font-medium leading-relaxed">The model is tuned. Next, generate a comprehensive strategic brief for stakeholders.</p>
                            </div>
                            <div className="flex gap-4 w-full md:w-auto">
                                <button
                                    onClick={() => setData(null)}
                                    className="flex-1 md:flex-none px-8 py-4 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 font-bold text-sm transition-all"
                                >
                                    Reconfigure
                                </button>
                                <button
                                    onClick={() => router.push("/report")}
                                    className="flex-1 md:flex-none flex items-center justify-center gap-2 px-10 py-4 rounded-2xl bg-indigo-600 hover:bg-indigo-500 font-bold text-sm transition-all shadow-xl shadow-indigo-500/20"
                                >
                                    Generate Brief
                                    <ArrowRight className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
