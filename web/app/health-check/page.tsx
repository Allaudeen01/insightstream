"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  ArrowRight,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  Sparkles,
  Table2,
  X,
  Settings2,
  RotateCcw,
  Eye,
  Check,
  ShieldCheck,
  Activity,
  Layers,
  Search,
  ChevronRight,
  Zap,
  BadgeCheck
} from "lucide-react";
import Navbar from "@/components/Navbar";
import KPICard from "@/components/KPICard";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface IssueItem {
  column: string;
  issue_type: string;
  severity: string;
  description: string;
  count: number;
  percentage: number;
}

interface HealthCheckData {
  session_id: string;
  quality_score: string;
  row_count: number;
  column_count: number;
  duplicate_rows: number;
  issues: IssueItem[];
}

interface SessionData {
  session_id: string;
  filename: string;
  row_count: number;
  column_count: number;
}

interface RawDataResponse {
  session_id: string;
  columns: string[];
  data: Record<string, unknown>[];
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
}

interface CleaningAction {
  action: string;
  column?: string;
  enabled: boolean;
  label: string;
  recommended: boolean;
}

interface PreviewData {
  before_rows: number;
  before_columns: number;
  before_score: string;
  after_rows: number;
  after_columns: number;
  after_score: string;
  row_delta: number;
  column_delta: number;
  changes: string[];
}

export default function HealthCheckPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [healthData, setHealthData] = useState<HealthCheckData | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);
  const [showRawData, setShowRawData] = useState(false);
  const [rawData, setRawData] = useState<RawDataResponse | null>(null);
  const [loadingRawData, setLoadingRawData] = useState(false);

  // Cleaning Modal State
  const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleaningActions, setCleaningActions] = useState<CleaningAction[]>([]);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);


  useEffect(() => {
    const stored = localStorage.getItem("analysis_session");
    if (!stored) { router.push("/upload"); return; }
    const session = JSON.parse(stored);
    setSessionData(session);
    fetchHealthCheck(session.session_id);
  }, [router]);

  const fetchHealthCheck = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/health-check/${sessionId}`);
      if (!response.ok) throw new Error("Diagnostic array failed to sync");
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Protocol breach detected");
    } finally {
      setLoading(false);
    }
  };

  const openCleanModal = () => {
    if (!healthData) return;
    const actions: CleaningAction[] = [];
    if (healthData.duplicate_rows > 0) {
      actions.push({
        action: "drop_duplicates",
        enabled: true,
        label: `Purge ${healthData.duplicate_rows} duplicate records`,
        recommended: true
      });
    }
    healthData.issues.forEach(issue => {
      if (issue.issue_type === "missing") {
        const isHighMissing = issue.percentage > 70;
        actions.push({
          action: isHighMissing ? "drop_column" : "impute_median",
          column: issue.column,
          enabled: !isHighMissing,
          label: isHighMissing
            ? `Decommission column '${issue.column}' (${issue.percentage.toFixed(1)}% nullity)`
            : `Synthesize ${issue.count} missing '${issue.column}' values via median`,
          recommended: !isHighMissing
        });
      } else if (issue.issue_type === "outlier") {
        actions.push({
          action: "cap_outliers",
          column: issue.column,
          enabled: false,
          label: `Clamp extreme outliers in '${issue.column}' (${issue.count} detected)`,
          recommended: false
        });
      }
    });
    setCleaningActions(actions);
    setPreviewData(null);
    setShowCleanModal(true);
  };

  const fetchPreview = async () => {
    if (!healthData) return;
    setLoadingPreview(true);
    try {
      const enabledActions = cleaningActions
        .filter(a => a.enabled)
        .map(a => ({ action: a.action, column: a.column, enabled: true }));
      const response = await fetch(`${API_BASE}/preview-clean/${healthData.session_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(enabledActions)
      });
      if (!response.ok) throw new Error("Simulation pipeline failure");
      const data = await response.json();
      setPreviewData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Simulation failure");
    } finally {
      setLoadingPreview(false);
    }
  };

  const applyCleaning = async () => {
    if (!healthData) return;
    setCleaning(true);
    try {
      const enabledActions = cleaningActions
        .filter(a => a.enabled)
        .map(a => ({ action: a.action, column: a.column, enabled: true }));
      const response = await fetch(`${API_BASE}/clean/${healthData.session_id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(enabledActions)
      });
      if (!response.ok) throw new Error("Sanitization protocol failed");
      const data = await response.json();
      setCanUndo(data.can_undo);
      setShowCleanModal(false);
      setSuccessMessage(`Dataset sanitized. ${data.changes?.length || 0} modifications applied.`);
      setTimeout(() => setSuccessMessage(null), 10000);
      await fetchHealthCheck(healthData.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sanitization failed");
    } finally {
      setCleaning(false);
    }
  };

  const undoClean = async () => {
    if (!healthData) return;
    try {
      const response = await fetch(`${API_BASE}/undo-clean/${healthData.session_id}`, {
        method: "POST"
      });
      if (!response.ok) throw new Error("Rollback failed");
      setCanUndo(false);
      setSuccessMessage("Dataset reverted to initial state.");
      setTimeout(() => setSuccessMessage(null), 5000);
      await fetchHealthCheck(healthData.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rollback failure");
    }
  };

  const handleViewRawData = async () => {
    if (!sessionData) return;
    setShowRawData(true);
    setLoadingRawData(true);
    try {
      const response = await fetch(`${API_BASE}/data/${sessionData.session_id}?page_size=100`);
      if (!response.ok) throw new Error("Vault access denied");
      const data = await response.json();
      setRawData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Data fetch failure");
    } finally {
      setLoadingRawData(false);
    }
  };

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case "high": return "bg-red-500/10 border-red-500/20 text-red-400";
      case "medium": return "bg-orange-500/10 border-orange-500/20 text-orange-400";
      default: return "bg-blue-500/10 border-blue-500/20 text-blue-400";
    }
  };

  const getScoreColor = (score: string) => {
    switch (score) {
      case "A": return "text-emerald-400 bg-emerald-500/20 border-emerald-500/30";
      case "B": return "text-blue-400 bg-blue-500/20 border-blue-500/30";
      case "C": return "text-orange-400 bg-orange-500/20 border-orange-500/30";
      default: return "text-red-400 bg-red-500/20 border-red-500/30";
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center">
        <Activity className="w-12 h-12 text-purple-500 animate-pulse mb-6" />
        <p className="text-muted-foreground font-bold uppercase tracking-widest text-xs">Scanning Data Integrity...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col overflow-hidden">
      <Navbar />

      <main className="flex-1 overflow-y-auto px-6 py-10">
        <div className="max-w-7xl mx-auto space-y-10">
          {error && (
            <div className="p-5 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 animate-page-enter">
              {error}
            </div>
          )}

          {healthData && (
            <div className="animate-page-enter">
              {/* Summary Section */}
              <div className="flex flex-col md:flex-row items-end justify-between gap-8 mb-10">
                <div className="space-y-2">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-3 py-1 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-widest border border-purple-500/20">System Diagnostics</span>
                  </div>
                  <h1 className="text-4xl font-black tracking-tight">Data Health Core</h1>
                  <p className="text-muted-foreground max-w-xl">Deep structural audit of dataset integrity and relational consistency. Score represents overall usability for modeling.</p>
                </div>

                <div className={`p-8 rounded-3xl border backdrop-blur-xl flex flex-col items-center gap-2 ${getScoreColor(healthData.quality_score)}`}>
                  <span className="text-[10px] font-black uppercase tracking-[0.2em] opacity-60">Integrity Score</span>
                  <span className="text-6xl font-black leading-none">{healthData.quality_score}</span>
                </div>
              </div>

              {/* Grid Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
                <KPICard title="Ingested Rows" value={healthData.row_count.toLocaleString()} icon={<Layers className="w-5 h-5" />} trend="Operational" />
                <KPICard title="Total Dimensions" value={healthData.column_count.toString()} icon={<Table2 className="w-5 h-5" />} trend="Standard" />
                <KPICard title="Duplicate Nodes" value={healthData.duplicate_rows.toString()} icon={<Trash2 className="w-5 h-5" />} trend={healthData.duplicate_rows > 0 ? "Purge Needed" : "0% (Clean)"} />
                <KPICard title="Issue Count" value={healthData.issues.length.toString()} icon={<ShieldCheck className="w-5 h-5" />} trend={healthData.issues.length > 5 ? "Critical" : "Stable"} />
              </div>

              {/* Action Bar */}
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold flex items-center gap-3">
                  <Search className="w-6 h-6 text-purple-500" />
                  Diagnostic Log
                </h2>
                <div className="flex gap-4">
                  <button onClick={handleViewRawData} className="flex items-center gap-2 px-5 py-2.5 bg-white/5 border border-white/10 hover:bg-white/10 rounded-2xl font-bold transition-all text-sm">
                    <Eye className="w-4 h-4" />
                    Vault Preview
                  </button>
                  {canUndo && (
                    <button onClick={undoClean} className="flex items-center gap-2 px-5 py-2.5 bg-orange-500/10 border border-orange-500/20 hover:bg-orange-500/20 text-orange-400 rounded-2xl font-bold transition-all text-sm">
                      <RotateCcw className="w-4 h-4" />
                      Rollback
                    </button>
                  )}
                  <button onClick={openCleanModal} className="flex items-center gap-2 px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-2xl font-bold transition-all shadow-xl shadow-purple-500/20 text-sm">
                    <Sparkles className="w-4 h-4" />
                    Auto-Sanitize
                  </button>
                </div>
              </div>

              {/* Issues List */}
              <div className="grid grid-cols-1 gap-4">
                {healthData.issues.length === 0 ? (
                  <div className="p-12 rounded-3xl bg-emerald-500/5 border border-emerald-500/20 flex flex-col items-center text-center">
                    <ShieldCheck className="w-12 h-12 text-emerald-400 mb-4" />
                    <h3 className="text-xl font-bold text-emerald-400 mb-2">Protocol Perfect</h3>
                    <p className="text-muted-foreground">Zero structural anomalies detected in current ingestion stream.</p>
                  </div>
                ) : (
                  healthData.issues.map((issue, i) => (
                    <div key={i} className={`group p-5 rounded-3xl border backdrop-blur-sm transition-all hover:translate-x-2 ${getSeverityStyles(issue.severity)}`}>
                      <div className="flex items-center gap-5">
                        <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
                          {issue.severity === "high" ? <AlertTriangle className="w-6 h-6" /> : <Layers className="w-6 h-6" />}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-1">
                            <span className="font-bold text-lg">{issue.column === "_all_" ? "Global Entity" : issue.column}</span>
                            <span className="px-2 py-0.5 rounded-full bg-white/10 text-[10px] font-black uppercase tracking-widest">{issue.severity} priority</span>
                          </div>
                          <p className="text-sm opacity-70 leading-relaxed font-medium">{issue.description}</p>
                        </div>
                        <div className="text-right">
                          <div className="text-xl font-black">{issue.count.toLocaleString()}</div>
                          <div className="text-[10px] font-black uppercase tracking-widest opacity-50">Impacted Units</div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Navigation */}
              <div className="mt-12 flex justify-between p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl">
                <div className="flex flex-col gap-1">
                    <h3 className="font-bold">Next Phase Integration</h3>
                    <p className="text-xs text-muted-foreground">Proceed to Exploratory Data Array once diagnostics are stable.</p>
                </div>
                <div className="flex gap-4">
                  <button onClick={() => router.push("/upload")} className="px-6 py-3 rounded-2xl bg-white/5 hover:bg-white/10 border border-white/10 font-bold transition-all">Ingest New Data</button>
                  <button onClick={() => router.push("/eda")} className="group flex items-center gap-2 px-8 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-2xl font-bold transition-all shadow-xl shadow-purple-500/20">
                    Propagate to EDA
                    <ChevronRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Modern Dialogs */}
      {showRawData && (
        <div className="fixed inset-0 bg-background/90 backdrop-blur-2xl flex items-center justify-center z-[100] p-6 animate-in fade-in zoom-in duration-300">
          <div className="bg-background border border-white/10 rounded-[2.5rem] w-full max-w-7xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="p-8 border-b border-white/5 flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-2xl font-black tracking-tight">Raw Data Vault</h3>
                <p className="text-xs text-muted-foreground uppercase font-black tracking-widest">Read-only transcript of primary ingestion</p>
              </div>
              <button onClick={() => setShowRawData(false)} className="p-3 bg-white/5 hover:bg-white/10 rounded-2xl transition-all border border-white/10 text-muted-foreground"><X className="w-6 h-6" /></button>
            </div>
            <div className="flex-1 overflow-auto p-1">
              {loadingRawData ? (
                <div className="flex flex-col items-center justify-center h-full gap-4">
                    <Loader2 className="w-12 h-12 animate-spin text-purple-500" />
                    <span className="text-xs font-bold uppercase tracking-widest text-muted-foreground">Accessing Records...</span>
                </div>
              ) : rawData && (
                <table className="w-full text-sm border-spacing-0 border-separate">
                  <thead className="sticky top-0 z-20">
                    <tr className="bg-background/80 backdrop-blur-md">
                      {rawData.columns.map((col) => (
                        <th key={col} className="px-6 py-5 text-left font-black text-xs uppercase tracking-widest border-b border-white/5 whitespace-nowrap text-muted-foreground">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {rawData.data.map((row, i) => (
                      <tr key={i} className="hover:bg-white/[0.02] transition-colors group">
                        {rawData.columns.map((col) => (
                          <td key={col} className="px-6 py-4 whitespace-nowrap text-foreground/70 font-medium group-hover:text-foreground">
                            {row[col] !== null && row[col] !== undefined ? String(row[col]) : <span className="text-red-500/50 font-bold italic">null</span>}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}

      {showCleanModal && (
        <div className="fixed inset-0 bg-background/90 backdrop-blur-2xl flex items-center justify-center z-[100] p-6 animate-in fade-in zoom-in duration-300">
            <div className="bg-background border border-white/10 rounded-[2.5rem] w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
                <div className="p-8 border-b border-white/5 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="p-3 rounded-2xl bg-purple-500/10 text-purple-500 border border-purple-500/20"><Settings2 className="w-6 h-6" /></div>
                        <div>
                            <h3 className="text-2xl font-black">Sanitization Logic</h3>
                            <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Select automated refinement protocols</p>
                        </div>
                    </div>
                    <button onClick={() => setShowCleanModal(false)} className="p-2 text-muted-foreground hover:text-foreground"><X className="w-6 h-6" /></button>
                </div>

                <div className="flex-1 overflow-y-auto p-8 space-y-6">
                    {cleaningActions.map((action, i) => (
                        <button key={i} onClick={() => {
                            const newActions = [...cleaningActions];
                            newActions[i].enabled = !newActions[i].enabled;
                            setCleaningActions(newActions);
                            setPreviewData(null);
                        }} className={`w-full text-left p-5 rounded-3xl border transition-all flex items-center gap-4 ${action.enabled ? 'bg-purple-600/10 border-purple-500/30' : 'bg-white/5 border-white/10 opacity-70 hover:opacity-100 hover:bg-white/10'}`}>
                            <div className={`w-6 h-6 rounded-lg border-2 flex items-center justify-center shrink-0 transition-all ${action.enabled ? 'bg-purple-500 border-purple-500 text-white' : 'border-white/20'}`}>
                                {action.enabled && <Check className="w-4 h-4" />}
                            </div>
                            <div className="flex-1">
                                <p className="font-bold">{action.label}</p>
                                {action.recommended && <span className="text-[8px] font-black uppercase tracking-widest text-emerald-500">System Recommended</span>}
                            </div>
                        </button>
                    ))}

                    <div className="pt-4">
                        <button onClick={fetchPreview} disabled={loadingPreview || !cleaningActions.some(a => a.enabled)} className="w-full flex items-center justify-center gap-3 p-4 bg-white/5 border border-white/10 rounded-2xl font-bold hover:bg-white/10 transition-all text-sm disabled:opacity-30">
                            {loadingPreview ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5 text-yellow-500" />}
                            Simulate Changes
                        </button>

                        {previewData && (
                            <div className="mt-6 p-6 rounded-[2rem] bg-indigo-500/5 border border-indigo-500/20 animate-in fade-in slide-in-from-top-4">
                                <div className="grid grid-cols-2 gap-8 mb-6">
                                    <div className="space-y-1">
                                        <div className="text-[10px] font-black uppercase tracking-widest opacity-50">Ingress State</div>
                                        <div className="font-bold flex items-center gap-2">
                                            {previewData.before_rows} Rows
                                            <span className={`px-2 py-0.5 rounded-lg text-[10px] border ${getScoreColor(previewData.before_score)}`}>{previewData.before_score}</span>
                                        </div>
                                    </div>
                                    <div className="space-y-1">
                                        <div className="text-[10px] font-black uppercase tracking-widest text-purple-400">Post-Process</div>
                                        <div className="font-bold flex items-center gap-2">
                                            {previewData.after_rows} Rows
                                            <span className={`px-2 py-0.5 rounded-lg text-[10px] border ${getScoreColor(previewData.after_score)}`}>{previewData.after_score}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-2 border-t border-indigo-500/10 pt-4">
                                    {previewData.changes.map((change, i) => (
                                        <div key={i} className="flex items-center gap-2 text-xs font-medium text-foreground/70 tracking-tight">
                                            <ChevronRight className="w-3 h-3 text-purple-500" />
                                            {change}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                <div className="p-8 border-t border-white/5 bg-white/[0.02]">
                    <button onClick={applyCleaning} disabled={cleaning || !cleaningActions.some(a => a.enabled)} className="w-full flex items-center justify-center gap-3 p-5 bg-purple-600 hover:bg-purple-500 text-white rounded-[1.5rem] font-black text-lg transition-all shadow-2xl shadow-purple-500/20 disabled:opacity-50">
                        {cleaning ? <Loader2 className="w-6 h-6 animate-spin" /> : <ShieldCheck className="w-6 h-6" />}
                        Apply Sanitization Protocol
                    </button>
                </div>
            </div>
        </div>
      )}

      {/* Success Toast */}
      {successMessage && (
        <div className="fixed bottom-10 left-1/2 -translate-x-1/2 bg-foreground text-background px-8 py-5 rounded-[2rem] shadow-2xl flex items-center gap-4 z-[200] animate-in slide-in-from-bottom-10">
          <BadgeCheck className="w-6 h-6" />
          <span className="font-bold text-sm tracking-tight">{successMessage}</span>
          <button onClick={() => setSuccessMessage(null)} className="ml-4 p-2 hover:bg-background/10 rounded-xl transition-all"><X className="w-4 h-4" /></button>
        </div>
      )}
    </div>
  );
}

