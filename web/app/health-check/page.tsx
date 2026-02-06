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
  Sparkles
} from "lucide-react";

const API_BASE = "http://localhost:8000";

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

export default function HealthCheckPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [healthData, setHealthData] = useState<HealthCheckData | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("analysis_session");
    if (!stored) {
      router.push("/upload");
      return;
    }
    
    const session = JSON.parse(stored);
    setSessionData(session);
    fetchHealthCheck(session.session_id);
  }, [router]);

  const fetchHealthCheck = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/health-check/${sessionId}`);
      if (!response.ok) throw new Error("Failed to fetch health check");
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleAutoClean = async () => {
    if (!healthData) return;
    setCleaning(true);
    
    try {
      // Build cleaning actions based on issues
      const actions: { action: string; column?: string }[] = [];
      
      // Add drop duplicates if there are duplicates
      if (healthData.duplicate_rows > 0) {
        actions.push({ action: "drop_duplicates" });
      }
      
      // Add impute median for numeric columns with missing values
      healthData.issues
        .filter(i => i.issue_type === "missing")
        .forEach(issue => {
          actions.push({ action: "impute_median", column: issue.column });
        });
      
      if (actions.length > 0) {
        const response = await fetch(`${API_BASE}/clean/${healthData.session_id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(actions)
        });
        
        if (!response.ok) throw new Error("Cleaning failed");
        
        // Refresh health check
        await fetchHealthCheck(healthData.session_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cleaning failed");
    } finally {
      setCleaning(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "text-red-400 bg-red-500/10 border-red-500/20";
      case "medium": return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
      default: return "text-blue-400 bg-blue-500/10 border-blue-500/20";
    }
  };

  const getScoreColor = (score: string) => {
    switch (score) {
      case "A": return "text-green-400 bg-green-500/20";
      case "B": return "text-blue-400 bg-blue-500/20";
      case "C": return "text-yellow-400 bg-yellow-500/20";
      default: return "text-red-400 bg-red-500/20";
    }
  };

  const getIssueIcon = (type: string) => {
    switch (type) {
      case "missing": return <XCircle className="w-4 h-4" />;
      case "duplicate": return <Trash2 className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
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
          <Link href="/upload" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
            <ArrowLeft className="w-5 h-5" />
            <span className="font-medium">Back</span>
          </Link>
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <span className="font-bold text-lg">V</span>
            </div>
            <span className="font-bold text-lg tracking-tight">Data Health Check</span>
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

        {healthData && (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <p className="text-slate-400 text-sm mb-1">Quality Score</p>
                <div className={`text-4xl font-bold ${getScoreColor(healthData.quality_score)} w-14 h-14 rounded-xl flex items-center justify-center`}>
                  {healthData.quality_score}
                </div>
              </div>
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <p className="text-slate-400 text-sm mb-1">Rows</p>
                <p className="text-3xl font-bold">{healthData.row_count.toLocaleString()}</p>
              </div>
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <p className="text-slate-400 text-sm mb-1">Columns</p>
                <p className="text-3xl font-bold">{healthData.column_count}</p>
              </div>
              <div className="p-6 rounded-2xl bg-slate-900 border border-white/10">
                <p className="text-slate-400 text-sm mb-1">Issues Found</p>
                <p className="text-3xl font-bold">{healthData.issues.length}</p>
              </div>
            </div>

            {/* Issues Panel */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Issues Detected</h2>
                {healthData.issues.length > 0 && (
                  <button
                    onClick={handleAutoClean}
                    disabled={cleaning}
                    className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-medium transition-colors disabled:opacity-50"
                  >
                    {cleaning ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    Auto Clean Data
                  </button>
                )}
              </div>

              {healthData.issues.length === 0 ? (
                <div className="p-8 rounded-2xl bg-green-500/10 border border-green-500/20 text-center">
                  <CheckCircle2 className="w-12 h-12 text-green-400 mx-auto mb-3" />
                  <p className="text-green-400 font-medium">No issues detected! Your data is clean.</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {healthData.issues.map((issue, i) => (
                    <div
                      key={i}
                      className={`p-4 rounded-xl border ${getSeverityColor(issue.severity)} flex items-center gap-4`}
                    >
                      <div className="flex-shrink-0">
                        {getIssueIcon(issue.issue_type)}
                      </div>
                      <div className="flex-grow">
                        <p className="font-medium">
                          {issue.column === "_all_" ? "All Rows" : issue.column}
                        </p>
                        <p className="text-sm opacity-80">{issue.description}</p>
                      </div>
                      <div className="flex-shrink-0 text-right">
                        <span className="text-xs uppercase font-medium px-2 py-1 rounded bg-white/10">
                          {issue.issue_type}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Navigation */}
            <div className="flex justify-between">
              <button
                onClick={() => router.push("/upload")}
                className="px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 font-medium transition-colors"
              >
                Upload New Data
              </button>
              <button
                onClick={() => router.push("/eda")}
                className="flex items-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors"
              >
                Continue to EDA
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
