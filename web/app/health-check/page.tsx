// web/app/health-check/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  ChevronRight,
  Eye,
  Loader2,
  RotateCcw,
  Sparkles,
  Table2,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/* ============================================================
   Types (unchanged from original)
   ============================================================ */
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

/* ============================================================
   Page
   ============================================================ */
export default function HealthCheckPage() {
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [healthData, setHealthData] = useState<HealthCheckData | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Cleaning state
  const [cleaning, setCleaning] = useState(false);
  const [showCleanModal, setShowCleanModal] = useState(false);
  const [cleaningActions, setCleaningActions] = useState<CleaningAction[]>([]);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Raw data preview
  const [showRawData, setShowRawData] = useState(false);
  const [rawData, setRawData] = useState<RawDataResponse | null>(null);
  const [loadingRawData, setLoadingRawData] = useState(false);

  // ----- Init -----
  useEffect(() => {
    const stored = localStorage.getItem("analysis_session");
    if (!stored) { router.push("/upload"); return; }
    const session = JSON.parse(stored);
    setSessionData(session);
    fetchHealthCheck(session.session_id);
  }, [router]);

  // ----- API: health check -----
  const fetchHealthCheck = async (sessionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/health-check/${sessionId}`);
      if (!response.ok) throw new Error("Couldn't load the quality check. Please try again.");
      const data = await response.json();
      setHealthData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  // ----- API: open clean modal (build action list from issues) -----
  const openCleanModal = () => {
    if (!healthData) return;
    const actions: CleaningAction[] = [];
    if (healthData.duplicate_rows > 0) {
      actions.push({
        action: "drop_duplicates",
        enabled: true,
        label: `Remove ${healthData.duplicate_rows.toLocaleString()} duplicate rows`,
        recommended: true,
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
            ? `Drop column "${issue.column}" (${issue.percentage.toFixed(1)}% missing)`
            : `Fill ${issue.count.toLocaleString()} missing values in "${issue.column}" with the median`,
          recommended: !isHighMissing,
        });
      } else if (issue.issue_type === "outlier") {
        actions.push({
          action: "cap_outliers",
          column: issue.column,
          enabled: false,
          label: `Cap ${issue.count.toLocaleString()} outliers in "${issue.column}"`,
          recommended: false,
        });
      }
    });
    setCleaningActions(actions);
    setPreviewData(null);
    setShowCleanModal(true);
  };

  // ----- API: preview -----
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
        body: JSON.stringify(enabledActions),
      });
      if (!response.ok) throw new Error("Couldn't preview the changes.");
      setPreviewData(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview failed.");
    } finally {
      setLoadingPreview(false);
    }
  };

  // ----- API: apply -----
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
        body: JSON.stringify(enabledActions),
      });
      if (!response.ok) throw new Error("Cleaning failed. Please try again.");
      const data = await response.json();
      setCanUndo(data.can_undo);
      setShowCleanModal(false);
      setSuccessMessage(`Data cleaned. ${data.changes?.length || 0} change${data.changes?.length === 1 ? "" : "s"} applied.`);
      setTimeout(() => setSuccessMessage(null), 8000);
      await fetchHealthCheck(healthData.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Cleaning failed.");
    } finally {
      setCleaning(false);
    }
  };

  // ----- API: undo -----
  const undoClean = async () => {
    if (!healthData) return;
    try {
      const response = await fetch(`${API_BASE}/undo-clean/${healthData.session_id}`, { method: "POST" });
      if (!response.ok) throw new Error("Couldn't undo. Please try again.");
      setCanUndo(false);
      setSuccessMessage("Changes reverted.");
      setTimeout(() => setSuccessMessage(null), 4000);
      await fetchHealthCheck(healthData.session_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Undo failed.");
    }
  };

  // ----- API: raw data -----
  const handleViewRawData = async () => {
    if (!sessionData) return;
    setShowRawData(true);
    setLoadingRawData(true);
    try {
      const response = await fetch(`${API_BASE}/data/${sessionData.session_id}?page_size=100`);
      if (!response.ok) throw new Error("Couldn't load preview data.");
      setRawData(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview failed.");
    } finally {
      setLoadingRawData(false);
    }
  };

  // ----- Loading -----
  if (loading) {
    return (
      <div className="flex h-screen bg-white">
        <Sidebar />
        <div className="flex flex-1 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-zinc-400" strokeWidth={1.75} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-white text-zinc-900 antialiased">
      <Sidebar />

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-6 backdrop-blur">
          <div className="flex items-center gap-3 min-w-0">
            <h1 className="text-[17px] font-semibold tracking-[-0.01em]">Data quality check</h1>
            {sessionData?.filename && (
              <>
                <span className="text-zinc-300">/</span>
                <span className="truncate text-[13px] text-zinc-500">{sessionData.filename}</span>
              </>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleViewRawData}
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50"
            >
              <Eye className="h-3.5 w-3.5" strokeWidth={1.75} />
              Preview data
            </button>
            {canUndo && (
              <button
                onClick={undoClean}
                className="inline-flex h-8 items-center gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-3 text-[13px] font-medium text-amber-700 hover:bg-amber-100"
              >
                <RotateCcw className="h-3.5 w-3.5" strokeWidth={1.75} />
                Undo
              </button>
            )}
            <button
              onClick={openCleanModal}
              disabled={!healthData}
              className="inline-flex h-8 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-3 text-[13px] font-medium text-white hover:bg-[#5b4be0] disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5" strokeWidth={1.75} />
              Clean data
            </button>
          </div>
        </header>

        {/* Body */}
        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1200px] px-6 py-6">
            {error && (
              <div className="mb-6 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
                <div className="flex-1">{error}</div>
                <button onClick={() => setError(null)} className="text-red-400 hover:text-red-700">
                  <X className="h-4 w-4" strokeWidth={1.75} />
                </button>
              </div>
            )}

            {healthData && (
              <>
                {/* Header summary */}
                <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm">
                  <div className="flex items-start justify-between gap-6">
                    <div className="min-w-0 flex-1">
                      <h2 className="text-[20px] font-semibold tracking-[-0.01em]">
                        Quality summary
                      </h2>
                      <p className="mt-1 text-[13px] leading-relaxed text-zinc-600">
                        We scanned your data for missing values, duplicates, and outliers.
                        Use the cleaning tools to fix issues before analysis.
                      </p>
                    </div>
                    <ScoreBadge score={healthData.quality_score} />
                  </div>

                  {/* Stats row */}
                  <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-zinc-200 bg-zinc-200 sm:grid-cols-4">
                    <Stat
                      label="Rows"
                      value={healthData.row_count.toLocaleString()}
                    />
                    <Stat
                      label="Columns"
                      value={healthData.column_count.toLocaleString()}
                    />
                    <Stat
                      label="Duplicates"
                      value={healthData.duplicate_rows.toLocaleString()}
                      tone={healthData.duplicate_rows > 0 ? "warn" : "ok"}
                    />
                    <Stat
                      label="Issues"
                      value={healthData.issues.length.toLocaleString()}
                      tone={healthData.issues.length > 5 ? "warn" : healthData.issues.length === 0 ? "ok" : "neutral"}
                    />
                  </div>
                </section>

                {/* Issues */}
                <section className="mt-8">
                  <h2 className="mb-3 text-[14px] font-semibold tracking-[-0.01em]">
                    Issues found
                  </h2>

                  {healthData.issues.length === 0 ? (
                    <div className="flex flex-col items-center justify-center rounded-xl border border-emerald-200 bg-emerald-50 px-6 py-10 text-center">
                      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-white">
                        <CheckCircle2 className="h-5 w-5 text-emerald-600" strokeWidth={1.75} />
                      </div>
                      <h3 className="text-[15px] font-semibold text-emerald-900">No issues found</h3>
                      <p className="mt-1 max-w-sm text-[13px] text-emerald-700">
                        Your data looks clean and ready for analysis.
                      </p>
                    </div>
                  ) : (
                    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
                      {healthData.issues.map((issue, i) => (
                        <IssueRow key={i} issue={issue} divider={i > 0} />
                      ))}
                    </div>
                  )}
                </section>

                {/* Footer nav */}
                <section className="mt-8 flex flex-col gap-3 rounded-xl border border-zinc-200 bg-zinc-50 p-5 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0 flex-1">
                    <h3 className="text-[14px] font-semibold text-zinc-900">Ready to analyze?</h3>
                    <p className="mt-0.5 text-[13px] text-zinc-600">
                      Continue to exploratory data analysis once you're happy with the data.
                    </p>
                  </div>
                  <div className="flex shrink-0 gap-2">
                    <button
                      onClick={() => router.push("/upload")}
                      className="inline-flex h-9 items-center rounded-md border border-zinc-200 bg-white px-4 text-[13px] font-medium text-zinc-700 hover:bg-white/80"
                    >
                      Upload another file
                    </button>
                    <button
                      onClick={() => router.push("/eda")}
                      className="inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0]"
                    >
                      Continue to EDA
                      <ArrowRight className="h-3.5 w-3.5" strokeWidth={1.75} />
                    </button>
                  </div>
                </section>
              </>
            )}
          </div>
        </main>
      </div>

      {/* ============= Preview data modal ============= */}
      {showRawData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 p-4 backdrop-blur-sm">
          <div className="flex max-h-[88vh] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-lg">
            <div className="flex shrink-0 items-center justify-between border-b border-zinc-100 px-5 py-4">
              <div>
                <h3 className="text-[15px] font-semibold tracking-tight">Preview data</h3>
                <p className="mt-0.5 text-[12px] text-zinc-500">
                  {rawData
                    ? `Showing first ${rawData.data.length.toLocaleString()} of ${rawData.total_rows.toLocaleString()} rows`
                    : "Loading…"}
                </p>
              </div>
              <button
                onClick={() => setShowRawData(false)}
                className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600"
                aria-label="Close"
              >
                <X className="h-4 w-4" strokeWidth={1.75} />
              </button>
            </div>

            <div className="flex-1 overflow-auto">
              {loadingRawData ? (
                <div className="flex h-64 items-center justify-center">
                  <Loader2 className="h-5 w-5 animate-spin text-zinc-400" strokeWidth={1.75} />
                </div>
              ) : rawData && (
                <table className="w-full border-separate border-spacing-0 text-[13px]">
                  <thead className="sticky top-0 z-10 bg-zinc-50">
                    <tr>
                      {rawData.columns.map((col) => (
                        <th
                          key={col}
                          className="whitespace-nowrap border-b border-zinc-200 px-4 py-2.5 text-left text-[12px] font-medium text-zinc-700"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {rawData.data.map((row, i) => (
                      <tr key={i} className="hover:bg-zinc-50">
                        {rawData.columns.map((col) => (
                          <td
                            key={col}
                            className="whitespace-nowrap border-b border-zinc-100 px-4 py-2 text-zinc-800"
                          >
                            {row[col] !== null && row[col] !== undefined ? (
                              String(row[col])
                            ) : (
                              <span className="italic text-zinc-400">null</span>
                            )}
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

      {/* ============= Clean data modal ============= */}
      {showCleanModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 p-4 backdrop-blur-sm">
          <div className="flex max-h-[85vh] w-full max-w-xl flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-lg">
            <div className="flex shrink-0 items-center justify-between border-b border-zinc-100 px-5 py-4">
              <div>
                <h3 className="text-[15px] font-semibold tracking-tight">Clean data</h3>
                <p className="mt-0.5 text-[12px] text-zinc-500">
                  Pick the fixes you want to apply.
                </p>
              </div>
              <button
                onClick={() => setShowCleanModal(false)}
                className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600"
                aria-label="Close"
              >
                <X className="h-4 w-4" strokeWidth={1.75} />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-5">
              {cleaningActions.length === 0 ? (
                <div className="flex flex-col items-center py-10 text-center">
                  <CheckCircle2 className="mb-3 h-6 w-6 text-emerald-600" strokeWidth={1.75} />
                  <p className="text-[14px] font-medium text-zinc-900">Nothing to clean</p>
                  <p className="mt-1 text-[13px] text-zinc-600">Your data has no fixable issues.</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {cleaningActions.map((action, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        const next = [...cleaningActions];
                        next[i].enabled = !next[i].enabled;
                        setCleaningActions(next);
                        setPreviewData(null);
                      }}
                      className={[
                        "flex w-full items-start gap-3 rounded-lg border px-4 py-3 text-left transition-colors",
                        action.enabled
                          ? "border-[#c8c1ff] bg-[#f1efff]"
                          : "border-zinc-200 bg-white hover:bg-zinc-50",
                      ].join(" ")}
                    >
                      <Checkbox checked={action.enabled} />
                      <div className="min-w-0 flex-1">
                        <p className="text-[13.5px] font-medium text-zinc-900">{action.label}</p>
                        {action.recommended && (
                          <span className="mt-0.5 inline-block text-[11px] font-medium text-emerald-700">
                            Recommended
                          </span>
                        )}
                      </div>
                    </button>
                  ))}

                  <button
                    onClick={fetchPreview}
                    disabled={loadingPreview || !cleaningActions.some(a => a.enabled)}
                    className="mt-3 inline-flex h-9 w-full items-center justify-center gap-1.5 rounded-md border border-zinc-200 bg-white text-[13px] font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
                  >
                    {loadingPreview
                      ? <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.75} />
                      : <Zap className="h-3.5 w-3.5" strokeWidth={1.75} />}
                    Preview changes
                  </button>

                  {previewData && (
                    <div className="mt-3 rounded-lg border border-zinc-200 bg-zinc-50 p-4">
                      <div className="grid grid-cols-2 gap-4 pb-3">
                        <PreviewSide
                          label="Before"
                          rows={previewData.before_rows}
                          score={previewData.before_score}
                          tone="muted"
                        />
                        <PreviewSide
                          label="After"
                          rows={previewData.after_rows}
                          score={previewData.after_score}
                          tone="accent"
                        />
                      </div>
                      {previewData.changes.length > 0 && (
                        <div className="space-y-1.5 border-t border-zinc-200 pt-3">
                          {previewData.changes.map((change, i) => (
                            <div key={i} className="flex items-start gap-1.5 text-[12.5px] text-zinc-700">
                              <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-[#6d5ef5]" strokeWidth={1.75} />
                              {change}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="shrink-0 border-t border-zinc-100 bg-zinc-50 px-5 py-4">
              <div className="flex justify-end gap-2">
                <button
                  onClick={() => setShowCleanModal(false)}
                  className="h-9 rounded-md border border-zinc-200 bg-white px-4 text-[13px] font-medium text-zinc-700 hover:bg-white/80"
                >
                  Cancel
                </button>
                <button
                  onClick={applyCleaning}
                  disabled={cleaning || !cleaningActions.some(a => a.enabled)}
                  className="inline-flex h-9 items-center gap-1.5 rounded-md bg-[#6d5ef5] px-4 text-[13px] font-medium text-white hover:bg-[#5b4be0] disabled:opacity-50"
                >
                  {cleaning && <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.75} />}
                  Apply changes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============= Success toast ============= */}
      {successMessage && (
        <div className="fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-lg border border-zinc-200 bg-white px-4 py-3 shadow-lg">
          <CheckCircle2 className="h-4 w-4 text-emerald-600" strokeWidth={1.75} />
          <span className="text-[13px] font-medium text-zinc-900">{successMessage}</span>
          <button
            onClick={() => setSuccessMessage(null)}
            className="rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600"
            aria-label="Dismiss"
          >
            <X className="h-3.5 w-3.5" strokeWidth={1.75} />
          </button>
        </div>
      )}
    </div>
  );
}

/* ============================================================
   Subcomponents
   ============================================================ */

function ScoreBadge({ score }: { score: string }) {
  const map: Record<string, { bg: string; ring: string; text: string; label: string }> = {
    A: { bg: "bg-emerald-50", ring: "ring-emerald-200", text: "text-emerald-700", label: "Excellent" },
    B: { bg: "bg-sky-50", ring: "ring-sky-200", text: "text-sky-700", label: "Good" },
    C: { bg: "bg-amber-50", ring: "ring-amber-200", text: "text-amber-700", label: "Fair" },
    D: { bg: "bg-red-50", ring: "ring-red-200", text: "text-red-700", label: "Needs work" },
    F: { bg: "bg-red-50", ring: "ring-red-200", text: "text-red-700", label: "Poor" },
  };
  const s = map[score] ?? map.C;
  return (
    <div className={`flex shrink-0 flex-col items-center rounded-xl px-5 py-3 ring-1 ${s.bg} ${s.ring}`}>
      <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Quality score</span>
      <span className={`mt-0.5 text-3xl font-semibold leading-none ${s.text}`}>{score}</span>
      <span className={`mt-1 text-[11px] font-medium ${s.text}`}>{s.label}</span>
    </div>
  );
}

function Stat({
  label, value, tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "ok" | "warn";
}) {
  const valueTone =
    tone === "warn" ? "text-amber-700"
      : tone === "ok" ? "text-emerald-700"
        : "text-zinc-900";
  return (
    <div className="bg-white px-4 py-3.5">
      <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-0.5 text-[20px] font-semibold tabular-nums tracking-tight ${valueTone}`}>
        {value}
      </div>
    </div>
  );
}

function IssueRow({ issue, divider }: { issue: IssueItem; divider: boolean }) {
  const sevStyle: Record<string, { dot: string; pill: string; label: string }> = {
    high: { dot: "bg-red-500", pill: "bg-red-50 text-red-700 border-red-200", label: "High" },
    medium: { dot: "bg-amber-500", pill: "bg-amber-50 text-amber-700 border-amber-200", label: "Medium" },
    low: { dot: "bg-sky-500", pill: "bg-sky-50 text-sky-700 border-sky-200", label: "Low" },
  };
  const sev = sevStyle[issue.severity] ?? sevStyle.low;
  const columnLabel = issue.column === "_all_" ? "All columns" : issue.column;

  return (
    <div className={`flex items-start gap-4 px-5 py-4 ${divider ? "border-t border-zinc-100" : ""}`}>
      <div className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${sev.dot}`} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[14px] font-semibold text-zinc-900">{columnLabel}</span>
          <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${sev.pill}`}>
            {sev.label}
          </span>
          <span className="text-[12px] text-zinc-500">
            {issue.issue_type.replace(/_/g, " ")}
          </span>
        </div>
        <p className="mt-1 text-[13px] leading-relaxed text-zinc-600">{issue.description}</p>
      </div>
      <div className="shrink-0 text-right">
        <div className="text-[16px] font-semibold tabular-nums text-zinc-900">
          {issue.count.toLocaleString()}
        </div>
        <div className="text-[11px] text-zinc-500">
          {issue.percentage.toFixed(1)}% of rows
        </div>
      </div>
    </div>
  );
}

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <div
      className={[
        "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors",
        checked
          ? "border-[#6d5ef5] bg-[#6d5ef5] text-white"
          : "border-zinc-300 bg-white",
      ].join(" ")}
      aria-hidden="true"
    >
      {checked && <Check className="h-3 w-3" strokeWidth={2.5} />}
    </div>
  );
}

function PreviewSide({
  label, rows, score, tone,
}: {
  label: string;
  rows: number;
  score: string;
  tone: "muted" | "accent";
}) {
  const labelClass = tone === "accent" ? "text-[#6d5ef5]" : "text-zinc-500";
  return (
    <div>
      <div className={`text-[11px] font-medium uppercase tracking-wide ${labelClass}`}>{label}</div>
      <div className="mt-1 flex items-center gap-2">
        <span className="text-[14px] font-semibold text-zinc-900 tabular-nums">
          {rows.toLocaleString()} rows
        </span>
        <span className="rounded border border-zinc-200 bg-white px-1.5 py-0.5 text-[11px] font-semibold text-zinc-700">
          {score}
        </span>
      </div>
    </div>
  );
}