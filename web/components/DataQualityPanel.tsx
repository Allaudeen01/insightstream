"use client";

import React, { useState } from "react";
import { apiFetch } from "@/lib/api";

interface QualityIssue {
  issue_type: string;
  severity: string;
  column: string;
  count: number;
  description: string;
  percentage?: number;
}

interface QualitySummary {
  total_rows: number;
  clean_rows: number;
  issue_count: number;
  critical: number;
  medium: number;
  can_analyze: boolean;
}

interface QualityReport {
  summary: QualitySummary;
  issues: QualityIssue[];
}

interface OutlierData {
  column: string;
  total_outliers: number;
  pct: number;
  lower_bound: number;
  upper_bound: number;
  Q1: number;
  Q3: number;
  page: number;
  page_size: number;
  total_pages: number;
  rows: Record<string, unknown>[];
  columns: string[];
}

interface Props {
  report: QualityReport | null | undefined;
  sessionId?: number;
}

const severityStyle: Record<string, string> = {
  high:   "bg-red-50 border-red-300 text-red-800",
  medium: "bg-yellow-50 border-yellow-300 text-yellow-800",
  low:    "bg-blue-50 border-blue-300 text-blue-800",
};

function IssueRow({
  issue,
  sessionId,
}: {
  issue: QualityIssue;
  sessionId?: number;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<OutlierData | null>(null);
  const [loading, setLoading] = useState(false);
  const [page, setPage] = useState(1);

  const isOutlier = issue.issue_type === "outlier";
  const style = severityStyle[issue.severity] ?? "bg-gray-50 border-gray-200";

  async function fetchOutliers(p: number) {
    if (!sessionId) return;
    setLoading(true);
    try {
      const res = await apiFetch(
        `/eda/${sessionId}/outliers/${encodeURIComponent(issue.column)}?page=${p}&page_size=50`
      );
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setPage(p);
      }
    } finally {
      setLoading(false);
    }
  }

  function handleToggle() {
    if (!isOutlier || !sessionId) return;
    if (!open && !data) fetchOutliers(1);
    setOpen((v) => !v);
  }

  const displayColumns = data
    ? data.columns.filter((c) => !c.startsWith("_"))
    : [];

  return (
    <div className={`border rounded-lg text-sm ${style}`}>
      <button
        type="button"
        className="w-full text-left p-3 flex items-start justify-between gap-2 hover:opacity-80"
        style={isOutlier && sessionId ? { cursor: "pointer" } : { cursor: "default" }}
        onClick={(e) => {
          e.stopPropagation();
          handleToggle();
        }}
      >
        <div className="flex-1 min-w-0">
          <div className="font-semibold">
            {issue.issue_type.replace(/_/g, " ")} — {issue.column}
          </div>
          <div className="mt-0.5 opacity-90">{issue.description}</div>
          {issue.percentage != null && (
            <div className="text-xs opacity-70 mt-0.5">
              {issue.count.toLocaleString()} rows ({issue.percentage.toFixed(1)}%)
            </div>
          )}
        </div>
        {isOutlier && sessionId && (
          <span className="text-xs font-medium shrink-0 mt-0.5 opacity-70">
            {open ? "▲ hide" : "▼ view rows"}
          </span>
        )}
      </button>

      {open && isOutlier && (
        <div className="border-t border-current border-opacity-20 px-3 pb-3 pt-2">
          {loading && !data && (
            <div className="text-xs opacity-60 py-2">Loading…</div>
          )}

          {data && (
            <>
              <div className="flex flex-wrap gap-4 text-xs font-medium mb-2 opacity-80">
                <span>{data.total_outliers} outlier rows ({data.pct}%)</span>
                <span>
                  Normal range: {data.lower_bound.toLocaleString()} –{" "}
                  {data.upper_bound.toLocaleString()}
                </span>
                <span>Q1: {data.Q1.toLocaleString()}</span>
                <span>Q3: {data.Q3.toLocaleString()}</span>
              </div>

              {data.rows.length > 0 ? (
                <div className="overflow-x-auto rounded border border-current border-opacity-20">
                  <table className="text-xs w-full min-w-max">
                    <thead>
                      <tr className="bg-black bg-opacity-5">
                        <th className="px-2 py-1 text-left font-semibold">Direction</th>
                        <th className="px-2 py-1 text-right font-semibold">
                          {issue.column}
                        </th>
                        {displayColumns
                          .filter((c) => c !== issue.column)
                          .slice(0, 6)
                          .map((c) => (
                            <th key={c} className="px-2 py-1 text-left font-semibold">
                              {c}
                            </th>
                          ))}
                      </tr>
                    </thead>
                    <tbody>
                      {data.rows.map((row, i) => (
                        <tr
                          key={i}
                          className={i % 2 === 0 ? "bg-white bg-opacity-40" : ""}
                        >
                          <td className="px-2 py-1">
                            <span
                              className={`font-semibold ${
                                row._outlier_direction === "high"
                                  ? "text-red-600"
                                  : "text-blue-600"
                              }`}
                            >
                              {String(row._outlier_direction).toUpperCase()}
                            </span>
                          </td>
                          <td className="px-2 py-1 text-right font-mono">
                            {Number(row._outlier_value).toLocaleString()}
                          </td>
                          {displayColumns
                            .filter((c) => c !== issue.column)
                            .slice(0, 6)
                            .map((c) => (
                              <td key={c} className="px-2 py-1 truncate max-w-[120px]">
                                {row[c] == null ? "—" : String(row[c])}
                              </td>
                            ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-xs opacity-60 py-2">No rows on this page.</div>
              )}

              {data.total_pages > 1 && (
                <div className="flex items-center gap-3 mt-2 text-xs">
                  <button
                    disabled={page <= 1 || loading}
                    onClick={() => fetchOutliers(page - 1)}
                    className="px-2 py-0.5 rounded border border-current border-opacity-30 disabled:opacity-40 hover:bg-black hover:bg-opacity-5"
                  >
                    Prev
                  </button>
                  <span className="opacity-70">
                    Page {page} / {data.total_pages}
                  </span>
                  <button
                    disabled={page >= data.total_pages || loading}
                    onClick={() => fetchOutliers(page + 1)}
                    className="px-2 py-0.5 rounded border border-current border-opacity-30 disabled:opacity-40 hover:bg-black hover:bg-opacity-5"
                  >
                    Next
                  </button>
                  {loading && <span className="opacity-50">Loading…</span>}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function DataQualityPanel({ report, sessionId }: Props) {
  if (!report) return null;

  if (report.summary.issue_count === 0) {
    return (
      <div className="rounded-xl border border-green-200 bg-green-50 p-4 mt-4 flex items-center gap-3 shadow-sm">
        <span className="text-green-600 text-lg font-bold">✓</span>
        <div>
          <p className="font-bold text-sm text-green-800">Data Quality: Clean</p>
          <p className="text-xs text-green-700">
            {report.summary.total_rows.toLocaleString()} rows scanned — no issues found.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 mt-4 space-y-3 shadow-sm">
      <h3 className="font-bold text-base text-gray-900">Data Quality Issues Found</h3>

      <div className="flex flex-wrap gap-4 text-sm">
        <span className="text-red-600 font-semibold">
          {report.summary.critical} Critical
        </span>
        <span className="text-yellow-600 font-semibold">
          {report.summary.medium} Warnings
        </span>
        <span className="text-gray-500">
          {report.summary.clean_rows.toLocaleString()} clean rows remaining
        </span>
        <span className="text-gray-500">
          {report.summary.total_rows.toLocaleString()} total rows
        </span>
      </div>

      <div className="space-y-2">
        {report.issues.map((issue, i) => (
          <IssueRow key={i} issue={issue} sessionId={sessionId} />
        ))}
      </div>

    </div>
  );
}
