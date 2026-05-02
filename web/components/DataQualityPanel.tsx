import React from 'react';

interface QualityIssue {
  type: string;
  severity: 'critical' | 'medium';
  column: string;
  count: number;
  message: string;
  rows?: number[];
  values?: string[];
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

interface Props {
  report: QualityReport | null | undefined;
}

const severityStyle: Record<string, string> = {
  critical: 'bg-red-50 border-red-300 text-red-800',
  medium:   'bg-yellow-50 border-yellow-300 text-yellow-800',
};

export function DataQualityPanel({ report }: Props) {
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
          <div
            key={i}
            className={`border rounded-lg p-3 text-sm ${severityStyle[issue.severity] ?? 'bg-gray-50 border-gray-200'}`}
          >
            <div className="font-semibold">
              {issue.type.replace(/_/g, ' ')} — {issue.column}
            </div>
            <div className="mt-0.5">{issue.message}</div>
            {issue.values && issue.values.length > 0 && (
              <div className="mt-1 text-xs opacity-75">
                Bad values: {issue.values.slice(0, 8).join(', ')}
                {issue.values.length > 8 ? ` +${issue.values.length - 8} more` : ''}
              </div>
            )}
            {issue.rows && issue.rows.length > 0 && (
              <div className="text-xs opacity-75">
                Affected rows: {issue.rows.slice(0, 5).join(', ')}
                {issue.rows.length > 5 ? ` +${issue.rows.length - 5} more` : ''}
              </div>
            )}
          </div>
        ))}
      </div>

      {!report.summary.can_analyze && (
        <div className="bg-red-50 border border-red-300 rounded-lg p-3 text-red-700 font-semibold text-sm">
          Analysis blocked. Fix critical issues in your file and re-upload.
        </div>
      )}
    </div>
  );
}
