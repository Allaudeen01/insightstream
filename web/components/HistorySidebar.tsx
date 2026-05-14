"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiFetch } from "@/lib/api";

interface SessionSummary {
  id: number;
  original_filename: string;
  detected_domain: string | null;
  row_count: number | null;
  status: string;
  created_at: string;
}

interface Props {
  currentSessionId?: number;
  onSessionSelect: (id: number) => void;
}

export default function HistorySidebar({ currentSessionId, onSessionSelect }: Props) {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  useEffect(() => { fetchSessions(); }, []);

  async function fetchSessions() {
    try {
      const res = await apiFetch("/sessions?page=1&page_size=30");
      if (res.ok) {
        const data = await res.json();
        setSessions(data.items);
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(e: React.MouseEvent, sessionId: number) {
    e.stopPropagation();
    if (!confirm("Delete this analysis? This cannot be undone.")) return;

    setDeletingId(sessionId);
    const res = await apiFetch(`/sessions/${sessionId}`, { method: "DELETE" });

    if (res.ok) {
      setSessions(prev => prev.filter(s => s.id !== sessionId));
    }
    setDeletingId(null);
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString("en-IN", {
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"
    });
  }

  function statusBadge(status: string) {
    const colours: Record<string, string> = {
      complete: "bg-green-100 text-green-700",
      processing: "bg-yellow-100 text-yellow-700",
      failed: "bg-red-100 text-red-700",
      pending: "bg-gray-100 text-gray-600",
    };
    return `text-xs px-2 py-0.5 rounded-full font-medium ${colours[status] ?? colours.pending}`;
  }

  return (
    <aside className="w-64 bg-white border-r h-full flex flex-col">
      <div className="p-4 border-b flex items-center justify-between">
        <span className="font-semibold text-gray-800 text-sm">Recent Analyses</span>
        <button
          onClick={fetchSessions}
          className="text-xs text-indigo-600 hover:underline"
        >
          Refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading && (
          <p className="text-xs text-gray-400 p-4">Loading...</p>
        )}

        {!loading && sessions.length === 0 && (
          <p className="text-xs text-gray-400 p-4">No analyses yet. Upload a file to get started.</p>
        )}

        {sessions.map(session => (
          <div
            key={session.id}
            onClick={() => onSessionSelect(session.id)}
            className={`
              p-3 border-b cursor-pointer hover:bg-gray-50 transition-colors group
              ${currentSessionId === session.id ? "bg-indigo-50 border-l-2 border-l-indigo-500" : ""}
            `}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">
                  {session.original_filename}
                </p>
                <p className="text-xs text-gray-400 mt-0.5">
                  {formatDate(session.created_at)}
                </p>
                {session.row_count && (
                  <p className="text-xs text-gray-400">
                    {session.row_count.toLocaleString()} rows
                    {session.detected_domain && ` · ${session.detected_domain}`}
                  </p>
                )}
                <span className={statusBadge(session.status)}>
                  {session.status}
                </span>
              </div>

              <button
                onClick={e => handleDelete(e, session.id)}
                disabled={deletingId === session.id}
                className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 text-xs p-1 transition-opacity"
                title="Delete"
              >
                {deletingId === session.id ? "..." : "✕"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
