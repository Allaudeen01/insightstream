"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import dynamic from "next/dynamic";
import { Loader2, LayoutDashboard, Lock } from "lucide-react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ChartData {
  chart_id: string;
  title: string;
  plotly_json: {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  };
}

interface DashboardData {
  pinned_chart_ids: string[];
  text_blocks: { id: string; content: string }[];
  charts?: ChartData[];
}

export default function SharedDashboardPage() {
  const params = useParams<{ id: string }>();
  const id = params?.id;
  const [loading, setLoading] = useState(true);
  const [charts, setCharts] = useState<ChartData[]>([]);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);

  useEffect(() => {
    if (!id) return;
    fetch(`${API_BASE}/share/dashboard/${id}`)
      .then((r) => r.json())
      .then((payload) => {
        setCharts(payload.charts || []);
        setDashboard(payload);
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin" /></div>;
  }

  const itemIds = dashboard?.pinned_chart_ids || [];

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="border-b border-white/10 bg-slate-950/60 backdrop-blur-xl">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <LayoutDashboard className="w-5 h-5 text-indigo-400" />
            <span className="font-semibold">Shared Dashboard</span>
          </div>
          <div className="text-xs text-slate-400 flex items-center gap-1"><Lock className="w-3.5 h-3.5" />Read-only view</div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {itemIds.map((itemId) => {
          const textBlock = (dashboard?.text_blocks || []).find((t) => t.id === itemId);
          if (textBlock) {
            return (
              <div key={itemId} className="rounded-2xl bg-slate-900 border border-white/10 p-5 whitespace-pre-wrap text-slate-200 text-sm">
                {textBlock.content}
              </div>
            );
          }

          const chart = charts.find((c) => c.chart_id === itemId);
          if (!chart) return null;

          return (
            <div key={itemId} className="rounded-2xl bg-slate-900 border border-white/10 p-4">
              <h3 className="font-semibold mb-3">{chart.title}</h3>
              <div className="h-[320px]">
                <Plot
                  data={chart.plotly_json.data}
                  layout={{ ...chart.plotly_json.layout, autosize: true, paper_bgcolor: "transparent", plot_bgcolor: "rgba(30,41,59,0.5)" }}
                  config={{ displayModeBar: false, responsive: true, displaylogo: false }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
            </div>
          );
        })}
      </main>
    </div>
  );
}
