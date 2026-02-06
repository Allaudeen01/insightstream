"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
    ArrowLeft,
    FileText,
    Loader2,
    Download,
    Share2,
    Calendar,
    CheckCircle
} from "lucide-react";

const API_BASE = "http://localhost:8000";

interface ReportSection {
    title: string;
    content: string;
    chart_type?: string;
    chart_data?: Record<string, unknown>;
}

interface ReportData {
    session_id: string;
    title: string;
    generated_at: string;
    sections: ReportSection[];
}

export default function ReportPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<ReportData | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [exporting, setExporting] = useState(false);

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) {
            router.push("/upload");
            return;
        }
        const session = JSON.parse(stored);
        fetchReport(session.session_id);
    }, [router]);

    const fetchReport = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/report/${sessionId}`);
            if (!response.ok) throw new Error("Failed to generate report");
            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setLoading(false);
        }
    };

    const exportAsPDF = () => {
        setExporting(true);

        // Create printable content
        const printContent = `
      <html>
        <head>
          <title>${data?.title || "Analysis Report"}</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
            h1 { color: #4F46E5; border-bottom: 2px solid #4F46E5; padding-bottom: 10px; }
            h2 { color: #1E293B; margin-top: 30px; }
            p { color: #475569; line-height: 1.6; }
            .meta { color: #94A3B8; font-size: 14px; margin-bottom: 30px; }
            .section { margin-bottom: 30px; padding: 20px; background: #F8FAFC; border-radius: 8px; }
            pre { white-space: pre-wrap; }
          </style>
        </head>
        <body>
          <h1>${data?.title || "Analysis Report"}</h1>
          <p class="meta">Generated: ${data?.generated_at ? new Date(data.generated_at).toLocaleString() : "N/A"}</p>
          ${data?.sections.map(s => `
            <div class="section">
              <h2>${s.title}</h2>
              <pre>${s.content}</pre>
            </div>
          `).join("") || ""}
        </body>
      </html>
    `;

        const printWindow = window.open("", "_blank");
        if (printWindow) {
            printWindow.document.write(printContent);
            printWindow.document.close();
            printWindow.print();
        }

        setExporting(false);
    };

    const exportAsText = () => {
        if (!data) return;

        let content = `${data.title}\n`;
        content += `Generated: ${new Date(data.generated_at).toLocaleString()}\n`;
        content += "=".repeat(60) + "\n\n";

        data.sections.forEach(section => {
            content += `## ${section.title}\n\n`;
            content += `${section.content}\n\n`;
        });

        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "analysis-report.txt";
        a.click();
        URL.revokeObjectURL(url);
    };

    const copyShareLink = () => {
        navigator.clipboard.writeText(window.location.href);
        alert("Report link copied to clipboard!");
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-500 mx-auto mb-4" />
                    <p className="text-slate-400">Generating report...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-10">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/modeling" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <FileText className="w-5 h-5" />
                        </div>
                        <span className="font-bold text-lg tracking-tight">Analysis Report</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={exportAsText}
                            className="px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-colors flex items-center gap-1"
                        >
                            <Download className="w-4 h-4" />
                            TXT
                        </button>
                        <button
                            onClick={exportAsPDF}
                            disabled={exporting}
                            className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-sm transition-colors flex items-center gap-1"
                        >
                            <Download className="w-4 h-4" />
                            PDF
                        </button>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-10 max-w-3xl">
                {error && (
                    <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                        {error}
                    </div>
                )}

                {data && (
                    <>
                        {/* Report Header */}
                        <div className="mb-10 text-center">
                            <h1 className="text-3xl font-bold mb-3">{data.title}</h1>
                            <div className="flex items-center justify-center gap-2 text-slate-400">
                                <Calendar className="w-4 h-4" />
                                <span>{new Date(data.generated_at).toLocaleString()}</span>
                            </div>
                        </div>

                        {/* Report Sections */}
                        <div className="space-y-6">
                            {data.sections.map((section, i) => (
                                <div
                                    key={i}
                                    className="p-6 rounded-2xl bg-slate-900 border border-white/10"
                                >
                                    <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
                                        <CheckCircle className="w-5 h-5 text-green-400" />
                                        {section.title}
                                    </h2>
                                    <div className="text-slate-300 whitespace-pre-wrap leading-relaxed">
                                        {section.content}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Actions */}
                        <div className="mt-10 p-6 rounded-2xl bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
                            <h3 className="font-semibold mb-4">Share & Export</h3>
                            <div className="flex flex-wrap gap-3">
                                <button
                                    onClick={exportAsPDF}
                                    className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors flex items-center gap-2"
                                >
                                    <Download className="w-4 h-4" />
                                    Download PDF
                                </button>
                                <button
                                    onClick={exportAsText}
                                    className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 font-medium transition-colors flex items-center gap-2"
                                >
                                    <FileText className="w-4 h-4" />
                                    Download TXT
                                </button>
                                <button
                                    onClick={copyShareLink}
                                    className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 font-medium transition-colors flex items-center gap-2"
                                >
                                    <Share2 className="w-4 h-4" />
                                    Copy Link
                                </button>
                            </div>
                        </div>

                        {/* Completion Banner */}
                        <div className="mt-10 p-6 rounded-2xl bg-green-500/10 border border-green-500/20 text-center">
                            <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
                            <h3 className="text-xl font-semibold mb-2">Analysis Complete!</h3>
                            <p className="text-slate-400 mb-4">
                                You've completed the full data science workflow: Upload → Clean → Explore → Analyze → Model → Report
                            </p>
                            <button
                                onClick={() => router.push("/upload")}
                                className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition-colors"
                            >
                                Analyze New Dataset
                            </button>
                        </div>
                    </>
                )}
            </main>
        </div>
    );
}
