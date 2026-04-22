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
    CheckCircle,
    FileDown,
    Printer,
    BadgeCheck,
    ArrowRight,
    Sparkles
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        fetchReport(session.session_id);
    }, [router]);

    const fetchReport = async (sessionId: string) => {
        try {
            const response = await fetch(`${API_BASE}/report/${sessionId}`);
            if (!response.ok) throw new Error("Failed to compile executive brief");
            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Protocol synchronization failed");
        } finally {
            setLoading(false);
        }
    };

    const exportAsPDF = () => {
        setExporting(true);
        const printContent = `
            <html>
                <head>
                    <title>${data?.title || "Strategic Brief"}</title>
                    <style>
                        body { font-family: 'Inter', sans-serif; padding: 60px; max-width: 900px; margin: 0 auto; color: #1e293b; line-height: 1.6; }
                        h1 { font-size: 32px; font-weight: 800; color: #7c3aed; margin-bottom: 8px; }
                        .date { color: #64748b; font-size: 14px; margin-bottom: 40px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
                        h2 { font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 40px; border-left: 4px solid #7c3aed; padding-left: 16px; }
                        p { font-size: 16px; margin-bottom: 24px; }
                        .section { margin-bottom: 40px; p { white-space: pre-wrap; } }
                    </style>
                </head>
                <body>
                    <h1>${data?.title || "Strategic Brief"}</h1>
                    <p class="date">Authored by InsightStream AI • ${data ? new Date(data.generated_at).toLocaleDateString() : ""}</p>
                    ${data?.sections.map(s => `
                        <div class="section">
                            <h2>${s.title}</h2>
                            <p>${s.content}</p>
                        </div>
                    `).join("")}
                </body>
            </html>
        `;

        try {
            const printWindow = window.open("", "_blank");
            if (printWindow) {
                printWindow.document.write(printContent);
                printWindow.document.close();
                printWindow.print();
            }
        } finally {
            setExporting(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center">
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-purple-500/20 border-t-purple-500 rounded-lg animate-spin" />
                    <FileText className="w-6 h-6 text-purple-400 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
                </div>
                <p className="mt-6 text-muted-foreground font-bold uppercase tracking-widest text-xs animate-pulse">Compiling Strategic Brief...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            <Navbar />

            <main className="container mx-auto px-6 py-12 max-w-4xl">
                {error && (
                    <div className="mb-12 p-5 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-400 font-medium">
                        {error}
                    </div>
                )}

                {data && (
                    <div className="animate-page-enter">
                        {/* Report Header */}
                        <div className="mb-16">
                            <div className="flex items-center gap-2 mb-4">
                                <span className="px-3 py-1 rounded-full bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-[0.2em] border border-purple-500/20">
                                    Official Transcript
                                </span>
                            </div>
                            <h1 className="text-4xl md:text-5xl font-black text-foreground tracking-tight mb-6">
                                {data.title}
                            </h1>
                            <div className="flex items-center gap-6 text-sm text-muted-foreground font-medium">
                                <div className="flex items-center gap-2">
                                    <Calendar className="w-4 h-4" />
                                    <span>{new Date(data.generated_at).toLocaleDateString(undefined, { dateStyle: 'long' })}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <BadgeCheck className="w-4 h-4 text-emerald-400" />
                                    <span>AI Verified Analysis</span>
                                </div>
                            </div>
                        </div>

                        {/* Report Sections */}
                        <div className="space-y-12">
                            {data.sections.map((section, i) => (
                                <section key={i} className="relative pl-8 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-px before:bg-gradient-to-b before:from-purple-500 before:to-transparent">
                                    <h2 className="text-xl font-bold text-foreground mb-4 uppercase tracking-widest text-xs opacity-50">
                                        Section {i + 1}: {section.title}
                                    </h2>
                                    <div className="text-lg text-foreground/80 leading-relaxed whitespace-pre-wrap font-serif">
                                        {section.content}
                                    </div>
                                </section>
                            ))}
                        </div>

                        {/* Export Actions Overlay */}
                        <div className="mt-20 p-8 rounded-3xl bg-white/5 border border-white/10 backdrop-blur-xl">
                            <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                                <div>
                                    <h3 className="text-xl font-bold text-foreground mb-2">Finalize Documentation</h3>
                                    <p className="text-muted-foreground text-sm">Download your findings in professional formats</p>
                                </div>
                                <div className="flex gap-4">
                                    <button
                                        onClick={exportAsPDF}
                                        disabled={exporting}
                                        className="flex items-center gap-3 px-8 py-4 bg-purple-600 hover:bg-purple-500 text-white rounded-2xl font-bold transition-all shadow-xl shadow-purple-500/20"
                                    >
                                        <Printer className="w-5 h-5" />
                                        Print Brief
                                    </button>
                                    <button
                                        onClick={() => window.print()}
                                        className="p-4 bg-white/5 hover:bg-white/10 border border-white/10 text-foreground rounded-2xl transition-all"
                                    >
                                        <Share2 className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Workflow Completion */}
                        <div className="mt-12 p-8 rounded-3xl bg-emerald-500/10 border border-emerald-500/20 text-center">
                            <div className="inline-flex p-3 rounded-2xl bg-emerald-500/20 mb-6">
                                <Sparkles className="w-8 h-8 text-emerald-400" />
                            </div>
                            <h2 className="text-2xl font-bold text-foreground mb-3">Cycle Complete</h2>
                            <p className="text-muted-foreground mb-8 max-w-lg mx-auto">
                                You have successfully navigated the end-to-end intelligence pipeline. Ready to ingest another dimension?
                            </p>
                            <button
                                onClick={() => router.push("/upload")}
                                className="inline-flex items-center gap-2 px-8 py-3 bg-foreground text-background hover:bg-foreground/90 rounded-2xl font-bold transition-all"
                            >
                                Start New Session
                                <ArrowRight className="w-5 h-5" />
                            </button>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}

