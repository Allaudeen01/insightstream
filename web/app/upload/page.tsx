// web/app/upload/page.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
    Upload,
    Plus,
    LayoutDashboard,
    Sparkles,
    MessageSquare,
    FileText,
    Loader2,
    AlertTriangle,
    CheckCircle2,
    X,
    ChevronRight,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";
import { DataQualityPanel } from "@/components/DataQualityPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const NAV = [
    { id: "upload", href: "/upload", label: "New analysis", icon: Plus, primary: true },
    { id: "dashboard", href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { id: "insights", href: "/insights", label: "Insights", icon: Sparkles },
    { id: "chat", href: "/chat", label: "Chat", icon: MessageSquare },
];

const RECENTS = [
    "Q3 sales analysis",
    "Customer churn 2025",
    "Marketing spend ROI",
];

const SAMPLES = [
    "Retail sales 2024.csv",
    "Marketing spend.xlsx",
    "Customer cohort.csv",
];

export default function UploadPage() {
    const router = useRouter();
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showSheetModal, setShowSheetModal] = useState(false);
    const [availableSheets, setAvailableSheets] = useState<string[]>([]);
    const [pendingFile, setPendingFile] = useState<File | null>(null);
    const [qualityReport, setQualityReport] = useState<any>(null);
    const qualityPanelRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // ----- API health (silent unless offline) -----
    const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
    useEffect(() => {
        const ctl = new AbortController();
        fetch(`${API_BASE}/health`, { signal: ctl.signal })
            .then(r => setApiStatus(r.ok ? "online" : "offline"))
            .catch(() => setApiStatus("offline"));
        return () => ctl.abort();
    }, []);

    // ----- Upload handler (logic preserved) -----
    const handleFileUpload = async (file: File, sheetName?: string) => {
        setIsUploading(true);
        setError(null);
        setQualityReport(null);

        if (!sheetName) {
            setPendingFile(null);
            setShowSheetModal(false);
        }

        const formData = new FormData();
        formData.append("file", file);
        if (sheetName) formData.append("sheet_name", sheetName);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || "Upload failed");

            const qr = data.quality_report ?? null;
            setQualityReport(qr);
            if (qr) {
                setTimeout(() => {
                    qualityPanelRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
                }, 100);
            }

            if (data.requires_selection && data.sheets) {
                setPendingFile(file);
                setAvailableSheets(data.sheets);
                setShowSheetModal(true);
                return;
            }

            if (qr?.summary?.can_analyze === false) return;

            localStorage.setItem("analysis_session", JSON.stringify(data));
            router.push("/health-check");
        } catch (err) {
            setError(err instanceof Error ? err.message : "Something went wrong. Try again.");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="flex h-screen bg-white text-zinc-900 antialiased">
            {/* ============= SIDEBAR ============= */}
            <Sidebar />

            {/* ============= MAIN ============= */}
            <div className="flex min-w-0 flex-1 flex-col">
                {/* Top bar */}
                <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-6 backdrop-blur">
                    <h1 className="text-[17px] font-semibold tracking-[-0.01em]">Upload</h1>
                    {apiStatus === "offline" && (
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-700">
                            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                            API offline
                        </span>
                    )}
                </header>

                {/* Body */}
                <main className="flex-1 overflow-auto">
                    <div className="mx-auto max-w-2xl px-8 py-12">
                        <h2 className="text-[30px] font-semibold leading-tight tracking-[-0.02em]">
                            Upload a file
                        </h2>
                        <p className="mt-2 text-sm leading-relaxed text-zinc-600">
                            Drop a CSV or Excel file. We&rsquo;ll analyze the data and show insights, charts, and a quality report.
                        </p>

                        {/* Dropzone */}
                        <div
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={() => setIsDragging(false)}
                            onDrop={(e) => {
                                e.preventDefault();
                                setIsDragging(false);
                                if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
                            }}
                            onClick={() => !isUploading && inputRef.current?.click()}
                            className={[
                                "mt-8 cursor-pointer rounded-xl px-6 py-14 text-center transition-colors",
                                isDragging
                                    ? "border-2 border-dashed border-[#6d5ef5] bg-[#f1efff]"
                                    : "border border-dashed border-zinc-300 bg-zinc-50 hover:bg-zinc-100",
                                isUploading && "pointer-events-none",
                            ].filter(Boolean).join(" ")}
                        >
                            <input
                                ref={inputRef}
                                type="file"
                                accept=".csv,.xlsx,.xls"
                                className="hidden"
                                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                                disabled={isUploading}
                            />

                            <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-[10px] border border-zinc-200 bg-white">
                                {isUploading ? (
                                    <Loader2 className="h-5 w-5 animate-spin text-zinc-500" strokeWidth={1.75} />
                                ) : (
                                    <Upload className="h-5 w-5 text-zinc-500" strokeWidth={1.75} />
                                )}
                            </div>
                            <div className="text-[15px] font-semibold text-zinc-900">
                                {isUploading ? "Analyzing your file…" : "Drop your file here or click to browse"}
                            </div>
                            <div className="mt-1 text-[13px] text-zinc-500">
                                Supports CSV and Excel (.xlsx, .xls) up to 50&nbsp;MB
                            </div>
                        </div>

                        {/* Error */}
                        {error && (
                            <div className="mt-4 flex items-start gap-2.5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
                                <div>
                                    <div className="font-medium">Upload failed</div>
                                    <div className="text-red-600">{error}</div>
                                </div>
                            </div>
                        )}

                        {/* Samples */}
                        <div className="mt-6">
                            <div className="mb-2.5 text-xs font-medium text-zinc-400">Or try a sample</div>
                            <div className="flex flex-wrap gap-2">
                                {SAMPLES.map((s) => (
                                    <button
                                        key={s}
                                        onClick={() => {/* sample-load hook here */ }}
                                        disabled={isUploading}
                                        className="inline-flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-[13px] text-zinc-700 transition-colors hover:bg-zinc-50 disabled:opacity-50"
                                    >
                                        <FileText className="h-3.5 w-3.5 text-zinc-500" strokeWidth={1.75} />
                                        {s}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Quality report */}
                        <div ref={qualityPanelRef} className="mt-8">
                            <DataQualityPanel report={qualityReport} />
                        </div>
                    </div>
                </main>
            </div>

            {/* ============= SHEET MODAL ============= */}
            {showSheetModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/30 backdrop-blur-sm p-4">
                    <div className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-6 shadow-lg">
                        <div className="mb-4 flex items-start justify-between">
                            <div>
                                <h3 className="text-lg font-semibold tracking-tight">Pick a sheet</h3>
                                <p className="mt-1 text-sm text-zinc-600">
                                    This file has multiple sheets. Choose the one you want to analyze.
                                </p>
                            </div>
                            <button
                                onClick={() => { setShowSheetModal(false); setPendingFile(null); setIsUploading(false); }}
                                className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600"
                                aria-label="Close"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>

                        <div className="mb-4 max-h-[280px] space-y-1 overflow-y-auto">
                            {availableSheets.map((sheet) => (
                                <button
                                    key={sheet}
                                    onClick={() => handleFileUpload(pendingFile!, sheet)}
                                    className="group flex w-full items-center justify-between rounded-lg border border-zinc-200 bg-white px-4 py-3 text-left text-sm text-zinc-900 transition-colors hover:border-[#c8c1ff] hover:bg-[#f1efff]"
                                >
                                    <span className="font-medium">{sheet}</span>
                                    <ChevronRight className="h-4 w-4 text-zinc-400 group-hover:text-[#6d5ef5]" strokeWidth={1.75} />
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={() => { setShowSheetModal(false); setPendingFile(null); setIsUploading(false); }}
                            className="w-full rounded-lg border border-zinc-200 bg-white px-4 py-2.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}