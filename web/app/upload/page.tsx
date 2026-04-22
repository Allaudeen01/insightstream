"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { 
    Upload, 
    FileSpreadsheet, 
    ArrowLeft, 
    Loader2, 
    FileText, 
    Sparkles, 
    CheckCircle2, 
    AlertCircle,
    ChevronRight,
    Search,
    CloudUpload,
    LayoutGrid
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ColumnInfo {
    name: string;
    dtype: string;
    missing_count: number;
    missing_pct: number;
    unique_count: number;
}

export default function UploadPage() {
    const router = useRouter();
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [errorDetail, setErrorDetail] = useState<string | null>(null);
    const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
    const [showSheetModal, setShowSheetModal] = useState(false);
    const [availableSheets, setAvailableSheets] = useState<string[]>([]);
    const [pendingFile, setPendingFile] = useState<File | null>(null);

    useEffect(() => {
        const controller = new AbortController();
        const timeoutId = window.setTimeout(() => controller.abort(), 5000);

        const checkHealth = async () => {
            try {
                const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
                setApiStatus(response.ok ? "online" : "offline");
            } catch (err) {
                setApiStatus("offline");
            } finally {
                window.clearTimeout(timeoutId);
            }
        };

        checkHealth();
        return () => {
            controller.abort();
            window.clearTimeout(timeoutId);
        };
    }, []);

    const handleFileUpload = async (file: File, sheetName?: string) => {
        setIsUploading(true);
        setError(null);
        setErrorDetail(null);

        if (!sheetName) {
            setPendingFile(null);
            setShowSheetModal(false);
        }

        const formData = new FormData();
        formData.append("file", file);
        if (sheetName) {
            formData.append("sheet_name", sheetName);
        }

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Upload failed");
            }

            const data = await response.json();

            if (data.requires_selection && data.sheets) {
                setPendingFile(file);
                setAvailableSheets(data.sheets);
                setShowSheetModal(true);
                return;
            }

            localStorage.setItem("analysis_session", JSON.stringify(data));
            router.push("/health-check");

        } catch (err) {
            if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
                setError("Neural connection timeout.");
                setErrorDetail(`Ensure terminal 8000 is running.`);
            } else {
                setError(err instanceof Error ? err.message : "An error occurred");
                setErrorDetail(null);
            }
        } finally {
            setIsUploading(false);
        }
    };

    const handleSampleData = async () => {
        setIsUploading(true);
        setError(null);
        setErrorDetail(null);

        try {
            const response = await fetch("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv");
            const csvText = await response.text();
            const blob = new Blob([csvText], { type: "text/csv" });
            const file = new File([blob], "titanic.csv", { type: "text/csv" });
            await handleFileUpload(file);
        } catch (err) {
            setError("Failed to fetch tactical sample.");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background pb-20">
            <Navbar />

            <main className="container mx-auto px-6 py-20 max-w-4xl">
                {/* Header Section */}
                <div className="text-center mb-16 animate-page-enter">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-black uppercase tracking-widest mb-6">
                        <CloudUpload className="w-3 h-3" />
                        Intelligence Intake
                    </div>
                    <h1 className="text-4xl md:text-6xl font-black text-foreground mb-6 tracking-tighter">
                        Deploy Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 italic">Data Stream</span>
                    </h1>
                    <p className="text-lg text-muted-foreground max-w-xl mx-auto font-medium">
                        Supported tactical formats: <span className="text-foreground">CSV, Excel, Parablocks</span>. Our engine will autonomously map schemas and detect anomalies.
                    </p>
                </div>

                {/* API Status Indicator */}
                <div className="flex justify-center mb-12 animate-page-enter">
                    <div className={`flex items-center gap-2 px-4 py-2 rounded-2xl border transition-all ${apiStatus === 'online' ? 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400' : 'bg-amber-500/5 border-amber-500/20 text-amber-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${apiStatus === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                        <span className="text-[10px] font-black uppercase tracking-widest leading-none">
                            Engine Status: {apiStatus}
                        </span>
                    </div>
                </div>

                {error && (
                    <div className="max-w-xl mx-auto mb-10 p-6 rounded-3xl bg-red-500/5 border border-red-500/20 text-red-400 animate-page-enter">
                        <div className="flex items-center gap-3 mb-2">
                            <AlertCircle className="w-5 h-5" />
                            <span className="font-bold">{error}</span>
                        </div>
                        {errorDetail && <p className="text-sm text-red-400/70 ml-8">{errorDetail}</p>}
                    </div>
                )}

                {/* Drop Zone */}
                <div className="max-w-xl mx-auto animate-page-enter">
                    <div
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onDrop={(e) => { e.preventDefault(); setIsDragging(false); if(e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]); }}
                        className={`
                            relative group p-12 rounded-[3rem] border-2 border-dashed transition-all duration-500
                            ${isDragging 
                                ? "border-indigo-500 bg-indigo-500/10 scale-105" 
                                : "border-white/10 hover:border-indigo-500/30 bg-white/5 hover:bg-white/10"
                            }
                        `}
                    >
                        <input
                            type="file"
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                            onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
                            accept=".csv, .xlsx, .xls"
                            disabled={isUploading}
                        />

                        <div className="flex flex-col items-center text-center gap-6">
                            <div className={`p-6 rounded-3xl transition-all duration-500 ${isUploading ? 'bg-indigo-500/20 rotate-180' : 'bg-white/5 group-hover:bg-indigo-500/10 group-hover:scale-110'}`}>
                                {isUploading ? (
                                    <Loader2 className="w-12 h-12 text-indigo-400 animate-spin" />
                                ) : (
                                    <FileSpreadsheet className="w-12 h-12 text-indigo-400 group-hover:text-indigo-300" />
                                )}
                            </div>
                            <div>
                                <h3 className="text-2xl font-black mb-2 tracking-tight group-hover:text-indigo-400 transition-colors">
                                    {isUploading ? "Processing Nexus..." : "Initiate Uplink"}
                                </h3>
                                <p className="text-muted-foreground font-medium">
                                    {isUploading ? "Parsing structure and executing initial diagnostics" : "Drag files or click to browse local storage"}
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="mt-12 flex flex-col items-center gap-6">
                        <div className="flex items-center gap-4 w-full">
                            <div className="h-px bg-white/10 flex-1" />
                            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Baseline Scenario</span>
                            <div className="h-px bg-white/10 flex-1" />
                        </div>
                        
                        <button
                            onClick={handleSampleData}
                            disabled={isUploading}
                            className="group flex items-center gap-3 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-2xl font-black text-xs uppercase tracking-widest transition-all disabled:opacity-50"
                        >
                            <Sparkles className="w-4 h-4 text-indigo-400 group-hover:animate-pulse" />
                            Inject Sample Data (Titanic Suite)
                        </button>
                    </div>
                </div>
            </main>

            {/* Sheet Selection Modal */}
            {showSheetModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/90 backdrop-blur-xl animate-fade-in p-6">
                    <div className="bg-white/5 border border-white/10 rounded-[3rem] p-10 w-full max-w-lg shadow-[0_32px_64px_-16px_rgba(0,0,0,0.5)] animate-page-enter">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-3 rounded-2xl bg-indigo-500/20 text-indigo-400">
                                <LayoutGrid className="w-6 h-6" />
                            </div>
                            <div>
                                <h2 className="text-2xl font-black tracking-tight leading-none mb-1">Architecture Detected</h2>
                                <p className="text-xs text-muted-foreground font-bold uppercase tracking-widest">Multi-sheet detection active</p>
                            </div>
                        </div>
                        
                        <p className="text-muted-foreground font-medium mb-8">
                            This tactical archive contains multiple streams. Select the primary workspace for analysis.
                        </p>

                        <div className="grid grid-cols-1 gap-3 max-h-[300px] overflow-y-auto mb-10 pr-2 custom-scrollbar">
                            {availableSheets.map((sheet) => (
                                <button
                                    key={sheet}
                                    onClick={() => handleFileUpload(pendingFile!, sheet)}
                                    className="w-full flex items-center justify-between px-6 py-4 rounded-2xl bg-white/5 hover:bg-indigo-600 border border-white/10 hover:border-indigo-400 transition-all group"
                                >
                                    <span className="font-bold text-sm tracking-tight">{sheet}</span>
                                    <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-white transition-colors" />
                                </button>
                            ))}
                        </div>

                        <button
                            onClick={() => { setShowSheetModal(false); setPendingFile(null); setIsUploading(false); }}
                            className="w-full py-4 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-white transition-colors"
                        >
                            Abort Uplink
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
