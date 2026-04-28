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
    LayoutGrid,
    Settings2,
    Palette
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TEMPLATES = [
    { id: "modern", name: "Modern Analytics", color: "from-indigo-500 to-purple-500", desc: "Sleek, minimalist business look" },
    { id: "executive", name: "Executive Brief", color: "from-slate-700 to-slate-900", desc: "Formal, high-contrast report" },
    { id: "creative", name: "Creative Edge", color: "from-pink-500 to-orange-500", desc: "Vibrant and high-energy" }
];

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
    const [selectedTemplate, setSelectedTemplate] = useState("modern");

    useEffect(() => {
        const controller = new AbortController();
        const checkHealth = async () => {
            try {
                const response = await fetch(`${API_BASE}/health`, { signal: controller.signal });
                setApiStatus(response.ok ? "online" : "offline");
            } catch (err) {
                setApiStatus("offline");
            }
        };
        checkHealth();
        return () => controller.abort();
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

            // Save template preference
            localStorage.setItem("report_template", selectedTemplate);
            localStorage.setItem("analysis_session", JSON.stringify(data));
            router.push("/health-check");

        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-background pb-20">
            <Navbar />

            <main className="container mx-auto px-6 py-20 max-w-4xl">
                <div className="text-center mb-16 animate-page-enter">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-black uppercase tracking-widest mb-6">
                        <CloudUpload className="w-3 h-3" />
                        Intelligence Intake
                    </div>
                    <h1 className="text-4xl md:text-6xl font-black text-foreground mb-6 tracking-tighter">
                        Deploy Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 italic">Data Stream</span>
                    </h1>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-page-enter">
                    {/* Left: Configuration */}
                    <div className="lg:col-span-5 space-y-6">
                        <div className="p-6 rounded-[2rem] bg-white/5 border border-white/10">
                            <div className="flex items-center gap-2 mb-6">
                                <Palette className="w-4 h-4 text-indigo-400" />
                                <h3 className="text-xs font-black uppercase tracking-widest text-muted-foreground">Report Template</h3>
                            </div>
                            <div className="space-y-3">
                                {TEMPLATES.map((t) => (
                                    <button
                                        key={t.id}
                                        onClick={() => setSelectedTemplate(t.id)}
                                        className={`w-full text-left p-4 rounded-2xl border transition-all ${
                                            selectedTemplate === t.id 
                                            ? "bg-indigo-500/10 border-indigo-500/40 ring-1 ring-indigo-500/20" 
                                            : "bg-white/5 border-white/5 hover:border-white/10"
                                        }`}
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className={`w-3 h-3 rounded-full bg-gradient-to-br ${t.color}`} />
                                            <span className="font-bold text-sm">{t.name}</span>
                                        </div>
                                        <p className="text-[10px] text-muted-foreground mt-1 ml-6">{t.desc}</p>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right: Upload Zone */}
                    <div className="lg:col-span-7">
                        <div
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={() => setIsDragging(false)}
                            onDrop={(e) => { e.preventDefault(); setIsDragging(false); if(e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]); }}
                            className={`
                                relative group p-12 rounded-[3rem] border-2 border-dashed transition-all duration-500 h-full flex flex-col items-center justify-center
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
                                    <p className="text-muted-foreground font-medium text-sm">
                                        Drag files or click to browse local storage
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>

            {showSheetModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/90 backdrop-blur-xl p-6">
                    <div className="bg-white/5 border border-white/10 rounded-[3rem] p-10 w-full max-w-lg animate-page-enter">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-3 rounded-2xl bg-indigo-500/20 text-indigo-400">
                                <LayoutGrid className="w-6 h-6" />
                            </div>
                            <h2 className="text-2xl font-black tracking-tight">Select Workspace</h2>
                        </div>
                        <div className="grid grid-cols-1 gap-3 max-h-[300px] overflow-y-auto mb-10 pr-2">
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
                            className="w-full py-4 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-white"
                        >
                            Abort
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
