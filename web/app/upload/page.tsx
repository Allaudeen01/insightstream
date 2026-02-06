"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Upload, FileSpreadsheet, ArrowLeft, Loader2 } from "lucide-react";

const API_BASE = "http://localhost:8000";

interface ColumnInfo {
    name: string;
    dtype: string;
    missing_count: number;
    missing_pct: number;
    unique_count: number;
}

interface DatasetSchema {
    session_id: string;
    filename: string;
    row_count: number;
    column_count: number;
    columns: ColumnInfo[];
    preview: Record<string, unknown>[];
}

export default function UploadPage() {
    const router = useRouter();
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = async (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    };

    const handleFileUpload = async (file: File) => {
        setIsUploading(true);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`${API_BASE}/upload`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Upload failed");
            }

            const data: DatasetSchema = await response.json();

            // Store session in localStorage for now
            localStorage.setItem("analysis_session", JSON.stringify(data));

            // Navigate to Health Check (Screen 3)
            router.push("/health-check");

        } catch (err) {
            setError(err instanceof Error ? err.message : "An error occurred");
            setIsUploading(false);
        }
    };

    const handleSampleData = async () => {
        // Download a sample CSV from a public source
        setIsUploading(true);
        setError(null);

        try {
            // Using the classic Titanic dataset
            const response = await fetch("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv");
            const csvText = await response.text();
            const blob = new Blob([csvText], { type: "text/csv" });
            const file = new File([blob], "titanic.csv", { type: "text/csv" });

            await handleFileUpload(file);
        } catch (err) {
            setError("Failed to load sample dataset");
            setIsUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white selection:bg-indigo-500/30">
            {/* Navbar */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back to Home</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <span className="font-bold text-lg">V</span>
                        </div>
                        <span className="font-bold text-lg tracking-tight">VirtualScientist</span>
                    </div>
                    <div className="w-20" /> {/* Spacer for centering */}
                </div>
            </header>

            <main className="container mx-auto px-4 pt-20 pb-20">
                <div className="max-w-2xl mx-auto text-center">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-indigo-500/10 text-indigo-400 mb-6">
                        <Upload className="w-6 h-6" />
                    </div>
                    <h1 className="text-3xl font-bold mb-4">Upload your dataset</h1>
                    <p className="text-slate-400 mb-10">
                        Supported formats: CSV, Excel (.xlsx). <br />
                        We will automatically detect columns and data types.
                    </p>

                    {error && (
                        <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400">
                            {error}
                        </div>
                    )}

                    {/* Drop Zone */}
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`
              relative group border-2 border-dashed rounded-2xl p-12 transition-all duration-300
              ${isDragging
                                ? "border-indigo-500 bg-indigo-500/10 scale-[1.02]"
                                : "border-slate-800 hover:border-slate-700 bg-slate-900/50 hover:bg-slate-900"
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

                        <div className="flex flex-col items-center gap-4">
                            <div className="p-4 rounded-full bg-slate-800 group-hover:bg-slate-700 transition-colors">
                                {isUploading ? (
                                    <Loader2 className="w-8 h-8 text-indigo-500 animate-spin" />
                                ) : (
                                    <FileSpreadsheet className="w-8 h-8 text-indigo-400" />
                                )}
                            </div>
                            <div>
                                <p className="text-lg font-medium text-white mb-1">
                                    {isUploading ? "Analyzing file..." : "Drag and drop your file here"}
                                </p>
                                {!isUploading && (
                                    <p className="text-sm text-slate-500">
                                        or click to browse from your computer
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 flex items-center justify-center gap-4">
                        <div className="h-px bg-slate-800 w-full max-w-[100px]" />
                        <span className="text-slate-500 text-sm">OR</span>
                        <div className="h-px bg-slate-800 w-full max-w-[100px]" />
                    </div>

                    <button
                        onClick={handleSampleData}
                        disabled={isUploading}
                        className="mt-8 px-6 py-3 rounded-xl bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        Use Sample Dataset (Titanic.csv)
                    </button>
                </div>
            </main>
        </div>
    );
}
