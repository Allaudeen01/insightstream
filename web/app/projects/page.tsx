"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
    FolderOpen,
    Trash2,
    ArrowRight,
    Loader2,
    FileSpreadsheet,
    Clock,
    Search,
    Plus,
    Pencil,
    Check,
    X,
    BarChart3,
    Database,
    History,
    Calendar,
    ChevronRight,
    LayoutGrid,
    MoreVertical
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Project {
    id: string;
    name: string;
    filename: string;
    created_at: string;
    updated_at: string;
    row_count: number;
    column_count: number;
}

export default function ProjectsPage() {
    const router = useRouter();
    const [projects, setProjects] = useState<Project[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState("");
    const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [editName, setEditName] = useState("");
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchProjects();
    }, []);

    const fetchProjects = async () => {
        try {
            const response = await fetch(`${API_BASE}/projects`);
            if (!response.ok) throw new Error("Failed to fetch projects");
            const data = await response.json();
            setProjects(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load projects");
        } finally {
            setLoading(false);
        }
    };

    const handleResume = (project: Project) => {
        localStorage.setItem("analysis_session", JSON.stringify({
            session_id: project.id,
            filename: project.filename,
            row_count: project.row_count,
            column_count: project.column_count,
        }));
        router.push("/insights");
    };

    const handleDelete = async (id: string) => {
        try {
            const response = await fetch(`${API_BASE}/projects/${id}`, { method: "DELETE" });
            if (!response.ok) throw new Error("Delete failed");
            setProjects(prev => prev.filter(p => p.id !== id));
            setDeleteConfirm(null);
        } catch (err) {
            setError("Failed to delete project");
        }
    };

    const handleRename = async (id: string) => {
        if (!editName.trim()) return;
        try {
            const response = await fetch(`${API_BASE}/projects/${id}`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: editName.trim() }),
            });
            if (!response.ok) throw new Error("Rename failed");
            setProjects(prev => prev.map(p => p.id === id ? { ...p, name: editName.trim() } : p));
            setEditingId(null);
        } catch (err) {
            setError("Failed to rename project");
        }
    };

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        const now = new Date();
        const diff = now.getTime() - d.getTime();
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return "Just now";
        if (mins < 60) return `${mins}m ago`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days}d ago`;
        return d.toLocaleDateString();
    };

    const filtered = projects.filter(p =>
        p.name.toLowerCase().includes(search.toLowerCase()) ||
        p.filename.toLowerCase().includes(search.toLowerCase())
    );

    if (loading) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
                <Loader2 className="w-12 h-12 animate-spin text-purple-600 mb-4" />
                <p className="text-muted-foreground font-black uppercase tracking-widest text-[10px]">Accessing Vault...</p>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background pb-20">
            <Navbar />

            <main className="container mx-auto px-6 py-10 max-w-6xl">
                {/* Hero Section */}
                <div className="relative mb-12 rounded-[2.5rem] overflow-hidden border border-white/10 bg-white/5 p-8 md:p-12">
                    <div className="absolute top-0 right-0 w-1/3 h-full bg-purple-500/10 blur-[100px] -z-10" />
                    <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                        <div>
                            <div className="flex items-center gap-2 mb-4">
                                <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400">
                                    <Database className="w-4 h-4" />
                                </div>
                                <span className="text-sm font-bold text-purple-400 uppercase tracking-widest">Project Archive</span>
                            </div>
                            <h1 className="text-4xl md:text-5xl font-black text-foreground mb-4 tracking-tighter">
                                Neural <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-400">Workspace</span>
                            </h1>
                            <p className="text-lg text-muted-foreground max-w-xl">
                                Access your previously synthesized findings. Each session stores the full analytical context, allowing you to resume intelligence generation instantly.
                            </p>
                        </div>
                        <Link
                            href="/upload"
                            className="flex items-center gap-3 px-8 py-4 bg-purple-600 hover:bg-purple-500 rounded-2xl font-black text-sm uppercase tracking-tighter transition-all shadow-2xl shadow-purple-500/20"
                        >
                            <Plus className="w-5 h-5" />
                            New Exploration
                        </Link>
                    </div>
                </div>

                {error && (
                    <div className="mb-8 p-6 rounded-2xl bg-red-500/5 border border-red-500/20 text-red-400 animate-page-enter">
                        <div className="flex items-center justify-between">
                            <span className="font-bold">{error}</span>
                            <button onClick={() => setError(null)}><X className="w-4 h-4" /></button>
                        </div>
                    </div>
                )}

                {/* Search & Stats Bar */}
                <div className="flex flex-col md:flex-row gap-4 mb-12">
                    <div className="relative flex-1 group">
                        <Search className="absolute left-5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-purple-400 transition-colors" />
                        <input
                            type="text"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            placeholder="Locate intelligence stream..."
                            className="w-full bg-white/5 border border-white/10 rounded-2xl pl-12 pr-4 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/50 placeholder:text-muted-foreground/50 transition-all font-medium"
                        />
                    </div>
                    <div className="flex items-center gap-4 px-6 bg-white/5 rounded-2xl border border-white/10 text-xs font-bold uppercase tracking-widest text-muted-foreground">
                        <span className="text-purple-400">{projects.length}</span> Total Streams
                    </div>
                </div>

                {/* Projects Grid */}
                {filtered.length === 0 ? (
                    <div className="text-center py-32 rounded-[3rem] border border-dashed border-white/10 animate-page-enter">
                        <FolderOpen className="w-16 h-16 text-white/5 mx-auto mb-6" />
                        <h2 className="text-2xl font-black tracking-tight mb-2">Workspace Empty</h2>
                        <p className="text-muted-foreground mb-8">No matching tactical datasets were found in our neural vault.</p>
                        <Link
                            href="/upload"
                            className="inline-flex items-center gap-2 px-8 py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-2xl font-bold transition-all"
                        >
                            Upload Master Dataset
                        </Link>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-page-enter">
                        {filtered.map((project) => (
                            <div
                                key={project.id}
                                className="group relative flex flex-col p-6 rounded-[2rem] bg-white/5 border border-white/10 hover:border-purple-500/40 transition-all duration-300 hover:-translate-y-1"
                            >
                                <div className="flex items-start justify-between mb-6">
                                    <div className="flex items-center gap-3 min-w-0">
                                        <div className="p-3 rounded-2xl bg-purple-500/10 text-purple-400 group-hover:scale-110 transition-transform">
                                            <FileSpreadsheet className="w-6 h-6" />
                                        </div>
                                        <div className="min-w-0">
                                            {editingId === project.id ? (
                                                <div className="flex items-center gap-2">
                                                    <input
                                                        autoFocus
                                                        value={editName}
                                                        onChange={(e) => setEditName(e.target.value)}
                                                        onKeyDown={(e) => {
                                                            if (e.key === "Enter") handleRename(project.id);
                                                            if (e.key === "Escape") setEditingId(null);
                                                        }}
                                                        className="bg-background border border-purple-500 rounded-lg px-2 py-1 text-sm font-bold"
                                                    />
                                                </div>
                                            ) : (
                                                <h3 className="font-bold text-foreground truncate group-hover:text-purple-400 transition-colors">{project.name}</h3>
                                            )}
                                            <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground truncate">{project.filename}</p>
                                        </div>
                                    </div>
                                    
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button
                                            onClick={() => { setEditingId(project.id); setEditName(project.name); }}
                                            className="p-2 hover:bg-white/10 rounded-xl text-muted-foreground hover:text-white transition-colors"
                                        >
                                            <Pencil className="w-4 h-4" />
                                        </button>
                                        <button
                                            onClick={() => setDeleteConfirm(project.id)}
                                            className="p-2 hover:bg-red-500/10 rounded-xl text-muted-foreground hover:text-red-400 transition-colors"
                                        >
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-3 mb-6">
                                    <div className="p-3 rounded-2xl bg-white/5 border border-white/5">
                                        <p className="text-[8px] font-black uppercase tracking-widest text-muted-foreground mb-1">Rows</p>
                                        <p className="font-bold tracking-tight">{project.row_count?.toLocaleString()}</p>
                                    </div>
                                    <div className="p-3 rounded-2xl bg-white/5 border border-white/5">
                                        <p className="text-[8px] font-black uppercase tracking-widest text-muted-foreground mb-1">Impact</p>
                                        <p className="font-bold tracking-tight text-emerald-400">Optimal</p>
                                    </div>
                                </div>

                                <div className="mt-auto flex items-center justify-between pt-6 border-t border-white/5">
                                    <div className="flex items-center gap-2 text-xs font-bold text-muted-foreground">
                                        <Clock className="w-3.5 h-3.5" />
                                        {formatDate(project.updated_at)}
                                    </div>
                                    <button
                                        onClick={() => handleResume(project)}
                                        className="flex items-center gap-2 text-xs font-black uppercase tracking-widest text-purple-400 hover:text-purple-300 transition-colors"
                                    >
                                        Engage
                                        <ChevronRight className="w-4 h-4" />
                                    </button>
                                </div>

                                {deleteConfirm === project.id && (
                                    <div className="absolute inset-0 z-10 bg-background/90 backdrop-blur-sm flex flex-col items-center justify-center p-6 rounded-[2rem] animate-page-enter">
                                        <p className="text-sm font-bold mb-4 text-center">Permanently purge this intelligence stream?</p>
                                        <div className="flex gap-3 w-full">
                                            <button
                                                onClick={() => handleDelete(project.id)}
                                                className="flex-1 py-3 bg-red-600 hover:bg-red-500 rounded-xl text-xs font-black uppercase tracking-widest"
                                            >
                                                Purge
                                            </button>
                                            <button
                                                onClick={() => setDeleteConfirm(null)}
                                                className="flex-1 py-3 bg-white/10 hover:bg-white/20 rounded-xl text-xs font-black uppercase tracking-widest"
                                            >
                                                Abort
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </main>
        </div>
    );
}
