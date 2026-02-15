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
    History
} from "lucide-react";

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
        // Store session info same way upload page does
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
            <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
                <div className="text-center">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-500 mx-auto mb-4" />
                    <p className="text-slate-400">Loading projects...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl sticky top-0 z-50">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Link href="/" className="flex items-center gap-2">
                            <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                                <span className="font-bold text-lg">V</span>
                            </div>
                            <span className="font-bold text-lg tracking-tight">VirtualScientist</span>
                        </Link>
                    </div>
                    <Link
                        href="/upload"
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-medium transition-all shadow-lg shadow-indigo-500/20"
                    >
                        <Plus className="w-4 h-4" />
                        New Analysis
                    </Link>
                </div>
            </header>

            <main className="container mx-auto px-4 py-10 max-w-5xl">
                {/* Title Section */}
                <div className="mb-8">
                    <div className="flex items-center gap-3 mb-2">
                        <FolderOpen className="w-6 h-6 text-indigo-400" />
                        <h1 className="text-2xl font-bold">My Projects</h1>
                    </div>
                    <p className="text-slate-400">
                        Your saved analyses. Click any project to resume where you left off.
                    </p>
                </div>

                {error && (
                    <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 flex items-center justify-between">
                        <span>{error}</span>
                        <button onClick={() => setError(null)} className="text-red-300 hover:text-white">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                )}

                {/* Search */}
                {projects.length > 0 && (
                    <div className="relative mb-6">
                        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                        <input
                            type="text"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            placeholder="Search projects..."
                            className="w-full bg-slate-900 border border-white/10 rounded-xl pl-11 pr-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-slate-500"
                        />
                    </div>
                )}

                {/* Projects Grid */}
                {filtered.length === 0 ? (
                    <div className="text-center py-20">
                        <Database className="w-16 h-16 text-slate-700 mx-auto mb-4" />
                        <h2 className="text-xl font-semibold mb-2">
                            {projects.length === 0 ? "No projects yet" : "No matching projects"}
                        </h2>
                        <p className="text-slate-400 mb-6">
                            {projects.length === 0
                                ? "Upload a dataset to create your first project."
                                : "Try a different search term."}
                        </p>
                        {projects.length === 0 && (
                            <Link
                                href="/upload"
                                className="inline-flex items-center gap-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-xl font-medium transition-all"
                            >
                                <Plus className="w-4 h-4" />
                                Upload Dataset
                            </Link>
                        )}
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {filtered.map((project) => (
                            <div
                                key={project.id}
                                className="group p-5 rounded-2xl bg-slate-900 border border-white/10 hover:border-indigo-500/30 transition-all duration-200 hover:shadow-lg hover:shadow-indigo-500/5"
                            >
                                <div className="flex items-start justify-between mb-3">
                                    <div className="flex items-center gap-2 min-w-0 flex-1">
                                        <FileSpreadsheet className="w-5 h-5 text-indigo-400 flex-shrink-0" />
                                        {editingId === project.id ? (
                                            <div className="flex items-center gap-1.5 flex-1">
                                                <input
                                                    autoFocus
                                                    value={editName}
                                                    onChange={(e) => setEditName(e.target.value)}
                                                    onKeyDown={(e) => {
                                                        if (e.key === "Enter") handleRename(project.id);
                                                        if (e.key === "Escape") setEditingId(null);
                                                    }}
                                                    className="flex-1 bg-slate-800 border border-indigo-500/50 rounded px-2 py-1 text-sm focus:outline-none"
                                                />
                                                <button onClick={() => handleRename(project.id)} className="p-1 hover:bg-white/10 rounded text-emerald-400">
                                                    <Check className="w-3.5 h-3.5" />
                                                </button>
                                                <button onClick={() => setEditingId(null)} className="p-1 hover:bg-white/10 rounded text-slate-400">
                                                    <X className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        ) : (
                                            <h3 className="font-semibold text-white truncate">{project.name}</h3>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-1 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setEditingId(project.id);
                                                setEditName(project.name);
                                            }}
                                            className="p-1.5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"
                                            title="Rename"
                                        >
                                            <Pencil className="w-3.5 h-3.5" />
                                        </button>
                                        {deleteConfirm === project.id ? (
                                            <div className="flex items-center gap-1">
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); handleDelete(project.id); }}
                                                    className="px-2 py-1 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 transition-colors"
                                                >
                                                    Confirm
                                                </button>
                                                <button
                                                    onClick={(e) => { e.stopPropagation(); setDeleteConfirm(null); }}
                                                    className="px-2 py-1 text-xs bg-slate-700 text-slate-300 rounded hover:bg-slate-600 transition-colors"
                                                >
                                                    Cancel
                                                </button>
                                            </div>
                                        ) : (
                                            <button
                                                onClick={(e) => { e.stopPropagation(); setDeleteConfirm(project.id); }}
                                                className="p-1.5 hover:bg-red-500/10 rounded-lg text-slate-400 hover:text-red-400 transition-colors"
                                                title="Delete"
                                            >
                                                <Trash2 className="w-3.5 h-3.5" />
                                            </button>
                                        )}
                                    </div>
                                </div>

                                <div className="text-xs text-slate-500 mb-3">{project.filename}</div>

                                <div className="flex items-center gap-4 mb-4 text-xs text-slate-400">
                                    <span className="flex items-center gap-1">
                                        <BarChart3 className="w-3 h-3" />
                                        {project.row_count?.toLocaleString() || "?"} rows
                                    </span>
                                    <span>{project.column_count || "?"} cols</span>
                                    <span className="flex items-center gap-1 ml-auto">
                                        <Clock className="w-3 h-3" />
                                        {formatDate(project.updated_at)}
                                    </span>
                                </div>

                                <button
                                    onClick={() => handleResume(project)}
                                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600/10 hover:bg-indigo-600 text-indigo-400 hover:text-white text-sm font-medium transition-all duration-200 border border-indigo-500/20 hover:border-indigo-500"
                                >
                                    Resume Analysis
                                    <ArrowRight className="w-4 h-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </main>
        </div>
    );
}
