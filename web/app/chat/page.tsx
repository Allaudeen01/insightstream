"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
    ArrowLeft,
    Send,
    Loader2,
    MessageSquare,
    User,
    Bot,
    Download,
    Copy,
    Check,
    BarChart3,
    Sparkles,
    Terminal,
    ChevronRight,
    Command
} from "lucide-react";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    chart_type?: string;
    chart_data?: { labels: string[]; values: number[] };
    sql_equivalent?: string;
}

export default function ChatPage() {
    const router = useRouter();
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [copied, setCopied] = useState<number | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        setSessionId(session.session_id);

        setMessages([{
            role: "assistant",
            content: `I've initialized the brain for "${session.filename}". I can run multi-dimensional queries, synthesize trends, or explain specific anomalies.\n\nWhat would you like to explore?`
        }]);
    }, [router]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || !sessionId || loading) return;

        const userMessage: ChatMessage = { role: "user", content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE}/chat/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: input })
            });

            if (!response.ok) throw new Error("Terminal link severed");

            const data = await response.json();
            const assistantMessage: ChatMessage = {
                role: "assistant",
                content: data.answer,
                chart_type: data.chart_type,
                chart_data: data.chart_data,
                sql_equivalent: data.sql_equivalent
            };
            setMessages(prev => [...prev, assistantMessage]);
        } catch {
            setMessages(prev => [...prev, {
                role: "assistant",
                content: "Data pipeline interrupted. Please verify backend stability."
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const copyToClipboard = (text: string, index: number) => {
        navigator.clipboard.writeText(text);
        setCopied(index);
        setTimeout(() => setCopied(null), 2000);
    };

    const exportChat = () => {
        const content = messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n\n");
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "InsightStream_Internal_Log.txt";
        a.click();
    };

    const renderChart = (msg: ChatMessage) => {
        if (!msg.chart_data || !msg.chart_type) return null;

        const { labels, values } = msg.chart_data;
        const maxVal = Math.max(...values);

        return (
            <div className="mt-4 p-5 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm">
                <div className="flex items-end gap-2 h-36">
                    {values.map((val, i) => (
                        <div key={i} className="flex-1 flex flex-col items-center gap-2 group/bar">
                            <div 
                                className="w-full bg-gradient-to-t from-purple-600 to-purple-400 rounded-lg transition-all group-hover/bar:brightness-125 group-hover/bar:scale-x-110"
                                style={{ height: `${(val / maxVal) * 100}%`, minHeight: '8px' }}
                            />
                            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest truncate w-full text-center">{labels[i]}</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="min-h-screen bg-background flex flex-col overflow-hidden">
            <Navbar />

            {/* Chat Body */}
            <div className="flex-1 overflow-y-auto pt-8 pb-32 px-6">
                <div className="max-w-4xl mx-auto space-y-8">
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex gap-5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                            <div className={`w-10 h-10 rounded-2xl flex items-center justify-center shrink-0 border transition-all ${
                                msg.role === "assistant" 
                                    ? "bg-purple-600 border-purple-500 shadow-lg shadow-purple-500/20" 
                                    : "bg-white/5 border-white/10"
                            }`}>
                                {msg.role === "assistant" ? <Sparkles className="w-5 h-5 text-white" /> : <User className="w-5 h-5 text-muted-foreground" />}
                            </div>

                            <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === "user" ? "items-end" : "items-start"}`}>
                                <div className={`px-6 py-4 rounded-3xl text-[15px] leading-relaxed relative ${
                                    msg.role === "user"
                                        ? "bg-purple-600 text-white rounded-tr-none"
                                        : "bg-white/5 border border-white/10 text-foreground rounded-tl-none backdrop-blur-sm"
                                }`}>
                                    <p className="whitespace-pre-wrap">{msg.content}</p>
                                    {renderChart(msg)}
                                    
                                    {msg.sql_equivalent && (
                                        <div className="mt-4 p-4 rounded-xl bg-black/40 border border-white/5 font-mono text-xs overflow-hidden group">
                                            <div className="flex items-center justify-between mb-2 text-muted-foreground/60">
                                                <div className="flex items-center gap-2">
                                                    <Terminal className="w-3 h-3" />
                                                    <span className="uppercase tracking-widest font-bold">Relational Logic</span>
                                                </div>
                                                <button 
                                                    onClick={() => copyToClipboard(msg.sql_equivalent!, i)}
                                                    className="opacity-0 group-hover:opacity-100 transition-opacity hover:text-purple-400"
                                                >
                                                    {copied === i ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                                                </button>
                                            </div>
                                            <code className="text-purple-300 block overflow-x-auto whitespace-pre">{msg.sql_equivalent}</code>
                                        </div>
                                    )}
                                </div>
                                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest px-2">
                                    {msg.role === "assistant" ? "Insight Engine" : "Internal Request"}
                                </span>
                            </div>
                        </div>
                    ))}
                    
                    {loading && (
                        <div className="flex gap-5">
                            <div className="w-10 h-10 rounded-2xl bg-purple-600 flex items-center justify-center border border-purple-500 animate-pulse">
                                <Sparkles className="w-5 h-5 text-white" />
                            </div>
                            <div className="bg-white/5 border border-white/10 rounded-3xl rounded-tl-none p-5">
                                <div className="flex gap-2">
                                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} className="h-4" />
                </div>
            </div>

            {/* Input Overlay */}
            <div className="fixed bottom-0 left-0 right-0 p-8 pt-0 bg-gradient-to-t from-background via-background to-transparent pointer-events-none">
                <div className="max-w-4xl mx-auto pointer-events-auto">
                    <div className="relative group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-purple-400 rounded-3xl blur opacity-20 group-focus-within:opacity-40 transition-opacity" />
                        <div className="relative flex items-center gap-3 bg-white/5 border border-white/10 rounded-3xl p-2 pl-6 backdrop-blur-xl">
                            <Command className="w-5 h-5 text-muted-foreground" />
                            <input 
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Sync data patterns or request multi-pivot analysis..."
                                className="flex-1 bg-transparent py-4 text-sm text-foreground focus:outline-none placeholder:text-muted-foreground/50"
                                disabled={loading}
                            />
                            <button 
                                onClick={handleSend}
                                disabled={loading || !input.trim()}
                                className="p-4 bg-purple-600 hover:bg-purple-500 text-white rounded-2xl transition-all disabled:opacity-50 disabled:grayscale group"
                            >
                                {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5 transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />}
                            </button>
                        </div>
                    </div>
                    
                    <div className="flex gap-2 mt-4 overflow-x-auto no-scrollbar pb-2">
                        {[
                            "Synthesize growth trends", 
                            "Identify outlier segments", 
                            "Verify profitability vs volume",
                            "Explain variance in Q3"
                        ].map((q) => (
                            <button
                                key={q}
                                onClick={() => setInput(q)}
                                className="px-4 py-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground hover:text-foreground bg-white/5 border border-white/10 rounded-xl hover:border-purple-500/50 transition-all whitespace-nowrap"
                            >
                                {q}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

