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
    BarChart3
} from "lucide-react";

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
        if (!stored) {
            router.push("/upload");
            return;
        }
        const session = JSON.parse(stored);
        setSessionId(session.session_id);

        // Add welcome message
        setMessages([{
            role: "assistant",
            content: `Welcome! I'm ready to analyze your dataset "${session.filename}".\n\nTry asking:\n• "What is the total [column]?"\n• "Show me top 5 [category]"\n• "Why did [metric] change?"\n• "How many [category] are there?"`
        }]);
    }, [router]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
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

            if (!response.ok) throw new Error("Chat request failed");

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
                content: "Sorry, I encountered an error processing your question. Please try again."
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
        const content = messages
            .map(m => `${m.role.toUpperCase()}: ${m.content}`)
            .join("\n\n");
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "chat-export.txt";
        a.click();
        URL.revokeObjectURL(url);
    };

    const renderChart = (msg: ChatMessage) => {
        if (!msg.chart_data || !msg.chart_type) return null;

        const { labels, values } = msg.chart_data;
        const maxVal = Math.max(...values);

        if (msg.chart_type === "bar") {
            return (
                <div className="mt-4 p-4 rounded-lg bg-slate-800/50">
                    <div className="flex items-end gap-2 h-32">
                        {values.map((val, i) => (
                            <div key={i} className="flex-1 flex flex-col items-center gap-1">
                                <span className="text-xs text-slate-400">{val.toLocaleString()}</span>
                                <div
                                    className="w-full bg-gradient-to-t from-indigo-600 to-indigo-400 rounded-t"
                                    style={{ height: `${(val / maxVal) * 100}%`, minHeight: '8px' }}
                                />
                                <span className="text-xs text-slate-500 truncate w-full text-center">{labels[i]?.slice(0, 10)}</span>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        if (msg.chart_type === "pie") {
            const total = values.reduce((a, b) => a + b, 0);
            return (
                <div className="mt-4 p-4 rounded-lg bg-slate-800/50">
                    <div className="space-y-2">
                        {labels.slice(0, 5).map((label, i) => {
                            const pct = Math.round((values[i] / total) * 100);
                            return (
                                <div key={i} className="flex items-center gap-2">
                                    <div className="w-24 truncate text-xs text-slate-400">{label}</div>
                                    <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-gradient-to-r from-indigo-600 to-purple-500"
                                            style={{ width: `${pct}%` }}
                                        />
                                    </div>
                                    <div className="w-12 text-xs text-slate-400 text-right">{pct}%</div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            );
        }

        return null;
    };

    return (
        <div className="min-h-screen bg-slate-950 text-white flex flex-col">
            {/* Header */}
            <header className="border-b border-white/10 bg-slate-950/50 backdrop-blur-xl flex-shrink-0">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <Link href="/insights" className="flex items-center gap-2 hover:text-indigo-400 transition-colors">
                        <ArrowLeft className="w-5 h-5" />
                        <span className="font-medium">Back</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <div className="h-8 w-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <MessageSquare className="w-5 h-5" />
                        </div>
                        <span className="font-bold text-lg tracking-tight">Chat with Data</span>
                    </div>
                    <button
                        onClick={exportChat}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-sm transition-colors"
                    >
                        <Download className="w-4 h-4" />
                        Export
                    </button>
                </div>
            </header>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4">
                <div className="max-w-3xl mx-auto space-y-4">
                    {messages.map((msg, i) => (
                        <div
                            key={i}
                            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            {msg.role === "assistant" && (
                                <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
                                    <Bot className="w-5 h-5" />
                                </div>
                            )}
                            <div
                                className={`max-w-[80%] rounded-2xl p-4 ${msg.role === "user"
                                    ? "bg-indigo-600 text-white"
                                    : "bg-slate-800 text-white"
                                    }`}
                            >
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                                {renderChart(msg)}
                                {msg.sql_equivalent && (
                                    <div className="mt-3 p-2 rounded bg-slate-900 text-xs">
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-slate-500">SQL Equivalent:</span>
                                            <button
                                                onClick={() => copyToClipboard(msg.sql_equivalent!, i)}
                                                className="text-slate-400 hover:text-white"
                                            >
                                                {copied === i ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                                            </button>
                                        </div>
                                        <code className="text-indigo-400">{msg.sql_equivalent}</code>
                                    </div>
                                )}
                            </div>
                            {msg.role === "user" && (
                                <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
                                    <User className="w-5 h-5" />
                                </div>
                            )}
                        </div>
                    ))}
                    {loading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
                                <Bot className="w-5 h-5" />
                            </div>
                            <div className="bg-slate-800 rounded-2xl p-4">
                                <Loader2 className="w-5 h-5 animate-spin text-indigo-400" />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area */}
            <div className="border-t border-white/10 bg-slate-900/50 p-4 flex-shrink-0">
                <div className="max-w-3xl mx-auto">
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Ask a question about your data..."
                            className="flex-1 bg-slate-800 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                            disabled={loading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={loading || !input.trim()}
                            className="px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl transition-colors"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </div>
                    <div className="flex gap-2 mt-3 flex-wrap">
                        {["What is the total?", "Show top 5", "Why did it change?", "Count by category"].map((q) => (
                            <button
                                key={q}
                                onClick={() => setInput(q)}
                                className="px-3 py-1 text-xs rounded-full bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
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
