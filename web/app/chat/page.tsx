// web/app/chat/page.tsx
"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import {
    Send,
    Loader2,
    User,
    Sparkles,
    Copy,
    Check,
    Download,
    Code2,
    ArrowUp,
} from "lucide-react";
import Sidebar from "@/components/Sidebar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const SUGGESTIONS = [
    "Show me sales trends over time",
    "Which segments drive the most revenue?",
    "Find anomalies in the last quarter",
    "Compare performance year over year",
];

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
    const [filename, setFilename] = useState<string>("");
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [copied, setCopied] = useState<number | null>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // ----- Load session -----
    useEffect(() => {
        const stored = localStorage.getItem("analysis_session");
        if (!stored) { router.push("/upload"); return; }
        const session = JSON.parse(stored);
        setSessionId(session.session_id);
        setFilename(session.filename || "your dataset");
        setMessages([{
            role: "assistant",
            content: `I've loaded "${session.filename}". Ask me anything — I can run queries, surface trends, or explain anomalies.`,
        }]);
    }, [router]);

    // ----- Auto-scroll to latest -----
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }, [messages, loading]);

    // ----- Auto-grow textarea -----
    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }, [input]);

    // ----- Send -----
    const handleSend = async () => {
        if (!input.trim() || !sessionId || loading) return;

        const userMessage: ChatMessage = { role: "user", content: input.trim() };
        setMessages(prev => [...prev, userMessage]);
        const question = input.trim();
        setInput("");
        setLoading(true);

        try {
            const response = await fetch(`${API_BASE}/chat/${sessionId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            if (!response.ok) throw new Error("Request failed");

            const data = await response.json();
            setMessages(prev => [...prev, {
                role: "assistant",
                content: data.answer,
                chart_type: data.chart_type,
                chart_data: data.chart_data,
                sql_equivalent: data.sql_equivalent,
            }]);
        } catch {
            setMessages(prev => [...prev, {
                role: "assistant",
                content: "Sorry — I couldn't reach the server. Please try again in a moment.",
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
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
            .map(m => `${m.role === "user" ? "You" : "InsightStream"}: ${m.content}`)
            .join("\n\n");
        const blob = new Blob([content], { type: "text/plain" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "insightstream-chat.txt";
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 100);
    };

    return (
        <div className="flex h-screen bg-white text-zinc-900 antialiased">
            <Sidebar />

            <div className="flex min-w-0 flex-1 flex-col">
                {/* Top bar */}
                <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b border-zinc-200 bg-white/85 px-6 backdrop-blur">
                    <div className="flex items-center gap-3 min-w-0">
                        <h1 className="text-[17px] font-semibold tracking-[-0.01em]">Chat</h1>
                        <span className="text-zinc-300">/</span>
                        <span className="truncate text-[13px] text-zinc-500">{filename}</span>
                    </div>
                    <button
                        onClick={exportChat}
                        disabled={messages.length <= 1}
                        className="inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-200 bg-white px-3 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
                    >
                        <Download className="h-3.5 w-3.5" strokeWidth={1.75} />
                        Export
                    </button>
                </header>

                {/* Conversation */}
                <main className="flex-1 overflow-y-auto">
                    <div className="mx-auto max-w-3xl px-6 pt-8 pb-44">
                        <div className="space-y-6">
                            {messages.map((msg, i) => (
                                <Message
                                    key={i}
                                    msg={msg}
                                    index={i}
                                    copied={copied === i}
                                    onCopy={(text) => copyToClipboard(text, i)}
                                />
                            ))}

                            {loading && <TypingIndicator />}
                            <div ref={messagesEndRef} />
                        </div>
                    </div>
                </main>

                {/* Composer */}
                <div className="sticky bottom-0 border-t border-zinc-200 bg-white/95 backdrop-blur">
                    <div className="mx-auto max-w-3xl px-6 py-4">
                        {/* Suggestion chips — only on the first turn */}
                        {messages.length <= 1 && (
                            <div className="mb-3 flex flex-wrap gap-2">
                                {SUGGESTIONS.map((s) => (
                                    <button
                                        key={s}
                                        onClick={() => setInput(s)}
                                        className="rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-[12.5px] text-zinc-700 transition-colors hover:border-zinc-300 hover:bg-zinc-50"
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        )}

                        <div className="flex items-end gap-2 rounded-2xl border border-zinc-200 bg-white p-2 shadow-sm focus-within:border-[#6d5ef5] focus-within:ring-2 focus-within:ring-[#6d5ef5]/15 transition-shadow">
                            <textarea
                                ref={textareaRef}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask about your data…"
                                rows={1}
                                disabled={loading}
                                className="flex-1 resize-none bg-transparent px-3 py-2 text-[14px] leading-relaxed text-zinc-900 placeholder:text-zinc-400 focus:outline-none disabled:opacity-50"
                            />
                            <button
                                onClick={handleSend}
                                disabled={loading || !input.trim()}
                                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#6d5ef5] text-white transition-colors hover:bg-[#5b4be0] disabled:bg-zinc-200 disabled:text-zinc-400"
                                aria-label="Send"
                            >
                                {loading
                                    ? <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
                                    : <ArrowUp className="h-4 w-4" strokeWidth={2.25} />}
                            </button>
                        </div>

                        <div className="mt-2 px-2 text-[11px] text-zinc-400">
                            Press <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1 font-mono text-[10px] text-zinc-600">Enter</kbd> to send,{" "}
                            <kbd className="rounded border border-zinc-200 bg-zinc-50 px-1 font-mono text-[10px] text-zinc-600">Shift+Enter</kbd> for newline
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

/* ============================================================
   Message
   ============================================================ */

function Message({
    msg, index, copied, onCopy,
}: {
    msg: ChatMessage;
    index: number;
    copied: boolean;
    onCopy: (text: string) => void;
}) {
    const isUser = msg.role === "user";
    return (
        <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}>
            <Avatar role={msg.role} />

            <div className={`flex max-w-[85%] flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>
                <div
                    className={[
                        "rounded-2xl px-4 py-2.5 text-[14px] leading-relaxed",
                        isUser
                            ? "bg-[#6d5ef5] text-white rounded-tr-sm"
                            : "bg-zinc-50 border border-zinc-200 text-zinc-900 rounded-tl-sm",
                    ].join(" ")}
                >
                    <p className="whitespace-pre-wrap">{msg.content}</p>

                    {msg.chart_data && msg.chart_type && (
                        <InlineChart labels={msg.chart_data.labels} values={msg.chart_data.values} />
                    )}

                    {msg.sql_equivalent && (
                        <SqlBlock
                            sql={msg.sql_equivalent}
                            copied={copied}
                            onCopy={() => onCopy(msg.sql_equivalent!)}
                        />
                    )}
                </div>

                <span className="px-1 text-[11px] text-zinc-400">
                    {isUser ? "You" : "InsightStream"}
                </span>
            </div>
        </div>
    );
}

/* ============================================================
   Avatar
   ============================================================ */

function Avatar({ role }: { role: "user" | "assistant" }) {
    if (role === "assistant") {
        return (
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#f1efff]">
                <Sparkles className="h-4 w-4 text-[#6d5ef5]" strokeWidth={1.75} />
            </div>
        );
    }
    return (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-zinc-200 bg-white">
            <User className="h-4 w-4 text-zinc-500" strokeWidth={1.75} />
        </div>
    );
}

/* ============================================================
   Inline chart (bars only — for chart_type === "bar")
   For other chart types, fall back to a labeled list.
   ============================================================ */

function InlineChart({ labels, values }: { labels: string[]; values: number[] }) {
    const maxVal = Math.max(...values, 1);
    return (
        <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-4">
            <div className="flex h-32 items-end gap-2">
                {values.map((val, i) => (
                    <div key={i} className="group/bar flex flex-1 flex-col items-center gap-1.5">
                        <div className="w-full text-center text-[10px] font-medium text-zinc-500 opacity-0 transition-opacity group-hover/bar:opacity-100 tabular-nums">
                            {val.toLocaleString()}
                        </div>
                        <div
                            className="w-full rounded-t-md bg-[#6d5ef5]/85 transition-all group-hover/bar:bg-[#6d5ef5]"
                            style={{ height: `${(val / maxVal) * 100}%`, minHeight: "4px" }}
                        />
                    </div>
                ))}
            </div>
            <div className="mt-2 flex gap-2">
                {labels.map((lbl, i) => (
                    <div
                        key={i}
                        className="flex-1 truncate text-center text-[11px] text-zinc-500"
                        title={lbl}
                    >
                        {lbl}
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ============================================================
   SQL block
   ============================================================ */

function SqlBlock({
    sql, copied, onCopy,
}: {
    sql: string;
    copied: boolean;
    onCopy: () => void;
}) {
    return (
        <div className="group/sql mt-3 overflow-hidden rounded-lg border border-zinc-200 bg-white">
            <div className="flex items-center justify-between border-b border-zinc-100 bg-zinc-50 px-3 py-1.5">
                <div className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-600">
                    <Code2 className="h-3 w-3" strokeWidth={1.75} />
                    SQL
                </div>
                <button
                    onClick={onCopy}
                    className="flex items-center gap-1 rounded px-2 py-0.5 text-[11px] font-medium text-zinc-500 hover:bg-white hover:text-zinc-900"
                >
                    {copied
                        ? <><Check className="h-3 w-3 text-emerald-600" strokeWidth={2} />Copied</>
                        : <><Copy className="h-3 w-3" strokeWidth={1.75} />Copy</>}
                </button>
            </div>
            <pre className="overflow-x-auto p-3 font-mono text-[12px] leading-relaxed text-zinc-800">
                <code>{sql}</code>
            </pre>
        </div>
    );
}

/* ============================================================
   Typing indicator
   ============================================================ */

function TypingIndicator() {
    return (
        <div className="flex gap-3">
            <Avatar role="assistant" />
            <div className="rounded-2xl rounded-tl-sm border border-zinc-200 bg-zinc-50 px-4 py-3">
                <div className="flex gap-1">
                    <Dot delay={0} />
                    <Dot delay={150} />
                    <Dot delay={300} />
                </div>
            </div>
        </div>
    );
}

function Dot({ delay }: { delay: number }) {
    return (
        <span
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400"
            style={{ animationDelay: `${delay}ms`, animationDuration: "1s" }}
        />
    );
}