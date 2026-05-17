"use client";
import { useEffect, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";

interface Message {
  id?: number;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
}

interface Props {
  sessionId: number;
}

const SUGGESTED = [
  "What's the most important action I should take this week?",
  "Which customer segment should I focus on first?",
  "What would happen to revenue if repeat rate improved to 25%?",
  "Why is September revenue always lower?",
];

export default function ChatPanel({ sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sessionId) loadHistory();
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function loadHistory() {
    const res = await apiFetch(`/sessions/${sessionId}/chat`);
    if (res.ok) {
      const data = await res.json();
      setMessages(data);
    }
  }

  async function sendMessage(text?: string) {
    const msg = (text ?? input).trim();
    if (!msg || streaming) return;
    setInput("");
    setStreaming(true);

    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setMessages(prev => [...prev, { role: "assistant", content: "", pending: true }]);

    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/sessions/${sessionId}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          credentials: "include",
          body: JSON.stringify({ message: msg }),
        }
      );

      if (!res.ok) throw new Error("Chat request failed");

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === "text") {
              setMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === "assistant") {
                  updated[updated.length - 1] = {
                    ...last,
                    content: last.content + event.content,
                    pending: false,
                  };
                }
                return updated;
              });
            }

            if (event.type === "error") {
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  role: "assistant",
                  content: `⚠️ ${event.content}`,
                  pending: false,
                };
                return updated;
              });
            }
          } catch {}
        }
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Something went wrong. Please try again.",
          pending: false,
        };
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="flex flex-col h-full bg-white rounded-xl border overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b bg-gray-50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm font-semibold text-gray-800">
            Ask InsightStream AI
          </span>
          <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full">
            Gemini 2.0 Flash
          </span>
        </div>
        <button
          onClick={loadHistory}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          Refresh
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-sm text-gray-400 text-center">
              Ask anything about this analysis
            </p>
            <div className="space-y-2">
              {SUGGESTED.map(q => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="w-full text-left text-xs text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded-lg px-3 py-2 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={[
                "max-w-[80%] rounded-xl px-4 py-2.5 text-sm leading-relaxed",
                msg.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-800",
                msg.pending ? "animate-pulse" : "",
              ].join(" ")}
            >
              {msg.content || (msg.pending ? "Thinking..." : "")}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t p-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Ask about your data..."
          disabled={streaming}
          maxLength={2000}
          className="flex-1 text-sm border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
        />
        <button
          onClick={() => sendMessage()}
          disabled={streaming || !input.trim()}
          className="bg-indigo-600 text-white rounded-lg px-4 py-2 text-sm font-medium disabled:opacity-40 hover:bg-indigo-700 transition-colors"
        >
          {streaming ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
