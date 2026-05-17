"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
    Plus,
    LayoutDashboard,
    Sparkles,
    MessageSquare,
    FileText,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

type NavItem = {
    href: string;
    label: string;
    icon: LucideIcon;
    primary?: boolean;
    matchPrefix?: boolean;
};

const NAV: NavItem[] = [
    { href: "/upload", label: "New analysis", icon: Plus, primary: true },
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, matchPrefix: true },
    { href: "/insights", label: "Insights", icon: Sparkles, matchPrefix: true },
    { href: "/chat", label: "Chat", icon: MessageSquare, matchPrefix: true },
];

interface RecentSession {
    id: number;
    original_filename: string;
}

interface UserInfo {
    full_name: string;
    email: string;
}

export default function Sidebar() {
    const pathname = usePathname() ?? "";
    const router = useRouter();
    const [user, setUser] = useState<UserInfo | null>(null);
    const [recents, setRecents] = useState<RecentSession[]>([]);

    useEffect(() => {
        const token = localStorage.getItem("access_token");
        if (!token) return;

        const headers = { Authorization: `Bearer ${token}` };

        fetch("/api/auth/me", { headers, credentials: "include" })
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data) setUser(data); })
            .catch(() => {});

        fetch("/api/sessions?page=1&page_size=5", { headers, credentials: "include" })
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data?.items) setRecents(data.items); })
            .catch(() => {});
    }, []);

    const isActive = (item: NavItem) =>
        item.matchPrefix
            ? pathname === item.href || pathname.startsWith(item.href + "/")
            : pathname === item.href;

    const initials = user?.full_name
        ? user.full_name.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase()
        : "?";

    const handleLogout = async () => {
        const token = localStorage.getItem("access_token");
        await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/auth/logout`, {
            method: "POST",
            headers: { Authorization: `Bearer ${token}` },
            credentials: "include",
        }).catch(() => {});
        localStorage.removeItem("access_token");
        window.location.href = "/login";
    };

    return (
        <aside className="flex w-60 shrink-0 flex-col border-r border-zinc-200 bg-zinc-50">
            {/* Brand */}
            <Link href="/" className="flex items-center gap-2.5 px-4 pb-3 pt-4">
                <div className="flex h-[26px] w-[26px] items-center justify-center rounded-[7px] bg-[#6d5ef5] text-sm font-bold text-white">
                    I
                </div>
                <span className="text-[15px] font-semibold tracking-[-0.01em] text-zinc-900">
                    InsightStream
                </span>
            </Link>

            {/* Primary nav */}
            <nav className="flex flex-col gap-px px-2">
                {NAV.map((item) => {
                    const active = isActive(item);
                    const Icon = item.icon;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={[
                                "flex items-center gap-2.5 rounded-md px-3 py-2 text-[13.5px] transition-colors",
                                active
                                    ? "bg-[#f1efff] font-medium text-[#6d5ef5]"
                                    : item.primary
                                        ? "text-zinc-900 hover:bg-zinc-100"
                                        : "text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900",
                            ].join(" ")}
                        >
                            <Icon className="h-4 w-4" strokeWidth={1.75} />
                            {item.label}
                        </Link>
                    );
                })}
            </nav>

            {/* Recents */}
            {recents.length > 0 && (
                <>
                    <div className="px-4 pb-2 pt-5 text-[11px] font-medium tracking-wide text-zinc-400">
                        Recent
                    </div>
                    <nav className="flex flex-1 flex-col gap-px overflow-auto px-2">
                        {recents.map((s) => (
                            <button
                                key={s.id}
                                onClick={() => router.push(`/insights?session=${s.id}`)}
                                className="flex items-center gap-2.5 truncate rounded-md px-3 py-1.5 text-left text-[13px] text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
                            >
                                <FileText className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                                <span className="truncate">{s.original_filename}</span>
                            </button>
                        ))}
                    </nav>
                </>
            )}

            {/* Spacer when no recents */}
            {recents.length === 0 && <div className="flex-1" />}

            {/* User footer */}
            <div className="flex items-center justify-between border-t border-zinc-200 p-3">
                <div className="flex min-w-0 items-center gap-2.5">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-200 text-[11px] font-semibold text-zinc-700">
                        {initials}
                    </div>
                    <span className="truncate text-[13px] text-zinc-600">{user?.full_name ?? "..."}</span>
                </div>
                <button
                    onClick={handleLogout}
                    className="ml-2 shrink-0 text-[12px] text-zinc-400 hover:text-red-500 transition-colors"
                    title="Sign out"
                >
                    Sign out
                </button>
            </div>
        </aside>
    );
}
