"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
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
    /** True for entry-point actions like "New analysis" — rendered slightly stronger. */
    primary?: boolean;
    /** Match nested routes (e.g. /dashboard/123). Defaults to exact match. */
    matchPrefix?: boolean;
};

const NAV: NavItem[] = [
    { href: "/upload", label: "New analysis", icon: Plus, primary: true },
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, matchPrefix: true },
    { href: "/insights", label: "Insights", icon: Sparkles, matchPrefix: true },
    { href: "/chat", label: "Chat", icon: MessageSquare, matchPrefix: true },
];

interface SidebarProps {
    /** Recent analyses to render under the primary nav. Pass [] to hide the section. */
    recents?: string[];
    /** Current user — shown in the footer. Optional. */
    user?: { name: string; initials?: string };
}

export default function Sidebar({
    recents = ["Q3 sales analysis", "Customer churn 2025", "Marketing spend ROI"],
    user = { name: "Alex Singh", initials: "AS" },
}: SidebarProps) {
    const pathname = usePathname() ?? "";

    const isActive = (item: NavItem) =>
        item.matchPrefix
            ? pathname === item.href || pathname.startsWith(item.href + "/")
            : pathname === item.href;

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
                        {recents.map((r) => (
                            <button
                                key={r}
                                className="flex items-center gap-2.5 truncate rounded-md px-3 py-1.5 text-left text-[13px] text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900"
                            >
                                <FileText className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                                <span className="truncate">{r}</span>
                            </button>
                        ))}
                    </nav>
                </>
            )}

            {/* Spacer when no recents */}
            {recents.length === 0 && <div className="flex-1" />}

            {/* User footer */}
            <div className="border-t border-zinc-200 p-3">
                <button className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] text-zinc-600 hover:bg-zinc-100">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-zinc-200 text-[11px] font-semibold text-zinc-700">
                        {user.initials ?? user.name.slice(0, 2).toUpperCase()}
                    </div>
                    {user.name}
                </button>
            </div>
        </aside>
    );
}