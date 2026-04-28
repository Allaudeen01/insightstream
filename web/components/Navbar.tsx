"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Navbar = ({ sessionId }: { sessionId?: string }) => {
  const pathname = usePathname();

  const navLinks = [
    { label: 'Dashboard', href: '/dashboard' },
    { label: 'Insights',  href: '/insights'  },
    { label: 'EDA',       href: '/eda'       },
    { label: 'Chat',      href: '/chat'      },
    { label: 'Report',    href: '/report'    },
  ];

  return (
    <nav className="
      sticky top-0 z-50
      border-b border-white/10
      bg-background/80 backdrop-blur-md
    ">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="
            w-8 h-8 rounded-lg
            bg-purple-600 flex items-center 
            justify-center text-white font-bold
            text-sm group-hover:bg-purple-500 transition-colors
          ">
            IS
          </div>
          <span className="font-semibold text-foreground text-lg">
            InsightStream
          </span>
        </Link>

        {/* Nav links */}
        <div className="hidden md:flex items-center gap-1">
          {navLinks.map(({ label, href }) => (
            <Link key={href} href={href} className={`
              px-4 py-2 rounded-lg text-sm 
              font-medium transition-all
              ${pathname === href
                ? 'bg-purple-600/20 text-purple-300'
                : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
              }
            `}>
              {label}
            </Link>
          ))}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-3">
          <Link href="/report" className="
            text-sm px-4 py-2 rounded-lg
            border border-white/20
            text-foreground hover:bg-white/5
            transition-all font-medium hidden sm:block
          ">
            Export Report
          </Link>
          <Link href="/upload" className="
            text-sm px-4 py-2 rounded-lg
            bg-purple-600 hover:bg-purple-500
            text-white transition-all 
            font-medium shadow-lg 
            shadow-purple-500/25
          ">
            New Analysis
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
