import React from "react";
import Link from "next/link";
import { Upload, CheckCircle2, ArrowRight } from "lucide-react";
import Sidebar from "@/components/Sidebar";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-50 flex">
      <Sidebar />

      <main className="flex-1 px-8 py-10">
        <div className="max-w-3xl mx-auto">
          {/* Hero */}
          <header className="mb-10">
            <p className="text-xs font-medium text-indigo-700 uppercase tracking-wide mb-3">
              Get started
            </p>
            <h1 className="text-3xl md:text-4xl font-semibold text-slate-900 tracking-tight mb-3">
              Upload a file to begin
            </h1>
            <p className="text-base text-slate-500 leading-relaxed max-w-xl">
              Drop in a CSV or Excel file and we&apos;ll summarize the columns,
              check the data for issues, and surface insights worth your time.
            </p>
          </header>

          {/* Upload card */}
          <Link href="/upload" className="block group">
            <div className="rounded-lg border-2 border-dashed border-slate-300 bg-white p-10 text-center transition-colors group-hover:border-indigo-400 group-hover:bg-indigo-50/40">
              <div className="w-12 h-12 rounded-md bg-slate-50 border border-slate-200 flex items-center justify-center mx-auto mb-5 group-hover:bg-white group-hover:border-indigo-200 transition-colors">
                <Upload className="w-5 h-5 text-slate-600 group-hover:text-indigo-600" strokeWidth={1.75} />
              </div>
              <h2 className="text-lg font-semibold text-slate-900 mb-1.5">
                Drop your file here
              </h2>
              <p className="text-sm text-slate-500 mb-6">
                Supports CSV and Excel (.xlsx) files
              </p>
              <span className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-indigo-600 group-hover:bg-indigo-700 text-white text-sm font-medium transition-colors">
                Browse files
                <ArrowRight className="w-4 h-4" />
              </span>
            </div>
          </Link>

          {/* Feature list */}
          <section className="mt-12">
            <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-4">
              What you&apos;ll get
            </h3>
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3">
              {[
                "A summary of every column",
                "Data quality checks",
                "Plain-language insights",
                "Downloadable reports",
                "Ask questions in chat",
                "No code required",
              ].map(feature => (
                <li
                  key={feature}
                  className="flex items-center gap-2.5 text-sm text-slate-700"
                >
                  <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0" />
                  {feature}
                </li>
              ))}
            </ul>
          </section>
        </div>
      </main>
    </div>
  );
}