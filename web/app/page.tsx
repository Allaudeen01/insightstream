"use client";
import { useRouter } from "next/navigation";

export default function LandingPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Navbar */}
      <nav className="flex items-center justify-between px-8 py-4 border-b">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center">
            <span className="text-white text-xs font-bold">I</span>
          </div>
          <span className="font-semibold text-gray-900 text-lg">InsightStream</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/login")}
            className="text-sm text-gray-600 hover:text-gray-900 px-4 py-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            Log In
          </button>
          <button
            onClick={() => router.push("/register")}
            className="text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
          >
            Sign Up
          </button>
        </div>
      </nav>

      {/* Hero */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 text-center max-w-3xl mx-auto w-full">
        <div className="mb-6">
          <span className="inline-flex items-center gap-2 bg-indigo-50 text-indigo-700 text-xs font-medium px-3 py-1.5 rounded-full border border-indigo-100">
            ✨ AI-powered analytics — no code needed
          </span>
        </div>

        <h1 className="text-5xl font-bold text-gray-900 mb-4 leading-tight">
          What can I do<br />for you today?
        </h1>

        <p className="text-gray-500 text-lg mb-10 max-w-xl">
          Upload any CSV or Excel file. Get instant insights, charts, and a branded PDF report — powered by AI.
        </p>

        {/* Fake input CTA — clicking opens signup */}
        <div
          onClick={() => router.push("/register")}
          className="w-full max-w-xl border border-gray-200 rounded-xl px-5 py-4 text-left text-gray-400 cursor-pointer hover:border-indigo-300 hover:shadow-sm transition-all bg-white"
        >
          Upload a CSV or Excel file to get started...
        </div>

        {/* Feature chips */}
        <div className="flex flex-wrap items-center justify-center gap-2 mt-6">
          {["Revenue Analysis", "Customer Segmentation", "Cohort Retention", "CLV Forecast", "PDF Export"].map(f => (
            <span key={f} className="text-xs text-gray-500 bg-gray-100 px-3 py-1.5 rounded-full">
              {f}
            </span>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center text-xs text-gray-400 py-6">
        © 2026 InsightStream · AI Data Analyst
      </footer>
    </div>
  );
}
