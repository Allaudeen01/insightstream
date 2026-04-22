import React from 'react';
import { Upload, CheckCircle2, Sparkles } from 'lucide-react';
import Navbar from '@/components/Navbar';
import Link from 'next/link';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="
        flex flex-col items-center 
        justify-center p-6 pt-20
      ">
        {/* Hero */}
        <div className="text-center mb-10 max-w-2xl">
          <div className="
            inline-flex items-center gap-2
            px-3 py-1 rounded-full
            bg-purple-500/10 border 
            border-purple-500/20
            text-purple-300 text-sm 
            font-medium mb-6
          ">
            <span className="w-2 h-2 rounded-full 
                             bg-purple-400 
                             animate-pulse" />
            AI-Powered Data Analysis
          </div>
          <h1 className="
            text-4xl md:text-6xl font-bold text-foreground 
            mb-4 tracking-tight
          ">
            Turn your data into
            <span className="text-purple-400 px-2">insights</span>
          </h1>
          <p className="text-muted-foreground text-lg mb-8">
            Upload any CSV or Excel file. Get instant 
            AI-powered insights, charts, and recommendations.
          </p>
        </div>

        {/* Upload zone */}
        <Link href="/upload" className="w-full max-w-2xl">
          <div className="
            w-full
            border-2 border-dashed border-white/20
            rounded-2xl p-12 text-center
            hover:border-purple-500/50
            hover:bg-purple-500/5
            transition-all duration-300
            cursor-pointer group
          ">
            <div className="
              w-16 h-16 rounded-2xl
              bg-purple-500/20 
              flex items-center justify-center
              mx-auto mb-6
              group-hover:bg-purple-500/30
              transition-all
            ">
              <Upload className="w-8 h-8 text-purple-400" strokeWidth={1.5} />
            </div>
            <h3 className="text-xl font-semibold 
                           text-foreground mb-2">
              Drop your file here
            </h3>
            <p className="text-muted-foreground mb-6">
              Supports CSV and Excel (.xlsx) files
            </p>
            <div className="
              inline-block px-6 py-3 rounded-xl
              bg-purple-600 hover:bg-purple-500
              text-white font-medium
              shadow-lg shadow-purple-500/25
              transition-all
            ">
              Browse Files
            </div>
          </div>
        </Link>

        {/* Feature pills */}
        <div className="flex flex-wrap gap-3 
                        mt-12 justify-center max-w-3xl">
          {[
            'Auto EDA',
            'Contradiction Detection', 
            'Smart Insights',
            'Export Reports',
            'AI Chat',
            'No Code Required'
          ].map(feature => (
            <span key={feature} className="
              px-4 py-2 rounded-full
              bg-white/5 border border-white/10
              text-sm text-muted-foreground
              flex items-center gap-2
            ">
              <CheckCircle2 className="w-3.5 h-3.5 text-purple-400" />
              {feature}
            </span>
          ))}
        </div>
      </main>
    </div>
  );
}

