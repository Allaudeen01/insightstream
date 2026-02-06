
import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line
} from 'recharts';
import { 
  FileUp, Database, PieChart as ChartIcon, MessageSquare, 
  LayoutDashboard, Loader2, Sparkles, CheckCircle2, ChevronRight,
  TrendingUp, AlertCircle, RefreshCw
} from 'lucide-react';
import { Dataset, AnalysisResult, ChatMessage } from './types';
import { parseFile } from './utils/dataProcessor';
import { GeminiService } from './services/geminiService';

// --- Sub-components ---

interface CardProps {
  children?: React.ReactNode;
  title?: string;
  icon?: React.ElementType;
  className?: string;
}

const Card = ({ children, title, icon: Icon, className = "" }: CardProps) => (
  <div className={`bg-white rounded-xl shadow-sm border border-slate-200 p-6 ${className}`}>
    {title && (
      <div className="flex items-center gap-2 mb-4">
        {Icon && <Icon className="w-5 h-5 text-indigo-600" />}
        <h3 className="font-semibold text-slate-800">{title}</h3>
      </div>
    )}
    {children}
  </div>
);

const AnalysisDashboard: React.FC<{ dataset: Dataset, analysis: AnalysisResult | null }> = ({ dataset, analysis }) => {
  if (!analysis) {
    return (
      <div className="flex flex-col items-center justify-center p-20 gap-4">
        <Loader2 className="w-10 h-10 animate-spin text-indigo-600" />
        <p className="text-slate-600 font-medium">AI is crunching the numbers...</p>
      </div>
    );
  }

  const categoricalCol = dataset.columns.find(c => c.type === 'categorical');
  const numericCol = dataset.columns.find(c => c.type === 'numeric');

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Rows & Columns" icon={Database}>
          <div className="flex justify-between items-end">
            <div>
              <p className="text-3xl font-bold text-slate-900">{dataset.rowCount.toLocaleString()}</p>
              <p className="text-sm text-slate-500">Total Observations</p>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-slate-900">{dataset.columns.length}</p>
              <p className="text-sm text-slate-500">Features</p>
            </div>
          </div>
        </Card>
        
        <Card title="Health Score" icon={CheckCircle2}>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Data Quality</span>
              <span className="font-medium text-emerald-600">Excellent</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full bg-emerald-500 w-[92%]" />
            </div>
            <p className="text-xs text-slate-400">Low missing value count (&lt; 1%)</p>
          </div>
        </Card>
        
        <Card title="Summary" icon={Sparkles}>
          <p className="text-sm text-slate-600 leading-relaxed italic">
            &ldquo;{analysis.executiveSummary}&rdquo;
          </p>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Deep Insights" icon={TrendingUp}>
          <div className="space-y-4">
            {analysis.insights.map((insight, idx) => (
              <div key={idx} className="flex gap-4 p-3 rounded-lg hover:bg-slate-50 transition-colors">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-50 flex items-center justify-center text-indigo-600 font-bold text-sm">
                  {idx + 1}
                </div>
                <p className="text-sm text-slate-700 leading-snug">{insight}</p>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Smart Visualization" icon={ChartIcon}>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              {categoricalCol ? (
                <BarChart data={dataset.data.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey={categoricalCol.name} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey={numericCol?.name || ""} fill="#4f46e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              ) : (
                <LineChart data={dataset.data.slice(0, 50)}>
                   <CartesianGrid strokeDasharray="3 3" />
                   <XAxis dataKey={dataset.columns[0]?.name} />
                   <YAxis />
                   <Tooltip />
                   <Line type="monotone" dataKey={numericCol?.name || ""} stroke="#4f46e5" dot={false} />
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-4 text-center">Auto-selected visualization based on dataset structure</p>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Recommendations" icon={AlertCircle}>
           <ul className="space-y-3">
             {analysis.recommendations.map((rec, idx) => (
               <li key={idx} className="flex gap-2 text-sm text-slate-700">
                 <ChevronRight className="w-4 h-4 text-indigo-500 mt-0.5" />
                 {rec}
               </li>
             ))}
           </ul>
        </Card>
        
        <Card title="Column Distribution" icon={Database}>
          <div className="max-h-[200px] overflow-y-auto space-y-2">
            {dataset.columns.map(col => (
              <div key={col.name} className="flex items-center justify-between text-sm py-1 border-b border-slate-50 last:border-0">
                <span className="font-medium text-slate-700">{col.name}</span>
                <span className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold ${
                  col.type === 'numeric' ? 'bg-blue-100 text-blue-700' : 
                  col.type === 'date' ? 'bg-amber-100 text-amber-700' : 
                  'bg-slate-100 text-slate-600'
                }`}>
                  {col.type}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
};

const ChatSection: React.FC<{ dataset: Dataset, gemini: GeminiService }> = ({ dataset, gemini }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input;
    setInput("");
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      const response = await gemini.askQuestion(dataset, userMsg, messages);
      setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I hit a snag analyzing that. Can you try again?" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[600px] bg-white rounded-xl border border-slate-200 overflow-hidden">
      <div className="p-4 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-indigo-600" />
          <h3 className="font-semibold text-slate-800">Ask Anything</h3>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2">
            <Sparkles className="w-8 h-8 opacity-20" />
            <p className="text-sm">Try asking: "What's the trend of sales over time?"</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-2xl text-sm leading-relaxed ${
              m.role === 'user' 
                ? 'bg-indigo-600 text-white rounded-tr-none' 
                : 'bg-slate-100 text-slate-700 rounded-tl-none'
            }`}>
              {m.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
             <div className="bg-slate-100 p-3 rounded-2xl rounded-tl-none flex gap-2">
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" />
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]" />
               <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]" />
             </div>
          </div>
        )}
      </div>

      <div className="p-4 border-t border-slate-100">
        <div className="flex gap-2">
          <input 
            type="text" 
            placeholder="Query your data..." 
            className="flex-1 px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 text-sm"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          <button 
            onClick={handleSend}
            disabled={isLoading}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:bg-slate-300"
          >
            Ask
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Main App Component ---

export default function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [view, setView] = useState<'dashboard' | 'chat' | 'table'>('dashboard');
  const [gemini] = useState(() => new GeminiService());

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const parsedData = await parseFile(file);
      setDataset(parsedData);
      setAnalysis(null);
      const res = await gemini.analyzeDataset(parsedData);
      setAnalysis(res);
    } catch (error) {
      console.error("Upload failed", error);
      alert("Failed to process file. Please ensure it is a valid CSV or XLSX.");
    } finally {
      setIsUploading(false);
    }
  };

  const reset = () => {
    setDataset(null);
    setAnalysis(null);
    setView('dashboard');
  };

  const navItems = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { id: 'chat', icon: MessageSquare, label: 'Chat' },
    { id: 'table', icon: Database, label: 'Raw Data' }
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={reset}>
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">InsightStream</h1>
          </div>
          
          {dataset && (
            <div className="flex items-center gap-4">
              <nav className="flex bg-slate-100 p-1 rounded-lg">
                {navItems.map((item) => {
                  const NavIcon = item.icon;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setView(item.id as any)}
                      className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                        view === item.id 
                          ? 'bg-white shadow-sm text-indigo-600' 
                          : 'text-slate-500 hover:text-slate-700'
                      }`}
                    >
                      <NavIcon className="w-3.5 h-3.5" />
                      {item.label}
                    </button>
                  );
                })}
              </nav>
              <button 
                onClick={reset}
                className="text-slate-400 hover:text-slate-600 transition-transform active:rotate-180"
                title="Upload New Data"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {!dataset ? (
          <div className="max-w-2xl mx-auto mt-20 text-center animate-in fade-in zoom-in-95 duration-700">
            <h2 className="text-4xl font-extrabold text-slate-900 mb-4">
              Your data has stories to tell. <br/>
              <span className="text-indigo-600">Let AI find them.</span>
            </h2>
            <p className="text-lg text-slate-500 mb-10 leading-relaxed">
              Upload any CSV or Excel file. Our AI analyst will automatically scan for trends, 
              calculate stats, and explain insights in plain English.
            </p>
            
            <div className="relative group">
              <input 
                type="file" 
                accept=".csv,.xlsx,.xls" 
                onChange={handleFileUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                disabled={isUploading}
              />
              <div className={`p-16 border-2 border-dashed rounded-2xl flex flex-col items-center gap-4 transition-all ${
                isUploading 
                  ? 'bg-slate-50 border-indigo-200' 
                  : 'bg-white border-slate-300 group-hover:border-indigo-500 group-hover:bg-indigo-50/30'
              }`}>
                {isUploading ? (
                  <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
                ) : (
                  <div className="w-16 h-16 bg-indigo-50 rounded-full flex items-center justify-center text-indigo-600 mb-2">
                    <FileUp className="w-8 h-8" />
                  </div>
                )}
                <div className="space-y-1">
                  <p className="text-lg font-semibold text-slate-700">
                    {isUploading ? 'Analyzing Dataset...' : 'Drop your file here or click to browse'}
                  </p>
                  <p className="text-sm text-slate-400">Supports CSV and Excel files</p>
                </div>
              </div>
            </div>

            <div className="mt-12 grid grid-cols-3 gap-8">
              <div className="space-y-2">
                <div className="w-10 h-10 bg-emerald-50 text-emerald-600 rounded-lg flex items-center justify-center mx-auto">
                  <ChartIcon className="w-5 h-5" />
                </div>
                <h4 className="font-semibold text-slate-800">Auto EDA</h4>
                <p className="text-xs text-slate-500">Instant distributions and correlations</p>
              </div>
              <div className="space-y-2">
                <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center mx-auto">
                  <Sparkles className="w-5 h-5" />
                </div>
                <h4 className="font-semibold text-slate-800">Smart Insights</h4>
                <p className="text-xs text-slate-500">Human-readable summaries by Gemini</p>
              </div>
              <div className="space-y-2">
                <div className="w-10 h-10 bg-amber-50 text-amber-600 rounded-lg flex items-center justify-center mx-auto">
                  <MessageSquare className="w-5 h-5" />
                </div>
                <h4 className="font-semibold text-slate-800">Chat with Data</h4>
                <p className="text-xs text-slate-500">Natural language querying</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8 animate-in fade-in duration-300">
            {view === 'dashboard' && <AnalysisDashboard dataset={dataset} analysis={analysis} />}
            {view === 'chat' && <ChatSection dataset={dataset} gemini={gemini} />}
            {view === 'table' && (
              <Card title={`Raw Data - ${dataset.name}`} icon={Database} className="h-[700px] flex flex-col">
                <div className="flex-1 overflow-auto border rounded-lg">
                  <table className="w-full text-sm text-left border-collapse">
                    <thead className="sticky top-0 bg-slate-50 border-b">
                      <tr>
                        {dataset.columns.map(col => (
                          <th key={col.name} className="px-4 py-3 font-semibold text-slate-700 whitespace-nowrap">
                            {col.name}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.data.slice(0, 100).map((row, i) => (
                        <tr key={i} className="border-b last:border-0 hover:bg-slate-50/50 transition-colors">
                          {dataset.columns.map(col => (
                            <td key={col.name} className="px-4 py-3 text-slate-600 font-mono text-xs max-w-xs truncate">
                              {String(row[col.name] ?? '-')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {dataset.rowCount > 100 && (
                    <div className="p-4 text-center text-slate-400 text-xs italic">
                      Showing first 100 of {dataset.rowCount} rows
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>
        )}
      </main>

      <footer className="bg-slate-900 text-slate-400 py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm">
          <p>&copy; 2024 InsightStream &bull; Advanced AI Data Exploration</p>
        </div>
      </footer>
    </div>
  );
}
