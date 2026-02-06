
export type ColumnType = 'numeric' | 'categorical' | 'date' | 'string';

export interface ColumnStats {
  name: string;
  type: ColumnType;
  missing: number;
  unique: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
}

export interface Dataset {
  name: string;
  columns: ColumnStats[];
  data: any[];
  rowCount: number;
}

export interface AnalysisResult {
  executiveSummary: string;
  insights: string[];
  recommendations: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  chartData?: any[];
  chartType?: 'bar' | 'line' | 'scatter' | 'pie';
}
