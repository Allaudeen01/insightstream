
import { GoogleGenAI, Type } from "@google/genai";
import { Dataset, AnalysisResult } from "../types";

const MODEL_NAME = 'gemini-3-flash-preview';

export class GeminiService {
  // Always use process.env.API_KEY directly and create a fresh instance per request for reliability
  private createClient() {
    return new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async analyzeDataset(dataset: Dataset): Promise<AnalysisResult> {
    const ai = this.createClient();
    const context = this.createDatasetContext(dataset);
    
    const prompt = `
      You are a world-class Senior Data Analyst. Analyze this dataset and provide a high-level executive summary, key insights, and actionable recommendations.
      
      Dataset Context:
      ${context}
      
      Format the response in JSON with:
      - executiveSummary: A paragraph summarizing the dataset's purpose and health.
      - insights: An array of 4-6 specific insights found in the data (e.g., "Product X accounts for 40% of revenue but has the highest return rate").
      - recommendations: 2-3 business actions based on the data.
      
      Use simple English. Avoid technical jargon like "correlation coefficient" unless you explain it in plain terms.
    `;

    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            executiveSummary: { type: Type.STRING },
            insights: { type: Type.ARRAY, items: { type: Type.STRING } },
            recommendations: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: ["executiveSummary", "insights", "recommendations"]
        }
      }
    });

    try {
      // Correct extraction using the .text property as per guidelines
      const jsonStr = response.text || "";
      return JSON.parse(jsonStr);
    } catch (e) {
      console.error("Failed to parse Gemini response", e);
      return {
        executiveSummary: "Data processed but AI analysis encountered an error.",
        insights: ["Check your data for consistency.", "Look for trends manually."],
        recommendations: ["Retry the analysis."]
      };
    }
  }

  async askQuestion(dataset: Dataset, question: string, history: any[]): Promise<string> {
    const ai = this.createClient();
    const context = this.createDatasetContext(dataset);
    const chat = ai.chats.create({
      model: MODEL_NAME,
      config: {
        systemInstruction: `You are an AI Data Assistant named InsightStream. You have access to a dataset with the following schema and summary: ${context}. Answer user questions accurately based ONLY on the data provided. Be concise and helpful.`
      }
    });

    const response = await chat.sendMessage({ message: question });
    // Use the .text property directly
    return response.text || "I'm sorry, I couldn't generate a response.";
  }

  private createDatasetContext(dataset: Dataset): string {
    const sample = dataset.data.slice(0, 5);
    const columns = dataset.columns.map(c => 
      `${c.name} (${c.type}): ${c.unique} unique values, ${c.missing} missing. ${c.mean ? `Avg: ${c.mean.toFixed(2)}` : ''}`
    ).join('\n');

    return `
      Dataset Name: ${dataset.name}
      Total Rows: ${dataset.rowCount}
      Columns Info:
      ${columns}
      
      Data Sample (First 5 rows):
      ${JSON.stringify(sample, null, 2)}
    `;
  }
}
