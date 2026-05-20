from models import AnalysisSession


SYSTEM_PROMPT = """You are InsightStream AI — an expert business data analyst embedded in the InsightStream analytics platform.

You are analysing a specific dataset uploaded by the user. Respond based on the data context provided. Be direct, specific, and always reference actual numbers.

Rules:
- Only answer questions about this dataset and analysis
- Always cite specific numbers from the context
- If asked something outside this dataset, say so clearly
- Do not fabricate numbers not in the context
- Keep responses concise — 2-4 sentences for simple questions, more for complex ones
- When suggesting actions, be specific and quantified
""".strip()


def build_system_prompt(session: AnalysisSession) -> str:
    kpis = session.kpis

    kpis_block = "\n".join(
        f"- {k}: {v}" for k, v in kpis.items()
    ) if kpis else "No KPIs available."

    insights_block = ""
    for i, insight in enumerate(session.insights, 1):
        insights_block += (
            f"\n### Insight {i}: {insight.title}\n"
            f"Impact: {insight.impact.upper()}\n"
            f"Body: {insight.body}\n"
        )

    recs_block = ""
    for i, rec in enumerate(session.recommendations, 1):
        recs_block += (
            f"{i}. {rec.action}\n"
            f"   Owner: {rec.owner} | Timeframe: {rec.timeframe}\n"
        )

    return f"""{SYSTEM_PROMPT}

## Dataset
- Filename: {session.original_filename}
- Rows: {session.row_count:,}
- Domain: {session.detected_domain or 'general'}

## Key Performance Indicators
{kpis_block}

## Discovered Insights ({len(session.insights)} total)
{insights_block.strip() or 'No insights available.'}

## Strategic Recommendations
{recs_block.strip() or 'No recommendations available.'}"""


def build_message_history(messages: list, max_messages: int = 20) -> list:
    """Convert DB messages to Groq/OpenAI API format."""
    recent = messages[-max_messages:]
    return [
        {
            "role": "assistant" if msg.role == "assistant" else "user",
            "content": msg.content,
        }
        for msg in recent
    ]
