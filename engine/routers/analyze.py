from fastapi import APIRouter, UploadFile, Depends, HTTPException, Form
from sqlalchemy.ext.asyncio import AsyncSession
import io
import os
import pandas as pd
from db_async import get_db
from auth import get_current_user
from models import User
from services.session_service import create_session, save_results, mark_failed
from insight_engine import run_insight_engine
from analyzer import analyze_dataset, detect_domain

router = APIRouter(tags=["analyze"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")

@router.post("/analyze")
async def analyze(
    file: UploadFile,
    currency: str = Form("auto"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 1. Create session record immediately
    session_record = await create_session(
        db=db,
        user_id=current_user.id,
        filename=file.filename,
        original_filename=file.filename,
        row_count=0,
        column_count=0,
        file_size_bytes=0,
    )

    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        # Persist raw upload so the old /insights and /generate-viz endpoints can load it
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        safe_name = (file.filename or "upload").replace("/", "_").replace("\\", "_")
        upload_path = os.path.join(UPLOAD_DIR, f"session_{session_record.id}_{safe_name}")
        with open(upload_path, "wb") as fh:
            fh.write(contents)

        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin-1")
        else:
            raise HTTPException(400, "Unsupported file type. Please upload CSV or Excel.")

        # Reset explicit flag so auto-detect works when currency=="auto"
        from insight_engine import _set_currency_symbol
        import insight_engine as _ie
        _ie._CURRENCY_EXPLICIT = False

        # Override currency if user specified
        if currency != "auto":
            _CURRENCY_MAP = {
                "INR": "₹", "USD": "$", "GBP": "£",
                "EUR": "€", "AED": "AED", "SGD": "S$",
                "JPY": "¥",
            }
            sym = _CURRENCY_MAP.get(currency, "₹")
            _set_currency_symbol(sym, explicit=True)
            print(f"[CURRENCY] User override: {currency} → {sym}")
        else:
            try:
                from gemini_column_semantics import detect_currency_gemini
                detected_sym = detect_currency_gemini(df, list(df.columns))
                _set_currency_symbol(detected_sym)
                print(f"[Gemini Currency] Auto-detected: {detected_sym}")
            except Exception as _ce:
                print(f"[Gemini Currency] Failed: {_ce} — using rule-based")

        session_record.row_count = len(df)
        session_record.column_count = len(df.columns)
        session_record.file_size_bytes = len(contents)
        session_record.status = "processing"
        
        # Store currency selection in session
        if currency and currency != "auto":
            session_record.currency = currency
            print(f"[SESSION] Stored currency: {currency}")
        
        await db.commit()

        # ── STEP 9: Domain-based routing ─────────────────────────────────
        # Detect domain from column patterns BEFORE calling the engine.
        # Only a small set of explicitly validated datasets use the rule engine.
        # Everything else (including PSL, generic sports, HR, etc.) → LLM.
        detected = detect_domain(df, filename=file.filename or "")
        print(f"[ROUTER] domain={detected!r} for file={file.filename!r}")

        # Refine domain for logging/display (does not change routing)
        from analyzer import _refine_domain
        _refined = _refine_domain(detected, df)
        if _refined != detected:
            print(f"[ROUTER] Refined label: {_refined!r} (routing: {detected!r})")

        # Domains where the rule engine produces validated, clean output.
        # PSL and generic sports are intentionally excluded — the rule engine
        # outputs "revenue" language for sports data which has no revenue column.
        _RULE_ENGINE_DOMAINS = {
            "entertainment",   # Netflix / Disney+ / Amazon Prime (validated)
        }

        if detected in _RULE_ENGINE_DOMAINS:
            print(f"[ROUTER] {detected} → rule engine (run_insight_engine)")
            results = run_insight_engine(df)
        else:
            # LLM analyzer for sports, HR, housing, Titanic, PSL, and everything else
            print(f"[ROUTER] {detected} → LLM analyzer (analyze_dataset)")
            llm_results = analyze_dataset(df)

            # ── Persist LLM results to DB so PDF export can use them ─────
            # Convert Plotly figures to JSON strings now so the export
            # endpoint can use them directly without re-rendering.
            # chart_metas carries per-chart summaries generated by _generate_chart_summary.
            import plotly.io as pio
            chart_jsons = []
            _chart_metas = llm_results.get("chart_metas", [])
            for _i, _fig in enumerate(llm_results.get("charts", [])):
                try:
                    # Pull summary from chart_metas if available (index-aligned with charts)
                    _meta    = _chart_metas[_i] if _i < len(_chart_metas) else {}
                    _summary = _meta.get("summary", "")
                    chart_jsons.append({
                        "id":          f"llm_chart_{_i}",
                        "title":       _fig.layout.title.text or f"Chart {_i + 1}",
                        "plotly_json": pio.to_json(_fig),
                        "insight":     _summary,   # natural-language caption for PDF
                    })
                    if _summary:
                        print(f"[ROUTER] Chart {_i} summary: {_summary[:80]!r}")
                except Exception as _ce:
                    print(f"[ROUTER] Chart {_i} serialization failed: {_ce}")

            session_record.llm_results = {
                "insights":        llm_results.get("insights", []),
                "recommendations": llm_results.get("recommendations", []),
                "domain":          llm_results.get("domain", "general").lower(),
                "title":           llm_results.get("title", "Data Analysis Report"),
                "charts":          chart_jsons,
            }
            db.add(session_record)

            # Debug: log first insight to confirm text is populated
            _stored_insights = llm_results.get("insights", [])
            _stored_recs     = llm_results.get("recommendations", [])
            if _stored_insights:
                _first = _stored_insights[0]
                print(f"[ROUTER] First insight: title={_first.get('title')!r}, "
                      f"text_len={len(_first.get('text', ''))}, "
                      f"text_preview={_first.get('text', '')[:80]!r}")
            print(f"[ROUTER] Stored LLM results: "
                  f"{len(_stored_insights)} insights, "
                  f"{len(chart_jsons)} charts, "
                  f"{len(_stored_recs)} recommendations (filtered)")

            # Wrap LLM results into the run_insight_engine return shape so
            # save_results() and the frontend receive a consistent structure.
            # LLM insights are stored in strategic_brief; recommendations stay.
            llm_insights = llm_results.get("insights", [])
            llm_recs     = llm_results.get("recommendations", [])
            llm_domain   = llm_results.get("domain", "General")
            llm_title    = llm_results.get("title", "Data Analysis Report")

            # Convert LLM insight dicts to the strategic_brief shape
            strategic_brief = [
                {
                    "title":                  ins.get("title", ""),
                    "description":            ins.get("text", ""),
                    "why_it_matters":         "",
                    "evidence":               "",
                    "decision_implication":   "",
                    "impact":                 ins.get("impact", "IMPORTANT"),
                    "recommendation":         "",
                    "is_unexpected":          False,
                    "confidence_label":       "medium",
                    "confidence_explanation": "LLM-generated insight",
                    "score":                  0.5,
                    "chart_type":             None,
                    "chart_data":             None,
                    "qualified_segments":     [],
                    "excluded_segments":      [],
                    "rule_type":              "llm_generated",
                    "methodology":            "LLM code generation",
                    "narrative_hook":         ins.get("text", "")[:120],
                }
                for ins in llm_insights
            ]

            # Convert LLM recommendation dicts to the engine rec shape
            recommendations = [
                {
                    "action":    rec.get("text", ""),
                    "timeframe": rec.get("timeframe", "Next 30 days"),
                    "owner":     rec.get("owner", "Strategy team"),
                    "impact":    rec.get("impact", "Important"),
                }
                for rec in llm_recs
            ]

            results = {
                "domain": {
                    "name":       llm_domain,
                    "confidence": "medium",
                    "reason":     "LLM-detected domain",
                    "id":         llm_domain.lower(),
                },
                "target":           None,
                "key_drivers":      [],
                "profile": {
                    "identifiers":  [],
                    "numericals":   df.select_dtypes(include="number").columns.tolist(),
                    "categoricals": df.select_dtypes(exclude="number").columns.tolist(),
                    "temporals":    [],
                    "binaries":     [],
                },
                "computed_metrics": {},
                "strategic_brief":  strategic_brief,
                "recommendations":  recommendations,
                "executive_summary": llm_title,
                "warnings":         [],
                "sports_meta":      {},
                "column_coverage": {
                    "total_columns":    len(df.columns),
                    "analyzed_columns": len(df.columns),
                    "coverage_pct":     100.0,
                    "untouched_columns": [],
                    "high_value_missed": [],
                    "warning":          None,
                },
                # Pass LLM charts through for the frontend
                "llm_charts": [
                    fig.to_json() if hasattr(fig, "to_json") else None
                    for fig in llm_results.get("charts", [])
                ],
            }
        # ── END routing ───────────────────────────────────────────────────

        # Mock PDF generation for now, wire it properly later
        report_path = f"/tmp/Report_{session_record.id}.pdf"

        domain_raw = results.get("domain", {})
        domain_name = domain_raw.get("name", "general") if isinstance(domain_raw, dict) else str(domain_raw)

        await save_results(
            db=db,
            session_id=session_record.id,
            kpis=results.get("computed_metrics", {}),
            insights=results.get("strategic_brief", []),
            recommendations=results.get("recommendations", []),
            detected_domain=domain_name,
            report_path=report_path,
        )

        return {
            "session_id": session_record.id,
            **results
        }

    except Exception as e:
        await db.rollback()
        await mark_failed(db, session_record.id, str(e))
        raise HTTPException(500, f"Analysis failed: {str(e)}")
