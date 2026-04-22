#!/usr/bin/env python3
"""Generate a local HTML preview using engine.export_report for quick visual QA."""
from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import polars as pl


def _stub_gemini() -> None:
    google_mod = types.ModuleType("google")
    generative_mod = types.ModuleType("google.generativeai")
    generative_mod.configure = lambda *a, **k: None

    class DummyModel:
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):
            class Response:
                text = "{}"

            return Response()

    generative_mod.GenerativeModel = DummyModel
    sys.modules["google"] = google_mod
    sys.modules["google.generativeai"] = generative_mod


def main() -> None:
    _stub_gemini()
    sys.path.append("engine")
    import main as app_main  # noqa: WPS433

    session_id = str(uuid.uuid4())
    df = pl.DataFrame(
        {
            "Date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"],
            "Region": ["North", "South", "North", "South", "North", "South"],
            "Revenue": [12000, 15000, 18000, 14000, 22000, 21000],
            "Cost": [7000, 8000, 9000, 8500, 10000, 11000],
            "Pipeline": [30, 28, 35, 33, 40, 38],
        }
    )

    app_main.save_session(session_id, "sample_revenue.csv", df)
    response = app_main.export_report(
        session_id=session_id,
        project_name="Acme Growth",
        report_title="InsightStream Report",
        logo_data=None,
    )

    output = Path("/tmp/insightstream_report_preview.html")
    output.write_text(response.body.decode("utf-8"), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
