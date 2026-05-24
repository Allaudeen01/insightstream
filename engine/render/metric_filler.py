"""
engine/render/metric_filler.py
────────────────────────────────
Post-LLM renderer: resolves {{metric:COLUMN.STATISTIC}} placeholders
from the MetricStore, guaranteeing all numbers in the report come from
a single authoritative source.
"""
from __future__ import annotations

import re

from render.metric_store import MetricKey, MetricStore

_PLACEHOLDER = re.compile(
    r"\{\{(?:fmt:(?P<fmt>\w+):)?metric:(?P<col>[\w]+)\.(?P<stat>\w+)"
    r"(?:\|scope=(?P<scope_col>[\w]+)=(?P<scope_val>[^}]+))?\}\}"
)


class UnknownMetricError(ValueError):
    pass


def fill_metrics(text: str, store: MetricStore) -> str:
    """Replace all {{metric:...}} placeholders with values from store."""
    def _sub(m: re.Match) -> str:
        scope = (
            f"{m['scope_col']}={m['scope_val']}"
            if m["scope_col"] else None
        )
        key = MetricKey(m["col"], m["stat"], scope)
        try:
            return store.format(key, fmt=m["fmt"] or "auto")
        except KeyError:
            raise UnknownMetricError(f"Missing metric: {key}")

    return _PLACEHOLDER.sub(_sub, text)


def assert_no_bare_numerics_for(text: str, columns: list[str]) -> None:
    """
    Guard: raise if a column name appears within 30 chars of a bare number.
    Prevents LLM-hallucinated numbers from slipping through.
    """
    for col in columns:
        pat = re.compile(rf"\b{re.escape(col)}\b[^{{}}]{{0,30}}?(\d+\.?\d*)")
        for m in pat.finditer(text):
            raise ValueError(
                f"Bare numeric '{m.group(1)}' near column '{col}' in: "
                f"...{m.group(0)}... "
                f"All numbers must use {{{{metric:...}}}} placeholders."
            )
