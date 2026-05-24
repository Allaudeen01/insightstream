"""
engine/analysis/limitations.py
────────────────────────────────
Detects analytical questions the dataset cannot answer by checking for
missing canonical columns.

Organised into three domain-aware buckets:
  - "finance"  — only shown for FINANCE/CREDIT/ECOMMERCE domains
  - "temporal" — shown for any domain (time-series gaps are universal)
  - "entity"   — shown for any domain where rows are at item level

This prevents finance-specific limitations ("cannot compute credit utilization")
from appearing in entertainment or people datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# ── Domain-keyed canonical column definitions ─────────────────────────────────
CANONICAL_COLUMNS: dict[str, dict] = {
    # Finance bucket: only shown when domain is FINANCE/CREDIT/ECOMMERCE
    "finance": {
        "account_status": {
            "aliases": ["status", "acct_status", "account_state", "is_active",
                        "accountstatus", "acctstate"],
            "missing_impact": (
                "Cannot distinguish active from closed/dormant accounts. "
                "Aggregate metrics may include inactive accounts."
            ),
        },
        "utilization_rate": {
            "aliases": ["utilization", "util_rate", "credit_utilization",
                        "utilizationrate", "creditutil"],
            "missing_impact": (
                "Cannot compute credit utilization. "
                "credit_limit alone is not a risk signal without balance data."
            ),
        },
        "transaction_amount": {
            "aliases": ["amount", "txn_amount", "spend", "purchase_amount",
                        "transactionamount", "txnamount"],
            "missing_impact": (
                "Cannot analyze spending behavior or detect fraud patterns."
            ),
        },
    },
    # Temporal bucket: shown for any domain
    "temporal": {
        "transaction_date": {
            "aliases": ["date", "txn_date", "transaction_dt", "created_at",
                        "transactiondate", "txndate", "createdat"],
            "missing_impact": (
                "Cannot perform time-series analysis or detect seasonal patterns."
            ),
        },
    },
    # Entity bucket: shown for any domain
    "entity": {
        "customer_id": {
            "aliases": ["client_id", "cust_id", "customer_key", "client_num",
                        "customerid", "clientid", "custid"],
            "missing_impact": (
                "Cannot link items to customers. "
                "Per-customer aggregations are not possible without a join key."
            ),
        },
    },
}

# Domains where the finance bucket is relevant
_FINANCE_DOMAINS = frozenset({
    "FINANCE", "CREDIT", "FINANCE_CREDIT", "ECOMMERCE_TRANSACTIONS",
    "ECOMMERCE", "BANKING", "INSURANCE",
})


@dataclass
class Limitation:
    missing_concept: str   # e.g., "account_status"
    missing_impact: str    # human-readable consequence of absence


def _is_present(concept: str, aliases: list[str], cols_normalized: set[str]) -> bool:
    """
    Return True if the concept or any of its aliases appears in the DataFrame
    columns (case-insensitive, underscore-insensitive).
    """
    concept_norm = concept.lower().replace("_", "")
    if concept_norm in cols_normalized:
        return True
    for alias in aliases:
        if alias.lower().replace("_", "") in cols_normalized:
            return True
    return False


def detect_limitations(
    df: pd.DataFrame,
    semantics: dict,
    domain: Optional[str] = None,
) -> list[Limitation]:
    """
    Return a list of Limitation objects for canonical concepts absent from df.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect (not mutated).
    semantics : dict
        {col_name: ColumnSemantics} — not currently used for matching but
        available for future extension.
    domain : str | None
        The classified domain (e.g., "FINANCE", "CREDIT", "ENTERTAINMENT").
        When None or not a finance domain, the "finance" bucket is skipped.

    Returns
    -------
    list[Limitation]
        Empty list (not None) when all applicable concepts are present.
    """
    cols_normalized = {c.lower().replace("_", "") for c in df.columns}
    limitations: list[Limitation] = []
    domain_upper = (domain or "").upper().replace(" ", "_")

    # Finance bucket — only for finance/credit/ecommerce domains
    if domain_upper in _FINANCE_DOMAINS:
        for concept, meta in CANONICAL_COLUMNS["finance"].items():
            if not _is_present(concept, meta["aliases"], cols_normalized):
                limitations.append(Limitation(
                    missing_concept=concept,
                    missing_impact=meta["missing_impact"],
                ))

    # Temporal bucket — always check
    for concept, meta in CANONICAL_COLUMNS["temporal"].items():
        if not _is_present(concept, meta["aliases"], cols_normalized):
            limitations.append(Limitation(
                missing_concept=concept,
                missing_impact=meta["missing_impact"],
            ))

    # Entity bucket — always check
    for concept, meta in CANONICAL_COLUMNS["entity"].items():
        if not _is_present(concept, meta["aliases"], cols_normalized):
            limitations.append(Limitation(
                missing_concept=concept,
                missing_impact=meta["missing_impact"],
            ))

    return limitations
