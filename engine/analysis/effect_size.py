"""
engine/analysis/effect_size.py
────────────────────────────────
Computes η² (eta-squared) effect sizes for every (categorical_meaningful ×
monetary/numeric_meaningful) column pair via one-way ANOVA.

η² = SS_between / SS_total — the proportion of variance in the numeric column
explained by the categorical grouping. Range [0, 1].

Used to rank findings by actual explanatory power rather than spread ratio,
and to inject ground-truth effect sizes into the LLM prompt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class EffectSize:
    group_col: str       # categorical column name
    target_col: str      # numeric/monetary column name
    eta_squared: float   # 0..1, proportion of variance explained by group_col
    f_statistic: float   # one-way ANOVA F statistic
    p_value: float       # ANOVA p-value
    is_significant: bool # p_value < 0.05


def compute_effect_sizes(
    df: pd.DataFrame,
    semantics: dict,
) -> list[EffectSize]:
    """
    Compute one-way ANOVA η² for every (categorical_meaningful ×
    monetary/numeric_meaningful) column pair.

    Returns a list of EffectSize objects sorted by eta_squared descending.
    Returns [] (not None) when no valid pairs exist.
    Does NOT mutate df.

    Skips a pair and logs a warning when:
    - scipy.stats.f_oneway raises
    - f_statistic or p_value is NaN
    - fewer than 2 groups have ≥ 2 non-null observations
    """
    try:
        from scipy.stats import f_oneway
    except ImportError:
        print("[effect_size] scipy not available — skipping effect size computation")
        return []

    cat_cols = [c for c, s in semantics.items() if s.tag == "categorical_meaningful"]
    num_cols = [
        c for c, s in semantics.items()
        if s.tag in ("monetary", "numeric_meaningful")
    ]

    results: list[EffectSize] = []

    for group_col in cat_cols:
        if group_col not in df.columns:
            continue
        for target_col in num_cols:
            if target_col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                continue

            # Build per-group arrays; require ≥ 2 observations per group
            groups = []
            for val in df[group_col].dropna().unique():
                grp = df.loc[df[group_col] == val, target_col].dropna()
                if len(grp) >= 2:
                    groups.append(grp.values)

            if len(groups) < 2:
                continue

            try:
                f_stat, p_val = f_oneway(*groups)
            except Exception as e:
                print(f"[effect_size] f_oneway failed for {group_col}×{target_col}: {e}")
                continue

            if math.isnan(f_stat) or math.isnan(p_val):
                print(f"[effect_size] NaN result for {group_col}×{target_col} — skipping")
                continue

            # η² = SS_between / SS_total
            all_vals = df[target_col].dropna()
            grand_mean = float(all_vals.mean())
            ss_total = float(((all_vals - grand_mean) ** 2).sum())

            if ss_total == 0:
                eta_sq = 0.0
            else:
                ss_between = sum(
                    len(g) * (float(g.mean()) - grand_mean) ** 2
                    for g in groups
                )
                eta_sq = ss_between / ss_total

            # Clamp to [0, 1] to guard against floating-point edge cases
            eta_sq = max(0.0, min(1.0, eta_sq))

            results.append(EffectSize(
                group_col=group_col,
                target_col=target_col,
                eta_squared=round(eta_sq, 4),
                f_statistic=round(float(f_stat), 4),
                p_value=round(float(p_val), 6),
                is_significant=(p_val < 0.05),
            ))

    results.sort(key=lambda e: e.eta_squared, reverse=True)
    return results


def build_effect_size_block(effect_sizes: list[EffectSize]) -> str:
    """
    Return a prompt injection block with the top-3 η² pairs.
    Returns empty string when effect_sizes is empty.
    """
    if not effect_sizes:
        return ""

    top3 = effect_sizes[:3]
    lines = ["\n=== EFFECT SIZES (η² — proportion of variance explained) ==="]
    for e in top3:
        lines.append(
            f"  {e.group_col} explains {e.eta_squared:.0%} of {e.target_col} variance "
            f"(η²={e.eta_squared:.2f}, F={e.f_statistic:.1f}, p={e.p_value:.4f})"
        )
    lines.append(
        "INSTRUCTION: Cite these η² values in findings. "
        "The variable with the highest η² is the primary driver — "
        "lead with it in the Key Takeaway."
    )
    lines.append("=== END EFFECT SIZES ===\n")
    return "\n".join(lines)
