"""
engine/classifiers/policy.py
─────────────────────────────
Downstream handling rules keyed by SemanticTag.
"""

SEMANTICS_POLICY: dict[str, dict] = {
    "identifier": {
        "include_in_correlations": False,
        "include_in_group_by_means": False,
        "eligible_for_key_takeaway": False,
        "show_distribution_chart": False,
    },
    "random_token": {
        "include_in_correlations": False,
        "include_in_group_by_means": False,
        "eligible_for_key_takeaway": False,
        "show_distribution_chart": False,   # only if user explicitly asks
    },
    "categorical_degenerate": {
        "include_in_correlations": False,
        "include_in_group_by_means": False,
        "eligible_for_key_takeaway": False,
        "show_distribution_chart": False,   # flag in data-quality section
    },
    "temporal": {
        "include_in_correlations": True,
        "include_in_group_by_means": True,
        "eligible_for_key_takeaway": True,
        "show_distribution_chart": True,    # timeline chart
    },
    "monetary": {
        "include_in_correlations": True,
        "include_in_group_by_means": True,  # as target
        "eligible_for_key_takeaway": True,
        "show_distribution_chart": True,    # with $ formatting
    },
    "numeric_meaningful": {
        "include_in_correlations": True,
        "include_in_group_by_means": True,  # as target
        "eligible_for_key_takeaway": True,
        "show_distribution_chart": True,
    },
    "categorical_meaningful": {
        "include_in_correlations": True,    # Cramér's V
        "include_in_group_by_means": True,  # as grouper
        "eligible_for_key_takeaway": True,
        "show_distribution_chart": True,
    },
    "free_text": {
        "include_in_correlations": False,
        "include_in_group_by_means": False,
        "eligible_for_key_takeaway": False,
        "show_distribution_chart": False,   # only token cloud if requested
    },
}
