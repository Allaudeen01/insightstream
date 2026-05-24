# Implementation Plan: InsightStream 7.5 → 9.5 Upgrade

## Overview

Implement 10 targeted improvements across two tiers in Python. Tier 1 (Items 1–5) fixes
mechanical gaps in rendering, prose, formatting, and prompt rules. Tier 2 (Items 6–10) adds
new analysis modules (`effect_size.py`, `outlier_profile.py`, `limitations.py`) and a
cross-finding synthesis LLM pass. All changes are confined to `engine/analysis/`,
`engine/render/`, `analyzer.py`, and `report_generator.py`.

## Tasks

- [ ] 1. Create new analysis module stubs and package structure
  - Create `engine/analysis/` directory with `__init__.py` if it does not already exist
  - Create `engine/render/` directory with `__init__.py` if it does not already exist
  - Verify `scipy` is importable (it is a transitive dependency via scikit-learn)
  - Add `hypothesis` as a dev dependency in `requirements.txt` or `pyproject.toml`
  - _Requirements: 7.1, 8.1, 10.1_

- [ ] 2. Implement `engine/analysis/effect_size.py`
  - [ ] 2.1 Implement `EffectSize` dataclass and `compute_effect_sizes(df, semantics)` function
    - Define `EffectSize` as a Python dataclass with fields: `group_col`, `target_col`,
      `eta_squared`, `f_statistic`, `p_value`, `is_significant`
    - Iterate over all `(categorical_meaningful × monetary/numeric_meaningful)` column pairs
    - For each pair, call `scipy.stats.f_oneway`; skip and log a warning if it raises or
      returns NaN
    - Compute η² = SS_between / SS_total; clamp to [0, 1]
    - Return list sorted by `eta_squared` descending; return `[]` (not `None`) when no
      valid pairs exist
    - Do NOT mutate the input DataFrame
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [ ]* 2.2 Write property test for `compute_effect_sizes` — Property 1: Effect Size Bounds
    - **Property 1: Effect Size Bounds**
    - Use `hypothesis` to generate arbitrary DataFrames with ≥1 categorical and ≥1 numeric column
    - Assert every returned `eta_squared` ∈ [0, 1] and every `p_value` ∈ [0, 1]
    - **Validates: Requirements 7.3**

  - [ ]* 2.3 Write property test for `compute_effect_sizes` — Property 2: Effect Size Ordering
    - **Property 2: Effect Size Ordering**
    - Use `hypothesis` to generate arbitrary DataFrames
    - Assert the returned list is sorted by `eta_squared` descending (for all `i < j`,
      `result[i].eta_squared >= result[j].eta_squared`)
    - **Validates: Requirements 7.2**

  - [ ]* 2.4 Write property test for `compute_effect_sizes` — Property 10: DataFrame Immutability
    - **Property 10: Effect Size DataFrame Immutability**
    - Use `hypothesis` to generate arbitrary DataFrames
    - Assert that `df.shape`, `df.columns.tolist()`, and `df.values.tolist()` are identical
      before and after calling `compute_effect_sizes`
    - **Validates: Requirements 7.5**

  - [ ]* 2.5 Write unit tests for `compute_effect_sizes`
    - Test with a synthetic 3-group DataFrame; assert η² ∈ [0, 1] and list sorted descending
    - Test empty-pairs case returns `[]`
    - Test that a column pair where all values are identical in one group is skipped gracefully
    - _Requirements: 7.1, 7.4, 7.6_

- [ ] 3. Implement `engine/analysis/outlier_profile.py`
  - [ ] 3.1 Implement `OutlierProfile` dataclass and `profile_outliers(df, semantics, anomaly_indices)` function
    - Define `OutlierProfile` as a Python dataclass with fields: `n_outliers`,
      `pct_of_total`, `modal_profile`, `modal_pct`, `narrative`
    - Return `None` when `anomaly_indices` is `None` or empty
    - Filter `anomaly_indices` to valid `df.index` entries; log count of invalid indices
    - Compute modal value per categorical column (≤ 20 unique values) among outlier rows
    - Build `narrative` listing columns where ≥ 70% of outliers share the modal value,
      formatted as `"{pct:.0%} of outliers have {col}={val}"`
    - Set `pct_of_total = len(anomaly_indices) / len(df)` exactly
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 3.2 Write property test for `profile_outliers` — Property 5: Outlier Fraction Accuracy
    - **Property 5: Outlier Fraction Accuracy**
    - Use `hypothesis` to generate DataFrames and non-empty subsets of their index
    - Assert `profile.pct_of_total == len(anomaly_indices) / len(df)` exactly
    - **Validates: Requirements 8.3**

  - [ ]* 3.3 Write unit tests for `profile_outliers`
    - Test with synthetic outlier indices; assert `modal_profile` keys and `narrative` non-empty
    - Test `None` returned when `anomaly_indices` is empty
    - Test invalid indices are filtered and profile computed on valid subset
    - _Requirements: 8.1, 8.2, 8.4, 8.5_

- [ ] 4. Implement `engine/analysis/limitations.py`
  - [ ] 4.1 Implement `Limitation` dataclass, `CANONICAL_COLUMNS` dict, and `detect_limitations(df, semantics, domain=None)` function
    - Define `Limitation` as a Python dataclass with fields: `missing_concept`, `missing_impact`
    - Define `CANONICAL_COLUMNS` as a **domain-keyed dict** — not a flat list — so that
      finance-specific concepts are never shown for entertainment/people/sports datasets:
      ```python
      CANONICAL_COLUMNS = {
          # Shown for FINANCE, CREDIT, FINANCE_CREDIT, ECOMMERCE_TRANSACTIONS domains
          "finance": {
              "account_status":     {"aliases": [...], "missing_impact": "..."},
              "utilization_rate":   {"aliases": [...], "missing_impact": "..."},
              "transaction_amount": {"aliases": [...], "missing_impact": "..."},
          },
          # Shown for any domain that has a date column but no transaction history
          "temporal": {
              "transaction_date": {"aliases": [...], "missing_impact": "..."},
          },
          # Shown for any domain where rows are at item level (unit_notes present)
          "entity": {
              "customer_id": {"aliases": [...], "missing_impact": "..."},
          },
      }
      ```
    - `detect_limitations(df, semantics, domain=None)` signature:
      - When `domain` is `None` or not in `("FINANCE", "CREDIT", "FINANCE_CREDIT",
        "ECOMMERCE_TRANSACTIONS")`, skip the `"finance"` bucket entirely — return `[]`
        for finance concepts
      - Always check `"temporal"` bucket (relevant to any dataset with a date column)
      - Always check `"entity"` bucket (relevant to any dataset with repeated entity IDs)
      - Perform alias matching case-insensitively and underscore-insensitively
      - Return `[]` (not `None`) when all applicable concepts are present
      - Do NOT mutate `df`
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8_

  - [ ]* 4.2 Write property test for `detect_limitations` — Property 3: Limitations Completeness
    - **Property 3: Limitations Completeness**
    - Use `hypothesis` to generate DataFrames with arbitrary column names
    - Assert `len(detect_limitations(df, {})) + count_present_canonical(df) == len(CANONICAL_COLUMNS)` (== 5)
    - **Validates: Requirements 10.2, 10.3, 10.8**

  - [ ]* 4.3 Write property test for `detect_limitations` — Property 11: Alias Case-Insensitivity
    - **Property 11: Limitations Alias Case-Insensitivity**
    - Use `hypothesis` to generate case/underscore variants of canonical aliases
    - Assert that a DataFrame containing any such variant does NOT include that concept in
      the returned limitations list
    - **Validates: Requirements 10.4**

  - [ ]* 4.4 Write unit tests for `detect_limitations`
    - Test DataFrame with all canonical columns present → returns `[]`
    - Test DataFrame with no canonical columns → returns all 5 limitations
    - Test alias matching (e.g., `"Status"` matches `account_status`)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8_

- [ ] 5. Checkpoint — Ensure all new analysis module tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement `engine/render/prose_cleaner.py`
  - [ ] 6.1 Implement `clean_prose_artifacts(text)` function
    - Apply regex `r'\b(\w+)=([^\s,]+(?:\s+\([^)]+\))?)\b'` to find `COLUMN=VALUE` patterns
    - Replace each match with a natural-language equivalent: strip underscores from column
      name, combine with value in title case as fallback
    - Escape regex special characters in values before substitution; fall back to original
      text if substitution raises, without raising an exception
    - Return a string containing no substring matching `\b\w+=\S+`
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 6.2 Write property test for `clean_prose_artifacts` — Property 4: Prose Cleanliness
    - **Property 4: Prose Cleanliness**
    - Use `hypothesis` to generate arbitrary strings (including strings with regex special chars)
    - Assert that `re.search(r'\b\w+=\S+', clean_prose_artifacts(s))` returns `None` for all inputs
    - **Validates: Requirements 3.2**

  - [ ]* 6.3 Write unit tests for `clean_prose_artifacts`
    - Test `"card_type=Debit (Prepaid)"` → no `COLUMN=VALUE` pattern remains
    - Test `"has_chip=YES"` → natural-language output
    - Test string with regex special characters in value → no exception raised
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 7. Update `analyzer.py` — Tier 1 prompt and chart caption changes
  - [ ] 7.1 Replace Key Takeaway prompt rule and bump `CACHE_VERSION` to `v4`
    - Replace the existing Key Takeaway rule in `_generate_prompt` with the new rule
      requiring reference to ≥2 other findings by exact title and the through-line format
    - Add the FORBIDDEN phrases list to the Key Takeaway rule
    - Bump `CACHE_VERSION` from `v3` to `v4`
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ] 7.2 Replace Rule 10 (Recommendations) in `_generate_prompt`
    - Replace the existing Rule 10 with the stricter imperative-owner-timeframe structure
    - Add the FORBIDDEN phrases list: "investigate underlying drivers", "optimise operations",
      "consider reviewing", "may want to", "could potentially"
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ] 7.3 Update `_generate_chart_summary` to accept `semantics` and `store` parameters
    - Add `semantics=None` and `store=None` keyword parameters to `_generate_chart_summary`
    - When the axis column is tagged `monetary` in `semantics`, format values using
      `store.format(key, fmt="currency")`
    - Fall back to `f"{v:,.2f}"` when `semantics` or `store` is `None`
    - Update all call sites in `_render_from_spec` to pass `semantics` and `store`
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 7.4 Write unit tests for chart caption currency formatting
    - Mock `MetricStore` and `semantics`; assert monetary column caption shows `$18,558`
      (full number, not K-suffix) rather than `18558.23`
    - Assert fallback to `f"{v:,.2f}"` when `semantics=None`
    - **Note on formatting consistency:** Use full-number currency everywhere (`$18,558`),
      not K-suffix (`$18.6K`). The `MetricStore.format(fmt="currency")` already produces
      full numbers. Do NOT introduce K-suffix formatting in chart captions — it would
      create inconsistency with segmentation headlines in the same report.
    - _Requirements: 4.1, 4.3_

- [ ] 8. Update `analyzer.py` — Tier 2 module integration
  - [ ] 8.1 Integrate `compute_effect_sizes` into `analyze_dataset` and `_generate_prompt`
    - Import `compute_effect_sizes` and `build_effect_size_block` from `engine/analysis/effect_size.py`
    - Call `compute_effect_sizes(df, semantics)` in `analyze_dataset` after `MetricStore` is built
    - Store result in `results["effect_sizes"]` as a list of dicts
    - Pass the effect-size block into `_generate_prompt` as a new `effect_size_block` parameter,
      placed before the column list
    - _Requirements: 7.7, 7.8_

  - [ ] 8.2 Integrate `profile_outliers` into `analyze_dataset` and `_generate_prompt`
    - **Before writing any code, read `_detect_anomalies` in `analyzer.py` to confirm its
      return signature.** It currently returns `(scores_array, anomaly_indices)` where
      `anomaly_indices` is a Python list of integer row positions from `df.index[preds == -1]`.
      Pass this list directly to `profile_outliers`. Do NOT convert to a boolean mask.
    - Import `profile_outliers` and `build_outlier_profile_block` from `engine/analysis/outlier_profile.py`
    - Call `profile_outliers(df, semantics, anomaly_indices)` after `_detect_anomalies`
    - Store result in `results["outlier_profile"]`
    - Replace the existing generic anomaly count in `_generate_prompt` with the outlier
      profile block
    - _Requirements: 8.6_

  - [ ] 8.3 Integrate `detect_limitations` into `analyze_dataset` and `_attach_phase2_keys`
    - Import `detect_limitations` from `engine/analysis/limitations.py`
    - Call `detect_limitations(df, semantics)` in `analyze_dataset`
    - In `_attach_phase2_keys`, store `results["limitations"]` as a list of dicts with
      `concept` and `impact` keys
    - _Requirements: 10.5_

  - [ ] 8.4 Integrate domain taxonomy injection into `_generate_prompt`
    - Add `DOMAIN_TAXONOMY` dict (or import from `engine/classifiers/domain_taxonomy.py`)
      with the `FINANCE/CREDIT` block covering `credit_limit` semantics by `card_type`
    - Implement `inject_domain_taxonomy(domain_info, df, semantics)` that returns the
      taxonomy block when domain is `FINANCE`, `CREDIT`, or `FINANCE_CREDIT` AND at least
      one of `credit_limit`, `card_type`, `card_brand` is present
    - Inject the block into `_generate_prompt` before the column list; inject nothing when
      domain does not match
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ] 8.5 Implement `_generate_synthesis` and wire into `analyze_dataset`
    - Implement `_generate_synthesis(findings, client, model)` in `analyzer.py`
    - Return `""` immediately when `len(findings) < 2`
    - Make a second Groq call using `GROQ_MODELS["fast"]` with the top-8 findings
    - Catch all exceptions, log a warning, and return `""` on failure
    - After `_validate_results`, call `_generate_synthesis` and store result in
      `results["synthesis"]`
    - **Use this exact prompt template** (do not paraphrase or simplify):
      ```
      You are a senior data analyst writing the synthesis section of a report.
      Below are {n} findings from a dataset analysis.

      {findings_text}

      Write a 3-5 sentence synthesis paragraph. Rules:
      - Name specific columns, values, and statistics (use the effect size rankings if present)
      - Lead with the highest η² pair or the finding with the largest spread ratio
      - Contrast the top finding against a weaker one explicitly
      - End with one implication for analysis approach (e.g., "segment before modeling",
        "exclude outliers before computing means")
      - FORBIDDEN phrases: "these findings reveal", "important patterns",
        "multiple dimensions", "in conclusion", "overall", "it is worth noting"

      Return ONLY the paragraph text — no headers, no bullet points, no JSON.
      ```
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.8, 9.9_

  - [ ]* 8.6 Write property test for `_generate_synthesis` — Property 6: Synthesis Non-Crash
    - **Property 6: Synthesis Non-Crash**
    - Use `hypothesis` to generate arbitrary findings lists with `len >= 2`
    - Test with both a working mock Groq client and one that raises an exception
    - Assert `_generate_synthesis` always returns a `str` and never raises
    - **Validates: Requirements 9.3**

  - [ ] 8.7 Apply `clean_prose_artifacts` in `_attach_phase2_keys`
    - Import `clean_prose_artifacts` from `engine/render/prose_cleaner.py`
    - Apply it to every segmentation headline and every insight `text` field in
      `_attach_phase2_keys` before the results dict is returned
    - _Requirements: 3.5_

- [ ] 9. Checkpoint — Ensure all analyzer.py tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Update `report_generator.py` — new PDF sections
  - [ ] 10.1 Add `hypotheses` and `unit_notes` parameters to `build_from_assets` and render sections
    - Add `hypotheses=[]` and `unit_notes=[]` keyword parameters to `build_from_assets`
    - Implement `context_callout_box(unit_notes)` helper: tinted light-blue callout box,
      rendered after Executive Summary and before the first chart section; omit when empty
    - Implement `open_questions_section(hypotheses)` helper: "Open Questions" section header,
      each hypothesis as bold observation + bullet candidates + italic "To resolve:" note;
      rendered after Deep Insights section; omit when empty
    - Update the call site in `insight_engine.py` (or the route) to pass
      `results["hypotheses"]` and `results["unit_notes"]` into `build_from_assets`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 10.2 Write property test for hypothesis rendering — Property 7: Hypothesis Rendering Completeness
    - **Property 7: Hypothesis Rendering Completeness**
    - Use `hypothesis` to generate non-empty `hypotheses` lists
    - Call `build_from_assets` with the generated list and assert the output PDF bytes
      contain the text `b"Open Questions"`
    - **Validates: Requirements 1.1**

  - [ ]* 10.3 Write property test for unit notes rendering — Property 8: Unit Notes Rendering Completeness
    - **Property 8: Unit Notes Rendering Completeness**
    - Use `hypothesis` to generate non-empty `unit_notes` lists
    - Call `build_from_assets` with the generated list and assert the output PDF bytes
      contain the text `b"Context"`
    - **Validates: Requirements 1.2**

  - [ ] 10.4 Add `synthesis` parameter to `build_from_assets` and render Synthesis box
    - Add `synthesis=""` keyword parameter to `build_from_assets`
    - Implement `_build_synthesis_box(synthesis_text)` helper: light amber background
      (`#FEF3C7`), left border accent (`#F59E0B`, 3pt), rendered after Deep Insights section;
      omit when `synthesis` is empty or absent
    - Update the call site to pass `results.get("synthesis", "")` into `build_from_assets`
    - _Requirements: 9.6, 9.7_

  - [ ] 10.5 Add `limitations` parameter to `build_from_assets` and render "What We Don't Know" section
    - Add `limitations=[]` keyword parameter to `build_from_assets`
    - Implement `_build_limitations_section(limitations)` helper: "What We Don't Know"
      section after Recommendations, each limitation as a bullet `"• {concept}: {impact}"`;
      omit when empty
    - Update the call site to pass `results.get("limitations", [])` into `build_from_assets`
    - _Requirements: 10.6, 10.7_

  - [ ]* 10.6 Write property test for limitations rendering — Property 9: Limitations Rendering Completeness
    - **Property 9: Limitations Rendering Completeness**
    - Use `hypothesis` to generate non-empty `limitations` lists
    - Call `build_from_assets` with the generated list and assert the output PDF bytes
      contain the text `b"What We Don't Know"`
    - **Validates: Requirements 10.6**

  - [ ]* 10.7 Write unit tests for `build_from_assets` new sections
    - Test empty `hypotheses` → no "Open Questions" in PDF
    - Test empty `unit_notes` → no "Context" callout in PDF
    - Test empty `synthesis` → no Synthesis box in PDF
    - Test empty `limitations` → no "What We Don't Know" in PDF
    - _Requirements: 1.4, 1.5, 9.7, 10.7_

- [ ] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Real-LLM end-to-end verification gate
  - Create `scripts/verify_phase4.py` that runs the full pipeline on `cards_data.csv`
    with a live GROQ_API_KEY and asserts the following quality properties:
    ```python
    # D1: synthesis contains at least one column name AND one number AND no forbidden phrases
    assert any(col in synthesis for col in df.columns)
    assert re.search(r'\d', synthesis)
    for phrase in ["these findings reveal", "important patterns", "multiple dimensions",
                   "in conclusion", "overall", "it is worth noting"]:
        assert phrase not in synthesis.lower()

    # D2: limitations is empty OR contains only finance-relevant entries
    # (must NOT contain "cannot distinguish active from closed accounts" for non-finance datasets)
    # For cards_data (FINANCE domain): limitations may be non-empty and finance-relevant
    for lim in limitations:
        assert lim["concept"] in ("account_status", "utilization_rate",
                                   "transaction_amount", "transaction_date", "customer_id")

    # D3: effect_sizes[0].eta_squared > 0.10 (card_type dominates credit_limit variance)
    # Note: actual measured value is ~0.23 (22.7% explained variance, F=899).
    # The spec's original "0.73" estimate confused spread ratio with η² — they differ.
    assert effect_sizes[0]["eta_squared"] > 0.10, (
        f"Expected card_type to explain >10% of credit_limit variance, "
        f"got {effect_sizes[0]['eta_squared']:.2f}"
    )

    # D4: PDF contains all four new sections
    pdf_text = extract_text_from_pdf(pdf_path)
    assert "Open Questions" in pdf_text
    assert "Context" in pdf_text
    assert "What We Don't Know" in pdf_text
    assert "Synthesis" in pdf_text

    # D5: No COLUMN=VALUE patterns in any insight text
    for insight in insights:
        assert not re.search(r'\b\w+=\S+', insight.get("text", "")), (
            f"COLUMN=VALUE artifact in insight: {insight['text'][:100]}"
        )
    ```
  - Run: `python scripts/verify_phase4.py`
  - Must print `PHASE 4 PASSED` before declaring the upgrade complete
  - If any assertion fails, fix the responsible task before declaring done
  - _Requirements: 2.1, 3.2, 7.2, 9.3, 10.6_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use the `hypothesis` library (Python); add it as a dev dependency in step 1
- Unit tests use `pytest`
- `CACHE_VERSION` bump in task 7.1 will invalidate all existing cache entries — users will
  experience one cache miss per dataset after deployment
- The synthesis LLM call (task 8.5) adds ~1–3 seconds per analysis; it uses
  `GROQ_MODELS["fast"]` to minimise latency

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1", "4.1", "6.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "2.4", "2.5", "3.2", "3.3", "4.2", "4.3", "4.4", "6.2", "6.3"] },
    { "id": 2, "tasks": ["7.1", "7.2", "7.3", "8.1", "8.2", "8.3", "8.4"] },
    { "id": 3, "tasks": ["7.4", "8.5", "8.7"] },
    { "id": 4, "tasks": ["8.6", "10.1", "10.4", "10.5"] },
    { "id": 5, "tasks": ["10.2", "10.3", "10.6", "10.7"] },
    { "id": 6, "tasks": ["12"] }
  ]
}
```
