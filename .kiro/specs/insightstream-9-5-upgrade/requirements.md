# Requirements Document

## Introduction

InsightStream currently scores 7.5/10 on report quality. This upgrade implements 10 targeted
improvements across two tiers to reach 9.5/10. Tier 1 (Items 1–5) fixes mechanical gaps:
rendering computed-but-hidden data, enforcing synthesis in the Key Takeaway, cleaning prose
artifacts, applying currency formatting to chart captions, and tightening recommendation
structure. Tier 2 (Items 6–10) adds new reasoning modules: domain taxonomy injection, η²
effect-size computation, outlier characterization, cross-finding synthesis, and a limitations
section. All changes are confined to `engine/analysis/`, `engine/render/`, `analyzer.py`, and
`report_generator.py`.

## Glossary

- **Analyzer**: The `analyzer.py` module; orchestrates dataset analysis and LLM calls.
- **ReportGenerator**: The `report_generator.py` module; builds the PDF from the results dict.
- **build_from_assets**: The primary PDF-building function in `ReportGenerator`.
- **analyze_dataset**: The primary analysis function in `Analyzer`.
- **MetricStore**: In-memory store of computed column metrics with formatting support.
- **EffectSize**: Data structure holding η² (eta-squared), F-statistic, and p-value for a
  (categorical column × numeric column) pair.
- **OutlierProfile**: Data structure summarising the categorical modal profile of IsolationForest
  outlier rows.
- **Limitation**: Data structure representing a canonical analytical concept absent from the
  uploaded dataset.
- **ProseCleaner**: The module (or function) that removes `COLUMN=VALUE` artifacts from text.
- **SynthesisPass**: The second Groq LLM call that produces a cross-finding through-line paragraph.
- **DomainTaxonomy**: A domain-specific semantic block injected into the LLM prompt when the
  dataset is classified as FINANCE/CREDIT.
- **CANONICAL_COLUMNS**: The fixed set of five analytical concepts whose absence is reported as
  limitations (`account_status`, `transaction_date`, `customer_id`, `utilization_rate`,
  `transaction_amount`).
- **CACHE_VERSION**: A version string embedded in the prompt fingerprint; must be incremented
  after every prompt change to invalidate stale cache entries.
- **results dict**: The Python dict returned by `analyze_dataset` and consumed by
  `build_from_assets`.
- **η² (eta-squared)**: Proportion of variance in a numeric column explained by a categorical
  column, computed via one-way ANOVA; range [0, 1].

---

## Requirements

### Requirement 1: Render Hypotheses and Unit Notes in PDF

**User Story:** As a data analyst, I want the PDF report to display the computed hypotheses
and unit-of-analysis notes, so that readers can see open questions and data-context caveats
that are already computed but currently invisible.

#### Acceptance Criteria

1. WHEN `build_from_assets` is called with a non-empty `hypotheses` list, THE
   `ReportGenerator` SHALL render an "Open Questions" section in the PDF containing each
   hypothesis observation, its candidate explanations as bullet points, and a "To resolve:"
   note.

2. WHEN `build_from_assets` is called with a non-empty `unit_notes` list, THE
   `ReportGenerator` SHALL render a "Context" callout box in the PDF containing each note's
   text, placed after the Executive Summary and before the first chart section.

3. THE `ReportGenerator` SHALL accept `hypotheses` and `unit_notes` as keyword parameters
   in `build_from_assets`, defaulting to empty lists when not supplied.

4. WHEN `hypotheses` is an empty list, THE `ReportGenerator` SHALL omit the "Open Questions"
   section entirely from the PDF.

5. WHEN `unit_notes` is an empty list, THE `ReportGenerator` SHALL omit the "Context"
   callout box entirely from the PDF.

6. THE `Analyzer` SHALL pass `results["hypotheses"]` and `results["unit_notes"]` from the
   results dict into `build_from_assets` at the call site.

---

### Requirement 2: Key Takeaway Cross-Finding Synthesis

**User Story:** As a report reader, I want the Key Takeaway insight to synthesise multiple
findings rather than restate a single one, so that the opening insight delivers the most
important analytical conclusion.

#### Acceptance Criteria

1. THE `Analyzer` SHALL include a prompt rule requiring the first insight to be titled
   "Key Takeaway" and to reference at least two other findings by their exact titles.

2. WHEN the LLM generates the Key Takeaway, THE `Analyzer` SHALL enforce that the Key
   Takeaway states the through-line connecting the referenced findings in the format:
   "[Finding A title] and [Finding B title] both point to [through-line]: [one sentence
   with a specific number]."

3. THE `Analyzer` SHALL include in the prompt a list of phrases that are FORBIDDEN in the
   Key Takeaway, including generic restatements of a single finding.

4. WHEN the prompt is changed to enforce the new Key Takeaway rule, THE `Analyzer` SHALL
   increment `CACHE_VERSION` to invalidate stale cache entries.

---

### Requirement 3: Prose Artifact Cleaning

**User Story:** As a report reader, I want the PDF prose to use natural language rather than
raw `COLUMN=VALUE` syntax, so that the report reads professionally without technical
formatting artifacts.

#### Acceptance Criteria

1. THE `ProseCleaner` SHALL provide a `clean_prose_artifacts(text)` function that replaces
   all substrings matching the pattern `\b\w+=\S+` with a natural-language equivalent.

2. WHEN `clean_prose_artifacts` is applied to any string, THE `ProseCleaner` SHALL return a
   string containing no substring that matches `\b\w+=\S+`.

3. THE `ProseCleaner` SHALL derive the natural-language replacement by removing underscores
   from the column name and combining it with the value in title case as a fallback when no
   specific mapping exists.

4. IF a column value contains regex special characters, THEN THE `ProseCleaner` SHALL escape
   the value before substitution and fall back to the original text if substitution fails,
   without raising an exception.

5. THE `Analyzer` SHALL apply `clean_prose_artifacts` to every segmentation headline and
   every insight `text` field in `_attach_phase2_keys` before the results dict is returned.

---

### Requirement 4: Currency Formatting in Chart Captions

**User Story:** As a report reader, I want chart captions for monetary columns to display
values in currency format (e.g., `$18.6K`), so that financial figures are immediately
readable without manual unit conversion.

#### Acceptance Criteria

1. WHEN `_generate_chart_summary` is called for a chart whose axis column is tagged
   `monetary` in `semantics`, THE `Analyzer` SHALL format numeric values in that caption
   using `MetricStore.format(key, fmt="currency")`.

2. THE `Analyzer` SHALL update `_generate_chart_summary` to accept `semantics` and `store`
   as parameters.

3. WHEN `semantics` or `store` is `None`, THE `Analyzer` SHALL fall back to the default
   `f"{v:,.2f}"` formatting without raising an exception.

4. THE `Analyzer` SHALL pass `semantics` and `store` from `_render_from_spec` into every
   call to `_generate_chart_summary`.

---

### Requirement 5: Structured Recommendations

**User Story:** As a business stakeholder, I want each recommendation to name a specific
owner and timeframe, so that the report produces actionable items that can be assigned and
tracked.

#### Acceptance Criteria

1. THE `Analyzer` SHALL include a prompt rule (Rule 10) requiring every recommendation to
   begin with an imperative verb (e.g., Audit, Segment, Investigate, Build, Flag).

2. THE `Analyzer` SHALL include in the prompt rule that every recommendation must name the
   specific column(s) involved and state a concrete number from the data.

3. THE `Analyzer` SHALL include in the prompt rule that every recommendation must contain an
   "Owner:" label naming a team and a "Timeframe:" label naming a duration.

4. THE `Analyzer` SHALL include in the prompt rule a list of FORBIDDEN phrases, including
   "investigate underlying drivers", "optimise operations", "consider reviewing",
   "may want to", and "could potentially".

5. THE `Analyzer` SHALL generate between 3 and 5 recommendations per report.

---

### Requirement 6: Domain Taxonomy Injection

**User Story:** As a data analyst, I want the LLM to receive domain-specific semantic
context for FINANCE/CREDIT datasets, so that findings correctly interpret domain-specific
column semantics rather than producing misleading credit-risk conclusions.

#### Acceptance Criteria

1. THE `Analyzer` SHALL maintain a `DOMAIN_TAXONOMY` dict mapping domain categories to
   semantic context blocks, with at least a `FINANCE/CREDIT` entry covering `credit_limit`
   semantics by `card_type`.

2. WHEN `analyze_dataset` classifies a dataset as `FINANCE`, `CREDIT`, or
   `FINANCE_CREDIT` domain AND the dataset contains at least one of `credit_limit`,
   `card_type`, or `card_brand` columns, THE `Analyzer` SHALL inject the
   `FINANCE/CREDIT` taxonomy block into `_generate_prompt` before the column list.

3. WHEN the domain is not `FINANCE`, `CREDIT`, or `FINANCE_CREDIT`, THE `Analyzer` SHALL
   NOT inject any domain taxonomy block into the prompt.

4. THE `Analyzer` SHALL place the domain taxonomy block before the column list in the
   prompt so the LLM reads domain context before seeing the data.

---

### Requirement 7: η² Effect Size Module

**User Story:** As a data analyst, I want the LLM prompt to include η² effect sizes for
the top categorical-to-numeric relationships, so that findings are ranked by actual
explanatory power rather than by spread ratio alone.

#### Acceptance Criteria

1. THE `Analyzer` SHALL provide a `compute_effect_sizes(df, semantics)` function in
   `engine/analysis/effect_size.py` that computes one-way ANOVA η² for every
   (categorical_meaningful × monetary/numeric_meaningful) column pair.

2. WHEN `compute_effect_sizes` is called, THE `Analyzer` SHALL return a list of
   `EffectSize` objects sorted by `eta_squared` descending.

3. FOR ALL `EffectSize` objects returned by `compute_effect_sizes`, THE `Analyzer` SHALL
   ensure `eta_squared` is in the range [0, 1] and `p_value` is in the range [0, 1].

4. WHEN no valid (categorical × numeric) column pairs exist, THE `Analyzer` SHALL return
   an empty list (not `None`) from `compute_effect_sizes`.

5. THE `Analyzer` SHALL NOT mutate the input DataFrame `df` during `compute_effect_sizes`.

6. WHEN `scipy.stats.f_oneway` raises for a specific column pair (e.g., all values
   identical in a group), THE `Analyzer` SHALL skip that pair, log a warning, and
   continue processing remaining pairs.

7. THE `Analyzer` SHALL inject a prompt block containing the top-3 η² pairs into
   `_generate_prompt`, formatted as:
   `"{group_col} explains {eta_squared:.0%} of {target_col} variance (η²=..., F=..., p=...)"`.

8. THE `Analyzer` SHALL include in the injected block an instruction directing the LLM to
   cite η² values in findings and treat the highest-η² variable as the primary driver.

---

### Requirement 8: Outlier Characterization Module

**User Story:** As a data analyst, I want the LLM prompt to include a categorical profile
of the outlier rows, so that findings about anomalies describe who the outliers are rather
than just how many there are.

#### Acceptance Criteria

1. THE `Analyzer` SHALL provide a `profile_outliers(df, semantics, anomaly_indices)`
   function in `engine/analysis/outlier_profile.py` that computes the modal categorical
   value for each categorical column (with ≤ 20 unique values) among the outlier rows.

2. WHEN `anomaly_indices` is `None` or empty, THE `Analyzer` SHALL return `None` from
   `profile_outliers`.

3. FOR ANY non-empty `anomaly_indices` subset of `df.index`, THE `Analyzer` SHALL set
   `OutlierProfile.pct_of_total` equal to `len(anomaly_indices) / len(df)`.

4. THE `Analyzer` SHALL build a human-readable `narrative` string in `OutlierProfile`
   listing each categorical column where ≥ 70% of outliers share the modal value, in the
   format `"{pct:.0%} of outliers have {col}={val}"`.

5. WHEN `anomaly_indices` contains indices not present in `df.index`, THE `Analyzer` SHALL
   filter to valid indices before subsetting, log the count of invalid indices, and compute
   the profile on the valid subset.

6. THE `Analyzer` SHALL call `profile_outliers` in `analyze_dataset` after
   `_detect_anomalies` and inject the resulting block into `_generate_prompt`, replacing
   the existing generic anomaly count.

---

### Requirement 9: Cross-Finding Synthesis Pass

**User Story:** As a report reader, I want a synthesis paragraph that identifies the
through-line connecting the most important findings, so that the report tells a coherent
story rather than presenting isolated observations.

#### Acceptance Criteria

1. THE `Analyzer` SHALL provide a `_generate_synthesis(findings, client, model)` function
   that makes a second Groq LLM call after the primary analysis call, using the top-8
   findings as input.

2. WHEN `len(findings) < 2`, THE `Analyzer` SHALL return an empty string from
   `_generate_synthesis` without making an LLM call.

3. IF the Groq API call in `_generate_synthesis` raises any exception, THEN THE `Analyzer`
   SHALL catch the exception, log a warning, and return an empty string without re-raising.

4. THE `Analyzer` SHALL store the synthesis result in `results["synthesis"]` after the
   primary LLM call and `_validate_results`.

5. THE `Analyzer` SHALL use `GROQ_MODELS["fast"]` for the synthesis call to minimise
   latency and preserve the primary model's rate-limit budget.

6. WHEN `build_from_assets` is called with a non-empty `synthesis` string, THE
   `ReportGenerator` SHALL render a "Synthesis" highlighted box after the Deep Insights
   section, with a light amber background and left border accent.

7. WHEN `synthesis` is an empty string or absent, THE `ReportGenerator` SHALL omit the
   Synthesis box entirely from the PDF.

8. THE synthesis prompt SHALL instruct the LLM to identify the single through-line
   connecting the most important findings, name at least two findings by title, state what
   the data collectively reveals, and end with one actionable implication.

9. THE `_generate_synthesis` function SHALL return only plain prose — no JSON, no markdown
   headers, no bullet points.

---

### Requirement 10: Limitations Section

**User Story:** As a report reader, I want the PDF to include a "What We Don't Know"
section listing analytical questions the data cannot answer, so that I understand the
boundaries of the analysis and avoid drawing unsupported conclusions.

#### Acceptance Criteria

1. THE `Analyzer` SHALL provide a `detect_limitations(df, semantics)` function in
   `engine/analysis/limitations.py` that checks for the presence of five canonical
   analytical concepts: `account_status`, `transaction_date`, `customer_id`,
   `utilization_rate`, and `transaction_amount`.

2. WHEN a canonical concept is absent from `df` (neither the concept name nor any of its
   defined aliases is present as a column), THE `Analyzer` SHALL include a `Limitation`
   object for that concept in the returned list.

3. WHEN all canonical concepts are present in `df`, THE `Analyzer` SHALL return an empty
   list (not `None`) from `detect_limitations`.

4. THE `Analyzer` SHALL perform alias matching in `detect_limitations` in a
   case-insensitive and underscore-insensitive manner.

5. THE `Analyzer` SHALL store the limitations in `results["limitations"]` as a list of
   dicts with `concept` and `impact` keys via `_attach_phase2_keys`.

6. WHEN `build_from_assets` is called with a non-empty `limitations` list, THE
   `ReportGenerator` SHALL render a "What We Don't Know" section after the Recommendations
   section, listing each missing concept and its impact description as a bullet point.

7. WHEN `limitations` is an empty list or absent, THE `ReportGenerator` SHALL omit the
   "What We Don't Know" section entirely from the PDF.

8. WHEN `df` has no columns matching any canonical alias, THE `Analyzer` SHALL return all
   five canonical concepts as limitations.
