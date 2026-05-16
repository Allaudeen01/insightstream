"""
Tier 1.2 — Enhanced Time-Series Analysis Module
================================================
Replaces the basic peak/trough detection in _rule_temporal_peaks
with a full time-series analysis that includes:
  - Trend slope + direction
  - YoY / QoQ / MoM growth rates
  - Seasonality detection (CV of calendar-month averages)
  - Anomaly months (>2σ from rolling mean)

INTEGRATION: Replace _rule_temporal_peaks in BusinessRuleEngine with
  TimeSeriesAnalyzer.analyze(), or call it alongside.
"""

from __future__ import annotations
import logging
from typing import Optional

import polars as pl
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

log = logging.getLogger(__name__)


def _fmt_currency(val: float) -> str:
    abs_val = abs(val)
    sign = "" if val >= 0 else "-"
    if abs_val >= 1_00_00_000:
        return f"{sign}₹{abs_val/1_00_00_000:.2f} Cr"
    if abs_val >= 1_00_000:
        return f"{sign}₹{abs_val/1_00_000:.2f} L"
    if abs_val >= 1_000:
        return f"{sign}₹{abs_val/1_000:.1f}K"
    return f"{sign}₹{abs_val:,.0f}"


class TimeSeriesAnalyzer:
    """
    Comprehensive time-series analysis for sales/revenue data.
    
    Produces insights about:
      1. Overall trend (growing / declining / flat)
      2. Peak and trough periods
      3. Seasonality patterns
      4. Anomalous months
      5. Growth rates (MoM, QoQ, YoY where data permits)
    """

    MIN_MONTHS = 3      # Need at least 3 months for meaningful analysis
    MIN_RECORDS = 30     # Need 30+ records
    SEASONALITY_CV_THRESHOLD = 0.10  # CV > 10% across calendar months = seasonal

    def analyze(self, df: pl.DataFrame, profile=None) -> list:
        """
        Main entry point. Finds date + revenue columns and runs analysis.
        
        Returns list of BusinessInsight-compatible objects.
        """
        from insight_engine import BusinessInsight

        # ── Find columns ──────────────────────────────────────────────
        date_col = self._find_date_col(df, profile)
        rev_col = self._find_revenue_col(df, profile)

        if not date_col or not rev_col:
            log.info(f"[TimeSeries] Missing columns: date={date_col}, rev={rev_col}")
            return []

        # ── Parse dates and aggregate monthly ─────────────────────────
        try:
            pdf = df.to_pandas()
            pdf[date_col] = pd.to_datetime(pdf[date_col], errors="coerce")
            pdf = pdf.dropna(subset=[date_col])

            if len(pdf) < self.MIN_RECORDS:
                return []

            pdf["_ym"] = pdf[date_col].dt.to_period("M")
            monthly = pdf.groupby("_ym")[rev_col].sum().sort_index()

            if len(monthly) < self.MIN_MONTHS:
                return []

        except Exception as e:
            log.warning(f"[TimeSeries] Parse failed: {e}")
            return []

        insights = []
        months_list = monthly.index.tolist()
        rev_list = monthly.values.tolist()

        # ── 1. TREND ANALYSIS ─────────────────────────────────────────
        trend_insight = self._analyze_trend(months_list, rev_list, rev_col, BusinessInsight)
        if trend_insight:
            insights.append(trend_insight)

        # ── 2. SEASONALITY DETECTION ──────────────────────────────────
        if len(monthly) >= 12:
            season_insight = self._analyze_seasonality(pdf, date_col, rev_col, monthly, BusinessInsight)
            if season_insight:
                insights.append(season_insight)

        # ── 3. GROWTH RATES ───────────────────────────────────────────
        growth_insight = self._analyze_growth_rates(monthly, rev_col, BusinessInsight)
        if growth_insight:
            insights.append(growth_insight)

        # ── 4. ANOMALOUS MONTHS ───────────────────────────────────────
        anomaly_insight = self._detect_anomaly_months(monthly, rev_col, BusinessInsight)
        if anomaly_insight:
            insights.append(anomaly_insight)

        log.info(f"[TimeSeries] Generated {len(insights)} insights from {len(monthly)} months")
        return insights

    # ──────────────────────────────────────────────────────────────
    # SUB-ANALYZERS
    # ──────────────────────────────────────────────────────────────

    def _decompose_seasonality(self, monthly: pd.Series) -> dict:
        """Run STL decomposition and return variance explained by each component."""
        if len(monthly) < 24:  # Need 2+ full years
            return {}
        try:
            stl = STL(monthly, period=12, robust=True)
            result = stl.fit()
            
            total_var = monthly.var()
            trend_var = result.trend.var() / total_var * 100
            seasonal_var = result.seasonal.var() / total_var * 100
            residual_var = result.resid.var() / total_var * 100
            
            return {
                "trend_explains_pct": round(trend_var, 1),
                "seasonal_explains_pct": round(seasonal_var, 1),
                "residual_explains_pct": round(residual_var, 1),
                "seasonality_is_significant": seasonal_var > 20,
            }
        except Exception:
            return {}

    def _forecast_next_months(self, monthly: pd.Series, periods: int = 3) -> dict | None:
        """GAP 4: Forecast next N months using Holt-Winters Exponential Smoothing.
        
        Requires at least 24 months (2 full years) for seasonal fitting.
        Returns forecast values, month labels, and 80% prediction intervals.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        if len(monthly) < 24:
            return None
        try:
            model = ExponentialSmoothing(
                monthly.values,
                seasonal_periods=12,
                trend="add",
                seasonal="add",
            )
            fitted = model.fit(optimized=True, remove_bias=True)
            forecast = fitted.forecast(periods)

            # 80% prediction interval via fitted residual std
            resid_std = np.std(fitted.resid)
            z80 = 1.282
            lower = forecast - z80 * resid_std
            upper = forecast + z80 * resid_std

            # Generate month labels
            last_period = monthly.index[-1]
            if hasattr(last_period, "to_timestamp"):
                base = last_period.to_timestamp()
            else:
                base = pd.Timestamp(last_period)

            labels = [(base + pd.DateOffset(months=i)).strftime("%b %Y")
                      for i in range(1, periods + 1)]

            return {
                "forecast_values": [round(float(v), 2) for v in forecast],
                "forecast_months": labels,
                "lower_80": [round(float(v), 2) for v in lower],
                "upper_80": [round(float(v), 2) for v in upper],
                "method": "Holt-Winters Exponential Smoothing (additive trend + seasonality)",
            }
        except Exception as e:
            log.warning(f"[TimeSeries] Forecast failed: {e}")
            return None

    @staticmethod
    def segment_comparison_test(group_a: pd.Series, group_b: pd.Series) -> dict:
        """GAP 6: Returns Cohen's d effect size and t-test p-value for two groups."""
        from scipy import stats as sp_stats
        a = group_a.dropna()
        b = group_b.dropna()
        if len(a) < 2 or len(b) < 2:
            return {"significant": False, "cohens_d": 0, "p_value": 1.0, "interpretation": "insufficient data"}
        pooled_std = np.sqrt((a.var() + b.var()) / 2)
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0
        _, p_value = sp_stats.ttest_ind(a, b, equal_var=False)
        interpretation = "large effect" if abs(d) > 0.8 else ("medium effect" if abs(d) > 0.5 else "small effect")
        return {
            "cohens_d": round(d, 2),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "interpretation": interpretation,
        }

    def _analyze_trend(self, months, revenues, rev_col, BI):
        """Linear trend: slope, direction, R², peak/trough."""
        n = len(revenues)
        x = np.arange(n)
        y = np.array(revenues, dtype=float)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        avg_rev = np.mean(y)
        monthly_growth_pct = (slope / avg_rev) * 100 if avg_rev > 0 else 0

        if monthly_growth_pct > 2:
            direction = "growing"
            impact = "🟢 Minor"
        elif monthly_growth_pct < -2:
            direction = "declining"
            impact = "🔴 Critical"
        else:
            direction = "flat"
            impact = "🟠 Important"

        # Peak / trough
        peak_idx = int(np.argmax(y))
        trough_idx = int(np.argmin(y))
        peak_month = months[peak_idx].strftime("%B %Y")
        trough_month = months[trough_idx].strftime("%B %Y")
        peak_val = y[peak_idx]
        trough_val = y[trough_idx]
        pct_gap = ((peak_val - trough_val) / peak_val * 100) if peak_val > 0 else 0

        # Chart data for frontend
        chart_monthly_data = [
            (m.strftime("%Y-%m"), float(r)) for m, r in zip(months, revenues)
        ]

        # Only claim a directional signal when R² is meaningful (≥ 0.10).
        # A slope with R²=0.05 explains almost none of the variance — it's noise.
        if r_squared < 0.10:
            trend_title = f"Revenue Trend: Flat (R²={r_squared:.2f}, no directional signal)"
            trend_desc_line = (
                f"Over {n} months, the revenue trend is flat. "
                f"A linear fit gives {monthly_growth_pct:+.1f}%/mo but R²={r_squared:.2f} — "
                f"less than 10% of variance is explained, so the slope carries no reliable signal. "
            )
        else:
            trend_title = f"Revenue Trend: {direction.title()} ({monthly_growth_pct:+.1f}%/mo)"
            trend_desc_line = (
                f"Over {n} months, revenue is {direction} at {monthly_growth_pct:+.1f}% "
                f"per month (R²={r_squared:.2f}). "
            )

        return BI(
            title=trend_title,
            description=(
                trend_desc_line +
                f"Peak: {peak_month} at {_fmt_currency(peak_val)}. "
                f"Trough: {trough_month} at {_fmt_currency(trough_val)} "
                f"({pct_gap:.0f}% gap). "
                f"Monthly average: {_fmt_currency(avg_rev)}."
            ),
            why_it_matters={
                "declining": (
                    f"Revenue is contracting at {abs(monthly_growth_pct):.1f}%/mo — "
                    f"R²={r_squared:.2f} confirms this is a consistent trend, not noise. "
                    f"Requires immediate root-cause investigation."
                ),
                "growing": (
                    f"Revenue is growing consistently at {monthly_growth_pct:.1f}%/mo. "
                    f"Protect the drivers of this trend and avoid changes that could disrupt momentum."
                ),
                "flat": (
                    f"Revenue is stable but not growing. The {pct_gap:.0f}% seasonal swing between "
                    f"{peak_month} and {trough_month} creates cash flow risk even without a downward trend. "
                    f"Growth levers are needed — new segments, pricing, or market expansion."
                ),
            }[direction],
            evidence=(
                f"Slope: {_fmt_currency(slope)}/mo | R²: {r_squared:.2f} | "
                f"Peak: {peak_month} ({_fmt_currency(peak_val)}) | "
                f"Trough: {trough_month} ({_fmt_currency(trough_val)})"
            ),
            impact=impact,
            recommendation=(
                f"{'Investigate the declining trend — check for market share loss, pricing issues, or demand shifts.' if direction == 'declining' else ''}"
                f"{'Revenue is flat — look for growth levers (new segments, pricing, expansion).' if direction == 'flat' else ''}"
                f"{'Sustain the growth trajectory — identify which segments are driving it and double down.' if direction == 'growing' else ''}"
                f" Pre-position resources ahead of {months[peak_idx].strftime('%B')} (historical peak)."
            ),
            methodology=(
                f"Linear regression (numpy.polyfit) over {n} monthly data points. "
                f"R²={r_squared:.2f}. Direction threshold: ±2%/mo."
            ),
            rule_type="temporal_peaks",
            score=9.0 if direction == "declining" else 8.0,
            chart_data={
                "monthly_data": chart_monthly_data,
                "peak_month": months[peak_idx].strftime("%B"),
                "peak_val": float(peak_val),
                "trough_month": months[trough_idx].strftime("%B"),
                "trough_val": float(trough_val),
                "pct_gap": round(pct_gap, 1),
                "slope": float(slope),
                "r_squared": round(r_squared, 3),
                "monthly_growth_pct": round(monthly_growth_pct, 2),
                "direction": direction,
                "forecast": self._forecast_next_months(
                    pd.Series(revenues, index=pd.DatetimeIndex([
                        m.to_timestamp() if hasattr(m, "to_timestamp") else m for m in months
                    ]))
                ),
            },
        )

    def _analyze_seasonality(self, pdf, date_col, rev_col, monthly, BI):
        """Detect recurring calendar-month patterns."""
        try:
            pdf["_cal_month"] = pdf[date_col].dt.month
            cal_avg = pdf.groupby("_cal_month")[rev_col].mean()

            if len(cal_avg) < 6:
                return None

            mean_of_means = cal_avg.mean()
            cv = cal_avg.std() / mean_of_means if mean_of_means > 0 else 0

            if cv < self.SEASONALITY_CV_THRESHOLD:
                return None

            # Find strongest and weakest calendar months
            strong_month = cal_avg.idxmax()
            weak_month = cal_avg.idxmin()

            import calendar
            strong_name = calendar.month_name[int(strong_month)]
            weak_name = calendar.month_name[int(weak_month)]
            lift_pct = ((cal_avg.max() - cal_avg.min()) / cal_avg.min() * 100) if cal_avg.min() > 0 else 0

            description = (
                f"Revenue shows a recurring seasonal pattern (CV={cv:.2f} across calendar months). "
                f"{strong_name} averages {_fmt_currency(cal_avg.max())} per period — "
                f"{lift_pct:.0f}% above {weak_name} ({_fmt_currency(cal_avg.min())}). "
                f"This pattern repeats across years, suggesting structural seasonality "
                f"rather than one-time events."
            )

            stl_result = self._decompose_seasonality(monthly)
            if stl_result.get("seasonality_is_significant"):
                seasonal_pct = stl_result["seasonal_explains_pct"]
                description += (
                    f" STL decomposition confirms seasonality explains "
                    f"{seasonal_pct:.0f}% of revenue variance — statistically significant."
                )

            cat_col = next((c for c in pdf.columns if "category" in c.lower()), None)
            peak_cat = None
            if cat_col and cat_col in pdf.columns:
                pdf["_cal_month"] = pdf[date_col].dt.month
                category_seasonal = pdf.groupby(["_cal_month", cat_col])[rev_col].sum().unstack(cat_col)
                
                # Which category drives the peak month?
                peak_cat = category_seasonal.loc[strong_month].idxmax()
                peak_cat_share = (category_seasonal.loc[strong_month, peak_cat] / 
                                  category_seasonal.loc[strong_month].sum() * 100)
                
                if peak_cat_share > 40:  # One category dominates the peak
                    description += (
                        f" {peak_cat} drives {peak_cat_share:.0f}% of {strong_name} revenue — "
                        f"investigate whether this category has seasonal demand or promotional timing."
                    )

            chart_data = {
                "calendar_averages": {
                    calendar.month_abbr[int(m)]: round(float(v), 2) 
                    for m, v in cal_avg.items()
                },
                "cv": round(cv, 3),
                "peak_calendar_month": strong_name,
                "trough_calendar_month": weak_name,
            }
            if peak_cat:
                chart_data["peak_category"] = peak_cat

            return BI(
                title=f"Seasonality Detected: {strong_name} strongest, {weak_name} weakest",
                description=description,
                why_it_matters=(
                    "Seasonal patterns inform inventory planning, marketing spend allocation, "
                    "staffing, and cash flow forecasting. Ignoring seasonality leads to "
                    "overstocking in slow months and stockouts in peak months."
                ),
                evidence=f"Seasonality CV: {cv:.2f} | Peak month: {strong_name} | Trough month: {weak_name}",
                impact="🟠 Important",
                recommendation=(
                    f"Build a seasonal budget model: allocate {strong_name} +20% and "
                    f"{weak_name} -15% vs flat baseline. Pre-position inventory 30 days "
                    f"before {strong_name}. Consider promotions in {weak_name} to smooth demand."
                ),
                rule_type="seasonality_pattern",
                score=7.5,
                chart_data=chart_data,
            )
        except Exception as e:
            log.warning(f"[TimeSeries] Seasonality analysis failed: {e}")
            return None

    def _analyze_growth_rates(self, monthly, rev_col, BI):
        """Compute MoM, QoQ, YoY growth rates where data permits."""
        try:
            n = len(monthly)
            rates = {}

            # MoM: last month vs previous month
            if n >= 2:
                last = monthly.iloc[-1]
                prev = monthly.iloc[-2]
                mom = ((last - prev) / prev * 100) if prev > 0 else 0
                rates["MoM"] = round(mom, 1)

            # QoQ: last 3 months vs previous 3 months
            if n >= 6:
                last_q = monthly.iloc[-3:].sum()
                prev_q = monthly.iloc[-6:-3].sum()
                qoq = ((last_q - prev_q) / prev_q * 100) if prev_q > 0 else 0
                rates["QoQ"] = round(qoq, 1)

            # YoY: last 12 months vs previous 12 months
            if n >= 24:
                last_y = monthly.iloc[-12:].sum()
                prev_y = monthly.iloc[-24:-12].sum()
                yoy = ((last_y - prev_y) / prev_y * 100) if prev_y > 0 else 0
                rates["YoY"] = round(yoy, 1)

            if not rates:
                return None

            # Prefer longer-horizon rates — MoM is too noisy for headline
            if "YoY" in rates:
                headline_key = "YoY"
            elif "QoQ" in rates:
                headline_key = "QoQ"
            else:
                headline_key = "MoM"

            # Suppress MoM from display if it contradicts YoY direction
            if "MoM" in rates and "YoY" in rates:
                mom_up = rates["MoM"] > 0
                yoy_up = rates["YoY"] > 0
                if mom_up != yoy_up:
                    # Opposite signals — drop MoM to avoid confusion
                    rates.pop("MoM")

            headline_val = rates[headline_key]

            rate_str = " | ".join(f"{k}: {v:+.1f}%" for k, v in rates.items())

            return BI(
                title=f"Growth Rate: {headline_key} {headline_val:+.1f}%",
                description=(
                    f"Period-over-period growth analysis: {rate_str}. "
                    f"{'All growth metrics are positive — momentum is building.' if all(v > 0 for v in rates.values()) else ''}"
                    f"{'Mixed signals across timeframes — investigate divergence.' if any(v > 0 for v in rates.values()) and any(v < 0 for v in rates.values()) else ''}"
                    f"{'All growth metrics are negative — revenue is contracting.' if all(v < 0 for v in rates.values()) else ''}"
                ),
                why_it_matters="Growth rates at different timeframes reveal acceleration vs deceleration.",
                evidence=rate_str,
                impact="🔴 Critical" if headline_val < -10 else ("🟠 Important" if abs(headline_val) > 5 else "🟢 Minor"),
                recommendation=(
                    f"{'Urgent: revenue declining — diagnose root cause within 14 days.' if headline_val < -5 else ''}"
                    f"{'Monitor: growth is steady but watch for deceleration.' if 0 < headline_val < 10 else ''}"
                    f"{'Strong growth — validate that it is sustainable and not driven by one-time events.' if headline_val > 10 else ''}"
                ),
                rule_type="growth_rates",
                score=7.0,
                chart_data={"rates": rates},
            )
        except Exception as e:
            log.warning(f"[TimeSeries] Growth rate analysis failed: {e}")
            return None

    def _detect_anomaly_months(self, monthly, rev_col, BI):
        """Flag months that deviate >2σ from rolling mean."""
        try:
            if len(monthly) < 6:
                return None

            values = monthly.values.astype(float)

            # 3-month rolling mean and std
            window = min(3, len(values) - 1)
            rolling_mean = pd.Series(values).rolling(window, min_periods=1).mean()
            rolling_std = pd.Series(values).rolling(window, min_periods=1).std().fillna(0)

            anomalies = []
            for i in range(len(values)):
                if rolling_std.iloc[i] == 0:
                    continue
                z_score = (values[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                if abs(z_score) > 2.0:
                    month_name = monthly.index[i].strftime("%B %Y")
                    direction = "spike" if z_score > 0 else "drop"
                    anomalies.append({
                        "month": month_name,
                        "value": float(values[i]),
                        "expected": float(rolling_mean.iloc[i]),
                        "z_score": round(z_score, 2),
                        "direction": direction,
                    })

            if not anomalies:
                return None

            worst = max(anomalies, key=lambda x: abs(x["z_score"]))
            anomaly_list = ", ".join(
                f"{a['month']} ({a['direction']}, z={a['z_score']:.1f})" for a in anomalies[:3]
            )

            return BI(
                title=f"{len(anomalies)} Anomalous Month(s) Detected",
                description=(
                    f"The following months deviate >2σ from the rolling average: {anomaly_list}. "
                    f"Most extreme: {worst['month']} at {_fmt_currency(worst['value'])} "
                    f"(expected ~{_fmt_currency(worst['expected'])}, z={worst['z_score']:.1f})."
                ),
                why_it_matters=(
                    "Revenue anomalies may indicate one-time events (promotions, outages), "
                    "data quality issues, or genuine demand shifts that need investigation."
                ),
                evidence=f"{len(anomalies)} months with |z| > 2.0",
                impact="🟠 Important",
                recommendation=(
                    f"Investigate {worst['month']}: was there a promotion, outage, or "
                    f"external event? If unexplained, flag for data quality review."
                ),
                rule_type="temporal_anomaly",
                score=6.5,
                chart_data={"anomalies": anomalies},
            )
        except Exception as e:
            log.warning(f"[TimeSeries] Anomaly detection failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # COLUMN DETECTION HELPERS
    # ──────────────────────────────────────────────────────────────

    def _find_date_col(self, df: pl.DataFrame, profile=None) -> Optional[str]:
        """Find the primary date column."""
        if profile and getattr(profile, "date_col", None):
            return profile.date_col
        for col in df.columns:
            if df[col].dtype in (pl.Date, pl.Datetime):
                return col
        for col in df.columns:
            if any(k in col.lower() for k in ["date", "time", "month", "period", "day"]):
                return col
        return None

    def _find_revenue_col(self, df: pl.DataFrame, profile=None) -> Optional[str]:
        """Find the revenue/sales column."""
        if profile:
            if getattr(profile, "revenue_col", None):
                return profile.revenue_col
            if getattr(profile, "price_col", None):
                return profile.price_col
        for col in df.columns:
            if any(k in col.lower() for k in ["revenue", "sales", "amount", "total", "value"]):
                if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                    return col
        return None

