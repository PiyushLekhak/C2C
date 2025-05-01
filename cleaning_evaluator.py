import numpy as np
from event_logger import get_logger

logger = get_logger("cleaning_evaluator")


def flatten_skew(skew_dict):
    """Calculate average skewness magnitude and max skew for summary."""
    values = [abs(v) for v in skew_dict.values() if isinstance(v, (int, float))]
    if not values:
        return {"mean_abs_skew": 0.0, "max_skew": 0.0}
    return {
        "mean_abs_skew": round(np.mean(values), 3),
        "max_skew": round(np.max(values), 3),
    }


def evaluate_cleaning(profile_before, profile_after):
    """
    Compare two profiling summaries and compute data-centric improvement metrics.
    Returns a dictionary of key metrics for evaluation, adaptation, and reporting.
    """
    logger.info("🔍 Evaluating cleaning effectiveness...")

    # === Missing Values ===
    missing_before = profile_before.get("missing_values", {})
    missing_after = profile_after.get("missing_values", {})
    total_missing_before = sum(missing_before.values())
    total_missing_after = sum(missing_after.values())

    # Use shape to get total fields
    rows, cols = profile_before.get("profile_shape", (1, 1))  # fallback
    total_fields_before = rows * cols

    missing_pct_before = round(total_missing_before / total_fields_before, 4)
    missing_pct_after = round(total_missing_after / total_fields_before, 4)

    # === Duplicates ===
    duplicates_before = profile_before.get("duplicate_rows", 0)
    duplicates_after = profile_after.get("duplicate_rows", 0)
    duplicates_removed = duplicates_before - duplicates_after

    # === Outliers ===
    outlier_info_before = profile_before.get("outliers", {})
    outlier_info_after = profile_after.get("outliers", {})
    outliers_before = sum([v.get("count", 0) for v in outlier_info_before.values()])
    outliers_after = sum([v.get("count", 0) for v in outlier_info_after.values()])

    # === Skewness ===
    skew_before = profile_before.get("skewness", {})
    skew_after = profile_after.get("skewness", {})
    skew_stats_before = flatten_skew(skew_before)
    skew_stats_after = flatten_skew(skew_after)

    evaluation_summary = {
        "missing_pct_before": missing_pct_before,
        "missing_pct_after": missing_pct_after,
        "missing_pct_relative_reduction": round(
            (missing_pct_before - missing_pct_after) / (missing_pct_before + 1e-8), 4
        ),
        "duplicates_removed": duplicates_removed,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
        "outliers_change": outliers_before - outliers_after,
        "skew_mean_before": skew_stats_before["mean_abs_skew"],
        "skew_mean_after": skew_stats_after["mean_abs_skew"],
        "skew_mean_change": round(
            skew_stats_before["mean_abs_skew"] - skew_stats_after["mean_abs_skew"], 4
        ),
        "skew_max_before": skew_stats_before["max_skew"],
        "skew_max_after": skew_stats_after["max_skew"],
        "skew_max_change": round(
            skew_stats_before["max_skew"] - skew_stats_after["max_skew"], 4
        ),
    }

    logger.info("✅ Cleaning evaluation complete.")
    return evaluation_summary
