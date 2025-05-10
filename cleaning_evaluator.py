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
    Now also returns the original dataset shape so the adaptive controller
    can compute truly dynamic thresholds.
    """
    logger.info("🔍 Evaluating cleaning effectiveness...")

    # === Missing Values ===
    missing_before = profile_before.get("missing_values", {})
    missing_after = profile_after.get("missing_values", {})
    total_missing_before = sum(missing_before.values())
    total_missing_after = sum(missing_after.values())

    # Use the original shape to compute percentages
    rows, cols = profile_before.get("profile_shape", (1, 1))
    total_fields = rows * cols

    missing_pct_before = round(total_missing_before / total_fields, 4)
    missing_pct_after = round(total_missing_after / total_fields, 4)

    # === Duplicates ===
    dup_before = profile_before.get("duplicate_rows", 0)
    dup_after = profile_after.get("duplicate_rows", 0)
    duplicates_removed = dup_before - dup_after

    # === Outliers ===
    info_before = profile_before.get("outliers", {})
    info_after = profile_after.get("outliers", {})
    outliers_before = sum(v.get("count", 0) for v in info_before.values())
    outliers_after = sum(v.get("count", 0) for v in info_after.values())

    # === Skewness ===
    skew_before = profile_before.get("skewness", {})
    skew_after = profile_after.get("skewness", {})
    stats_before = flatten_skew(skew_before)
    stats_after = flatten_skew(skew_after)

    # === Build the summary dict ===
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
        "skew_mean_before": stats_before["mean_abs_skew"],
        "skew_mean_after": stats_after["mean_abs_skew"],
        "skew_mean_change": round(
            stats_before["mean_abs_skew"] - stats_after["mean_abs_skew"], 4
        ),
        "skew_max_before": stats_before["max_skew"],
        "skew_max_after": stats_after["max_skew"],
        "skew_max_change": round(stats_before["max_skew"] - stats_after["max_skew"], 4),
        "profile_shape": (rows, cols),
    }

    logger.info("✅ Cleaning evaluation complete.")
    return evaluation_summary
