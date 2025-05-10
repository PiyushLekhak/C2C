import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from event_logger import get_logger

logger = get_logger("data_profiler")
KURTOSIS_THRESHOLD = 3  # Used to decide between IQR vs Z-score for outlier detection


def profile_data(df, save_path="plots", target_column=None):
    logger.info("🔍 Profiling started...")
    os.makedirs(save_path, exist_ok=True)

    # === 1. Missing Values ===
    missing_values = df.isnull().sum().to_dict()
    missing_pct = df.isnull().mean().to_dict()
    total_missing_pct = round(df.isnull().sum().sum() / df.size, 4)

    # === 2. Duplicates ===
    duplicate_rows = int(df.duplicated().sum())

    # === 3. Data Types ===
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # === 4. Outlier Detection ===
    outliers = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        kurt = series.kurtosis()
        if kurt > KURTOSIS_THRESHOLD:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            method = "IQR"
        else:
            mean, std = series.mean(), series.std()
            lower, upper = mean - 3 * std, mean + 3 * std
            method = "zscore"
        count = int(((series < lower) | (series > upper)).sum())
        outliers[col] = {"method": method, "count": count}

    total_outliers = sum([v["count"] for v in outliers.values()])

    # === 5. Inconsistencies in Categorical Columns ===
    inconsistencies = {}
    for col in cat_cols:
        vals = df[col].dropna().astype(str).unique()
        cleaned = [v.strip().lower() for v in vals]
        if len(set(cleaned)) < len(vals):
            inconsistencies[col] = list(vals)

    # === 6. Skewness of Numeric Columns ===
    skewness = df[numeric_cols].skew().to_dict()

    # === 7. Correlation Heatmap ===
    avg_abs_skew = np.mean([abs(v) for v in skewness.values()])
    correlation_method = "spearman" if avg_abs_skew > 0.8 else "pearson"
    corr = df[numeric_cols].corr(method=correlation_method)
    heatmap_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title(f"Correlation Heatmap ({correlation_method.capitalize()})")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    heatmap_path = os.path.relpath(heatmap_path, start="reports")

    # === 8. Top Correlations ===
    corr_pairs = corr.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    top_pairs = corr_pairs.sort_values(ascending=False).drop_duplicates().head(5)
    top_correlations = [
        {"Feature 1": i, "Feature 2": j, "Correlation": round(v, 3)}
        for (i, j), v in top_pairs.items()
    ]

    # === 9. Class Distribution (if target is given) ===
    target_distribution_path = None
    if target_column and target_column in df.columns:
        series = df[target_column].dropna()
        counts = series.value_counts()
        if series.dtype.name in ["object", "category"]:
            plt.figure(figsize=(6, 6))
            counts.plot.pie(
                autopct="%1.1f%%", startangle=90, explode=[0.05] * len(counts)
            )
            plt.title(f"Distribution of {target_column}")
            plt.ylabel("")
            target_path = os.path.join(save_path, "target_distribution.png")
            plt.tight_layout()
            plt.savefig(target_path)
            plt.close()
            target_distribution_path = os.path.relpath(target_path, start="reports")

    # === 10. Summary Stats ===
    summary_stats = df.describe(include="all").to_dict()

    # === 11. Final Packaging ===
    profiling_report = {
        "missing_values": missing_values,
        "missing_pct": missing_pct,
        "total_missing_pct": total_missing_pct,
        "duplicate_rows": duplicate_rows,
        "outliers": outliers,
        "total_outliers": total_outliers,
        "inconsistencies": inconsistencies,
        "skewness": skewness,
        "summary_stats": summary_stats,
        "correlation_heatmap_path": heatmap_path,
        "top_correlations": top_correlations,
        "target_distribution_path": target_distribution_path,
        "profile_shape": df.shape,
    }

    logger.info("✅ Profiling complete.")
    return profiling_report
