import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from event_logger import get_logger

logger = get_logger("data_profiler")
KURTOSIS_THRESHOLD = 0


def profile_data(df, save_path="plots", target_column=None):
    logger.info("🔍 Profiling started...")
    os.makedirs(save_path, exist_ok=True)

    # missing values and duplicates
    missing_values = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())

    # placeholders for profiling
    outliers = {}
    inconsistencies = {}
    summary_stats = df.describe(include="all").to_dict()

    # identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # detect outliers per column
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
            mu, sigma = series.mean(), series.std()
            lower, upper = mu - 3 * sigma, mu + 3 * sigma
            method = "zscore"
        count = int(((series < lower) | (series > upper)).sum())
        outliers[col] = {"method": method, "count": count}

    # detect string inconsistencies
    for col in cat_cols:
        vals = df[col].dropna().astype(str).unique()
        cleaned = [v.strip().lower() for v in vals]
        if len(set(cleaned)) < len(vals):
            inconsistencies[col] = list(vals)

    # correlation heatmap
    corr = df[numeric_cols].corr()
    heatmap_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    heatmap_path = os.path.relpath(heatmap_path, start="reports")

    # top correlations
    corr_pairs = corr.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    top_pairs = corr_pairs.sort_values(ascending=False).drop_duplicates().head(5)
    top_list = []
    for (c1, c2), val in top_pairs.items():
        top_list.append({"Feature 1": c1, "Feature 2": c2, "Correlation": val})

    # target distribution
    target_distribution_path = None
    if target_column and target_column in df.columns:
        if df[target_column].dtype.name in ["object", "category"]:
            counts = df[target_column].value_counts()
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

    profiling_report = {
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "outliers": outliers,
        "inconsistencies": inconsistencies,
        "summary_stats": summary_stats,
        "correlation_heatmap_path": heatmap_path,
        "top_correlations": top_list,
        "target_distribution_path": target_distribution_path,
    }

    logger.info("✅ Profiling complete.")
    return profiling_report
