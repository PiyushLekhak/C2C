import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from event_logger import get_logger

logger = get_logger("data_profiler")


def profile_data(df, save_path="plots", target_column=None):
    logger.info("🔍 Profiling started...")

    os.makedirs(save_path, exist_ok=True)

    missing_values = df.isnull().sum().to_dict()
    duplicates = df.duplicated().sum()

    outlier_summary = {}
    inconsistencies_summary = {}
    summary_stats = df.describe(include="all").to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # ✅ Correlation heatmap
    corr = df[numeric_cols].corr()
    correlation_heatmap_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(correlation_heatmap_path)
    plt.close()
    correlation_heatmap_path = os.path.relpath(
        correlation_heatmap_path, start="reports"
    )

    # ✅ Top correlation pairs (excluding self-correlations)
    corr_pairs = corr.abs().unstack()
    corr_pairs = corr_pairs[corr_pairs < 1.0]
    top_correlation_pairs = (
        corr_pairs.sort_values(ascending=False).drop_duplicates().head(5)
    )
    top_correlation_pairs = top_correlation_pairs.reset_index()
    top_correlation_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]

    # ✅ Target distribution pie chart (if target exists)
    target_pie_chart_path = None
    if target_column and target_column in df.columns:
        if (
            df[target_column].dtype == "object"
            or df[target_column].dtype.name == "category"
        ):
            target_counts = df[target_column].value_counts()
            plt.figure(figsize=(6, 6))
            target_counts.plot.pie(
                autopct="%1.1f%%", startangle=90, explode=[0.05] * len(target_counts)
            )
            plt.title(f"Distribution of {target_column}")
            plt.ylabel("")
            target_pie_chart_path = os.path.join(save_path, "target_distribution.png")
            plt.tight_layout()
            plt.savefig(target_pie_chart_path)
            plt.close()
            target_pie_chart_path = os.path.relpath(
                target_pie_chart_path, start="reports"
            )

    profiling_summary = {
        "missing_values": missing_values,
        "duplicate_rows": duplicates,
        "outliers": outlier_summary,
        "inconsistencies": inconsistencies_summary,
        "summary_stats": summary_stats,
        "correlation_heatmap_path": correlation_heatmap_path,
        "top_correlations": top_correlation_pairs.to_dict(orient="records"),
        "target_distribution_path": target_pie_chart_path,
    }

    logger.info("✅ Profiling complete.")
    return profiling_summary
