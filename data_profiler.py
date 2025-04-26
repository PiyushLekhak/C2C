import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import zscore

warnings.filterwarnings("ignore")


def detect_missing_values_count(df):
    return df.isnull().sum().to_dict()


def detect_duplicates(df):
    return df[df.duplicated()]


def detect_outliers(df):
    outlier_info = {}
    numeric_df = df.select_dtypes(include=[np.number])

    for column in numeric_df.columns:
        col_data = numeric_df[column].dropna()
        skew = col_data.skew()

        if abs(skew) > 1:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            mask = (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
            method = "IQR"
        else:
            z_scores = zscore(col_data)
            mask = np.abs(z_scores) > 3
            method = "Z-score"

        outlier_info[column] = {
            "count": int(mask.sum()),
            "method": method,
        }

    return outlier_info


def check_inconsistencies(df, category_threshold=10):
    inconsistencies = {}
    categorical = df.select_dtypes(include=["object", "category"])

    for col in categorical.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= category_threshold:
            normalized = [str(v).strip().lower() for v in unique_values]
            if len(set(normalized)) != len(unique_values):
                inconsistencies[col] = list(unique_values)

    return inconsistencies


def generate_summary_statistics(df):
    return df.describe(include="all").to_dict()


def visualize_distributions(df, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)
    saved_plots = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")

        plt.tight_layout()
        plot_path = os.path.join(save_path, f"{col}_distribution.png")
        plt.savefig(plot_path)
        plt.close()

        saved_plots.append(plot_path)

    return saved_plots


def visualize_correlation_heatmap(df, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plot_path = os.path.join(save_path, "correlation_heatmap.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def profile_data(df, save_path="plots"):
    print("🔍 Profiling started...\n")

    missing = detect_missing_values_count(df)
    print("✅ Missing values detected.")

    duplicates = detect_duplicates(df)
    print("✅ Duplicates detected.")

    outliers = detect_outliers(df)
    print("✅ Outliers flagged.")

    inconsistencies = check_inconsistencies(df)
    print("✅ Inconsistencies checked.")

    stats = generate_summary_statistics(df)
    print("✅ Summary stats done.")

    dist_plots = visualize_distributions(df, save_path)
    print("✅ Distribution plots saved.")

    corr_plot = visualize_correlation_heatmap(df, save_path)
    print("✅ Correlation heatmap saved.\n")

    print("✅ Profiling complete.")

    return {
        "missing_values": missing,
        "duplicate_rows": len(duplicates),
        "outliers": outliers,
        "inconsistencies": inconsistencies,
        "summary_stats": stats,
        "distribution_plots": dist_plots,
        "correlation_plot": corr_plot,
    }
