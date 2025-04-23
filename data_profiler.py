import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import zscore

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")


def detect_missing_values_count(df):
    return df.isnull().sum()


def detect_duplicates(df):
    return df[df.duplicated()]


def detect_outliers(df):
    """
    Detects outliers in numerical columns using IQR or Z-score based on skewness.

    For each numeric column:
        - If skewness > 1 or < -1, uses the IQR method.
        - Otherwise, uses the Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric columns.

    Returns:
        dict: A dictionary where keys are column names and values are dictionaries
                with outlier count and the method used.
        Format: { 'column_name': { 'count': int, 'method': str } }
    """
    outlier_info = {}
    numeric_df = df.select_dtypes(include=[np.number])

    for column in numeric_df.columns:
        col_data = numeric_df[column].dropna()
        skew = col_data.skew()

        if abs(skew) > 1:
            # Use IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            mask = (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
            method = "IQR"
        else:
            # Use Z-score
            z_scores = zscore(col_data)
            mask = np.abs(z_scores) > 3
            method = "Z-score"

        outlier_info[column] = {
            "count": int(mask.sum()),
            "method": method,
        }

    return outlier_info


def check_inconsistencies(df, category_threshold=10):
    """
    Detects inconsistent categorical values in a DataFrame based on case and whitespace differences.

    Args:
        df (pd.DataFrame): The input DataFrame to check.
        category_threshold (int, optional): Maximum number of unique values in a column
            to consider it potentially categorical. Defaults to 10.

    Returns:
        dict: A dictionary where keys are column names with inconsistencies,
            and values are lists of original unique values in those columns.
    """
    inconsistencies = {}
    categorical = df.select_dtypes(include=["object", "category"])

    for col in categorical.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) <= category_threshold:
            normalized_values = [str(v).strip().lower() for v in unique_values]
            if len(set(normalized_values)) != len(unique_values):
                inconsistencies[col] = list(unique_values)

    return inconsistencies


def generate_summary_statistics(df):
    return df.describe(include="all")


def visualize_distributions(df, save_path="plots"):
    os.makedirs(save_path, exist_ok=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{col}_distribution.png"))
        plt.close()


def profile_data(df):
    print("🔍 Profiling started...\n")

    missing = detect_missing_values_count(df)
    print("✅ Missing values detected.")

    duplicates = detect_duplicates(df)
    print("✅ Duplicates detected.")

    outliers = detect_outliers(df)
    print("✅ Outliers flagged using IQR/Z-score based on skewness.")

    inconsistencies = check_inconsistencies(df)
    print("✅ Categorical inconsistencies flagged.")

    stats = generate_summary_statistics(df)
    print("✅ Summary statistics generated.")

    visualize_distributions(df)
    print("✅ Distributions visualized and saved in /plots folder.\n")

    print("✅ Profiling complete.")

    return {
        "missing_values": missing,
        "duplicate_rows": len(duplicates),
        "outliers": outliers,
        "inconsistencies": inconsistencies,
        "summary_stats": stats,
    }
