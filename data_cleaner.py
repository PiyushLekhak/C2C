import pandas as pd
import numpy as np
import warnings


def clean_missing_values(df, strategy="mean", threshold=0.5):
    """
    Handles missing values by:
    - Dropping columns with too many missing values
    - Imputing remaining missing values using the chosen strategy

    Args:
        df (pd.DataFrame): The input DataFrame
        strategy (str): 'mean', 'median', or 'mode'
        threshold (float): Columns with missing ratio > threshold will be dropped

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()
    total_rows = len(df)

    # Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    df.drop(columns=missing_frac[missing_frac > threshold].index, inplace=True)

    # Impute remaining
    for col in df.columns:
        num_missing = df[col].isnull().sum()
        if num_missing > 0:
            if np.issubdtype(df[col].dtype, np.number):
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

            # Warn if a large fraction is imputed
            if num_missing / total_rows > 0.5:
                warnings.warn(f"⚠️ Over 50% of values in column '{col}' were imputed.")

    return df


def clean_duplicates(df):
    return df.drop_duplicates()


def clean_outliers(df, profiling_results, method="remove"):
    """
    Handles outliers using profiling results.
    Can either remove outlier rows or cap outlier values.

    Args:
        df (pd.DataFrame): The input DataFrame
        profiling_results (dict): Dictionary with outlier info per column
        method (str): 'remove' to delete rows, 'cap' to cap values

    Returns:
        pd.DataFrame: Outlier-handled DataFrame
    """
    df = df.copy()
    for col, info in profiling_results.items():
        if info["count"] == 0:
            continue

        if info["method"] == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            higher_fence = Q3 + 1.5 * IQR
        else:  # Z-score
            mean = df[col].mean()
            std = df[col].std()
            lower_fence = mean - 3 * std
            higher_fence = mean + 3 * std

        if method == "remove":
            df = df[(df[col] >= lower_fence) & (df[col] <= higher_fence)]
        elif method == "cap":
            df[col] = np.where(
                df[col] < lower_fence,
                lower_fence,
                np.where(df[col] > higher_fence, higher_fence, df[col]),
            )

    return df


def fix_inconsistencies(df, inconsistencies):
    """
    Standardizes string-based categorical values by:
    - Lowercasing
    - Stripping whitespace

    Args:
        df (pd.DataFrame): The input DataFrame
        inconsistencies (dict): Output from profiler indicating affected columns

    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    df = df.copy()
    for col in inconsistencies:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def clean_data(
    df, profiling_report, strategy="mean", missing_thresh=0.5, outlier_method="remove"
):
    print("🧼 Starting full cleaning pipeline...")

    df = clean_missing_values(df, strategy=strategy, threshold=missing_thresh)
    print("✅ Missing values cleaned.")

    df = clean_duplicates(df)
    print("✅ Duplicates removed.")

    df = clean_outliers(df, profiling_report["outliers"], method=outlier_method)
    print(f"✅ Outliers handled with method: {outlier_method}")

    df = fix_inconsistencies(df, profiling_report["inconsistencies"])
    print("✅ String inconsistencies fixed.")

    print("🎉 Data cleaning complete.")
    return df
