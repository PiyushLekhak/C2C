import pandas as pd
import numpy as np
from event_logger import get_logger

logger = get_logger("data_cleaner")


def impute_missing_values(df, strategy="mean", threshold=0.5, ranked_features=None):

    df = df.copy()
    total_rows = len(df)
    cleaning_summary = {"dropped_columns": [], "imputed": {}}

    missing_frac = df.isnull().mean()
    dropped = missing_frac[missing_frac > threshold].index.tolist()
    df.drop(columns=dropped, inplace=True)
    cleaning_summary["dropped_columns"] = dropped

    if dropped:
        logger.info(f"Dropped columns with >{threshold*100:.0f}% missing: {dropped}")

    columns = ranked_features if ranked_features else df.columns
    for col in columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            if np.issubdtype(df[col].dtype, np.number):
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    df[col] = df[col].fillna(df[col].median())
                    method = "median (due to high skew)"
                else:
                    df[col] = df[col].fillna(df[col].mean())
                    method = "mean"
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                method = "mode"

            cleaning_summary["imputed"][col] = method
            logger.info(f"Imputed missing values in column '{col}' using {method}")

    return df, cleaning_summary


def clean_duplicates(df):
    df = df.copy()
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    removed = before - after

    logger.info(f"Removed {removed} duplicate rows")

    return df_clean, removed


def clean_outliers(df, profiling_results, method="remove", ranked_features=None):
    df = df.copy()
    columns = ranked_features if ranked_features else profiling_results.keys()
    outlier_cols_handled = []

    for col in columns:
        if col not in profiling_results or col not in df.columns:
            continue

        info = profiling_results[col]
        if info["count"] == 0:
            continue

        if info["method"] == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_fence = Q1 - 1.5 * IQR
            higher_fence = Q3 + 1.5 * IQR
        else:
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

        outlier_cols_handled.append(col)

    logger.info(
        f"Handled outliers in columns: {outlier_cols_handled} using method: {method}"
    )

    return df, {"outlier_method": method, "columns": outlier_cols_handled}


def fix_inconsistencies(df, inconsistencies, ranked_features=None):
    df = df.copy()
    columns = ranked_features if ranked_features else inconsistencies.keys()
    fixed_cols = []

    for col in columns:
        if col in inconsistencies and col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            fixed_cols.append(col)

    if fixed_cols:
        logger.info(f"Fixed string inconsistencies in columns: {fixed_cols}")

    return df, fixed_cols


def clean_data(
    df,
    profiling_report,
    strategy="mean",
    missing_thresh=0.5,
    outlier_method="remove",
    ranked_features=None,
):
    """
    Full prioritized data cleaning pipeline.
    Returns both cleaned dataframe and a cleaning summary dictionary.
    """
    logger.info("🧼 Starting prioritized cleaning pipeline...")

    # Step-by-step cleaning
    df, missing_summary = impute_missing_values(
        df, strategy=strategy, threshold=missing_thresh, ranked_features=ranked_features
    )
    df, duplicates_removed = clean_duplicates(df)
    df, outlier_summary = clean_outliers(
        df,
        profiling_report["outliers"],
        method=outlier_method,
        ranked_features=ranked_features,
    )
    df, inconsistency_summary = fix_inconsistencies(
        df, profiling_report["inconsistencies"], ranked_features=ranked_features
    )

    # Final safety check: Ensure no numeric NaNs left
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if df[numeric_cols].isnull().any().any():
        logger.info(
            "✅ Filling any remaining NaNs in numeric columns with mean (final step)."
        )
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    logger.info("🎉 Prioritized data cleaning complete.")

    cleaning_summary = {
        "missing_handling": missing_summary,
        "duplicates_removed": duplicates_removed,
        "outlier_handling": outlier_summary,
        "inconsistencies_fixed": inconsistency_summary,
    }

    return df, cleaning_summary
