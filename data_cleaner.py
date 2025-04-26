import pandas as pd
import numpy as np
import warnings
import logging
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
    logger.log(
        logging.INFO, f"Dropped columns with >{threshold*100:.0f}% missing: {dropped}"
    )

    columns = ranked_features if ranked_features else df.columns
    for col in columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            if np.issubdtype(df[col].dtype, np.number):
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                    method = "mean"
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                    method = "median"
                else:
                    method = "unknown"
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                method = "mode"

            cleaning_summary["imputed"][col] = method
            logger.log(
                logging.INFO, f"Imputed missing values in column '{col}' using {method}"
            )

    return df, cleaning_summary


def clean_duplicates(df):
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    removed = before - after
    logger.log(logging.INFO, f"Removed {removed} duplicate rows")
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

    logger.log(
        logging.INFO,
        f"Handled outliers in columns: {outlier_cols_handled} using method: {method}",
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

    logger.log(logging.INFO, "🎉 Prioritized data cleaning complete.")

    return df, fixed_cols


def clean_data(
    df,
    profiling_report,
    strategy="mean",
    missing_thresh=0.5,
    outlier_method="remove",
    ranked_features=None,
):
    logger.info("🧼 Starting prioritized cleaning pipeline...")

    summary = {}

    df, missing_summary = impute_missing_values(
        df, strategy=strategy, threshold=missing_thresh, ranked_features=ranked_features
    )
    summary["missing_handling"] = missing_summary

    df, duplicates_removed = clean_duplicates(df)
    summary["duplicates_removed"] = duplicates_removed

    df, outlier_summary = clean_outliers(
        df,
        profiling_report["outliers"],
        method=outlier_method,
        ranked_features=ranked_features,
    )
    summary["outlier_handling"] = outlier_summary

    df, fixed_inconsistencies = fix_inconsistencies(
        df, profiling_report["inconsistencies"], ranked_features=ranked_features
    )
    summary["inconsistencies_fixed"] = fixed_inconsistencies

    logger.info("🎉 Prioritized data cleaning complete.")
    return df, summary
