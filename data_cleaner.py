import pandas as pd
import numpy as np
from event_logger import get_logger

logger = get_logger("data_cleaner")


def impute_missing_values(df, threshold=0.5, imputation_strategy=None):
    df = df.copy()
    cleaning_summary = {
        "dropped_columns": [],
        "imputed_values": {},
        "imputation_counts": {},
    }

    # === Drop columns with too much missing data ===
    missing_frac = df.isnull().mean()
    dropped = missing_frac[missing_frac > threshold].index.tolist()
    df.drop(columns=dropped, inplace=True)
    cleaning_summary["dropped_columns"] = dropped

    if dropped:
        logger.info(f"🗑️ Dropped columns with >{threshold*100:.0f}% missing: {dropped}")

    # === Impute remaining missing values ===
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # Skip columns with no missing values

        count = df[col].isnull().sum()  # Number of missing values

        if np.issubdtype(df[col].dtype, np.number):  # Numeric columns
            skewness = df[col].skew()

            # If imputation strategy is provided, use it
            if imputation_strategy == "median":
                impute_val = df[col].median()
                method = "median (from policy)"
            elif imputation_strategy == "mean":
                impute_val = df[col].mean()
                method = "mean (from policy)"
            else:
                # Fallback rule-based imputation based on skewness
                if abs(skewness) > 1:
                    impute_val = df[col].median()
                    method = "median (skew-aware)"
                else:
                    impute_val = df[col].mean()
                    method = "mean (skew-aware)"
        else:  # Non-numeric columns (categorical)
            impute_val = df[col].mode()[0]
            method = "mode"

        # Impute missing values and record the summary
        df[col] = df[col].fillna(impute_val)
        cleaning_summary["imputed_values"][col] = method
        cleaning_summary["imputation_counts"][col] = int(count)

        logger.info(f"🧩 Imputed {count} missing values in '{col}' using {method}")

    return df, cleaning_summary


def clean_duplicates(df):
    df = df.copy()
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    removed = before - after
    logger.info(f"🚿 Removed {removed} duplicate rows")
    return df_clean, removed


def clean_outliers(df, profiling_results, method="cap", skip_cols=None):
    """
    Handles outliers by either capping, removing, or skipping.
    Returns:
        - df_cleaned: DataFrame after outlier treatment
        - summary: dict with:
            - outlier_method: "cap", "remove", or "skip"
            - columns: list of columns processed
            - capped_counts (if cap): dict of {col: # capped}
            - rows_removed (if remove): total rows dropped
    """
    if method == "skip":
        logger.info("⏭️ Outlier handling skipped by policy.")
        return df.copy(), {
            "outlier_method": "skip",
            "columns": [],
        }

    df_copy = df.copy()
    outlier_cols_handled = []
    capped_counts = {}
    skip_cols = set(skip_cols or [])

    initial_rows = len(df_copy)

    for col, info in profiling_results.items():
        if col not in df_copy.columns or info.get("count", 0) == 0:
            continue
        if col in skip_cols:
            logger.info(f"⏭️ Skipping outlier treatment for column: {col}")
            continue

        # Determine cutoffs
        if info["method"] == "IQR":
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:  # Z-score
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            lower, upper = mean - 3 * std, mean + 3 * std

        if method == "cap":
            original = df_copy[col].copy()
            df_copy[col] = df_copy[col].clip(lower, upper)
            capped_counts[col] = int((original != df_copy[col]).sum())
        elif method == "remove":
            df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]

        outlier_cols_handled.append(col)

    summary = {
        "outlier_method": method,
        "columns": outlier_cols_handled,
    }

    if method == "cap":
        summary["capped_counts"] = capped_counts
    elif method == "remove":
        final_rows = len(df_copy)
        summary["rows_removed"] = initial_rows - final_rows

    logger.info(
        f"Outlier handling ({method}) done on {outlier_cols_handled}."
        + (
            f" Capped: {capped_counts}"
            if method == "cap"
            else f" Rows removed: {summary.get('rows_removed', 0)}"
        )
    )

    return df_copy, summary


def fix_inconsistencies(df, inconsistencies):
    df = df.copy()
    fixed_cols = []

    for col in inconsistencies:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            fixed_cols.append(col)

    if fixed_cols:
        logger.info(f"🔠 Fixed string inconsistencies in: {fixed_cols}")

    return df, fixed_cols


def clean_data(
    df,
    profiling_report,
    missing_thresh=0.5,
    outlier_method="cap",
    imputation_strategy=None,
):
    """
    Full data cleaning pipeline for missing, duplicate, outlier, and inconsistency handling.
    Returns cleaned dataframe and structured summary.
    """
    logger.info("🧼 Starting data cleaning pipeline...")

    # Step 1: Impute missing values (after dropping high-missing columns)
    df, missing_summary = impute_missing_values(
        df, threshold=missing_thresh, imputation_strategy=imputation_strategy
    )

    # Step 2: Remove duplicates
    df, duplicates_removed = clean_duplicates(df)

    # Step 3: Handle outliers
    df, outlier_summary = clean_outliers(
        df,
        profiling_report.get("outliers", {}),
        method=outlier_method,
        skip_cols=profiling_report.get("outlier_skip_cols", []),
    )

    # Step 4: Fix categorical inconsistencies
    df, inconsistencies_fixed = fix_inconsistencies(
        df,
        profiling_report.get("inconsistencies", {}),
    )

    # Step 5: Ensure no remaining numeric NaNs (last safety)
    num_cols = df.select_dtypes(include="number").columns
    remaining_nans = df[num_cols].isnull().sum()

    for col, count in remaining_nans.items():
        if count > 0:
            skew = df[col].dropna().skew()  # Calculate skewness
            if abs(skew) > 1:  # If skew is significant, use median
                fill_val = df[col].median()
                method = "median (skew-aware final)"
            else:  # Otherwise, use mean
                fill_val = df[col].mean()
                method = "mean (final)"
            df[col].fillna(fill_val, inplace=True)
            logger.info(f"🧯 Final fill of {count} NaNs in '{col}' using {method}")

    cleaning_summary = {
        "missing_handling": missing_summary,
        "duplicates_removed": duplicates_removed,
        "outlier_handling": outlier_summary,
        "inconsistencies_fixed": inconsistencies_fixed,
        "final_shape": df.shape,
    }

    logger.info("✅ Data cleaning pipeline complete.")
    return df, cleaning_summary
