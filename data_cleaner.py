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


def clean_outliers(df, profiling_results, method="cap"):
    """
    Handles outliers by either capping or removing rows.
    Returns:
        - df_cleaned: DataFrame after outlier treatment
        - summary: dict with
            - outlier_method: "cap" or "remove"
            - columns: list of columns processed
            - capped_counts (if cap): dict of {col: # capped}
            - rows_removed (if remove): total rows dropped
    """
    df_copy = df.copy()
    outlier_cols_handled = []
    capped_counts = {}

    # === If remove mode, capture initial row count ===
    initial_rows = len(df_copy)

    for col, info in profiling_results.items():
        if col not in df_copy.columns or info.get("count", 0) == 0:
            continue

        # Determine cutoffs
        if info["method"] == "IQR":
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        else:  # "zscore"
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            lower, upper = mean - 3 * std, mean + 3 * std

        # Apply chosen method
        if method == "cap":
            # Clip values and count how many changed
            original = df_copy[col].copy()
            df_copy[col] = df_copy[col].clip(lower, upper)
            capped_counts[col] = int((original != df_copy[col]).sum())

        else:  # method == "remove"
            # Filter rows, will drop any out-of-bounds
            df_copy = df_copy[(df_copy[col] >= lower) & (df_copy[col] <= upper)]

        outlier_cols_handled.append(col)

    # === Build summary ===
    summary = {
        "outlier_method": method,
        "columns": outlier_cols_handled,
    }

    if method == "cap":
        summary["capped_counts"] = capped_counts
    else:  # remove
        final_rows = len(df_copy)
        summary["rows_removed"] = initial_rows - final_rows

    logger.info(
        f"Outlier handling ({method}) done on {outlier_cols_handled}."
        + (
            f" Capped: {capped_counts}"
            if method == "cap"
            else f" Rows removed: {summary['rows_removed']}"
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
    )

    # Step 4: Fix categorical inconsistencies
    df, inconsistencies_fixed = fix_inconsistencies(
        df,
        profiling_report.get("inconsistencies", {}),
    )

    # Step 5: Ensure no remaining numeric NaNs (last safety)
    num_cols = df.select_dtypes(include="number").columns
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        logger.info("🧯 Final fill of remaining numeric NaNs with column means.")

    cleaning_summary = {
        "missing_handling": missing_summary,
        "duplicates_removed": duplicates_removed,
        "outlier_handling": outlier_summary,
        "inconsistencies_fixed": inconsistencies_fixed,
        "final_shape": df.shape,
    }

    logger.info("✅ Data cleaning pipeline complete.")
    return df, cleaning_summary
