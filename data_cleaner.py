import pandas as pd
import numpy as np
import warnings
from event_logger import get_logger

# Initialize a module-specific logger
logger = get_logger(module_name="data_cleaner")


def impute_missing_values(df, strategy="mean", threshold=0.5, ranked_features=None):
    """
    Handles missing values by:
    - Dropping columns with too many missing values
    - Imputing remaining missing values using the chosen strategy
    - Prioritizing columns based on feature ranking (if provided)

    Args:
        df (pd.DataFrame): The input DataFrame
        strategy (str): 'mean', 'median', or 'mode'
        threshold (float): Columns with missing ratio > threshold will be dropped
        ranked_features (list): Optional list of features to prioritize cleaning

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()
    total_rows = len(df)

    missing_frac = df.isnull().mean()
    dropped = missing_frac[missing_frac > threshold].index.tolist()
    df.drop(columns=dropped, inplace=True)
    logger.log("info", f"Dropped columns with >{threshold*100:.0f}% missing: {dropped}")

    columns = ranked_features if ranked_features else df.columns
    for col in columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            if np.issubdtype(df[col].dtype, np.number):
                if strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

            frac = df[col].isnull().sum() / total_rows
            if frac > 0.5:
                warnings.warn(f"⚠️ Over 50% of values in column '{col}' were imputed.")
            logger.log("info", f"Imputed missing values in column '{col}'")

    return df


def clean_duplicates(df):
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    removed = before - after
    logger.log("info", f"Removed {removed} duplicate rows")
    return df_clean


def clean_outliers(df, profiling_results, method="remove", ranked_features=None):
    """
    Handles outliers using profiling results.
    Can either remove outlier rows or cap outlier values.
    Prioritizes columns based on feature ranking if provided.

    Args:
        df (pd.DataFrame): The input DataFrame
        profiling_results (dict): Dictionary with outlier info per column
        method (str): 'remove' to delete rows, 'cap' to cap values
        ranked_features (list): Optional list of features to prioritize cleaning

    Returns:
        pd.DataFrame: Outlier-handled DataFrame
    """
    df = df.copy()
    columns = ranked_features if ranked_features else profiling_results.keys()
    outlier_cols = []

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
        outlier_cols.append(col)

    logger.log(
        "info", f"Handled outliers in columns: {outlier_cols} using method: {method}"
    )
    return df


def fix_inconsistencies(df, inconsistencies, ranked_features=None):
    """
    Standardizes string-based categorical values by:
    - Lowercasing
    - Stripping whitespace
    - Prioritizing based on ranked features if provided

    Args:
        df (pd.DataFrame): The input DataFrame
        inconsistencies (dict): Output from profiler indicating affected columns
        ranked_features (list): Optional list of features to prioritize cleaning

    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    df = df.copy()
    columns = ranked_features if ranked_features else inconsistencies.keys()
    fixed_cols = []

    for col in columns:
        if col in inconsistencies and col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            fixed_cols.append(col)

    logger.log("info", f"Fixed string inconsistencies in columns: {fixed_cols}")
    return df


def clean_data(
    df,
    profiling_report,
    strategy="mean",
    missing_thresh=0.5,
    outlier_method="remove",
    ranked_features=None,
):
    """
    Full rule-based data cleaning pipeline with optional ranked feature prioritization.
    This includes:
    - Imputation of missing values
    - Duplicate removal
    - Outlier handling
    - String inconsistency fixing

    Args:
        df (pd.DataFrame): Raw dataset.
        profiling_report (dict): Output from data_profiler.
        strategy (str): Missing value imputation strategy.
        missing_thresh (float): Threshold to drop columns with too many missing values.
        outlier_method (str): 'remove' or 'cap' for outliers.
        ranked_features (list): Optional list of feature names in priority order.

    Returns:
        pd.DataFrame: Fully cleaned DataFrame
    """
    logger.log("info", "🧼 Starting prioritized cleaning pipeline...")

    df = impute_missing_values(
        df, strategy=strategy, threshold=missing_thresh, ranked_features=ranked_features
    )

    df = clean_duplicates(df)

    df = clean_outliers(
        df,
        profiling_report["outliers"],
        method=outlier_method,
        ranked_features=ranked_features,
    )

    df = fix_inconsistencies(
        df, profiling_report["inconsistencies"], ranked_features=ranked_features
    )

    logger.log("info", "🎉 Prioritized data cleaning complete.")
    return df
