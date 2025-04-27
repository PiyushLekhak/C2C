import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from event_logger import get_logger

logger = get_logger("model_evaluator")

# Threshold for missing values (% above which we impute instead of drop)
MISSING_THRESHOLD = 0.05


def encode_categorical_features(X):
    """
    Encodes categorical features (object or category dtype) using Label Encoding.
    Args:
        X (pd.DataFrame): Feature matrix.
    Returns:
        pd.DataFrame: Encoded feature matrix.
    """
    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    return X_encoded


def evaluate_model(y_true, y_pred):
    target_type = type_of_target(y_true)

    if target_type in ["binary", "multiclass"]:
        return evaluate_classification(y_true, y_pred)
    elif target_type in ["continuous", "continuous-multioutput"]:
        return evaluate_regression(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def evaluate_classification(y_true, y_pred):
    metrics = {
        "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Accuracy": accuracy_score(y_true, y_pred),
    }
    logger.info("Classification evaluation completed.")
    return metrics


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": rmse,
        "R² Score": r2,
    }
    logger.info("Regression evaluation completed.")
    return metrics


def minimally_clean_raw_data(X, y):
    """
    Prepares X and y for modeling by
      1) dropping any rows where y is NaN,
      2) dropping rows for features with low missingness,
      3) imputing features with high missingness,
      4) finally dropping any remaining rows with NaNs.

    Returns:
        X_clean (pd.DataFrame): No-NaN feature matrix
        y_clean (pd.Series): No-NaN target vector, aligned
    """
    # 1) Drop any rows where the target itself is missing
    mask_y = y.notnull()
    X_clean = X.loc[mask_y].copy()
    y_clean = y.loc[mask_y].copy()

    # 2) Identify features to drop rows on vs. impute
    missing_ratios = X_clean.isnull().mean()
    cols_to_impute = missing_ratios[missing_ratios > MISSING_THRESHOLD].index.tolist()
    cols_to_dropna = missing_ratios[missing_ratios <= MISSING_THRESHOLD].index.tolist()

    # 3) Drop rows where any of the low-missingness cols are NaN
    if cols_to_dropna:
        mask_rows = X_clean[cols_to_dropna].notnull().all(axis=1)
        X_clean = X_clean.loc[mask_rows]
        y_clean = y_clean.loc[mask_rows]

    # 4) Impute the high-missingness columns
    if cols_to_impute:
        imputer = SimpleImputer(strategy="mean")
        X_clean[cols_to_impute] = imputer.fit_transform(X_clean[cols_to_impute])

    # 5) Final sweep: drop any row that still has a NaN (should be none)
    final_mask = X_clean.notnull().all(axis=1)
    X_clean = X_clean.loc[final_mask]
    y_clean = y_clean.loc[final_mask]

    return X_clean, y_clean


def plot_evaluation_comparison(
    raw_metrics, clean_metrics, task_type, save_path="plots/performance_comparison.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if task_type in ["binary", "multiclass"]:
        metrics_to_plot = ["Accuracy", "Weighted F1 Score"]
    else:  # regression
        metrics_to_plot = ["RMSE", "R² Score"]

    raw_values = [raw_metrics.get(metric, np.nan) for metric in metrics_to_plot]
    clean_values = [clean_metrics.get(metric, np.nan) for metric in metrics_to_plot]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, raw_values, width, label="Raw Data")
    plt.bar(x + width / 2, clean_values, width, label="Cleaned Data")
    plt.xticks(x, metrics_to_plot)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"📊 Model performance comparison plot saved to {save_path}")
    return save_path


def evaluate_with_random_forest(
    X_raw, y_raw, X_clean, y_clean, test_size=0.2, random_state=42, save_dir="plots"
):
    target_type = type_of_target(y_raw)
    performance_diff = {}

    if target_type in ["binary", "multiclass"]:
        ModelClass = RandomForestClassifier
        evaluator = evaluate_classification
    elif target_type in ["continuous", "continuous-multioutput"]:
        ModelClass = RandomForestRegressor
        evaluator = evaluate_regression
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    # === 1. Encode Categorical Features ===
    X_raw = encode_categorical_features(X_raw)
    X_clean = encode_categorical_features(X_clean)

    # === 2. Pre-Cleaned (Raw) Data Evaluation ===
    try:
        X_raw_ready, y_raw_ready = minimally_clean_raw_data(X_raw, y_raw)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw_ready, y_raw_ready, test_size=test_size, random_state=random_state
        )
        model_raw = ModelClass(random_state=random_state)
        model_raw.fit(X_train_raw, y_train_raw)
        y_pred_raw = model_raw.predict(X_test_raw)
        raw_metrics = evaluator(y_test_raw, y_pred_raw)
    except Exception as e:
        logger.error(f"❌ Raw data evaluation failed: {str(e)}")
        raw_metrics = {
            "Error": "Raw data evaluation failed due to missing values or format issues."
        }

    # === 3. Cleaned Data Evaluation ===
    try:
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=random_state
        )
        model_clean = ModelClass(random_state=random_state)
        model_clean.fit(X_train_clean, y_train_clean)
        y_pred_clean = model_clean.predict(X_test_clean)
        clean_metrics = evaluator(y_test_clean, y_pred_clean)
    except Exception as e:
        logger.error(f"❌ Cleaned data evaluation failed: {str(e)}")
        clean_metrics = {"Error": "Cleaned data evaluation failed."}

    logger.info("Model evaluation on raw and cleaned data complete.")

    # === 4. Calculate performance differences ===
    if "Error" not in raw_metrics and "Error" not in clean_metrics:
        if target_type in ["binary", "multiclass"]:
            metrics_to_compare = ["Accuracy", "Weighted F1 Score"]
        else:
            metrics_to_compare = ["RMSE", "R² Score"]

        for metric in metrics_to_compare:
            raw_val = raw_metrics.get(metric)
            clean_val = clean_metrics.get(metric)

            if raw_val is not None and clean_val is not None:
                try:
                    if metric == "RMSE":
                        # Lower values are better
                        diff = ((raw_val - clean_val) / raw_val) * 100
                    else:
                        # Higher values are better
                        diff = ((clean_val - raw_val) / raw_val) * 100
                    performance_diff[f"{metric} % Difference"] = round(diff, 2)
                except ZeroDivisionError:
                    performance_diff[f"{metric} % Difference"] = "N/A (div by zero)"
    else:
        performance_diff["Error"] = "Could not calculate differences"

    # === 5. Plotting comparison ===
    try:
        perf_plot_path = plot_evaluation_comparison(
            raw_metrics,
            clean_metrics,
            target_type,
            save_path=os.path.join(save_dir, "performance_comparison.png"),
        )
    except Exception as e:
        logger.error(f"❌ Failed to generate performance plot: {str(e)}")
        perf_plot_path = None

    return {
        "Raw Data Evaluation": raw_metrics,
        "Cleaned Data Evaluation": clean_metrics,
        "Performance Difference (%)": performance_diff,
        "Performance Plot": perf_plot_path,
    }
