import os
import numpy as np
import pandas as pd
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
from event_logger import get_logger

logger = get_logger("model_evaluator")
MISSING_THRESHOLD = 0.05


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
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }
    logger.info("Classification evaluation completed.")
    return metrics


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics = {
        "RMSE": rmse,
        "R² Score": r2_score(y_true, y_pred),
    }
    logger.info("Regression evaluation completed.")
    return metrics


def minimally_clean_raw_data(X, y):
    # 1) Drop rows where the target is missing
    mask = y.notnull()
    Xc = X.loc[mask].copy()
    yc = y.loc[mask].copy()

    # 2) Decide which columns to impute vs. drop on
    miss_frac = Xc.isnull().mean()
    to_impute = miss_frac[miss_frac > MISSING_THRESHOLD].index.tolist()
    to_dropna = miss_frac[miss_frac <= MISSING_THRESHOLD].index.tolist()

    # 3) Drop any row with NaN in low-missingness cols
    if to_dropna:
        rows_ok = Xc[to_dropna].notnull().all(axis=1)
        Xc = Xc.loc[rows_ok]
        yc = yc.loc[rows_ok]

    # 4) Impute high-missingness cols, splitting by dtype
    if to_impute:
        # numeric columns → mean
        num_cols = Xc[to_impute].select_dtypes(include=[np.number]).columns.tolist()
        # categorical (everything else) → most frequent
        cat_cols = [c for c in to_impute if c not in num_cols]

        if num_cols:
            num_imp = SimpleImputer(strategy="mean")
            Xc[num_cols] = num_imp.fit_transform(Xc[num_cols])

        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            Xc[cat_cols] = cat_imp.fit_transform(Xc[cat_cols])

    # 5) Final sweep: drop any rows still containing NaNs
    final_ok = Xc.notnull().all(axis=1)
    Xc = Xc.loc[final_ok]
    yc = yc.loc[final_ok]

    return Xc, yc


def plot_evaluation_comparison(
    raw_metrics, clean_metrics, task_type, save_path="plots/performance_comparison.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if task_type in ["binary", "multiclass"]:
        metrics = ["Accuracy", "Weighted F1 Score"]
    else:
        metrics = ["RMSE", "R² Score"]

    raw_vals = [raw_metrics.get(m, np.nan) for m in metrics]
    clean_vals = [clean_metrics.get(m, np.nan) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, raw_vals, width, label="Raw Data")
    plt.bar(x + width / 2, clean_vals, width, label="Cleaned Data")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"📊 Performance plot saved to {save_path}")
    return save_path


def evaluate_with_random_forest(
    X_raw, y_raw, X_clean, y_clean, test_size=0.2, random_state=42, save_dir="plots"
):
    target_type = type_of_target(y_raw)
    if target_type in ["binary", "multiclass"]:
        ModelClass = RandomForestClassifier
    elif target_type in ["continuous", "continuous-multioutput"]:
        ModelClass = RandomForestRegressor
    else:
        raise ValueError(f"Unsupported target type: {target_type}")

    # 1. Raw data cleaning & encoding
    try:
        X_raw_ready, y_raw_ready = minimally_clean_raw_data(X_raw, y_raw)
        X_raw_ready = pd.get_dummies(X_raw_ready, drop_first=True)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw_ready, y_raw_ready, test_size=test_size, random_state=random_state
        )
        model_raw = ModelClass(random_state=random_state)
        model_raw.fit(X_train_raw, y_train_raw)
        y_pred_raw = model_raw.predict(X_test_raw)
        raw_metrics = evaluate_model(y_test_raw, y_pred_raw)
    except Exception as e:
        logger.error(f"❌ Raw data evaluation failed: {e}")
        raw_metrics = {"Error": "Raw evaluation failed."}

    # 2. Cleaned data encoding
    try:
        X_clean_enc = pd.get_dummies(X_clean, drop_first=True)
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
            X_clean_enc, y_clean, test_size=test_size, random_state=random_state
        )
        model_clean = ModelClass(random_state=random_state)
        model_clean.fit(X_train_clean, y_train_clean)
        y_pred_clean = model_clean.predict(X_test_clean)
        clean_metrics = evaluate_model(y_test_clean, y_pred_clean)
    except Exception as e:
        logger.error(f"❌ Clean data evaluation failed: {e}")
        clean_metrics = {"Error": "Clean evaluation failed."}

    logger.info("Model evaluation complete.")

    # 3. Performance diff
    perf_diff = {}
    if "Error" not in raw_metrics and "Error" not in clean_metrics:
        if target_type in ["binary", "multiclass"]:
            comps = ["Accuracy", "Weighted F1 Score"]
        else:
            comps = ["RMSE", "R² Score"]
        for m in comps:
            rv = raw_metrics.get(m)
            cv = clean_metrics.get(m)
            if rv is not None and cv is not None:
                try:
                    if m == "RMSE":
                        diff = ((rv - cv) / rv) * 100
                    else:
                        diff = ((cv - rv) / rv) * 100
                    perf_diff[f"{m} % Difference"] = round(diff, 2)
                except ZeroDivisionError:
                    perf_diff[f"{m} % Difference"] = None
    else:
        perf_diff["Error"] = "Could not compute differences"

    # 4. Plot comparison
    try:
        plot_path = plot_evaluation_comparison(
            raw_metrics,
            clean_metrics,
            target_type,
            save_path=os.path.join(save_dir, "performance_comparison.png"),
        )
    except Exception as e:
        logger.error(f"❌ Plot generation failed: {e}")
        plot_path = None

    return {
        "Raw Data Evaluation": raw_metrics,
        "Cleaned Data Evaluation": clean_metrics,
        "Performance Difference (%)": perf_diff,
        "Performance Plot": plot_path,
    }
