import os
import pandas as pd

from data_loader import load_data
from data_profiler import profile_data
from data_cleaner import clean_data
from anomaly_detector import detect_anomalies_with_knn
from feature_ranker import rank_features
from model_evaluator import evaluate_with_random_forest
from adaptive_controller import reflect_and_adapt, log_and_reflect_adaptation
from report_generator import generate_html_report

# === Configurations ===
DATA_PATH = "datasets/sample_data.csv"  # <-- change this if needed
TARGET_COLUMN = "target"  # <-- change depending on your dataset
SAVE_DIR = "plots"
USE_ADAPTIVE_POLICY = True  # Toggle: use default or adaptive


def main():
    # === 1. Load Data ===
    df = load_data(DATA_PATH, header=0)
    df.columns = df.columns.str.strip()
    print("✅ Dataset loaded.")

    # Split features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # === 2. Profile Data ===
    profiling_summary = profile_data(df, save_path=SAVE_DIR)
    print("✅ Data profiling completed.")

    # === 3. Feature Ranking (Before Cleaning) ===
    ranked_before, importance_before_path = rank_features(
        X, y, save_dir=SAVE_DIR, plot_name="feature_importance_before.png"
    )
    print("✅ Feature importance (before cleaning) ranked.")

    # === 4. Clean Data ===
    dataset_size = len(df)

    if USE_ADAPTIVE_POLICY:
        policy = reflect_and_adapt()
    else:
        policy = {
            "imputation_strategy": "mean",
            "outlier_method": "remove",
        }

    # ✅ Modify cleaning aggressiveness based on dataset size
    if dataset_size < 100:
        print(
            f"⚠️ Small dataset detected ({dataset_size} rows) — using softer cleaning policies."
        )
        policy["outlier_method"] = "cap"  # Cap instead of remove
        anomaly_detection_enabled = False  # Skip anomaly detection entirely
    else:
        anomaly_detection_enabled = True  # Normal cleaning

    print(f"🔧 Using Policy: {policy}")

    X_clean, cleaning_summary = clean_data(
        X,
        profiling_summary,
        strategy=policy["imputation_strategy"],
        outlier_method=policy["outlier_method"],
        ranked_features=ranked_before.index.tolist(),
    )
    print("✅ Data cleaning completed.")

    # === 5. Anomaly Detection ===
    if anomaly_detection_enabled:
        X_clean_post_anomaly, anomaly_summary, _ = detect_anomalies_with_knn(
            X_clean, post_action=policy["outlier_method"], save_path=SAVE_DIR
        )
        print("✅ Anomaly detection completed.")
    else:
        X_clean_post_anomaly = X_clean.copy()
        anomaly_summary = {
            "total_anomalies_flagged": 0,
            "post_action": "skipped (small dataset)",
            "anomaly_plot_path": None,
            "contamination_rate": 0,
        }
        print("⚠️ Anomaly detection skipped for small dataset.")

    # Align y with filtered X after anomaly detection
    y_clean_post_anomaly = y.loc[X_clean_post_anomaly.index]

    # === 6. Feature Ranking (After Cleaning) ===
    ranked_after, importance_after_path = rank_features(
        X_clean_post_anomaly,
        y_clean_post_anomaly,
        save_dir=SAVE_DIR,
        plot_name="feature_importance_after.png",
    )
    print("✅ Feature importance (after cleaning) ranked.")

    # === 7. Evaluate Model (Raw vs Cleaned) ===
    evaluation_summary = evaluate_with_random_forest(
        X_raw=X,
        y_raw=y,
        X_clean=X_clean_post_anomaly,
        y_clean=y_clean_post_anomaly,
        save_dir=SAVE_DIR,
    )
    print("✅ Model evaluation completed.")

    # === 8. Log run + reflect adaptation ===
    if USE_ADAPTIVE_POLICY:
        log_and_reflect_adaptation(
            evaluation=evaluation_summary["Cleaned Data Evaluation"],
            policy=policy,
            decision="Reflect after run",
            extra_info={"dataset": os.path.basename(DATA_PATH)},
        )
    print("✅ Adaptive controller reflection done.")

    # === 9. Generate Final HTML Report ===
    report_path = generate_html_report(
        profiling_summary=profiling_summary,
        cleaning_summary=cleaning_summary,
        anomaly_summary=anomaly_summary,
        feature_importance_before_path=importance_before_path,
        feature_importance_after_path=importance_after_path,
        evaluation_summary=evaluation_summary,
        policy_info=policy,
        save_dir="reports",
        template_dir="templates",
    )
    print(f"🎉 Report generated at: {report_path}")


if __name__ == "__main__":
    main()
# This script is designed to be run in a terminal or command line interface.
