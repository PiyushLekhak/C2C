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
DATA_PATH = "datasets/adult_data.xlsx"  # <-- change this if needed
TARGET_COLUMN = "income"  # <-- change depending on your dataset
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
    profiling_summary = profile_data(
        df, save_path=SAVE_DIR, target_column=TARGET_COLUMN
    )
    print("✅ Data profiling completed.")

    # === 3. Feature Ranking (Before Cleaning) ===
    # Use a copy, drop rows with any NaN in features or target
    X_rank = X.copy()
    y_rank = y.copy()
    mask_rank = X_rank.notnull().all(axis=1) & y_rank.notnull()
    X_rank = X_rank.loc[mask_rank]
    y_rank = y_rank.loc[mask_rank]
    task_type_rank = "regression" if y_rank.dtype.kind in "ifu" else "classification"
    ranked_before, importance_before_path = rank_features(
        X_rank,
        y_rank,
        task_type=task_type_rank,
        save_dir=SAVE_DIR,
        plot_name="feature_importance_before.png",
    )
    print("✅ Feature importance (before cleaning) ranked.")

    # === 4. Clean Data ===
    dataset_size = len(df)
    if USE_ADAPTIVE_POLICY:
        policy = reflect_and_adapt()
    else:
        policy = {"outlier_method": "cap"}

    # Modify cleaning based on dataset size
    if dataset_size < 100:
        print(
            f"⚠️ Small dataset detected ({dataset_size} rows) — using softer cleaning."
        )
        policy["outlier_method"] = "cap"
        anomaly_detection_enabled = False
    else:
        anomaly_detection_enabled = True

    print(f"🔧 Using Policy: {policy}")

    X_clean, cleaning_summary = clean_data(
        X,
        profiling_summary,
        outlier_method=policy["outlier_method"],
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

    # === Align and drop NaNs in target for subsequent steps ===
    y_aligned = y.reindex(X_clean_post_anomaly.index)
    mask_valid = y_aligned.notnull()
    X_final = X_clean_post_anomaly.loc[mask_valid]
    y_final = y_aligned.loc[mask_valid]

    # === 6. Feature Importance (Post-Cleaning) ===
    task_type = "regression" if y_final.dtype.kind in "ifu" else "classification"
    ranked_after, importance_after_path = rank_features(
        X_final,
        y_final,
        task_type=task_type,
        save_dir=SAVE_DIR,
        plot_name="feature_importance_after.png",
    )
    print("✅ Feature importance (after cleaning) ranked.")

    # === Prepare data for evaluation ===
    raw_mask = y.notnull()
    X_raw_eval = X.loc[raw_mask]
    y_raw_eval = y.loc[raw_mask]

    # === 7. Evaluate Model (Raw vs Cleaned) ===
    evaluation_summary = evaluate_with_random_forest(
        X_raw=X_raw_eval,
        y_raw=y_raw_eval,
        X_clean=X_final,
        y_clean=y_final,
        save_dir=SAVE_DIR,
    )
    print("✅ Model evaluation completed.")

    # === 8. Log run + reflect ===
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
