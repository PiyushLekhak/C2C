import os
import pandas as pd
from data_loader import load_data
from data_profiler import profile_data
from data_cleaner import clean_data
from anomaly_detector import detect_anomalies_with_isolation_forest
from feature_ranker import rank_features
from adaptive_controller import log_and_reflect_adaptation
from cleaning_evaluator import evaluate_cleaning
from report_generator import generate_html_report

# === Config ===
DATA_PATH = "datasets/adult_data.xlsx"  # Change as needed
TARGET_COLUMN = "income"  # Change depending on dataset
SAVE_DIR = "plots"
USE_ADAPTIVE_POLICY = True  # Enable policy reflection


def main():
    print("📥 Loading data...")
    df = load_data(DATA_PATH, header=0)
    df.columns = df.columns.str.strip()

    # === 1. Initial Profiling ===
    print("🔍 Profiling original data...")
    profile_before = profile_data(df, save_path=SAVE_DIR, target_column=TARGET_COLUMN)

    # === 2. Split features & target ===
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 3. Detect anomalies first (Isolation Forest)
    print("🧬 Detecting anomalies...")
    df_with_anomalies, anomaly_summary, full_anomaly_scores = (
        detect_anomalies_with_isolation_forest(
            X,
            contamination=0.05,
            save_path=SAVE_DIR,
            target_column=TARGET_COLUMN,
            target_values=y,
        )
    )

    # Combine features with anomaly scores
    df_anomaly_tagged = X.copy()
    df_anomaly_tagged["anomaly_score"] = df_with_anomalies["anomaly_score"]
    df_anomaly_tagged["is_anomaly"] = df_with_anomalies["is_anomaly"]
    df_anomaly_tagged[TARGET_COLUMN] = y

    # 4. Separate normal and anomalous records
    X_normal = df_anomaly_tagged[~df_anomaly_tagged["is_anomaly"]].drop(
        columns=["anomaly_score", "is_anomaly"]
    )
    X_anomalous = df_anomaly_tagged[df_anomaly_tagged["is_anomaly"]].drop(
        columns=["anomaly_score", "is_anomaly"]
    )
    y_normal = y.loc[X_normal.index]
    y_anomalous = y.loc[X_anomalous.index]

    # === 5. Policy Selection ===
    print("🧠 Deciding cleaning strategy...")
    dummy_eval = {
        "missing_pct_before": profile_before.get("total_missing_pct", 0),
        "skew_mean_before": round(
            sum(abs(v) for v in profile_before.get("skewness", {}).values())
            / max(len(profile_before.get("skewness", {})), 1),
            4,
        ),
        "outliers_before": profile_before.get("total_outliers", 0),
        "profile_shape": df.shape,
    }

    if USE_ADAPTIVE_POLICY:
        policy = log_and_reflect_adaptation(dummy_eval)
    else:
        policy = {
            "imputation_strategy": "mean",
            "outlier_method": "cap",
        }

    print(f"🛠️ Using policy: {policy}")

    # === 6. Clean only normal records ===
    print("🧼 Cleaning normal (non-anomalous) records...")
    X_normal_cleaned, cleaning_summary = clean_data(
        X_normal,
        profiling_report=profile_before,
        outlier_method=policy.get("outlier_method", "cap"),
        missing_thresh=0.5,
        imputation_strategy=policy.get("imputation_strategy", None),
    )

    # === 7. Recombine cleaned normal + untouched anomalies ===
    X_cleaned = pd.concat([X_normal_cleaned, X_anomalous], axis=0).sort_index()
    y_cleaned = pd.concat([y_normal, y_anomalous], axis=0).sort_index()
    df_cleaned_full = X_cleaned.copy()
    df_cleaned_full[TARGET_COLUMN] = y_cleaned

    # === 8. Profile After Cleaning ===
    print("🔎 Profiling cleaned data...")
    profile_after = profile_data(
        df_cleaned_full, save_path=SAVE_DIR, target_column=TARGET_COLUMN
    )

    # === 9. Evaluate Cleaning Effectiveness ===
    print("📈 Evaluating cleaning...")
    cleaning_eval = evaluate_cleaning(profile_before, profile_after)

    # === 10. Feature Ranking ===
    print("🏅 Ranking features...")
    task_type = "regression" if y_cleaned.dtype.kind in "ifu" else "classification"
    feature_scores, top_features, importance_path = rank_features(
        df_cleaned_full.drop(columns=[TARGET_COLUMN]),
        y_cleaned,
        task_type=task_type,
        save_dir=SAVE_DIR,
        plot_name="feature_importance_after.png",
    )

    # === 11. Report Generation ===
    print("📝 Generating report...")

    top_anomalies = full_anomaly_scores[full_anomaly_scores["is_anomaly"]].head(5)

    report_path = generate_html_report(
        profiling_summary_before=profile_before,
        profiling_summary_after=profile_after,
        cleaning_summary=cleaning_summary,
        cleaning_evaluation=cleaning_eval,
        anomaly_summary=anomaly_summary,
        anomaly_report=top_anomalies,
        feature_importance_path=importance_path,
        policy_info=policy,
        save_dir="reports",
        template_dir="templates",
    )

    print(f"✅ Report generated: {report_path}")


if __name__ == "__main__":
    main()
