import os
from data_loader import load_data
from data_profiler import profile_data
from data_cleaner import clean_data
from anomaly_detector import detect_anomalies_with_knn
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

    # === 1. Profile Before Cleaning ===
    print("🔍 Profiling original data...")
    profile_before = profile_data(df, save_path=SAVE_DIR, target_column=TARGET_COLUMN)

    # === 2. Split features & target ===
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # === 3. Cleaning Policy (adaptive or default) ===
    print("🧠 Deciding cleaning strategy...")
    dummy_eval = {
        "missing_pct_before": profile_before.get("total_missing_pct", 0),
        "skew_mean_before": round(
            sum(abs(v) for v in profile_before.get("skewness", {}).values())
            / max(len(profile_before.get("skewness", {})), 1),
            4,
        ),
        "outliers_before": profile_before.get("total_outliers", 0),
    }

    if USE_ADAPTIVE_POLICY:
        policy = log_and_reflect_adaptation(dummy_eval)
    else:
        policy = {
            "imputation_strategy": "mean",
            "outlier_method": "cap",
            "scale_method": "standard",
        }

    print(f"🛠️ Using policy: {policy}")

    # === 4. Clean Data ===
    print("🧼 Cleaning data...")
    X_cleaned, cleaning_summary = clean_data(
        X,
        profiling_report=profile_before,
        outlier_method=policy.get("outlier_method", "cap"),
        missing_thresh=0.5,
        imputation_strategy=policy.get("imputation_strategy", None),
    )

    # === 5. Reattach target for profiling after cleaning ===
    df_cleaned_full = X_cleaned.copy()
    y_aligned = y.reindex(X_cleaned.index)
    df_cleaned_full[TARGET_COLUMN] = y_aligned

    # === 6. Profile After Cleaning ===
    print("🔎 Profiling cleaned data...")
    profile_after = profile_data(
        df_cleaned_full, save_path=SAVE_DIR, target_column=TARGET_COLUMN
    )

    # === 7. Evaluate Cleaning Effectiveness ===
    print("📈 Evaluating cleaning...")
    cleaning_eval = evaluate_cleaning(profile_before, profile_after)

    # === 8. Anomaly Detection (KNN) ===
    print("🧬 Detecting anomalies...")
    df_with_anomalies, anomaly_summary, anomaly_report = detect_anomalies_with_knn(
        X_cleaned,
        scale_method=policy.get("scale_method", "standard"),
        save_path=SAVE_DIR,
    )
    top_anomalies = df_with_anomalies.copy()
    top_anomalies = top_anomalies.sort_values(by="anomaly_score", ascending=False).head(
        5
    )

    # === 9. Feature Ranking (after full cleaning only) ===
    print("🏅 Ranking features...")
    task_type = "regression" if y_aligned.dtype.kind in "ifu" else "classification"
    feature_scores, top_features, importance_path = rank_features(
        df_with_anomalies,
        y_aligned.loc[df_with_anomalies.index],
        task_type=task_type,
        save_dir=SAVE_DIR,
        plot_name="feature_importance_after.png",
    )

    # === 10. Generate HTML Report ===
    print("📝 Generating report...")
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
