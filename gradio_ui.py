import os
import pandas as pd
import gradio as gr

from data_loader import load_data
from data_profiler import profile_data
from data_cleaner import clean_data
from anomaly_detector import get_anomaly_flags
from feature_ranker import rank_features
from adaptive_controller import log_and_reflect_adaptation
from cleaning_evaluator import evaluate_cleaning
from report_generator import generate_html_report


# ───────── Helper ─────────
def parse_comma_list(text: str, default=None):
    if not text:
        return default
    items = [x.strip() for x in text.split(",") if x.strip()]
    return items if items else default


# ───────── Pipeline ─────────
def run_pipeline(
    dataset_file,
    target_column,
    column_names_text,
    missing_values_text,
    header_choice,
    skiprows,
    use_adaptive,
    imputation_strategy,
    missing_thresh,
    outlier_method,
    skip_cols_text,
    contamination_rate,
    top_n_anomalies,
    clean_anomalies,
):
    try:
        # 1) Parse inputs
        col_names = parse_comma_list(column_names_text)
        missing_vals = parse_comma_list(missing_values_text, default=["?"])
        header = None if header_choice == "None" else int(header_choice)
        skiprows = None if skiprows == 0 else skiprows
        skip_cols = parse_comma_list(skip_cols_text, default=[])

        # 2) Load raw data
        ext = os.path.splitext(dataset_file.name)[-1].lower()
        if ext not in [".csv", ".xlsx", ".xls", ".json"]:
            raise ValueError("Unsupported file format.")
        df_raw = load_data(
            dataset_file.name,
            column_names=col_names,
            missing_values=missing_vals,
            header=header,
            skiprows=skiprows,
        )
        df_raw.columns = df_raw.columns.str.strip()
        if target_column not in df_raw.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # 3) Profile before cleaning
        profile_before = profile_data(
            df_raw, save_path="plots", target_column=target_column
        )

        # 4) Split X/y
        X_raw = df_raw.drop(columns=[target_column])
        y_raw = df_raw[target_column]

        # 5) Get anomaly flags (no imputation leaks)
        anomaly_flags = get_anomaly_flags(
            X_raw, contamination=contamination_rate, random_state=42
        )
        anomaly_summary = {
            "total_anomalies_flagged": int(anomaly_flags.sum()),
            "contamination_rate": contamination_rate,
            "anomaly_plot_path": "plots/anomaly_score_distribution.png",
        }
        # Build a full anomaly report (with flags + original values)
        anomaly_report_full = pd.concat(
            [X_raw, y_raw.rename("__target__"), anomaly_flags.rename("is_anomaly")],
            axis=1,
        )

        # 6) Select raw data for cleaning
        if clean_anomalies:
            df_selected = df_raw.loc[~anomaly_flags].copy()
        else:
            df_selected = df_raw.copy()

        X_to_clean = df_selected.drop(columns=[target_column])
        y_to_clean = df_selected[target_column]

        # 7) Build policy
        dummy_metrics = {
            "missing_pct_before": profile_before.get("total_missing_pct", 0),
            "skew_mean_before": round(
                sum(abs(v) for v in profile_before.get("skewness", {}).values())
                / max(len(profile_before.get("skewness", {})), 1),
                4,
            ),
            "outliers_before": profile_before.get("total_outliers", 0),
            "profile_shape": df_raw.shape,
        }
        if use_adaptive:
            policy = log_and_reflect_adaptation(dummy_metrics)
            policy["used_adaptive"] = True
        else:
            policy = {
                "imputation_strategy": imputation_strategy,
                "outlier_method": outlier_method,
                "used_adaptive": False,
            }

        # 8) Clean data
        X_cleaned, cleaning_summary = clean_data(
            X_to_clean,
            profiling_report=profile_before,
            outlier_method=policy["outlier_method"],
            missing_thresh=missing_thresh,
            imputation_strategy=policy["imputation_strategy"],
        )
        cleaned_full = X_cleaned.copy()
        cleaned_full[target_column] = y_to_clean

        # 9) Profile after cleaning & evaluate
        profile_after = profile_data(
            cleaned_full, save_path="plots", target_column=target_column
        )
        cleaning_eval = evaluate_cleaning(profile_before, profile_after)

        # 10) Feature ranking
        X_rank = cleaned_full.drop(columns=[target_column])
        task_type = "regression" if y_raw.dtype.kind in "ifu" else "classification"
        feature_scores, top_features, feat_plot = rank_features(
            X_rank,
            cleaned_full[target_column],
            task_type=task_type,
            save_dir="plots",
            plot_name="feature_importance.png",
        )

        # 11) Top anomalies for report
        top_anoms = anomaly_report_full[anomaly_report_full["is_anomaly"]].head(
            top_n_anomalies
        )

        # 12) Generate HTML report
        report_path = generate_html_report(
            profiling_summary_before=profile_before,
            profiling_summary_after=profile_after,
            cleaning_summary=cleaning_summary,
            cleaning_evaluation=cleaning_eval,
            anomaly_summary=anomaly_summary,
            anomaly_report=top_anoms,
            feature_importance_path=feat_plot,
            policy_info=policy,
            save_dir="reports",
            template_dir="templates",
        )

        # 13) Save CSV outputs
        os.makedirs("outputs", exist_ok=True)
        cleaned_file = "outputs/cleaned_data.csv"
        anomaly_file = "outputs/anomaly_data.csv"
        cleaned_full.to_csv(cleaned_file, index=False)
        anomaly_report_full.to_csv(anomaly_file, index=False)

        return (
            "✅ Pipeline completed successfully!",
            cleaned_file,
            anomaly_file,
            top_features,
            report_path,
        )

    except Exception as e:
        return (
            f"❌ Pipeline failed: {e}",
            None,
            None,
            [],
            None,
        )


# ───────── Gradio App ─────────
with gr.Blocks() as demo:
    gr.Markdown("# 🧼 C2C: Agentic Data Cleaning AI")

    with gr.Tabs():
        with gr.Tab("Configuration"):
            dataset_file = gr.File(label="Upload Dataset (CSV/Excel/JSON)")
            target_column = gr.Textbox(label="Target Column", placeholder="e.g. income")
            column_names_text = gr.Textbox(
                label="Custom Column Names (comma-separated)"
            )
            missing_values_text = gr.Textbox(label="Missing Values Flags", value="?,NA")
            header_choice = gr.Dropdown(
                ["None", "0", "1"], value="0", label="Header Row"
            )
            skiprows = gr.Slider(0, 100, step=1, label="Skip Initial Rows")
            use_adaptive = gr.Checkbox(label="Use Adaptive Policy", value=True)
            imputation_strategy = gr.Dropdown(
                ["mean", "median", "mode"], value="mean", label="Imputation Strategy"
            )
            missing_thresh = gr.Slider(
                0.0, 1.0, step=0.05, value=0.5, label="Missing Threshold"
            )
            outlier_method = gr.Dropdown(
                ["cap", "remove", "skip"], value="cap", label="Outlier Method"
            )
            skip_cols_text = gr.Textbox(label="Columns to Skip Outlier Handling")
            contamination_rate = gr.Slider(
                0.01, 0.5, step=0.01, value=0.05, label="Anomaly Contamination Rate"
            )
            top_n_anomalies = gr.Slider(
                1, 50, step=1, value=10, label="Top N Anomalies to Show"
            )
            clean_anomalies = gr.Checkbox(
                label="Clean Anomalous Records Too", value=False
            )
            status_box = gr.Textbox(label="Status", interactive=False)
            run_btn = gr.Button("Run Pipeline")

        with gr.Tab("Results"):
            cleaned_out = gr.File(label="Download Cleaned Data")
            anomaly_out = gr.File(label="Download Anomaly Data")
            report_file = gr.File(label="Download HTML Report")
            feature_out = gr.Dataframe(
                headers=["Top Features"], label="Top Ranked Features"
            )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            dataset_file,
            target_column,
            column_names_text,
            missing_values_text,
            header_choice,
            skiprows,
            use_adaptive,
            imputation_strategy,
            missing_thresh,
            outlier_method,
            skip_cols_text,
            contamination_rate,
            top_n_anomalies,
            clean_anomalies,
        ],
        outputs=[status_box, cleaned_out, anomaly_out, feature_out, report_file],
        show_progress=True,
    )

demo.launch()
