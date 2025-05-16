import os
import pandas as pd
import gradio as gr
from data_loader import load_data
from data_profiler import profile_data
from data_cleaner import clean_data
from anomaly_detector import detect_anomalies_with_isolation_forest
from feature_ranker import rank_features
from adaptive_controller import log_and_reflect_adaptation
from cleaning_evaluator import evaluate_cleaning
from report_generator import generate_html_report


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
    clean_anomalies,  # ✅ New param
):
    try:

        def parse_list(text):
            return [item.strip() for item in text.split(",") if item.strip()]

        col_names = parse_list(column_names_text) if column_names_text else None
        missing_vals = parse_list(missing_values_text) or ["?"]
        header = None if header_choice == "None" else int(header_choice)
        skiprows = None if skiprows == 0 else skiprows
        skip_cols = parse_list(skip_cols_text)

        ext = os.path.splitext(dataset_file.name)[-1].lower()
        if ext not in [".csv", ".xlsx", ".xls", ".json"]:
            raise ValueError("Unsupported file format.")

        df = load_data(
            dataset_file.name,
            column_names=col_names,
            missing_values=missing_vals,
            header=header,
            skiprows=skiprows,
        )
        df.columns = df.columns.str.strip()

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        profile_before = profile_data(
            df, save_path="plots", target_column=target_column
        )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        df_with_anomalies, anomaly_summary, anomaly_report_full = (
            detect_anomalies_with_isolation_forest(
                X, contamination=contamination_rate, save_path="plots"
            )
        )

        X_with_flags = df_with_anomalies.copy()
        X_with_flags["__target__"] = y

        if clean_anomalies:
            X_to_clean = X_with_flags.drop(
                columns=["anomaly_score", "is_anomaly", "__target__"]
            )
            y_to_clean = X_with_flags["__target__"]
        else:
            normal_mask = ~X_with_flags["is_anomaly"]
            X_to_clean = X_with_flags.loc[normal_mask].drop(
                columns=["anomaly_score", "is_anomaly", "__target__"]
            )
            y_to_clean = X_with_flags.loc[normal_mask, "__target__"]

        anomalies_df = df_with_anomalies.loc[X_with_flags["is_anomaly"]].copy()
        anomalies_df[target_column] = y.loc[anomalies_df.index]

        dummy_metrics = {
            "missing_pct_before": profile_before.get("total_missing_pct", 0),
            "skew_mean_before": round(
                sum(abs(v) for v in profile_before.get("skewness", {}).values())
                / max(len(profile_before.get("skewness", {})), 1),
                4,
            ),
            "outliers_before": profile_before.get("total_outliers", 0),
            "profile_shape": df.shape,
        }

        policy = (
            log_and_reflect_adaptation(dummy_metrics)
            if use_adaptive
            else {
                "imputation_strategy": imputation_strategy,
                "outlier_method": outlier_method,
            }
        )

        X_cleaned, cleaning_summary = clean_data(
            X_to_clean,
            profiling_report=profile_before,
            outlier_method=policy.get("outlier_method", "cap"),
            missing_thresh=missing_thresh,
            imputation_strategy=policy.get("imputation_strategy", "mean"),
        )

        cleaned_full = X_cleaned.copy()
        cleaned_full[target_column] = y_to_clean

        profile_after = profile_data(
            cleaned_full, save_path="plots", target_column=target_column
        )
        cleaning_eval = evaluate_cleaning(profile_before, profile_after)

        X_rank = cleaned_full.drop(columns=[target_column])
        task_type = "regression" if y.dtype.kind in "ifu" else "classification"
        feature_scores, top_features, feat_plot = rank_features(
            X_rank,
            cleaned_full[target_column],
            task_type=task_type,
            save_dir="plots",
            plot_name="feature_importance.png",
        )

        top_anoms = anomalies_df.sort_values(by="anomaly_score", ascending=False).head(
            top_n_anomalies
        )

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

        cleaned_file = "outputs/cleaned_data.csv"
        anomaly_file = "outputs/anomaly_data.csv"
        os.makedirs(os.path.dirname(cleaned_file), exist_ok=True)
        cleaned_full.to_csv(cleaned_file, index=False)
        top_anoms.to_csv(anomaly_file, index=False)

        return (
            f"✅ Pipeline completed successfully!",
            cleaned_file,
            anomaly_file,
            top_features,
            report_path,
        )
    except Exception as e:
        return (f"❌ Pipeline failed: {str(e)}", None, None, [], None)


# === Gradio Interface ===

with gr.Blocks() as demo:
    gr.Markdown("# 🧼 C2C: Agentic Data Cleaning UI")
    with gr.Tabs() as tabs:
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
                1, 20, step=1, value=10, label="Top N Anomalies to Show"
            )
            clean_anomalies = gr.Checkbox(  # ✅ New checkbox
                label="Clean Anomalous Records Too", value=False
            )
            status_box = gr.Textbox(label="Status", interactive=False)
            run_btn = gr.Button("Run Pipeline")

        with gr.Tab("Results"):
            report_file = gr.File(label="Download HTML Report")
            cleaned_out = gr.File(label="Download Cleaned Data")
            anomaly_out = gr.File(label="Download Anomaly Data")
            feature_out = gr.Dataframe(
                headers=["Top Features"], label="Top Ranked Features"
            )

    def pipeline_with_ui_feedback(*args):
        try:
            message, cleaned_file, anomaly_file, top_features, report_path = (
                run_pipeline(*args)
            )
            return message, cleaned_file, anomaly_file, top_features, report_path
        except Exception as e:
            return f"❌ Pipeline failed: {str(e)}", None, None, [], None

    run_btn.click(
        fn=pipeline_with_ui_feedback,
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
            clean_anomalies,  # ✅ Included here
        ],
        outputs=[
            status_box,
            cleaned_out,
            anomaly_out,
            feature_out,
            report_file,
        ],
        show_progress=True,
    )

demo.launch()
