import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def generate_html_report(
    profiling_summary_before,
    profiling_summary_after,
    cleaning_summary,
    cleaning_evaluation,
    anomaly_summary,
    anomaly_report,
    feature_importance_path,
    policy_info,
    save_dir="reports",
    template_dir="templates",
):
    os.makedirs(save_dir, exist_ok=True)

    # Set up template
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(save_dir, report_filename)

    # Extract and relativize chart paths
    target_distribution_path = profiling_summary_after.get("target_distribution_path")
    correlation_heatmap_path = profiling_summary_after.get("correlation_heatmap_path")

    if target_distribution_path:
        target_distribution_path = os.path.relpath(
            target_distribution_path, start=save_dir
        )
    if correlation_heatmap_path:
        correlation_heatmap_path = os.path.relpath(
            correlation_heatmap_path, start=save_dir
        )

    # Convert paths to relative for portability
    feature_importance_path = os.path.relpath(feature_importance_path, start=save_dir)
    anomaly_plot_path = anomaly_summary.get("anomaly_plot_path", None)
    if anomaly_plot_path:
        anomaly_plot_path = os.path.relpath(anomaly_plot_path, start=save_dir)

    # Cleaned summaries
    dropped_columns = cleaning_summary.get("missing_handling", {}).get(
        "dropped_columns", []
    )
    imputed_columns = cleaning_summary.get("missing_handling", {}).get(
        "imputed_values", {}
    )
    outlier_handling = cleaning_summary.get("outlier_handling", {})
    final_shape = cleaning_summary.get("final_shape", ("?", "?"))

    # Evaluation Metrics
    eval_metrics = cleaning_evaluation.copy()
    eval_metrics["final_shape"] = final_shape

    # === HTML Context ===
    context = {
        "timestamp": timestamp,
        "profiling_before": profiling_summary_before,
        "profiling_after": profiling_summary_after,
        "cleaning_summary": cleaning_summary,
        "cleaning_eval": eval_metrics,
        "cleaning_dropped": dropped_columns,
        "cleaning_imputed": imputed_columns,
        "outlier_handling": outlier_handling,
        "anomaly_summary": anomaly_summary,
        "anomaly_report": (
            anomaly_report.to_dict(orient="records")
            if anomaly_report is not None
            else None
        ),
        "target_distribution_path": target_distribution_path,
        "correlation_heatmap_path": correlation_heatmap_path,
        "feature_importance_path": feature_importance_path,
        "anomaly_plot_path": anomaly_plot_path,
        "policy_info": policy_info,
    }

    # === Render & Write ===
    html = template.render(**context)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
