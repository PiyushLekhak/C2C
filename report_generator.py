import os
import base64
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def embed_image(path, save_dir):
    if not path:
        return None
    # Try absolute path first (e.g., plots/feature_importance.png)
    if os.path.exists(path):
        img_path = path
    else:
        # Fallback: join with save_dir
        img_path = os.path.normpath(os.path.join(save_dir, path))
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as img_f:
        encoded = base64.b64encode(img_f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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

    # Set up template environment
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(save_dir, report_filename)

    # Resolve and embed images
    target_dist_rel = profiling_summary_after.get("target_distribution_path")
    corr_rel = profiling_summary_after.get("correlation_heatmap_path")
    feature_rel = feature_importance_path
    anomaly_plot_rel = anomaly_summary.get("anomaly_plot_path")

    target_dist_data = embed_image(target_dist_rel, save_dir)
    corr_data = embed_image(corr_rel, save_dir)
    feature_data = embed_image(feature_rel, save_dir)
    anomaly_plot_data = embed_image(anomaly_plot_rel, save_dir)

    # Clean summary fields
    dropped_columns = cleaning_summary.get("missing_handling", {}).get(
        "dropped_columns", []
    )
    imputed_columns = cleaning_summary.get("missing_handling", {}).get(
        "imputed_values", {}
    )
    outlier_handling = cleaning_summary.get("outlier_handling", {})
    final_shape = cleaning_summary.get("final_shape", ("?", "?"))

    # Evaluation Metrics with final shape
    eval_metrics = cleaning_evaluation.copy()
    eval_metrics["final_shape"] = final_shape

    anomaly_records = (
        anomaly_report.to_dict(orient="records") if anomaly_report is not None else None
    )

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
        "anomaly_report": anomaly_records,
        "target_distribution_data": target_dist_data,
        "correlation_heatmap_data": corr_data,
        "feature_importance_data": feature_data,
        "anomaly_plot_data": anomaly_plot_data,
        "policy_info": policy_info,
    }

    html = template.render(**context)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
