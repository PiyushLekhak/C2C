import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


def categorize_plots(plot_folder):
    plots = os.listdir(plot_folder)
    categorized = {
        "correlation_matrices": [],
        "anomaly_distributions": [],
        "feature_importances": [],
        "performance_plots": [],
    }

    for plot in plots:
        if "correlation" in plot:
            categorized["correlation_matrices"].append(os.path.join(plot_folder, plot))
        elif "anomaly_score" in plot:
            categorized["anomaly_distributions"].append(os.path.join(plot_folder, plot))
        elif "feature_importance" in plot:
            categorized["feature_importances"].append(os.path.join(plot_folder, plot))
        elif "performance_comparison" in plot:
            categorized["performance_plots"].append(os.path.join(plot_folder, plot))

    return categorized


def generate_html_report(
    profiling_summary,
    cleaning_summary,
    anomaly_summary,
    feature_importance_before_path,
    feature_importance_after_path,
    evaluation_summary,
    policy_info,
    save_dir="reports",
    template_dir="templates",
):

    os.makedirs(save_dir, exist_ok=True)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(save_dir, report_filename)

    feature_importance_before_path = os.path.relpath(
        feature_importance_before_path, start=save_dir
    )
    feature_importance_after_path = os.path.relpath(
        feature_importance_after_path, start=save_dir
    )
    perf_plot_path = os.path.relpath(
        evaluation_summary["Performance Plot"], start=save_dir
    )

    anomaly_plot_path = anomaly_summary.get("anomaly_plot_path", None)

    # Split cleaning_summary nicely
    dropped_columns = cleaning_summary.get("missing_handling", {}).get(
        "dropped_columns", []
    )
    imputed_columns = cleaning_summary.get("missing_handling", {}).get("imputed", {})
    outlier_handling = cleaning_summary.get("outlier_handling", {})

    context = {
        "timestamp": timestamp,
        "profiling": profiling_summary,
        "cleaning_summary": cleaning_summary,
        "cleaning_dropped": dropped_columns,
        "cleaning_imputed": imputed_columns,
        "outlier_handling": outlier_handling,
        "anomaly_summary": anomaly_summary,
        "feature_importance_before_path": feature_importance_before_path,
        "feature_importance_after_path": feature_importance_after_path,
        "evaluation": {
            "Raw Data Evaluation": evaluation_summary["Raw Data Evaluation"],
            "Cleaned Data Evaluation": evaluation_summary["Cleaned Data Evaluation"],
            "Performance Plot": perf_plot_path,
            "Performance_Difference": evaluation_summary.get(
                "Performance Difference (%)", {}
            ),
        },
        "policy_info": policy_info,
    }

    html = template.render(**context)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
