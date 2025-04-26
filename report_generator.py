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
    plot_dir="plots",
):
    os.makedirs(save_dir, exist_ok=True)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{timestamp}.html"
    report_path = os.path.join(save_dir, report_filename)

    # Make relative paths
    feature_importance_before_path = os.path.relpath(
        feature_importance_before_path, start=save_dir
    )
    feature_importance_after_path = os.path.relpath(
        feature_importance_after_path, start=save_dir
    )
    perf_plot_path = os.path.relpath(
        evaluation_summary["Performance Plot"], start=save_dir
    )

    # Get all categorized plots
    categorized_plots = categorize_plots(plot_dir)
    # Relativize all plots to report folder
    for key in categorized_plots:
        categorized_plots[key] = [
            os.path.relpath(p, start=save_dir) for p in categorized_plots[key]
        ]

    context = {
        "timestamp": timestamp,
        "profiling": profiling_summary,
        "cleaning": cleaning_summary,
        "anomaly": anomaly_summary,
        "feature_importance_before_path": feature_importance_before_path,
        "feature_importance_after_path": feature_importance_after_path,
        "evaluation": {
            "Raw Data Evaluation": evaluation_summary["Raw Data Evaluation"],
            "Cleaned Data Evaluation": evaluation_summary["Cleaned Data Evaluation"],
            "Performance Plot": perf_plot_path,
        },
        "policy": policy_info,
        "plots": categorized_plots,
    }

    html = template.render(**context)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
