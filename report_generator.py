import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime


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
    """
    Compiles all pipeline outputs into a static HTML report.

    Args:
        profiling_summary (dict): Summary from data_profiler.
        cleaning_summary (dict): Summary from data_cleaner.
        anomaly_summary (dict): Summary from anomaly_detector.
        feature_importance_before_path (str): Plot path (before cleaning).
        feature_importance_after_path (str): Plot path (after cleaning).
        evaluation_summary (dict): Raw and cleaned evaluation + plot path.
        policy_info (dict): Current policy and any adaptive decision.
        save_dir (str): Where to save final HTML report.
        template_dir (str): Where to load Jinja2 HTML templates from.

    Returns:
        str: Path to saved HTML report.
    """
    # Setup
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
    }

    # Render and save
    html = template.render(**context)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path
