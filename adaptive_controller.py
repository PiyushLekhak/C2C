import json
import os
from run_history_logger import log_run_summary
from event_logger import get_logger

logger = get_logger("adaptive_controller")

HISTORY_LOG_PATH = "logs/run_history.jsonl"

# Default policy
DEFAULT_POLICY = {
    "imputation_strategy": "mean",
    "outlier_method": "remove",
}

# Thresholds for adapting
THRESHOLDS = {
    "rmse": 100,
    "f1": 0.6,
}


def load_all_runs():
    if not os.path.exists(HISTORY_LOG_PATH):
        return []
    with open(HISTORY_LOG_PATH, "r") as f:
        return [json.loads(line) for line in f]


def compute_summary_metrics(run_data):
    """
    Extract and return RMSE or Weighted F1 Score depending on task type.
    """
    metrics = run_data.get("evaluation", {})
    if "RMSE" in metrics:
        return {"task": "regression", "rmse": metrics["RMSE"]}
    elif "Weighted F1 Score" in metrics:
        return {"task": "classification", "f1": metrics["Weighted F1 Score"]}
    return None


def reflect_and_adapt():
    """
    Reflects on the last 5 runs and adjusts cleaning policy if needed.
    Returns the policy to use for the next run.
    """
    runs = load_all_runs()
    if len(runs) < 3:
        logger.info("📌 Not enough history to reflect. Using default policy.")
        return DEFAULT_POLICY

    # Extract valid summary metrics
    summaries = []
    for run in runs:
        summary = compute_summary_metrics(run)
        if summary:
            summaries.append(summary)

    if not summaries:
        logger.warning("⚠️ No usable evaluation metrics found in run history.")
        return DEFAULT_POLICY

    recent = summaries[-5:]  # Only last 5 runs
    task_type = recent[0]["task"]

    if task_type == "regression":
        avg_rmse = sum(r["rmse"] for r in recent) / len(recent)
        logger.info(f"📉 Recent avg RMSE (last 5): {avg_rmse:.2f}")
        if avg_rmse > THRESHOLDS["rmse"]:
            logger.info("🔁 RMSE too high — switching to median/cap.")
            return {"imputation_strategy": "median", "outlier_method": "cap"}

    elif task_type == "classification":
        avg_f1 = sum(r["f1"] for r in recent) / len(recent)
        logger.info(f"📈 Recent avg F1 Score (last 5): {avg_f1:.2f}")
        if avg_f1 < THRESHOLDS["f1"]:
            logger.info("🔁 F1 too low — switching to mode/cap.")
            return {"imputation_strategy": "mode", "outlier_method": "cap"}

    return DEFAULT_POLICY


def log_and_reflect_adaptation(evaluation, policy, decision, extra_info=None):
    """
    Logs current run, reflects on recent history, and returns the next policy.
    """
    log_run_summary(evaluation, policy, decision, extra_info)
    return reflect_and_adapt()
