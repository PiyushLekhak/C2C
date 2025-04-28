import json
import os
from run_history_logger import log_run_summary
from event_logger import get_logger

logger = get_logger("adaptive_controller")

HISTORY_LOG_PATH = "logs/run_history.jsonl"

# We only adapt outlier handling; imputation is self-managed by the cleaner
DEFAULT_POLICY = {
    "outlier_method": "remove",
    "scale_method": "standard",
}

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
    metrics = run_data.get("evaluation", {})
    if "RMSE" in metrics:
        return {"task": "regression", "rmse": metrics["RMSE"]}
    if "Weighted F1 Score" in metrics:
        return {"task": "classification", "f1": metrics["Weighted F1 Score"]}
    return None


def reflect_and_adapt():
    runs = load_all_runs()
    if len(runs) < 3:
        logger.info("📌 Not enough history to reflect. Using default policy.")
        return DEFAULT_POLICY

    # Compute summaries for each run
    summaries = [compute_summary_metrics(r) for r in runs]
    # Drop any runs where compute_summary_metrics returned None
    summaries = [s for s in summaries if s]
    if not summaries:
        logger.warning("⚠️ No usable evaluation metrics found. Using default policy.")
        return DEFAULT_POLICY

    recent = summaries[-5:]
    task = recent[0]["task"]

    if task == "regression":
        # Keep only those with a real 'rmse' value
        reg = [r for r in recent if "rmse" in r]
        if reg:
            avg_rmse = sum(r["rmse"] for r in reg) / len(reg)
            logger.info(f"📉 Recent avg RMSE (last {len(reg)}): {avg_rmse:.2f}")
            if avg_rmse > THRESHOLDS["rmse"]:
                logger.info("🔁 RMSE too high — switching to median/cap.")
                return {"imputation_strategy": "median", "outlier_method": "cap"}
    else:  # classification
        # Keep only those with a real 'f1' value
        cls = [r for r in recent if "f1" in r]
        if cls:
            avg_f1 = sum(r["f1"] for r in cls) / len(cls)
            logger.info(f"📈 Recent avg F1 (last {len(cls)}): {avg_f1:.2f}")
            if avg_f1 < THRESHOLDS["f1"]:
                logger.info("🔁 F1 too low — switching to mode/cap.")
                return {"imputation_strategy": "mode", "outlier_method": "cap"}

    # Fallback
    return DEFAULT_POLICY


def log_and_reflect_adaptation(evaluation, policy, decision, extra_info=None):
    log_run_summary(evaluation, policy, decision, extra_info)
    return reflect_and_adapt()
