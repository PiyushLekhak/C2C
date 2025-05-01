import json
import os
import uuid
from datetime import datetime, timezone
from event_logger import get_logger

logger = get_logger("adaptive_controller")

HISTORY_LOG_PATH = "logs/cleaning_metrics.jsonl"

DEFAULT_POLICY = {
    "imputation_strategy": "mean",
    "outlier_method": "cap",
    "scale_method": "standard",
}

THRESHOLDS = {
    "missing_pct": 0.3,
    "skew_mean": 2.0,
    "outlier_count": 0.1 * 10000,  # assuming 10k rows
}


def load_all_metrics():
    if not os.path.exists(HISTORY_LOG_PATH):
        return []
    with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def reflect_and_adapt(latest_metrics):
    logger.info("🔄 Reflecting on past runs to adapt cleaning policy...")

    policy = DEFAULT_POLICY.copy()

    if latest_metrics.get("missing_pct_before", 0) > THRESHOLDS["missing_pct"]:
        policy["imputation_strategy"] = "median"
        logger.info("📌 High missing % → using median imputation")

    if latest_metrics.get("skew_mean_before", 0) > THRESHOLDS["skew_mean"]:
        policy["imputation_strategy"] = "median"
        logger.info("📌 High skewness → using median imputation")

    if latest_metrics.get("outliers_before", 0) > THRESHOLDS["outlier_count"]:
        policy["outlier_method"] = "remove"
        logger.info("📌 Too many outliers → switching to remove")

    logger.info(f"✅ Policy decided: {policy}")
    return policy


def save_run_log(metrics_before, policy_decision):
    log_entry = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_before": metrics_before,
        "policy_used": policy_decision,
    }

    os.makedirs(os.path.dirname(HISTORY_LOG_PATH), exist_ok=True)
    with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_and_reflect_adaptation(metric_summary):
    policy = reflect_and_adapt(metric_summary)
    save_run_log(metric_summary, policy)
    return policy
