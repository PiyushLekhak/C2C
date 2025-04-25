import json
import os
from datetime import datetime, timezone


class RunHistoryLogger:
    def __init__(self, log_file="logs/run_history.jsonl"):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

    def log_run_summary(self, evaluation, policy, decision, extra_info=None):
        """
        Logs a full pipeline run's evaluation, policy, and decision outcome.

        Args:
            evaluation (dict): Evaluation metrics like RMSE, F1, etc.
            policy (dict): Configuration used for this run.
            decision (str): What decision was made (e.g. keep_policy, change_imputation).
            extra_info (dict): Optional additional context (e.g. model type, notes).
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "evaluation": evaluation,
            "policy": policy,
            "decision": decision,
        }

        if extra_info:
            log_entry["extra_info"] = extra_info

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


# Singleton instance for easy access
run_logger = RunHistoryLogger()
log_run_summary = run_logger.log_run_summary
