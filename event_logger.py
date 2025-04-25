import logging
import json
from logging import Handler
from datetime import datetime, timezone
import os


class JSONLogHandler(Handler):
    def __init__(self, filename="logs/system_log.jsonl"):
        super().__init__()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

    def emit(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "details"):
            log_entry["details"] = record.details

        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = JSONLogHandler()
        logger.addHandler(handler)
    return logger
