from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    log_json: bool = False, log_dir: str | None = None, level: str = "INFO"
) -> logging.Logger:
    """Set up BlueBird logger with Rich console output and optional JSONL file."""
    logger = logging.getLogger("guanwu")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Optional JSONL file handler
    if log_json and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path
            / f"guanwu_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    return logger


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "job_id"):
            log_entry["job_id"] = record.job_id
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def get_logger() -> logging.Logger:
    """Get the BlueBird logger."""
    return logging.getLogger("guanwu")


def new_job_id() -> str:
    """Generate a new job ID."""
    return (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
