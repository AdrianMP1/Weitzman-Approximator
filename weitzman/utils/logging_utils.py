import logging
import logging.config

from pathlib import Path


class RunLoggerAdapter(logging.LoggerAdapter):
    """Injects run_id into every log record."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.setdefault("run_id", self.extra.get("run_id", "-"))
        kwargs["extra"] = extra
        return msg, kwargs


def configure_logging(verbosity: int, log_file: Path | None = None) -> None:
    """
    Configure the root logger.
    verbosity: 0 = WARNING, 1 = INFO, 2+ = DEBUG
    """
    level = "WARNING" if verbosity <= 0 else "INFO" if verbosity == 1 else "DEBUG"

    handlers: list[str] = ["console"]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append("file")

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": level, "handlers": handlers},
    }

    if log_file is not None:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": level,
            "formatter": "standard",
            "filename": str(log_file),
            "mode": "a",
            "encoding": "utf-8",
        }

    logging.config.dictConfig(config)


def get_logger(name: str, run_id: str | None = None) -> logging.Logger:
    base = logging.getLogger(name)
    if run_id is None:
        return base
    return RunLoggerAdapter(base, {"run_id": run_id})
