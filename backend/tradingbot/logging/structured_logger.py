"""Structured logging configuration for production."""
from __future__ import annotations
import logging
import structlog
from typing import Any, Dict
import os

from ..infra.build_info import build_id


def configure_logging(log_level: str = "INFO", json_logs: bool = True) -> None:
    """Configure structured logging for production.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output JSON logs
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_build_info,
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=getattr(logging, log_level.upper()),
    )


def add_build_info(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add build information to log entries.

    Args:
        logger: Logger instance
        method_name: Log method name
        event_dict: Event dictionary

    Returns:
        Updated event dictionary
    """
    event_dict["build_id"] = build_id()
    event_dict["service"] = "wallstreetbots"
    return event_dict


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Auto-configure logging if not already done
if not structlog.is_configured():
    json_logs = os.getenv("WSB_JSON_LOGS", "true").lower() == "true"
    log_level = os.getenv("WSB_LOG_LEVEL", "INFO")
    configure_logging(log_level=log_level, json_logs=json_logs)