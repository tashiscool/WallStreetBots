"""
Structured JSON Logging Configuration for Prediction Market Arbitrage.

Synthesized from:
- kalshi-polymarket-arbitrage-bot: JSON formatter, domain-specific files
- polymarket-arbitrage: Colored console output, custom log levels

Production-grade logging for observability.
"""

import logging
import logging.handlers
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

# Try to import python-json-logger
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


# =============================================================================
# Custom Log Levels
# =============================================================================

# Add custom log levels (from polymarket-arbitrage)
TRADE_LEVEL = 25  # Between INFO and WARNING
OPPORTUNITY_LEVEL = 26

logging.addLevelName(TRADE_LEVEL, "TRADE")
logging.addLevelName(OPPORTUNITY_LEVEL, "OPPORTUNITY")


def trade(self, message, *args, **kwargs):
    """Log trade events."""
    if self.isEnabledFor(TRADE_LEVEL):
        self._log(TRADE_LEVEL, message, args, **kwargs)


def opportunity(self, message, *args, **kwargs):
    """Log opportunity events."""
    if self.isEnabledFor(OPPORTUNITY_LEVEL):
        self._log(OPPORTUNITY_LEVEL, message, args, **kwargs)


# Add methods to Logger class
logging.Logger.trade = trade
logging.Logger.opportunity = opportunity


# =============================================================================
# JSON Formatter
# =============================================================================

class ArbitrageJsonFormatter(logging.Formatter):
    """
    JSON log formatter with Decimal support.

    From kalshi-polymarket-arbitrage-bot: Structured JSON logging.
    """

    def __init__(
        self,
        fmt_keys: Optional[Dict[str, str]] = None,
        datefmt: str = "%Y-%m-%dT%H:%M:%S.%f",
    ):
        super().__init__()
        self.fmt_keys = fmt_keys or {}
        self.datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_dict = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                log_dict[key] = self._serialize(value)

        return json.dumps(log_dict, default=self._json_default)

    def _serialize(self, value: Any) -> Any:
        """Serialize value for JSON."""
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if hasattr(value, "__dict__"):
            return str(value)
        return value

    def _json_default(self, obj: Any) -> Any:
        """JSON default handler."""
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


# =============================================================================
# Colored Console Formatter
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Colored console output formatter.

    From polymarket-arbitrage: ANSI color codes for log levels.
    """

    # ANSI color codes
    COLORS: ClassVar[dict] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "TRADE": "\033[34m",     # Blue
        "OPPORTUNITY": "\033[35m",  # Magenta
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt: str = "%H:%M:%S",
        use_colors: bool = True,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# =============================================================================
# Domain-Specific Loggers
# =============================================================================

@dataclass
class LoggingConfig:
    """
    Configuration for arbitrage logging.

    From kalshi-polymarket-arbitrage-bot: Domain-specific log files.
    """
    # Log directory
    log_dir: str = "logs/arbitrage"

    # File settings
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

    # Log levels by module
    levels: Dict[str, str] = field(default_factory=lambda: {
        "root": "INFO",
        "arbitrage": "DEBUG",
        "arbitrage.ingestion": "DEBUG",
        "arbitrage.strategies": "DEBUG",
        "arbitrage.execution": "INFO",
        "arbitrage.markets": "INFO",
        "httpx": "WARNING",
        "websockets": "WARNING",
        "asyncio": "WARNING",
    })

    # Enable file logging
    enable_file_logging: bool = True

    # Enable console logging
    enable_console_logging: bool = True

    # Use JSON format for files
    use_json_format: bool = True

    # Use colors for console
    use_colors: bool = True


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Setup logging configuration.

    Creates:
    - service.log: General application lifecycle
    - market_data.log: Market data and connections
    - trading.log: Trades and opportunities
    - Console output with colors
    """
    if config is None:
        config = LoggingConfig()

    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter(use_colors=config.use_colors))
        root_logger.addHandler(console_handler)

    # File handlers
    if config.enable_file_logging:
        # JSON or text formatter
        if config.use_json_format:
            file_formatter = ArbitrageJsonFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Service log (general)
        service_handler = logging.handlers.RotatingFileHandler(
            log_dir / "service.log",
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        service_handler.setLevel(logging.INFO)
        service_handler.setFormatter(file_formatter)
        root_logger.addHandler(service_handler)

        # Market data log
        market_handler = logging.handlers.RotatingFileHandler(
            log_dir / "market_data.log",
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        market_handler.setLevel(logging.DEBUG)
        market_handler.setFormatter(file_formatter)
        market_handler.addFilter(lambda r: "ingestion" in r.name or "market" in r.name)

        market_logger = logging.getLogger("arbitrage.ingestion")
        market_logger.addHandler(market_handler)

        # Trading log
        trading_handler = logging.handlers.RotatingFileHandler(
            log_dir / "trading.log",
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        trading_handler.setLevel(logging.DEBUG)
        trading_handler.setFormatter(file_formatter)

        trading_logger = logging.getLogger("arbitrage.trading")
        trading_logger.addHandler(trading_handler)

    # Set levels by module
    for module, level in config.levels.items():
        if module == "root":
            continue
        logging.getLogger(module).setLevel(getattr(logging, level.upper()))

    logger = logging.getLogger(__name__)
    logger.info("Logging configured", extra={"config": config.log_dir})


# =============================================================================
# Specialized Loggers
# =============================================================================

class TradeLogger:
    """
    Specialized logger for trade events.

    From kalshi-polymarket-arbitrage-bot: Structured trade logging.
    """

    def __init__(self, logger_name: str = "arbitrage.trading"):
        self._logger = logging.getLogger(logger_name)

    def log_order_placed(
        self,
        order_id: str,
        market_id: str,
        side: str,
        outcome: str,
        price: Decimal,
        size: int,
        strategy: str = "",
    ) -> None:
        """Log order placement."""
        self._logger.trade(
            f"Order placed: {order_id}",
            extra={
                "event": "order_placed",
                "order_id": order_id,
                "market_id": market_id,
                "side": side,
                "outcome": outcome,
                "price": float(price),
                "size": size,
                "strategy": strategy,
            }
        )

    def log_order_filled(
        self,
        trade_id: str,
        order_id: str,
        market_id: str,
        side: str,
        outcome: str,
        price: Decimal,
        size: int,
        fee: Decimal = Decimal("0"),
    ) -> None:
        """Log order fill."""
        self._logger.trade(
            f"Order filled: {order_id}",
            extra={
                "event": "order_filled",
                "trade_id": trade_id,
                "order_id": order_id,
                "market_id": market_id,
                "side": side,
                "outcome": outcome,
                "price": float(price),
                "size": size,
                "fee": float(fee),
            }
        )

    def log_order_cancelled(
        self,
        order_id: str,
        reason: str = "",
    ) -> None:
        """Log order cancellation."""
        self._logger.warning(
            f"Order cancelled: {order_id}",
            extra={
                "event": "order_cancelled",
                "order_id": order_id,
                "reason": reason,
            }
        )


class OpportunityLogger:
    """
    Specialized logger for arbitrage opportunities.

    From kalshi-polymarket-arbitrage-bot: Opportunity event logging.
    """

    def __init__(self, logger_name: str = "arbitrage.strategies"):
        self._logger = logging.getLogger(logger_name)

    def log_opportunity_detected(
        self,
        opportunity_id: str,
        market_id: str,
        strategy: str,
        edge: Decimal,
        total_cost: Decimal,
        suggested_size: int,
    ) -> None:
        """Log detected opportunity."""
        self._logger.opportunity(
            f"Opportunity detected: {opportunity_id}",
            extra={
                "event": "opportunity_detected",
                "opportunity_id": opportunity_id,
                "market_id": market_id,
                "strategy": strategy,
                "edge": float(edge),
                "total_cost": float(total_cost),
                "suggested_size": suggested_size,
            }
        )

    def log_opportunity_expired(
        self,
        opportunity_id: str,
        duration_ms: float,
        was_executed: bool,
    ) -> None:
        """Log expired opportunity."""
        self._logger.info(
            f"Opportunity expired: {opportunity_id}",
            extra={
                "event": "opportunity_expired",
                "opportunity_id": opportunity_id,
                "duration_ms": duration_ms,
                "was_executed": was_executed,
            }
        )


class PerformanceLogger:
    """
    Specialized logger for performance metrics.

    From kalshi-polymarket-arbitrage-bot: Portfolio and latency logging.
    """

    def __init__(self, logger_name: str = "arbitrage.performance"):
        self._logger = logging.getLogger(logger_name)

    def log_portfolio_snapshot(
        self,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        total_exposure: Decimal,
        positions_count: int,
        open_orders: int,
    ) -> None:
        """Log portfolio snapshot."""
        self._logger.info(
            f"Portfolio: PnL=${float(realized_pnl + unrealized_pnl):.2f}",
            extra={
                "event": "portfolio_snapshot",
                "realized_pnl": float(realized_pnl),
                "unrealized_pnl": float(unrealized_pnl),
                "total_pnl": float(realized_pnl + unrealized_pnl),
                "total_exposure": float(total_exposure),
                "positions_count": positions_count,
                "open_orders": open_orders,
            }
        )

    def log_latency(
        self,
        operation: str,
        latency_ms: float,
    ) -> None:
        """Log operation latency."""
        self._logger.debug(
            f"Latency: {operation} = {latency_ms:.2f}ms",
            extra={
                "event": "latency",
                "operation": operation,
                "latency_ms": latency_ms,
            }
        )


# =============================================================================
# Singleton Instances
# =============================================================================

_trade_logger: Optional[TradeLogger] = None
_opportunity_logger: Optional[OpportunityLogger] = None
_performance_logger: Optional[PerformanceLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create trade logger."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger


def get_opportunity_logger() -> OpportunityLogger:
    """Get or create opportunity logger."""
    global _opportunity_logger
    if _opportunity_logger is None:
        _opportunity_logger = OpportunityLogger()
    return _opportunity_logger


def get_performance_logger() -> PerformanceLogger:
    """Get or create performance logger."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger
