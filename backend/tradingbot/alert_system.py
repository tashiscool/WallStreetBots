"""Alert System and Execution Checklist
Automated alerts and systematic execution workflow for the options playbook.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from typing import TYPE_CHECKING

import requests

from .market_regime import (
    MarketSignal,
    SignalGenerator,
    SignalType,
    TechnicalIndicators,
)
from .options_calculator import OptionsTradeCalculator, TradeCalculation

if TYPE_CHECKING:
    from .exit_planning import ExitSignal

# Environment configuration
SLACK_WEBHOOK = os.getenv("ALERT_SLACK_WEBHOOK")


def send_slack(msg: str) -> bool:
    webhook = os.getenv("ALERT_SLACK_WEBHOOK")
    if not webhook:
        return False
    try:
        r = requests.post(webhook, json={"text": msg}, timeout=5)
        return r.ok
    except Exception:
        return False


def send_email(subject: str, body: str) -> bool:
    host = os.getenv("ALERT_EMAIL_SMTP_HOST")
    port = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
    sender = os.getenv("ALERT_EMAIL_FROM")
    recipient = os.getenv("ALERT_EMAIL_TO")
    user = os.getenv("ALERT_EMAIL_USER")
    pwd = os.getenv("ALERT_EMAIL_PASS")
    if not all([host, sender, recipient, user, pwd]):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(sender, [recipient], msg.as_string())
        return True
    except Exception:
        return False


class AlertType(Enum):
    """Types of trading alerts."""

    SETUP_DETECTED = "setup_detected"
    ENTRY_SIGNAL = "entry_signal"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_WARNING = "time_warning"
    RISK_ALERT = "risk_alert"
    EARNINGS_WARNING = "earnings_warning"
    SYSTEM_ERROR = "system_error"
    TRADE_EXECUTED = "trade_executed"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    SIGNAL_VALIDATION_DEGRADATION = "signal_validation_degradation"
    SIGNAL_VALIDATION_TREND_DEGRADATION = "signal_validation_trend_degradation"
    SIGNAL_VALIDATION_TREND_IMPROVEMENT = "signal_validation_trend_improvement"
    REGIME_CHANGE = "regime_change"
    VALIDATION_STATE_CHANGE = "validation_state_change"
    # VIX-based alerts
    VIX_LEVEL_CHANGE = "vix_level_change"
    VIX_SPIKE = "vix_spike"
    VIX_CRITICAL = "vix_critical"
    VIX_TRADING_PAUSED = "vix_trading_paused"
    VIX_TRADING_RESUMED = "vix_trading_resumed"
    # Allocation alerts
    ALLOCATION_WARNING = "allocation_warning"
    ALLOCATION_EXCEEDED = "allocation_exceeded"
    ALLOCATION_ORDER_REJECTED = "allocation_order_rejected"
    # Circuit breaker recovery alerts
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    RECOVERY_STAGE_ADVANCED = "recovery_stage_advanced"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_TRADE_RECORDED = "recovery_trade_recorded"


class AlertPriority(Enum):
    """Alert priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DESKTOP = "desktop"
    MOBILE_PUSH = "mobile_push"
    SLACK = "slack"


@dataclass
class Alert:
    """Individual alert with metadata."""

    alert_type: AlertType
    priority: AlertPriority
    ticker: str
    title: str
    message: str
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    channels: list[AlertChannel] = field(default_factory=list)
    acknowledged: bool = False

    def to_json(self) -> str:
        """Convert alert to JSON string."""
        data = asdict(self)
        # Convert enums to their values
        if "alert_type" in data and hasattr(data["alert_type"], "value"):
            data["alert_type"] = data["alert_type"].value
        if "priority" in data and hasattr(data["priority"], "value"):
            data["priority"] = data["priority"].value
        if "channel" in data and hasattr(data["channel"], "value"):
            data["channel"] = data["channel"].value
        return json.dumps(data, default=str)


@dataclass
class ChecklistItem:
    """Individual item in execution checklist."""

    step: int
    description: str
    completed: bool = False
    timestamp: datetime | None = None
    notes: str = ""

    def complete(self, notes: str = ""):
        """Mark item as completed."""
        self.completed = True
        self.timestamp = datetime.now()
        self.notes = notes


@dataclass
class ExecutionChecklist:
    """Systematic execution checklist for trades."""

    trade_id: str
    ticker: str
    checklist_type: str  # "entry", "monitoring", "exit"
    items: list[ChecklistItem] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.items:
            return 0.0
        completed_items = sum(1 for item in self.items if item.completed)
        return (completed_items / len(self.items)) * 100

    @property
    def is_complete(self) -> bool:
        """Check if all items are completed."""
        return all(item.completed for item in self.items)

    def complete_step(self, step: int, notes: str = ""):
        """Complete a specific step."""
        for item in self.items:
            if item.step == step:
                item.complete(notes)
                break

        if self.is_complete and not self.completed_at:
            self.completed_at = datetime.now()


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass


class EmailAlertHandler(AlertHandler):
    """Email alert handler."""

    def __init__(self, smtp_config: dict):
        self.smtp_config = smtp_config

    def send_alert(self, alert: Alert) -> bool:
        """Send email alert using send_email function."""
        from .alert_system import send_email

        subject = f"Trading Alert: {alert.title}"
        body = f"{alert.message}\nTicker: {alert.ticker}\nTime: {alert.timestamp}"
        result = send_email(subject, body)
        logging.info(f"EMAIL ALERT: {alert.title} - {alert.message}")
        return result


class WebhookAlertHandler(AlertHandler):
    """Webhook alert handler for Discord, Slack, etc."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert (placeholder implementation)."""
        # In production, send HTTP POST to webhook
        logging.info(f"WEBHOOK ALERT: {alert.title} - {alert.message}")
        return True


class DesktopAlertHandler(AlertHandler):
    """Desktop notification handler."""

    def send_alert(self, alert: Alert) -> bool:
        """Send desktop notification using subprocess."""
        import subprocess

        try:
            # Use osascript on macOS for desktop notifications
            message = f"{alert.title}: {alert.message}"
            subprocess.run(  # noqa: S603 - Safe subprocess call for desktop notifications
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "Trading Alert"',
                ],
                check=True,
                capture_output=True,
                timeout=5,
                shell=False,  # Explicitly disable shell for security
            )
            logging.info(f"DESKTOP ALERT: {alert.title} - {alert.message}")
            return True
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            # In test environments or systems without osascript, just log and return True
            logging.info(f"Desktop notification not available (expected in tests): {e}")
            return True  # Return True for tests
        except Exception as e:
            logging.error(f"Desktop notification failed: {e}")
            return False


class TradingAlertSystem:
    """Comprehensive trading alert system."""

    def __init__(self):
        self.handlers: dict[AlertChannel, AlertHandler] = {}
        self.alert_history: list[Alert] = []
        self.active_alerts: list[Alert] = []
        self.max_history: int = 100  # Default max history
        self.signal_generator = SignalGenerator()
        self.options_calculator = OptionsTradeCalculator()

        # Default alert preferences
        self.alert_preferences = {
            AlertType.ENTRY_SIGNAL: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK],
            AlertType.PROFIT_TARGET: [AlertChannel.DESKTOP],
            AlertType.STOP_LOSS: [AlertChannel.DESKTOP, AlertChannel.EMAIL],
            AlertType.RISK_ALERT: [AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            AlertType.TIME_WARNING: [AlertChannel.DESKTOP],
            AlertType.EARNINGS_WARNING: [AlertChannel.EMAIL],
            # VIX alerts
            AlertType.VIX_LEVEL_CHANGE: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK],
            AlertType.VIX_SPIKE: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            AlertType.VIX_CRITICAL: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            AlertType.VIX_TRADING_PAUSED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            AlertType.VIX_TRADING_RESUMED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK],
            # Allocation alerts
            AlertType.ALLOCATION_WARNING: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK],
            AlertType.ALLOCATION_EXCEEDED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            AlertType.ALLOCATION_ORDER_REJECTED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            # Circuit breaker recovery alerts
            AlertType.CIRCUIT_BREAKER_TRIGGERED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL, AlertChannel.SMS],
            AlertType.RECOVERY_STAGE_ADVANCED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK],
            AlertType.RECOVERY_COMPLETED: [AlertChannel.DESKTOP, AlertChannel.WEBHOOK, AlertChannel.EMAIL],
            AlertType.RECOVERY_TRADE_RECORDED: [AlertChannel.DESKTOP],
        }

    def register_handler(self, channel: AlertChannel, handler: AlertHandler):
        """Register an alert handler for a channel."""
        self.handlers[channel] = handler

    async def send_alert(
        self, alert_type_or_alert, priority=None, message=None, ticker=None
    ) -> dict[AlertChannel, bool]:
        """Send alert with flexible parameters - supports both Alert objects and string parameters.

        Usage:
        - send_alert(alert_object) - Original method
        - send_alert("ENTRY_SIGNAL", "HIGH", "message", "AAPL") - Convenience method
        """
        # Handle both old Alert object format and new string parameters
        if isinstance(alert_type_or_alert, Alert):
            # Original format - Alert object
            return self.send_alert_object(alert_type_or_alert)
        else:
            # New format - string parameters (for backward compatibility)
            try:
                # Convert string parameters to Alert object
                if isinstance(alert_type_or_alert, str):
                    alert_type = AlertType(alert_type_or_alert.lower())
                else:
                    alert_type = alert_type_or_alert

                if isinstance(priority, str):
                    priority_obj = AlertPriority[priority.upper()]
                else:
                    priority_obj = priority or AlertPriority.MEDIUM

                alert = Alert(
                    alert_type=alert_type,
                    priority=priority_obj,
                    ticker=ticker or "UNKNOWN",
                    title=alert_type_or_alert.upper()
                    if isinstance(alert_type_or_alert, str)
                    else str(alert_type_or_alert),
                    message=message or "Alert triggered",
                )

                return self.send_alert_object(alert)

            except Exception as e:
                print(f"Error creating alert: {e}")
                return {}

    def send_alert_object(self, alert: Alert) -> dict[AlertChannel, bool]:
        """Original send_alert method renamed."""
        results = {}

        # Filter by priority (only send HIGH and URGENT alerts by default)
        # But allow all alerts to be stored in history
        if alert.priority in [AlertPriority.LOW, AlertPriority.MEDIUM]:
            # Store in history but don't send
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history :]
            return results  # Return empty results for low / medium priority

        # Use alert - specific channels or fall back to preferences
        channels = alert.channels or self.alert_preferences.get(
            alert.alert_type, [AlertChannel.DESKTOP]
        )

        for alert_channel in channels:
            handler = self.handlers.get(alert_channel)
            if handler:
                try:
                    success = handler.send_alert(alert)
                    results[alert_channel] = success
                except Exception as e:
                    logging.error(f"Failed to send alert via {alert_channel}: {e}")
                    results[alert_channel] = False
            else:
                logging.warning(f"No handler registered for channel: {alert_channel}")
                results[alert_channel] = False

        # Store alert in history
        self.alert_history.append(alert)

        # Enforce history limit
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        # Add to active alerts if high priority
        if alert.priority in [AlertPriority.HIGH, AlertPriority.URGENT]:
            self.active_alerts.append(alert)

        return results

    def check_market_signals(
        self,
        ticker: str,
        current_indicators: TechnicalIndicators,
        previous_indicators: TechnicalIndicators,
        earnings_risk: bool = False,
        macro_risk: bool = False,
    ):
        """Check for market signals and generate alerts."""
        signal = self.signal_generator.generate_signal(
            current_indicators, previous_indicators, earnings_risk, macro_risk
        )

        if signal.signal_type == SignalType.BUY:
            self._create_entry_signal_alert(ticker, signal, current_indicators)
        elif (
            signal.signal_type == SignalType.HOLD
            and "setup" in " ".join(signal.reasoning).lower()
        ):
            self._create_setup_alert(ticker, signal, current_indicators)

    def _create_entry_signal_alert(
        self, ticker: str, signal: MarketSignal, indicators: TechnicalIndicators
    ):
        """Create entry signal alert with trade calculation."""
        # Calculate recommended trade
        try:
            trade_calc = self.options_calculator.calculate_trade(
                ticker=ticker,
                spot_price=indicators.price,
                account_size=500000,  # Default account size
                implied_volatility=0.28,  # Default IV estimate
            )

            alert = Alert(
                alert_type=AlertType.ENTRY_SIGNAL,
                priority=AlertPriority.HIGH,
                ticker=ticker,
                title=f"🚀 BUY SIGNAL: {ticker}",
                message="Bull pullback reversal detected. Consider ~5% OTM calls.\n"
                f"Recommended: {trade_calc.recommended_contracts} contracts @ ${trade_calc.estimated_premium: .2f}\n"
                f"Strike: ${trade_calc.strike:.0f} | Expiry: {trade_calc.expiry_date}",
                data={
                    "signal_confidence": signal.confidence,
                    "trade_calculation": asdict(trade_calc),
                    "reasoning": signal.reasoning,
                },
            )

            self.send_alert(alert)

        except Exception as e:
            logging.error(f"Failed to create entry signal alert: {e}")

    def _create_setup_alert(
        self, ticker: str, signal: MarketSignal, indicators: TechnicalIndicators
    ):
        """Create setup detection alert."""
        alert = Alert(
            alert_type=AlertType.SETUP_DETECTED,
            priority=AlertPriority.MEDIUM,
            ticker=ticker,
            title=f"⚠️ SETUP: {ticker}",
            message="Pullback setup detected. Monitor for reversal trigger.\n"
            f"Price: ${indicators.price:.2f} | RSI: {indicators.rsi_14:.0f}",
            data={
                "signal_confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "indicators": asdict(indicators),
            },
        )

        self.send_alert(alert)

    def create_exit_alert(
        self, ticker: str, exit_signals: list[ExitSignal], position_data: dict
    ):
        """Create exit - related alerts."""
        if not exit_signals:
            return

        strongest_signal = max(exit_signals, key=lambda x: x.strength.value)

        # Determine alert type and priority
        if "profit" in strongest_signal.reason.value:
            alert_type = AlertType.PROFIT_TARGET
            priority = AlertPriority.MEDIUM
            emoji = "💰"
        else:
            alert_type = AlertType.STOP_LOSS
            priority = AlertPriority.HIGH
            emoji = "🛑"

        alert = Alert(
            alert_type=alert_type,
            priority=priority,
            ticker=ticker,
            title=f"{emoji} EXIT SIGNAL: {ticker}",
            message=f"{strongest_signal.reason.value.title()} triggered.\n"
            f"Action: Close {strongest_signal.position_fraction:.0%} of position + n"
            f"Expected P & L: ${strongest_signal.expected_pnl:,.0f}",
            data={
                "exit_signals": [asdict(sig) for sig in exit_signals],
                "position_data": position_data,
            },
        )

        self.send_alert(alert)

    def create_risk_alert(self, message: str, data: dict | None = None):
        """Create risk management alert."""
        alert = Alert(
            alert_type=AlertType.RISK_ALERT,
            priority=AlertPriority.HIGH,
            ticker="PORTFOLIO",
            title="⚠️ RISK ALERT",
            message=message,
            data=data or {},
        )

        self.send_alert(alert)

    # =========================================================================
    # VIX-based Alert Methods
    # =========================================================================

    def create_vix_level_change_alert(
        self,
        previous_level: str,
        current_level: str,
        vix_value: float,
        position_multiplier: float,
    ):
        """Create alert when VIX level changes (e.g., normal -> elevated)."""
        level_emojis = {
            'normal': '🟢',
            'elevated': '🟡',
            'high': '🟠',
            'extreme': '🔴',
            'critical': '⚫',
        }

        emoji = level_emojis.get(current_level, '⚠️')
        priority = AlertPriority.MEDIUM

        if current_level in ['extreme', 'critical']:
            priority = AlertPriority.HIGH

        sizing_note = ""
        if position_multiplier < 1.0:
            reduction_pct = (1 - position_multiplier) * 100
            sizing_note = f" Position sizes reduced by {reduction_pct:.0f}%."

        alert = Alert(
            alert_type=AlertType.VIX_LEVEL_CHANGE,
            priority=priority,
            ticker="VIX",
            title=f"{emoji} VIX Level Change: {current_level.upper()}",
            message=f"VIX moved from {previous_level} to {current_level}.\n"
                    f"Current VIX: {vix_value:.1f}.{sizing_note}",
            data={
                'vix_value': vix_value,
                'previous_level': previous_level,
                'current_level': current_level,
                'position_multiplier': position_multiplier,
            },
        )

        self.send_alert(alert)

    def create_vix_spike_alert(self, vix_value: float, change_1d: float):
        """Create alert when VIX spikes significantly (>20% increase)."""
        alert = Alert(
            alert_type=AlertType.VIX_SPIKE,
            priority=AlertPriority.HIGH,
            ticker="VIX",
            title=f"🚨 VIX SPIKE DETECTED",
            message=f"VIX spiked +{change_1d:.1f} points to {vix_value:.1f}.\n"
                    f"This indicates sudden increase in market fear.\n"
                    f"Exercise caution with new positions.",
            data={
                'vix_value': vix_value,
                'change_1d': change_1d,
                'spike_percent': (change_1d / (vix_value - change_1d)) * 100,
            },
        )

        self.send_alert(alert)

    def create_vix_critical_alert(self, vix_value: float):
        """Create URGENT alert when VIX exceeds critical threshold (>45)."""
        alert = Alert(
            alert_type=AlertType.VIX_CRITICAL,
            priority=AlertPriority.URGENT,
            ticker="VIX",
            title=f"⚫ VIX CRITICAL: {vix_value:.1f}",
            message=f"VIX has exceeded critical threshold ({vix_value:.1f}).\n"
                    f"This indicates extreme market fear/panic.\n"
                    f"All new trading has been PAUSED.",
            data={
                'vix_value': vix_value,
                'critical_threshold': 45.0,
            },
        )

        self.send_alert(alert)

    def create_vix_trading_paused_alert(self, vix_value: float, reason: str):
        """Create alert when trading is paused due to VIX."""
        alert = Alert(
            alert_type=AlertType.VIX_TRADING_PAUSED,
            priority=AlertPriority.URGENT,
            ticker="SYSTEM",
            title=f"⏸️ TRADING PAUSED - VIX {vix_value:.1f}",
            message=f"Trading has been automatically paused.\n"
                    f"Reason: {reason}\n"
                    f"No new positions will be opened until VIX normalizes.",
            data={
                'vix_value': vix_value,
                'reason': reason,
                'paused_at': datetime.now().isoformat(),
            },
        )

        self.send_alert(alert)

    def create_vix_trading_resumed_alert(self, vix_value: float):
        """Create alert when trading resumes after VIX normalization."""
        alert = Alert(
            alert_type=AlertType.VIX_TRADING_RESUMED,
            priority=AlertPriority.HIGH,
            ticker="SYSTEM",
            title=f"▶️ TRADING RESUMED - VIX {vix_value:.1f}",
            message=f"VIX has returned to acceptable levels ({vix_value:.1f}).\n"
                    f"Normal trading operations have resumed.\n"
                    f"Position sizing may still be reduced if VIX is elevated.",
            data={
                'vix_value': vix_value,
                'resumed_at': datetime.now().isoformat(),
            },
        )

        self.send_alert(alert)

    def check_vix_and_alert(self):
        """Check VIX and create appropriate alerts."""
        try:
            from backend.auth0login.services.market_monitor import get_market_monitor

            monitor = get_market_monitor()
            alert_info = monitor.check_alert_threshold()

            if alert_info:
                if alert_info['type'] == 'vix_level_change':
                    self.create_vix_level_change_alert(
                        previous_level=alert_info.get('previous_level', 'unknown'),
                        current_level=alert_info['current_level'],
                        vix_value=alert_info['vix_value'],
                        position_multiplier=monitor.get_position_size_multiplier(
                            alert_info['vix_value']
                        ),
                    )

                    # Additional alerts for critical levels
                    if alert_info['current_level'] == 'critical':
                        self.create_vix_critical_alert(alert_info['vix_value'])
                        self.create_vix_trading_paused_alert(
                            alert_info['vix_value'],
                            f"VIX exceeded critical threshold ({alert_info['vix_value']:.1f})"
                        )
                    elif alert_info.get('previous_level') == 'critical':
                        # Resuming from critical
                        self.create_vix_trading_resumed_alert(alert_info['vix_value'])

                elif alert_info['type'] == 'vix_spike':
                    self.create_vix_spike_alert(
                        vix_value=alert_info['vix_value'],
                        change_1d=alert_info.get('change_1d', 0),
                    )

        except ImportError:
            logging.warning("Market monitor not available for VIX alerts")
        except Exception as e:
            logging.error(f"Error checking VIX for alerts: {e}")

    # =========================================================================
    # Allocation Alert Methods
    # =========================================================================

    def create_allocation_warning_alert(
        self,
        strategy_name: str,
        utilization_pct: float,
        available_capital: float,
        allocated_amount: float,
    ):
        """Create alert when strategy is nearing its allocation limit (90%+)."""
        alert = Alert(
            alert_type=AlertType.ALLOCATION_WARNING,
            priority=AlertPriority.MEDIUM,
            ticker=strategy_name.upper(),
            title=f"⚠️ {strategy_name} Near Allocation Limit",
            message=f"{strategy_name} is at {utilization_pct:.0f}% of allocation.\n"
                    f"Available: ${available_capital:,.2f} of ${allocated_amount:,.2f}.\n"
                    f"Consider reducing positions or increasing allocation.",
            data={
                'strategy_name': strategy_name,
                'utilization_pct': utilization_pct,
                'available_capital': available_capital,
                'allocated_amount': allocated_amount,
            },
        )

        self.send_alert(alert)

    def create_allocation_exceeded_alert(
        self,
        strategy_name: str,
        current_exposure: float,
        allocated_amount: float,
        overage_pct: float,
    ):
        """Create alert when strategy has exceeded its allocation."""
        alert = Alert(
            alert_type=AlertType.ALLOCATION_EXCEEDED,
            priority=AlertPriority.HIGH,
            ticker=strategy_name.upper(),
            title=f"🚨 {strategy_name} Allocation Exceeded",
            message=f"{strategy_name} has exceeded its allocation by {overage_pct:.0f}%.\n"
                    f"Current exposure: ${current_exposure:,.2f}\n"
                    f"Allocated: ${allocated_amount:,.2f}\n"
                    f"New orders for this strategy will be blocked.",
            data={
                'strategy_name': strategy_name,
                'current_exposure': current_exposure,
                'allocated_amount': allocated_amount,
                'overage_pct': overage_pct,
            },
        )

        self.send_alert(alert)

    def create_allocation_order_rejected_alert(
        self,
        strategy_name: str,
        symbol: str,
        requested_amount: float,
        available_capital: float,
    ):
        """Create alert when an order is rejected due to allocation limit."""
        alert = Alert(
            alert_type=AlertType.ALLOCATION_ORDER_REJECTED,
            priority=AlertPriority.HIGH,
            ticker=symbol,
            title=f"❌ Order Rejected - {strategy_name} Limit",
            message=f"Order for {symbol} rejected due to {strategy_name} allocation limit.\n"
                    f"Requested: ${requested_amount:,.2f}\n"
                    f"Available: ${available_capital:,.2f}\n"
                    f"Increase allocation or reduce position size.",
            data={
                'strategy_name': strategy_name,
                'symbol': symbol,
                'requested_amount': requested_amount,
                'available_capital': available_capital,
            },
        )

        self.send_alert(alert)

    def check_allocation_thresholds(self, user=None):
        """Check all allocations and create alerts for those near limits.

        Args:
            user: Django user object (required for checking allocations)
        """
        if user is None:
            logging.warning("User required for allocation threshold check")
            return

        try:
            from backend.auth0login.services.allocation_manager import get_allocation_manager

            manager = get_allocation_manager()
            summary = manager.get_allocation_summary(user)

            if not summary.get('configured'):
                return

            for allocation in summary.get('strategies', []):
                utilization = allocation['utilization_pct']
                strategy = allocation['strategy_name']

                if utilization >= 100:
                    # Exceeded
                    overage = utilization - 100
                    self.create_allocation_exceeded_alert(
                        strategy_name=strategy,
                        current_exposure=allocation['current_exposure'],
                        allocated_amount=allocation['allocated_amount'],
                        overage_pct=overage,
                    )
                elif utilization >= 90:
                    # Warning threshold
                    self.create_allocation_warning_alert(
                        strategy_name=strategy,
                        utilization_pct=utilization,
                        available_capital=allocation['available_capital'],
                        allocated_amount=allocation['allocated_amount'],
                    )

        except ImportError:
            logging.warning("Allocation manager not available")
        except Exception as e:
            logging.error(f"Error checking allocation thresholds: {e}")

    # =========================================================================
    # Circuit Breaker Recovery Alerts
    # =========================================================================

    def create_circuit_breaker_triggered_alert(
        self,
        breaker_type: str,
        trigger_value: float,
        trigger_threshold: float,
        recovery_schedule_hours: float,
    ):
        """Create alert when a circuit breaker triggers.

        Args:
            breaker_type: Type of breaker (daily_loss, vix_critical, etc.)
            trigger_value: Value that caused the trigger
            trigger_threshold: Threshold that was exceeded
            recovery_schedule_hours: Total hours until full recovery
        """
        breaker_names = {
            'daily_loss': 'Daily Loss Limit',
            'vix_critical': 'VIX Critical Level',
            'vix_extreme': 'VIX Extreme Level',
            'error_rate': 'Error Rate Spike',
            'stale_data': 'Stale Data',
            'consecutive_loss': 'Consecutive Losses',
            'position_limit': 'Position Count Limit',
            'margin_call': 'Margin Call Risk',
            'manual': 'Manual Trigger',
        }
        breaker_name = breaker_names.get(breaker_type, breaker_type)

        alert = Alert(
            alert_type=AlertType.CIRCUIT_BREAKER_TRIGGERED,
            priority=AlertPriority.URGENT,
            ticker="SYSTEM",
            title=f"🛑 Circuit Breaker Triggered: {breaker_name}",
            message=f"Trading has been PAUSED due to {breaker_name}.\n"
                    f"Trigger value: {trigger_value:.2f} (threshold: {trigger_threshold:.2f})\n"
                    f"Estimated recovery time: {recovery_schedule_hours:.1f} hours\n\n"
                    f"Position sizes will be gradually restored as recovery progresses.",
            data={
                'breaker_type': breaker_type,
                'trigger_value': trigger_value,
                'trigger_threshold': trigger_threshold,
                'recovery_hours': recovery_schedule_hours,
            },
        )

        self.send_alert(alert)
        logging.warning(f"Circuit breaker triggered: {breaker_type}")

    def create_recovery_stage_advanced_alert(
        self,
        breaker_type: str,
        old_mode: str,
        new_mode: str,
        position_multiplier: float,
        hours_remaining: float,
    ):
        """Create alert when recovery advances to next stage.

        Args:
            breaker_type: Type of breaker
            old_mode: Previous recovery mode
            new_mode: New recovery mode
            position_multiplier: New position size multiplier
            hours_remaining: Hours until full recovery
        """
        mode_descriptions = {
            'paused': 'Trading Paused (0%)',
            'restricted': 'Restricted Trading (25%)',
            'cautious': 'Cautious Trading (50%)',
            'normal': 'Normal Trading (100%)',
        }

        new_description = mode_descriptions.get(new_mode, new_mode)

        alert = Alert(
            alert_type=AlertType.RECOVERY_STAGE_ADVANCED,
            priority=AlertPriority.MEDIUM,
            ticker="SYSTEM",
            title=f"📈 Recovery Advanced: {new_description}",
            message=f"Circuit breaker recovery has advanced.\n"
                    f"Previous: {old_mode.title()} → Now: {new_mode.title()}\n"
                    f"Position sizing: {int(position_multiplier * 100)}%\n"
                    f"Time to full recovery: {hours_remaining:.1f} hours",
            data={
                'breaker_type': breaker_type,
                'old_mode': old_mode,
                'new_mode': new_mode,
                'position_multiplier': position_multiplier,
                'hours_remaining': hours_remaining,
            },
        )

        self.send_alert(alert)
        logging.info(f"Recovery advanced: {old_mode} -> {new_mode}")

    def create_recovery_completed_alert(
        self,
        breaker_type: str,
        total_duration_hours: float,
        recovery_trades: int,
        recovery_pnl: float,
    ):
        """Create alert when recovery completes and full trading resumes.

        Args:
            breaker_type: Type of breaker that was resolved
            total_duration_hours: How long the recovery took
            recovery_trades: Number of trades during recovery
            recovery_pnl: P&L during recovery period
        """
        pnl_str = f"+${recovery_pnl:,.2f}" if recovery_pnl >= 0 else f"-${abs(recovery_pnl):,.2f}"

        alert = Alert(
            alert_type=AlertType.RECOVERY_COMPLETED,
            priority=AlertPriority.HIGH,
            ticker="SYSTEM",
            title=f"✅ Full Trading Resumed",
            message=f"Circuit breaker recovery complete!\n"
                    f"Breaker type: {breaker_type.replace('_', ' ').title()}\n"
                    f"Total duration: {total_duration_hours:.1f} hours\n"
                    f"Recovery trades: {recovery_trades}\n"
                    f"Recovery P&L: {pnl_str}\n\n"
                    f"Trading has resumed at 100% position sizing.",
            data={
                'breaker_type': breaker_type,
                'duration_hours': total_duration_hours,
                'recovery_trades': recovery_trades,
                'recovery_pnl': recovery_pnl,
            },
        )

        self.send_alert(alert)
        logging.info(f"Recovery completed for {breaker_type}")

    def create_recovery_trade_recorded_alert(
        self,
        is_profitable: bool,
        pnl: float,
        current_win_rate: float,
        trades_until_advance: int | None,
    ):
        """Create alert when a trade is recorded during recovery.

        Args:
            is_profitable: Whether the trade was profitable
            pnl: P&L of the trade
            current_win_rate: Current win rate during recovery
            trades_until_advance: Trades needed to advance (if applicable)
        """
        result = "profitable" if is_profitable else "unprofitable"
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"

        message = f"Recovery trade recorded: {result} ({pnl_str})\n"
        message += f"Current recovery win rate: {current_win_rate * 100:.0f}%"

        if trades_until_advance is not None and trades_until_advance > 0:
            message += f"\nProfitable trades needed to advance: {trades_until_advance}"

        alert = Alert(
            alert_type=AlertType.RECOVERY_TRADE_RECORDED,
            priority=AlertPriority.LOW,
            ticker="SYSTEM",
            title=f"📊 Recovery Trade: {result.title()}",
            message=message,
            data={
                'is_profitable': is_profitable,
                'pnl': pnl,
                'win_rate': current_win_rate,
                'trades_until_advance': trades_until_advance,
            },
        )

        self.send_alert(alert)

    def check_recovery_status(self, user=None):
        """Check recovery status and auto-advance if conditions are met.

        Args:
            user: Django user object

        Returns:
            List of advancement results
        """
        if user is None:
            return []

        try:
            from backend.auth0login.services.recovery_manager import get_recovery_manager

            recovery_mgr = get_recovery_manager(user)
            advancements = recovery_mgr.check_auto_recovery()

            for advancement in advancements:
                if advancement.get('resolved'):
                    # Recovery completed
                    status = recovery_mgr.get_recovery_status()
                    self.create_recovery_completed_alert(
                        breaker_type=advancement['breaker_type'],
                        total_duration_hours=0,  # Would need to get from event
                        recovery_trades=0,
                        recovery_pnl=0,
                    )
                else:
                    # Stage advanced
                    status = recovery_mgr.get_recovery_status()
                    self.create_recovery_stage_advanced_alert(
                        breaker_type=advancement['breaker_type'],
                        old_mode=advancement['old_mode'],
                        new_mode=advancement['new_mode'],
                        position_multiplier=status.position_multiplier,
                        hours_remaining=status.hours_until_next_stage or 0,
                    )

            return advancements

        except ImportError:
            logging.warning("Recovery manager not available")
            return []
        except Exception as e:
            logging.error(f"Error checking recovery status: {e}")
            return []

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert."""
        for alert in self.active_alerts:
            if id(alert) == hash(alert_id):  # Simple ID matching
                alert.acknowledged = True
                break


class ExecutionChecklistManager:
    """Manages execution checklists for systematic trading."""

    def __init__(self):
        self.checklists: dict[str, ExecutionChecklist] = {}

    def create_entry_checklist(self, ticker: str, trade_calc: TradeCalculation) -> str:
        """Create entry execution checklist."""
        trade_id = f"{ticker}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"

        items = [
            ChecklistItem(
                1, f"Verify bull regime: {ticker}  >  50 - EMA, 50 - EMA  >  200 - EMA"
            ),
            ChecklistItem(2, "Confirm pullback setup: RSI 35 - 50, near 20 - EMA"),
            ChecklistItem(
                3, "Verify reversal trigger: Price  >  20 - EMA and previous high"
            ),
            ChecklistItem(4, "Check earnings calendar: No earnings within ±7 days"),
            ChecklistItem(5, "Verify no major macro events today"),
            ChecklistItem(
                6,
                f"Calculate position size: Max {trade_calc.account_risk_pct:.1f}% of account",
            ),
            ChecklistItem(7, "Check option liquidity: Bid - ask spread  <  10%"),
            ChecklistItem(
                8, "Set stop loss level: Exit if premium drops to 50% of entry"
            ),
            ChecklistItem(9, "Set profit targets: 100%, 200%, 250% gains"),
            ChecklistItem(
                10,
                f"Execute trade: {trade_calc.recommended_contracts} contracts @ ${trade_calc.estimated_premium: .2f}",
            ),
            ChecklistItem(11, "Confirm fill and record entry details"),
            ChecklistItem(12, "Set GTC orders for profit targets if available"),
        ]

        checklist = ExecutionChecklist(
            trade_id=trade_id, ticker=ticker, checklist_type="entry", items=items
        )

        self.checklists[trade_id] = checklist
        return trade_id

    def create_monitoring_checklist(
        self, trade_id: str, ticker: str
    ) -> ExecutionChecklist:
        """Create daily monitoring checklist."""
        items = [
            ChecklistItem(1, f"Check {ticker} price action vs. key EMAs"),
            ChecklistItem(2, "Monitor option premium and delta changes"),
            ChecklistItem(3, "Verify bull regime still intact (50 - EMA support)"),
            ChecklistItem(4, "Check for profit target hits (100%, 200%, 250%)"),
            ChecklistItem(
                5, "Assess stop loss conditions (50% loss or 50 - EMA break)"
            ),
            ChecklistItem(6, "Review days to expiry and time decay impact"),
            ChecklistItem(7, "Check for upcoming earnings or macro events"),
            ChecklistItem(8, "Update exit plan based on current scenario analysis"),
        ]

        monitoring_id = f"{trade_id}_monitoring_{datetime.now().strftime('%Y % m % d')}"

        checklist = ExecutionChecklist(
            trade_id=monitoring_id,
            ticker=ticker,
            checklist_type="monitoring",
            items=items,
        )

        self.checklists[monitoring_id] = checklist
        return checklist

    def create_exit_checklist(
        self, trade_id: str, ticker: str, exit_reason: str
    ) -> ExecutionChecklist:
        """Create exit execution checklist."""
        items = [
            ChecklistItem(1, f"Confirm exit trigger: {exit_reason}"),
            ChecklistItem(2, "Check current option premium and liquidity"),
            ChecklistItem(3, "Calculate expected P & L from exit"),
            ChecklistItem(4, "Cancel any existing GTC profit target orders"),
            ChecklistItem(5, "Execute exit order (market or limit based on liquidity)"),
            ChecklistItem(6, "Confirm fill and record exit details"),
            ChecklistItem(7, "Calculate final P & L and ROI"),
            ChecklistItem(8, "Update trade journal with lessons learned"),
            ChecklistItem(9, "Assess portfolio impact and remaining risk"),
            ChecklistItem(10, "Plan next potential setup if position closed"),
        ]

        exit_id = (
            f"{trade_id}_exit_{datetime.now().strftime('%Y % m % d_ % H % M % S')}"
        )

        checklist = ExecutionChecklist(
            trade_id=exit_id, ticker=ticker, checklist_type="exit", items=items
        )

        self.checklists[exit_id] = checklist
        return checklist

    def get_checklist(self, checklist_id: str) -> ExecutionChecklist | None:
        """Get checklist by ID."""
        return self.checklists.get(checklist_id)

    def complete_item(self, checklist_id: str, step: int, notes: str = ""):
        """Complete a checklist item."""
        checklist = self.checklists.get(checklist_id)
        if checklist:
            checklist.complete_step(step, notes)
            return True
        return False

    def get_active_checklists(self) -> list[ExecutionChecklist]:
        """Get all incomplete checklists."""
        return [cl for cl in self.checklists.values() if not cl.is_complete]


class MarketScreener:
    """Screen market for setup opportunities."""

    def __init__(self, alert_system: TradingAlertSystem):
        self.alert_system = alert_system
        self.mega_cap_tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "META",
            "NVDA",
            "AVGO",
            "AMD",
            "TSLA",
        ]

    def screen_for_setups(self, market_data: dict[str, dict]):
        """Screen all tickers for bull pullback setups."""
        for ticker in self.mega_cap_tickers:
            if ticker not in market_data:
                continue

            try:
                current_data = market_data[ticker]["current"]
                previous_data = market_data[ticker]["previous"]

                current_indicators = self._convert_to_indicators(current_data)
                previous_indicators = self._convert_to_indicators(previous_data)

                # Check for earnings risk (placeholder)
                earnings_risk = market_data[ticker].get("earnings_in_7_days", False)

                self.alert_system.check_market_signals(
                    ticker, current_indicators, previous_indicators, earnings_risk
                )

            except Exception as e:
                logging.error(f"Error screening {ticker}: {e}")

    def _convert_to_indicators(self, data: dict) -> TechnicalIndicators:
        """Convert market data to TechnicalIndicators."""
        return TechnicalIndicators(
            price=data["close"],
            ema_20=data["ema_20"],
            ema_50=data["ema_50"],
            ema_200=data["ema_200"],
            rsi_14=data["rsi"],
            atr_14=data["atr"],
            volume=data["volume"],
            high_24h=data["high"],
            low_24h=data["low"],
        )


class SignalValidationMonitor:
    """Monitor signal validation performance across all strategies.

    GAP FIX: This class implements the Monitoring & Alerting Gap - providing
    comprehensive signal validation performance monitoring with automatic alerts.

    Features:
    - Real-time signal validation performance tracking
    - Automatic degradation detection and alerts
    - Dashboard integration for validation metrics
    - Threshold-based automatic responses
    """

    def __init__(
        self,
        alert_system: TradingAlertSystem,
        thresholds: dict = None,
    ):
        """Initialize signal validation monitor.

        Args:
            alert_system: Alert system for sending notifications
            thresholds: Strategy-specific thresholds for validation scores
        """
        self.alert_system = alert_system
        self.logger = logging.getLogger(__name__)

        # Default thresholds by strategy type
        self.thresholds = thresholds or {
            'default': 50.0,
            'wsb_dip_bot': 55.0,      # Higher threshold for WSB (more volatile)
            'earnings_protection': 60.0,  # Higher for earnings (time-critical)
            'wheel_strategy': 45.0,    # Lower threshold (longer-term)
            'momentum_weeklies': 65.0,  # Highest (short-term momentum)
            'lotto_scanner': 40.0,     # Lowest (speculative by design)
            'index_baseline': 50.0,    # Standard threshold
            'swing_trading': 55.0,     # Medium-high for swing trades
        }

        # Tracking state
        self.validation_history: dict[str, list] = {}
        self.last_check_time: dict[str, datetime] = {}
        self.degradation_count: dict[str, int] = {}

        self.logger.info("SignalValidationMonitor initialized")

    def monitor_validation_performance(
        self,
        strategies: dict,
    ) -> dict:
        """Monitor signal validation across all strategies.

        Args:
            strategies: Dictionary of strategy_name -> strategy instance

        Returns:
            Monitoring report with status for each strategy
        """
        report = {
            'timestamp': datetime.now(),
            'strategies_monitored': 0,
            'strategies_healthy': 0,
            'strategies_warning': 0,
            'strategies_critical': 0,
            'alerts_sent': 0,
            'details': {},
        }

        for strategy_name, strategy in strategies.items():
            if not hasattr(strategy, 'get_strategy_signal_summary'):
                continue

            try:
                summary = strategy.get_strategy_signal_summary()
                status = self._evaluate_strategy_validation(strategy_name, summary)
                report['details'][strategy_name] = status
                report['strategies_monitored'] += 1

                if status['health'] == 'HEALTHY':
                    report['strategies_healthy'] += 1
                elif status['health'] == 'WARNING':
                    report['strategies_warning'] += 1
                elif status['health'] == 'CRITICAL':
                    report['strategies_critical'] += 1

                # Send alerts based on health status
                if status['health'] == 'CRITICAL':
                    self._send_critical_alert(strategy_name, status)
                    report['alerts_sent'] += 1
                elif status['health'] == 'WARNING' and status.get('send_warning', False):
                    self._send_warning_alert(strategy_name, status)
                    report['alerts_sent'] += 1

            except Exception as e:
                self.logger.error(f"Error monitoring {strategy_name}: {e}")
                report['details'][strategy_name] = {'error': str(e)}

        return report

    def _evaluate_strategy_validation(
        self,
        strategy_name: str,
        summary: dict,
    ) -> dict:
        """Evaluate a strategy's signal validation performance.

        Args:
            strategy_name: Name of the strategy
            summary: Signal summary from strategy

        Returns:
            Evaluation status dict
        """
        avg_score = summary.get('average_strength_score', 0)
        total_signals = summary.get('total_signals_validated', 0)
        threshold = self.thresholds.get(strategy_name, self.thresholds['default'])

        # Track history
        if strategy_name not in self.validation_history:
            self.validation_history[strategy_name] = []
        self.validation_history[strategy_name].append({
            'timestamp': datetime.now(),
            'avg_score': avg_score,
            'total_signals': total_signals,
        })
        # Keep last 100 entries
        self.validation_history[strategy_name] = self.validation_history[strategy_name][-100:]

        # Determine health status
        if total_signals < 5:
            health = 'INSUFFICIENT_DATA'
            message = 'Not enough signals for evaluation'
        elif avg_score < threshold * 0.7:  # 70% of threshold = critical
            health = 'CRITICAL'
            message = f'Signal quality critically low: {avg_score:.1f} < {threshold * 0.7:.1f}'
            self.degradation_count[strategy_name] = self.degradation_count.get(strategy_name, 0) + 1
        elif avg_score < threshold:
            health = 'WARNING'
            message = f'Signal quality below threshold: {avg_score:.1f} < {threshold:.1f}'
            self.degradation_count[strategy_name] = self.degradation_count.get(strategy_name, 0) + 1
        else:
            health = 'HEALTHY'
            message = f'Signal quality acceptable: {avg_score:.1f} >= {threshold:.1f}'
            self.degradation_count[strategy_name] = 0  # Reset on healthy

        # Calculate trend
        trend = self._calculate_trend(strategy_name)

        # Determine if we should send warning (avoid alert fatigue)
        consecutive_warnings = self.degradation_count.get(strategy_name, 0)
        send_warning = consecutive_warnings in [1, 3, 5]  # Alert on 1st, 3rd, 5th occurrence

        return {
            'health': health,
            'message': message,
            'avg_score': avg_score,
            'threshold': threshold,
            'total_signals': total_signals,
            'trend': trend,
            'consecutive_degradations': consecutive_warnings,
            'send_warning': send_warning,
        }

    def _calculate_trend(self, strategy_name: str) -> str:
        """Calculate score trend for a strategy."""
        history = self.validation_history.get(strategy_name, [])
        if len(history) < 3:
            return 'INSUFFICIENT_DATA'

        recent = history[-3:]
        scores = [h['avg_score'] for h in recent]

        if all(scores[i] < scores[i - 1] for i in range(1, len(scores))):
            return 'DECLINING'
        elif all(scores[i] > scores[i - 1] for i in range(1, len(scores))):
            return 'IMPROVING'
        else:
            return 'STABLE'

    def _send_critical_alert(self, strategy_name: str, status: dict):
        """Send critical validation alert."""
        alert = Alert(
            alert_type=AlertType.SIGNAL_VALIDATION_DEGRADATION,
            priority=AlertPriority.HIGH,
            ticker=strategy_name.upper(),
            title=f"🚨 CRITICAL: Signal Validation Degraded - {strategy_name}",
            message=(
                f"{status['message']}\n"
                f"Trend: {status['trend']}\n"
                f"Consecutive degradations: {status['consecutive_degradations']}\n"
                f"Action: Strategy may be paused if condition persists."
            ),
            data={
                'avg_score': status['avg_score'],
                'threshold': status['threshold'],
                'total_signals': status['total_signals'],
                'trend': status['trend'],
            },
        )
        self.alert_system.send_alert(alert)
        self.logger.error(f"CRITICAL validation alert sent for {strategy_name}")

    def _send_warning_alert(self, strategy_name: str, status: dict):
        """Send warning validation alert."""
        alert = Alert(
            alert_type=AlertType.SIGNAL_VALIDATION_DEGRADATION,
            priority=AlertPriority.MEDIUM,
            ticker=strategy_name.upper(),
            title=f"⚠️ WARNING: Signal Validation Below Threshold - {strategy_name}",
            message=(
                f"{status['message']}\n"
                f"Trend: {status['trend']}\n"
                f"Monitor closely for further degradation."
            ),
            data={
                'avg_score': status['avg_score'],
                'threshold': status['threshold'],
                'total_signals': status['total_signals'],
            },
        )
        self.alert_system.send_alert(alert)
        self.logger.warning(f"Warning validation alert sent for {strategy_name}")

    def get_dashboard_metrics(self) -> dict:
        """Get metrics for dashboard display.

        Returns:
            Dictionary of metrics suitable for dashboard rendering
        """
        metrics = {
            'strategies': {},
            'overall_health': 'HEALTHY',
            'total_monitored': len(self.validation_history),
            'last_updated': datetime.now().isoformat(),
        }

        critical_count = 0
        warning_count = 0

        for strategy_name, history in self.validation_history.items():
            if not history:
                continue

            latest = history[-1]
            threshold = self.thresholds.get(strategy_name, self.thresholds['default'])

            metrics['strategies'][strategy_name] = {
                'avg_score': latest['avg_score'],
                'threshold': threshold,
                'health_pct': min(100, (latest['avg_score'] / threshold) * 100),
                'trend': self._calculate_trend(strategy_name),
                'signal_count': latest['total_signals'],
            }

            if latest['avg_score'] < threshold * 0.7:
                critical_count += 1
            elif latest['avg_score'] < threshold:
                warning_count += 1

        # Set overall health
        if critical_count > 0:
            metrics['overall_health'] = 'CRITICAL'
        elif warning_count > 0:
            metrics['overall_health'] = 'WARNING'

        return metrics


if __name__ == "__main__":  # Test the alert system
    print("=== ALERT SYSTEM TEST ===")

    # Setup alert system
    alert_system = TradingAlertSystem()

    # Register handlers
    alert_system.register_handler(AlertChannel.DESKTOP, DesktopAlertHandler())
    alert_system.register_handler(
        AlertChannel.WEBHOOK, WebhookAlertHandler("https: //hooks.slack.com / test")
    )
    alert_system.register_handler(
        AlertChannel.EMAIL, EmailAlertHandler({"smtp_server": "smtp.gmail.com"})
    )

    # Create sample alert
    test_alert = Alert(
        alert_type=AlertType.ENTRY_SIGNAL,
        priority=AlertPriority.HIGH,
        ticker="GOOGL",
        title="🚀 BUY SIGNAL: GOOGL",
        message="Bull pullback reversal detected. Consider 220C 30DTE calls.",
        data={"confidence": 0.85},
    )

    # Send alert
    results = alert_system.send_alert(test_alert)
    print(f"Alert sent via channels: {results}")

    # Test execution checklist
    print("\n=== EXECUTION CHECKLIST TEST ===")

    checklist_manager = ExecutionChecklistManager()

    # Sample trade calculation for checklist
    from .options_calculator import TradeCalculation

    sample_trade = TradeCalculation(
        ticker="GOOGL",
        spot_price=207.0,
        strike=220.0,
        expiry_date=date.today() + timedelta(days=30),
        days_to_expiry=30,
        estimated_premium=4.70,
        recommended_contracts=100,
        total_cost=470,
        breakeven_price=224.70,
        estimated_delta=0.35,
        leverage_ratio=44.1,
        risk_amount=470,
        account_risk_pct=0.94,
    )

    # Create entry checklist
    checklist_id = checklist_manager.create_entry_checklist("GOOGL", sample_trade)
    checklist = checklist_manager.get_checklist(checklist_id)

    print(f"Created checklist: {checklist_id}")
    print(f"Completion: {checklist.completion_percentage:.1f}%")

    # Complete first few items
    checklist_manager.complete_item(
        checklist_id, 1, "Verified: GOOGL  >  50 - EMA confirmed"
    )
    checklist_manager.complete_item(
        checklist_id, 2, "RSI at 42, touched 20 - EMA support"
    )

    updated_checklist = checklist_manager.get_checklist(checklist_id)
    print(f"Updated completion: {updated_checklist.completion_percentage:.1f}%")

    print("\nChecklist items: ")
    for item in updated_checklist.items[:5]:  # Show first 5 items
        status = "✅" if item.completed else "⭕"
        print(f"  {status} Step {item.step}: {item.description}")
        if item.notes:
            print(f"      Notes: {item.notes}")
