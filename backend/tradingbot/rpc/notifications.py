"""
Discord and Slack Notifications Module.

Ported from lumibot's notification system.
Supports webhooks, rich embeds, and trading alerts.
"""

import asyncio
import aiohttp
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Optional
import io
import base64

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationType(Enum):
    """Types of trading notifications."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    POSITION_UPDATE = "position_update"
    ALERT = "alert"
    RISK_WARNING = "risk_warning"
    DAILY_SUMMARY = "daily_summary"
    ERROR = "error"
    SYSTEM = "system"
    SIGNAL = "signal"
    BALANCE_UPDATE = "balance_update"


@dataclass
class EmbedField:
    """Field in an embed message."""
    name: str
    value: str
    inline: bool = True


@dataclass
class NotificationEmbed:
    """Rich embed for notifications."""
    title: str
    description: str = ""
    color: int = 0x2196F3  # Blue
    fields: list[EmbedField] = field(default_factory=list)
    thumbnail_url: Optional[str] = None
    image_url: Optional[str] = None
    footer: Optional[str] = None
    timestamp: Optional[datetime] = None

    def add_field(self, name: str, value: str, inline: bool = True) -> "NotificationEmbed":
        """Add a field to the embed."""
        self.fields.append(EmbedField(name=name, value=value, inline=inline))
        return self

    def to_discord_dict(self) -> dict:
        """Convert to Discord embed format."""
        embed = {
            "title": self.title,
            "description": self.description,
            "color": self.color,
            "fields": [
                {"name": f.name, "value": f.value, "inline": f.inline}
                for f in self.fields
            ],
        }
        if self.thumbnail_url:
            embed["thumbnail"] = {"url": self.thumbnail_url}
        if self.image_url:
            embed["image"] = {"url": self.image_url}
        if self.footer:
            embed["footer"] = {"text": self.footer}
        if self.timestamp:
            embed["timestamp"] = self.timestamp.isoformat()
        return embed

    def to_slack_blocks(self) -> list[dict]:
        """Convert to Slack blocks format."""
        blocks = []

        # Header
        blocks.append({
            "type": "header",
            "text": {"type": "plain_text", "text": self.title}
        })

        # Description
        if self.description:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": self.description}
            })

        # Fields
        if self.fields:
            field_blocks = []
            for f in self.fields:
                field_blocks.append({
                    "type": "mrkdwn",
                    "text": f"*{f.name}*\n{f.value}"
                })

            # Slack allows max 10 fields per section
            for i in range(0, len(field_blocks), 10):
                blocks.append({
                    "type": "section",
                    "fields": field_blocks[i:i + 10]
                })

        # Image
        if self.image_url:
            blocks.append({
                "type": "image",
                "image_url": self.image_url,
                "alt_text": self.title
            })

        # Footer
        if self.footer:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": self.footer}]
            })

        return blocks


@dataclass
class NotificationConfig:
    """Configuration for notification service."""
    enabled: bool = True
    send_trade_entries: bool = True
    send_trade_exits: bool = True
    send_alerts: bool = True
    send_daily_summary: bool = True
    send_risk_warnings: bool = True
    send_errors: bool = True
    min_priority: NotificationPriority = NotificationPriority.LOW
    rate_limit_per_minute: int = 30
    batch_notifications: bool = False
    batch_interval_seconds: int = 60


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    def __init__(self, webhook_url: str, config: Optional[NotificationConfig] = None):
        self.webhook_url = webhook_url
        self.config = config or NotificationConfig()
        self._message_count = 0
        self._last_reset = datetime.now()

    @abstractmethod
    async def send(self, embed: NotificationEmbed) -> bool:
        """Send a notification."""
        pass

    @abstractmethod
    async def send_text(self, message: str) -> bool:
        """Send a plain text message."""
        pass

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()
        if (now - self._last_reset).total_seconds() >= 60:
            self._message_count = 0
            self._last_reset = now

        if self._message_count >= self.config.rate_limit_per_minute:
            logger.warning("Rate limit exceeded, notification dropped")
            return False

        self._message_count += 1
        return True


class DiscordNotifier(NotificationProvider):
    """Discord webhook notification provider."""

    # Color mapping for notification types
    COLORS: ClassVar[dict] = {
        NotificationType.TRADE_ENTRY: 0x4CAF50,  # Green
        NotificationType.TRADE_EXIT: 0x2196F3,  # Blue
        NotificationType.RISK_WARNING: 0xFF9800,  # Orange
        NotificationType.ERROR: 0xF44336,  # Red
        NotificationType.ALERT: 0x9C27B0,  # Purple
        NotificationType.DAILY_SUMMARY: 0x00BCD4,  # Cyan
        NotificationType.SIGNAL: 0xFFEB3B,  # Yellow
        NotificationType.SYSTEM: 0x607D8B,  # Gray
    }

    def __init__(
        self,
        webhook_url: str,
        username: str = "WallStreetBots",
        avatar_url: Optional[str] = None,
        config: Optional[NotificationConfig] = None,
    ):
        super().__init__(webhook_url, config)
        self.username = username
        self.avatar_url = avatar_url

    async def send(self, embed: NotificationEmbed) -> bool:
        """Send a Discord embed notification."""
        if not self.config.enabled or not self._check_rate_limit():
            return False

        payload = {
            "username": self.username,
            "embeds": [embed.to_discord_dict()],
        }
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 204:
                        return True
                    else:
                        logger.error(f"Discord webhook failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    async def send_text(self, message: str) -> bool:
        """Send a plain text Discord message."""
        if not self.config.enabled or not self._check_rate_limit():
            return False

        payload = {
            "username": self.username,
            "content": message,
        }
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    return response.status == 204
        except Exception as e:
            logger.error(f"Discord notification error: {e}")
            return False

    async def send_file(
        self,
        file_data: bytes,
        filename: str,
        message: Optional[str] = None
    ) -> bool:
        """Send a file to Discord."""
        if not self.config.enabled or not self._check_rate_limit():
            return False

        try:
            form = aiohttp.FormData()
            form.add_field(
                "file",
                file_data,
                filename=filename,
                content_type="application/octet-stream"
            )
            if message:
                form.add_field("content", message)
            form.add_field("username", self.username)

            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, data=form) as response:
                    return response.status in (200, 204)
        except Exception as e:
            logger.error(f"Discord file upload error: {e}")
            return False


class SlackNotifier(NotificationProvider):
    """Slack webhook notification provider."""

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "WallStreetBots",
        icon_emoji: str = ":chart_with_upwards_trend:",
        config: Optional[NotificationConfig] = None,
    ):
        super().__init__(webhook_url, config)
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    async def send(self, embed: NotificationEmbed) -> bool:
        """Send a Slack block notification."""
        if not self.config.enabled or not self._check_rate_limit():
            return False

        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": embed.to_slack_blocks(),
        }
        if self.channel:
            payload["channel"] = self.channel

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return False

    async def send_text(self, message: str) -> bool:
        """Send a plain text Slack message."""
        if not self.config.enabled or not self._check_rate_limit():
            return False

        payload = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "text": message,
        }
        if self.channel:
            payload["channel"] = self.channel

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
            return False


class TradingNotificationService:
    """
    High-level trading notification service.

    Manages multiple notification providers and creates
    trading-specific notification messages.
    """

    def __init__(self):
        self.providers: list[NotificationProvider] = []
        self._notification_queue: list[NotificationEmbed] = []
        self._batch_task: Optional[asyncio.Task] = None

    def add_discord(
        self,
        webhook_url: str,
        username: str = "WallStreetBots",
        avatar_url: Optional[str] = None,
        config: Optional[NotificationConfig] = None,
    ) -> "TradingNotificationService":
        """Add a Discord notification provider."""
        self.providers.append(DiscordNotifier(
            webhook_url, username, avatar_url, config
        ))
        return self

    def add_slack(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "WallStreetBots",
        icon_emoji: str = ":chart_with_upwards_trend:",
        config: Optional[NotificationConfig] = None,
    ) -> "TradingNotificationService":
        """Add a Slack notification provider."""
        self.providers.append(SlackNotifier(
            webhook_url, channel, username, icon_emoji, config
        ))
        return self

    async def _send_all(self, embed: NotificationEmbed) -> dict[str, bool]:
        """Send to all providers."""
        results = {}
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            results[provider_name] = await provider.send(embed)
        return results

    async def notify_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send trade entry notification."""
        embed = NotificationEmbed(
            title=f"📈 Trade Entry: {symbol}",
            description=f"New {side.upper()} position opened",
            color=DiscordNotifier.COLORS[NotificationType.TRADE_ENTRY],
            timestamp=datetime.now(),
        )
        embed.add_field("Symbol", symbol)
        embed.add_field("Side", side.upper())
        embed.add_field("Quantity", str(quantity))
        embed.add_field("Price", f"${price:.2f}")
        embed.add_field("Total Value", f"${quantity * price:.2f}")

        if strategy:
            embed.add_field("Strategy", strategy)
        if reason:
            embed.add_field("Reason", reason, inline=False)

        embed.footer = "WallStreetBots Trading System"

        return await self._send_all(embed)

    async def notify_trade_exit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        profit_loss: float,
        profit_pct: float,
        hold_time: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send trade exit notification."""
        is_profit = profit_loss >= 0
        emoji = "💰" if is_profit else "📉"
        color = DiscordNotifier.COLORS[NotificationType.TRADE_ENTRY] if is_profit else DiscordNotifier.COLORS[NotificationType.ERROR]

        embed = NotificationEmbed(
            title=f"{emoji} Trade Exit: {symbol}",
            description=f"Position closed - {'PROFIT' if is_profit else 'LOSS'}",
            color=color,
            timestamp=datetime.now(),
        )
        embed.add_field("Symbol", symbol)
        embed.add_field("Side", side.upper())
        embed.add_field("Quantity", str(quantity))
        embed.add_field("Entry Price", f"${entry_price:.2f}")
        embed.add_field("Exit Price", f"${exit_price:.2f}")
        embed.add_field("P/L", f"${profit_loss:+.2f}")
        embed.add_field("Return", f"{profit_pct:+.2f}%")

        if hold_time:
            embed.add_field("Hold Time", hold_time)
        if strategy:
            embed.add_field("Strategy", strategy)

        embed.footer = "WallStreetBots Trading System"

        return await self._send_all(embed)

    async def notify_alert(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        alert_type: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send alert notification."""
        priority_emojis = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.NORMAL: "⚡",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.URGENT: "🚨",
        }

        embed = NotificationEmbed(
            title=f"{priority_emojis[priority]} {title}",
            description=message,
            color=DiscordNotifier.COLORS[NotificationType.ALERT],
            timestamp=datetime.now(),
        )
        embed.add_field("Priority", priority.value.upper())
        if alert_type:
            embed.add_field("Type", alert_type)

        embed.footer = "WallStreetBots Trading System"

        return await self._send_all(embed)

    async def notify_risk_warning(
        self,
        warning_type: str,
        message: str,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        recommendation: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send risk warning notification."""
        embed = NotificationEmbed(
            title=f"⚠️ Risk Warning: {warning_type}",
            description=message,
            color=DiscordNotifier.COLORS[NotificationType.RISK_WARNING],
            timestamp=datetime.now(),
        )
        embed.add_field("Warning Type", warning_type)

        if current_value is not None:
            embed.add_field("Current Value", f"{current_value:.2f}")
        if threshold is not None:
            embed.add_field("Threshold", f"{threshold:.2f}")
        if recommendation:
            embed.add_field("Recommendation", recommendation, inline=False)

        embed.footer = "WallStreetBots Risk Management"

        return await self._send_all(embed)

    async def notify_daily_summary(
        self,
        date: datetime,
        total_trades: int,
        winning_trades: int,
        total_pnl: float,
        pnl_pct: float,
        portfolio_value: float,
        top_performers: Optional[list[tuple[str, float]]] = None,
        worst_performers: Optional[list[tuple[str, float]]] = None,
    ) -> dict[str, bool]:
        """Send daily summary notification."""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        is_positive = total_pnl >= 0
        emoji = "📊" if is_positive else "📉"

        embed = NotificationEmbed(
            title=f"{emoji} Daily Summary - {date.strftime('%Y-%m-%d')}",
            description="End of day trading report",
            color=DiscordNotifier.COLORS[NotificationType.DAILY_SUMMARY],
            timestamp=datetime.now(),
        )
        embed.add_field("Total Trades", str(total_trades))
        embed.add_field("Winning Trades", str(winning_trades))
        embed.add_field("Win Rate", f"{win_rate:.1f}%")
        embed.add_field("Daily P/L", f"${total_pnl:+.2f}")
        embed.add_field("Return", f"{pnl_pct:+.2f}%")
        embed.add_field("Portfolio Value", f"${portfolio_value:,.2f}")

        if top_performers:
            top_str = "\n".join([f"{s}: +{p:.2f}%" for s, p in top_performers[:3]])
            embed.add_field("Top Performers", top_str, inline=False)

        if worst_performers:
            worst_str = "\n".join([f"{s}: {p:.2f}%" for s, p in worst_performers[:3]])
            embed.add_field("Worst Performers", worst_str, inline=False)

        embed.footer = "WallStreetBots Daily Report"

        return await self._send_all(embed)

    async def notify_error(
        self,
        error_type: str,
        message: str,
        details: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send error notification."""
        embed = NotificationEmbed(
            title=f"🚨 Error: {error_type}",
            description=message,
            color=DiscordNotifier.COLORS[NotificationType.ERROR],
            timestamp=datetime.now(),
        )
        embed.add_field("Error Type", error_type)

        if details:
            embed.add_field("Details", details[:1024], inline=False)
        if stack_trace:
            # Truncate stack trace if too long
            truncated = stack_trace[:1000] + "..." if len(stack_trace) > 1000 else stack_trace
            embed.add_field("Stack Trace", f"```{truncated}```", inline=False)

        embed.footer = "WallStreetBots Error Handler"

        return await self._send_all(embed)

    async def notify_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        price: float,
        indicators: Optional[dict[str, Any]] = None,
        recommendation: Optional[str] = None,
    ) -> dict[str, bool]:
        """Send trading signal notification."""
        strength_emoji = "🟢" if strength > 0.7 else "🟡" if strength > 0.4 else "🔴"

        embed = NotificationEmbed(
            title=f"{strength_emoji} Signal: {symbol}",
            description=f"{signal_type.upper()} signal detected",
            color=DiscordNotifier.COLORS[NotificationType.SIGNAL],
            timestamp=datetime.now(),
        )
        embed.add_field("Symbol", symbol)
        embed.add_field("Signal", signal_type.upper())
        embed.add_field("Strength", f"{strength * 100:.0f}%")
        embed.add_field("Current Price", f"${price:.2f}")

        if indicators:
            for name, value in list(indicators.items())[:5]:
                if isinstance(value, float):
                    embed.add_field(name, f"{value:.2f}")
                else:
                    embed.add_field(name, str(value))

        if recommendation:
            embed.add_field("Recommendation", recommendation, inline=False)

        embed.footer = "WallStreetBots Signal Generator"

        return await self._send_all(embed)

    async def notify_balance_update(
        self,
        cash: float,
        equity: float,
        buying_power: float,
        margin_used: Optional[float] = None,
        day_change: Optional[float] = None,
        day_change_pct: Optional[float] = None,
    ) -> dict[str, bool]:
        """Send balance update notification."""
        embed = NotificationEmbed(
            title="💵 Balance Update",
            description="Account balance has been updated",
            color=DiscordNotifier.COLORS[NotificationType.SYSTEM],
            timestamp=datetime.now(),
        )
        embed.add_field("Cash", f"${cash:,.2f}")
        embed.add_field("Equity", f"${equity:,.2f}")
        embed.add_field("Buying Power", f"${buying_power:,.2f}")

        if margin_used is not None:
            embed.add_field("Margin Used", f"${margin_used:,.2f}")
        if day_change is not None:
            embed.add_field("Day Change", f"${day_change:+,.2f}")
        if day_change_pct is not None:
            embed.add_field("Day Change %", f"{day_change_pct:+.2f}%")

        embed.footer = "WallStreetBots Account Monitor"

        return await self._send_all(embed)


# Global notification service instance
notification_service = TradingNotificationService()


def create_notification_service() -> TradingNotificationService:
    """Factory function to create a notification service."""
    return TradingNotificationService()


# Sync wrapper for non-async contexts
class SyncNotificationService:
    """Synchronous wrapper for TradingNotificationService."""

    def __init__(self, service: Optional[TradingNotificationService] = None):
        self.service = service or notification_service

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Create a new loop in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    def notify_trade_entry(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_trade_entry(*args, **kwargs))

    def notify_trade_exit(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_trade_exit(*args, **kwargs))

    def notify_alert(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_alert(*args, **kwargs))

    def notify_risk_warning(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_risk_warning(*args, **kwargs))

    def notify_daily_summary(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_daily_summary(*args, **kwargs))

    def notify_error(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_error(*args, **kwargs))

    def notify_signal(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_signal(*args, **kwargs))

    def notify_balance_update(self, *args, **kwargs) -> dict[str, bool]:
        return self._run_async(self.service.notify_balance_update(*args, **kwargs))
