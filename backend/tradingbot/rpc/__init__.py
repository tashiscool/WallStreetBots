"""
RPC (Remote Procedure Call) Interface

Provides remote control and notification interfaces for the trading system.
Inspired by freqtrade's RPC system.

Includes:
- Telegram Bot for mobile control
- REST API for web interfaces
- Discord/Slack notifications
- WebSocket real-time streaming
"""

from .telegram_bot import TradingTelegramBot, TelegramConfig
from .message_types import RPCMessageType, RPCMessage

from .websocket import (
    MessageType,
    WebSocketMessage,
    WebSocketHub,
    TradingWebSocketConsumer,
    WebSocketBroadcaster,
    SyncWebSocketBroadcaster,
    websocket_broadcaster,
)

from .notifications import (
    NotificationPriority,
    NotificationType,
    EmbedField,
    NotificationEmbed,
    NotificationConfig,
    NotificationProvider,
    DiscordNotifier,
    SlackNotifier,
    TradingNotificationService,
    notification_service,
    create_notification_service,
    SyncNotificationService,
)

__all__ = [
    'DiscordNotifier',
    'EmbedField',
    # WebSocket
    'MessageType',
    'NotificationConfig',
    'NotificationEmbed',
    # Notifications
    'NotificationPriority',
    'NotificationProvider',
    'NotificationType',
    'RPCMessage',
    'RPCMessageType',
    'SlackNotifier',
    'SyncNotificationService',
    'SyncWebSocketBroadcaster',
    'TelegramConfig',
    'TradingNotificationService',
    # Telegram
    'TradingTelegramBot',
    'TradingWebSocketConsumer',
    'WebSocketBroadcaster',
    'WebSocketHub',
    'WebSocketMessage',
    'create_notification_service',
    'notification_service',
    'websocket_broadcaster',
]
