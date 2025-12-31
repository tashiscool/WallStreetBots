"""
WebSocket Real-time Streaming.

Ported from freqtrade's WebSocket implementation.
Provides real-time updates for trades, signals, and market data.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

logger = logging.getLogger(__name__)

# Try to import websocket libraries
try:
    from channels.generic.websocket import AsyncWebsocketConsumer
    from channels.layers import get_channel_layer
    from asgiref.sync import async_to_sync
    CHANNELS_AVAILABLE = True
except ImportError:
    CHANNELS_AVAILABLE = False
    logger.warning("Django Channels not available. WebSocket features disabled.")


class MessageType(Enum):
    """Types of WebSocket messages."""
    # Connection
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

    # Trading events
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    TRADE_UPDATE = "trade_update"
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # Market data
    PRICE_UPDATE = "price_update"
    NEW_CANDLE = "new_candle"
    ANALYZED_DF = "analyzed_df"

    # Status updates
    STATUS_UPDATE = "status_update"
    BALANCE_UPDATE = "balance_update"
    POSITION_UPDATE = "position_update"

    # Alerts and notifications
    ALERT = "alert"
    RISK_ALERT = "risk_alert"
    SIGNAL = "signal"

    # System
    HEARTBEAT = "heartbeat"
    WHITELIST_UPDATE = "whitelist_update"
    PROTECTION_TRIGGER = "protection_trigger"

    # Strategy messages
    STRATEGY_MSG = "strategy_msg"
    BACKTEST_PROGRESS = "backtest_progress"
    BACKTEST_COMPLETE = "backtest_complete"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
        }, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            message_id=data.get("message_id", str(uuid.uuid4())[:8]),
        )


class WebSocketHub:
    """
    Central hub for managing WebSocket connections and broadcasting.

    Features:
    - Connection management
    - Channel-based subscriptions
    - Broadcast to all or specific clients
    - Message queuing
    - Rate limiting
    """

    def __init__(self):
        """Initialize WebSocket hub."""
        self._connections: Dict[str, Any] = {}  # connection_id -> consumer
        self._subscriptions: Dict[str, Set[str]] = {}  # channel -> connection_ids
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._callbacks: Dict[MessageType, List[Callable]] = {}
        self._rate_limits: Dict[str, datetime] = {}
        self._is_running = False

    def register_connection(
        self,
        connection_id: str,
        consumer: Any,
        channels: Optional[List[str]] = None,
    ) -> None:
        """Register a new WebSocket connection."""
        self._connections[connection_id] = consumer
        channels = channels or ["default"]

        for channel in channels:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = set()
            self._subscriptions[channel].add(connection_id)

        logger.info(f"WebSocket connected: {connection_id}")

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        if connection_id in self._connections:
            del self._connections[connection_id]

        for channel in self._subscriptions.values():
            channel.discard(connection_id)

        logger.info(f"WebSocket disconnected: {connection_id}")

    def subscribe(self, connection_id: str, channel: str) -> None:
        """Subscribe a connection to a channel."""
        if channel not in self._subscriptions:
            self._subscriptions[channel] = set()
        self._subscriptions[channel].add(connection_id)

    def unsubscribe(self, connection_id: str, channel: str) -> None:
        """Unsubscribe a connection from a channel."""
        if channel in self._subscriptions:
            self._subscriptions[channel].discard(connection_id)

    async def broadcast(
        self,
        message: WebSocketMessage,
        channel: str = "default",
    ) -> None:
        """Broadcast message to all connections in a channel."""
        if channel not in self._subscriptions:
            return

        json_msg = message.to_json()
        connection_ids = list(self._subscriptions[channel])

        for conn_id in connection_ids:
            consumer = self._connections.get(conn_id)
            if consumer:
                try:
                    await consumer.send(text_data=json_msg)
                except Exception as e:
                    logger.error(f"Error sending to {conn_id}: {e}")
                    self.unregister_connection(conn_id)

    async def send_to_connection(
        self,
        connection_id: str,
        message: WebSocketMessage,
    ) -> None:
        """Send message to specific connection."""
        consumer = self._connections.get(connection_id)
        if consumer:
            try:
                await consumer.send(text_data=message.to_json())
            except Exception as e:
                logger.error(f"Error sending to {connection_id}: {e}")

    def on_message(
        self,
        message_type: MessageType,
        callback: Callable[[WebSocketMessage], None],
    ) -> None:
        """Register callback for message type."""
        if message_type not in self._callbacks:
            self._callbacks[message_type] = []
        self._callbacks[message_type].append(callback)

    async def process_message(self, message: WebSocketMessage) -> None:
        """Process incoming message and trigger callbacks."""
        callbacks = self._callbacks.get(message.type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# Global hub instance
_websocket_hub: Optional[WebSocketHub] = None


def get_websocket_hub() -> WebSocketHub:
    """Get or create the global WebSocket hub."""
    global _websocket_hub
    if _websocket_hub is None:
        _websocket_hub = WebSocketHub()
    return _websocket_hub


if CHANNELS_AVAILABLE:
    class TradingWebSocketConsumer(AsyncWebsocketConsumer):
        """
        Django Channels WebSocket consumer for trading updates.

        Handles:
        - Connection lifecycle
        - Authentication
        - Channel subscriptions
        - Message routing
        """

        async def connect(self):
            """Handle WebSocket connection."""
            self.connection_id = str(uuid.uuid4())[:12]
            self.user = self.scope.get("user")
            self.subscribed_channels = set()

            # Accept connection
            await self.accept()

            # Register with hub
            hub = get_websocket_hub()
            hub.register_connection(self.connection_id, self, ["default"])

            # Send connected message
            await self.send(text_data=WebSocketMessage(
                type=MessageType.CONNECTED,
                data={
                    "connection_id": self.connection_id,
                    "message": "Connected to WallStreetBots WebSocket",
                },
            ).to_json())

            # Join default group
            await self.channel_layer.group_add("trading_updates", self.channel_name)

            logger.info(f"WebSocket client connected: {self.connection_id}")

        async def disconnect(self, close_code):
            """Handle WebSocket disconnection."""
            hub = get_websocket_hub()
            hub.unregister_connection(self.connection_id)

            # Leave all groups
            await self.channel_layer.group_discard("trading_updates", self.channel_name)
            for channel in self.subscribed_channels:
                await self.channel_layer.group_discard(channel, self.channel_name)

            logger.info(f"WebSocket client disconnected: {self.connection_id}")

        async def receive(self, text_data):
            """Handle incoming WebSocket messages."""
            try:
                data = json.loads(text_data)
                action = data.get("action")

                if action == "subscribe":
                    channel = data.get("channel", "default")
                    await self.subscribe_channel(channel)

                elif action == "unsubscribe":
                    channel = data.get("channel")
                    if channel:
                        await self.unsubscribe_channel(channel)

                elif action == "ping":
                    await self.send(text_data=WebSocketMessage(
                        type=MessageType.HEARTBEAT,
                        data={"pong": True},
                    ).to_json())

                elif action == "get_status":
                    await self.send_status()

                else:
                    # Process as regular message
                    message = WebSocketMessage.from_json(text_data)
                    hub = get_websocket_hub()
                    await hub.process_message(message)

            except json.JSONDecodeError:
                await self.send(text_data=WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"error": "Invalid JSON"},
                ).to_json())
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                await self.send(text_data=WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"error": str(e)},
                ).to_json())

        async def subscribe_channel(self, channel: str):
            """Subscribe to a channel."""
            self.subscribed_channels.add(channel)
            await self.channel_layer.group_add(channel, self.channel_name)

            hub = get_websocket_hub()
            hub.subscribe(self.connection_id, channel)

            await self.send(text_data=WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={"subscribed": channel},
            ).to_json())

        async def unsubscribe_channel(self, channel: str):
            """Unsubscribe from a channel."""
            self.subscribed_channels.discard(channel)
            await self.channel_layer.group_discard(channel, self.channel_name)

            hub = get_websocket_hub()
            hub.unsubscribe(self.connection_id, channel)

            await self.send(text_data=WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={"unsubscribed": channel},
            ).to_json())

        async def send_status(self):
            """Send current status to client."""
            await self.send(text_data=WebSocketMessage(
                type=MessageType.STATUS_UPDATE,
                data={
                    "connection_id": self.connection_id,
                    "subscribed_channels": list(self.subscribed_channels),
                    "connected_at": datetime.now().isoformat(),
                },
            ).to_json())

        # Group message handlers
        async def trading_message(self, event):
            """Handle trading messages from channel layer."""
            await self.send(text_data=event["message"])

        async def trade_update(self, event):
            """Handle trade update messages."""
            await self.send(text_data=event["message"])

        async def price_update(self, event):
            """Handle price update messages."""
            await self.send(text_data=event["message"])

        async def alert_message(self, event):
            """Handle alert messages."""
            await self.send(text_data=event["message"])


class WebSocketBroadcaster:
    """
    Utility class for broadcasting messages to WebSocket clients.

    Use this from anywhere in the application to send real-time updates.
    """

    def __init__(self):
        """Initialize broadcaster."""
        self._channel_layer = None

    @property
    def channel_layer(self):
        """Get channel layer lazily."""
        if self._channel_layer is None and CHANNELS_AVAILABLE:
            self._channel_layer = get_channel_layer()
        return self._channel_layer

    async def broadcast_trade_entry(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        strategy: str = "",
        **extra,
    ) -> None:
        """Broadcast trade entry event."""
        message = WebSocketMessage(
            type=MessageType.TRADE_ENTRY,
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "strategy": strategy,
                **extra,
            },
        )
        await self._broadcast("trading_updates", "trade_update", message)

    async def broadcast_trade_exit(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        strategy: str = "",
        **extra,
    ) -> None:
        """Broadcast trade exit event."""
        message = WebSocketMessage(
            type=MessageType.TRADE_EXIT,
            data={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": (exit_price - entry_price) / entry_price * 100 if entry_price else 0,
                "strategy": strategy,
                **extra,
            },
        )
        await self._broadcast("trading_updates", "trade_update", message)

    async def broadcast_order_update(
        self,
        order_id: str,
        symbol: str,
        status: str,
        **extra,
    ) -> None:
        """Broadcast order update event."""
        type_map = {
            "created": MessageType.ORDER_CREATED,
            "filled": MessageType.ORDER_FILLED,
            "cancelled": MessageType.ORDER_CANCELLED,
        }
        msg_type = type_map.get(status.lower(), MessageType.ORDER_CREATED)

        message = WebSocketMessage(
            type=msg_type,
            data={
                "order_id": order_id,
                "symbol": symbol,
                "status": status,
                **extra,
            },
        )
        await self._broadcast("trading_updates", "trade_update", message)

    async def broadcast_price_update(
        self,
        symbol: str,
        price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        volume: Optional[int] = None,
    ) -> None:
        """Broadcast price update event."""
        message = WebSocketMessage(
            type=MessageType.PRICE_UPDATE,
            data={
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "volume": volume,
            },
        )
        await self._broadcast("price_updates", "price_update", message)

    async def broadcast_balance_update(
        self,
        total: float,
        available: float,
        in_positions: float,
        pnl_today: float = 0,
    ) -> None:
        """Broadcast balance update event."""
        message = WebSocketMessage(
            type=MessageType.BALANCE_UPDATE,
            data={
                "total": total,
                "available": available,
                "in_positions": in_positions,
                "pnl_today": pnl_today,
            },
        )
        await self._broadcast("trading_updates", "trading_message", message)

    async def broadcast_alert(
        self,
        alert_type: str,
        title: str,
        message: str,
        severity: str = "info",
        **extra,
    ) -> None:
        """Broadcast alert event."""
        ws_message = WebSocketMessage(
            type=MessageType.ALERT,
            data={
                "alert_type": alert_type,
                "title": title,
                "message": message,
                "severity": severity,
                **extra,
            },
        )
        await self._broadcast("alerts", "alert_message", ws_message)

    async def broadcast_risk_alert(
        self,
        risk_type: str,
        current_value: float,
        threshold: float,
        message: str,
    ) -> None:
        """Broadcast risk alert event."""
        ws_message = WebSocketMessage(
            type=MessageType.RISK_ALERT,
            data={
                "risk_type": risk_type,
                "current_value": current_value,
                "threshold": threshold,
                "message": message,
            },
        )
        await self._broadcast("alerts", "alert_message", ws_message)

    async def broadcast_backtest_progress(
        self,
        progress: float,
        current_date: str,
        total_days: int,
        processed_days: int,
        eta_seconds: Optional[int] = None,
    ) -> None:
        """Broadcast backtest progress event."""
        message = WebSocketMessage(
            type=MessageType.BACKTEST_PROGRESS,
            data={
                "progress": progress,
                "current_date": current_date,
                "total_days": total_days,
                "processed_days": processed_days,
                "eta_seconds": eta_seconds,
            },
        )
        await self._broadcast("backtest", "trading_message", message)

    async def broadcast_backtest_complete(
        self,
        results: Dict[str, Any],
    ) -> None:
        """Broadcast backtest completion event."""
        message = WebSocketMessage(
            type=MessageType.BACKTEST_COMPLETE,
            data=results,
        )
        await self._broadcast("backtest", "trading_message", message)

    async def broadcast_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        strategy: str = "",
        **extra,
    ) -> None:
        """Broadcast trading signal event."""
        message = WebSocketMessage(
            type=MessageType.SIGNAL,
            data={
                "symbol": symbol,
                "signal_type": signal_type,
                "strength": strength,
                "strategy": strategy,
                **extra,
            },
        )
        await self._broadcast("signals", "trading_message", message)

    async def _broadcast(
        self,
        group: str,
        message_type: str,
        message: WebSocketMessage,
    ) -> None:
        """Internal broadcast method."""
        if self.channel_layer:
            await self.channel_layer.group_send(
                group,
                {
                    "type": message_type,
                    "message": message.to_json(),
                },
            )


# Synchronous wrapper for use in non-async contexts
class SyncWebSocketBroadcaster:
    """Synchronous wrapper for WebSocketBroadcaster."""

    def __init__(self):
        """Initialize sync broadcaster."""
        self._async_broadcaster = WebSocketBroadcaster()

    def broadcast_trade_entry(self, **kwargs) -> None:
        """Broadcast trade entry (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_trade_entry)(**kwargs)

    def broadcast_trade_exit(self, **kwargs) -> None:
        """Broadcast trade exit (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_trade_exit)(**kwargs)

    def broadcast_order_update(self, **kwargs) -> None:
        """Broadcast order update (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_order_update)(**kwargs)

    def broadcast_price_update(self, **kwargs) -> None:
        """Broadcast price update (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_price_update)(**kwargs)

    def broadcast_balance_update(self, **kwargs) -> None:
        """Broadcast balance update (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_balance_update)(**kwargs)

    def broadcast_alert(self, **kwargs) -> None:
        """Broadcast alert (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_alert)(**kwargs)

    def broadcast_risk_alert(self, **kwargs) -> None:
        """Broadcast risk alert (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_risk_alert)(**kwargs)

    def broadcast_backtest_progress(self, **kwargs) -> None:
        """Broadcast backtest progress (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_backtest_progress)(**kwargs)

    def broadcast_signal(self, **kwargs) -> None:
        """Broadcast signal (sync)."""
        if CHANNELS_AVAILABLE:
            async_to_sync(self._async_broadcaster.broadcast_signal)(**kwargs)


# Global broadcaster instance
websocket_broadcaster = SyncWebSocketBroadcaster()
