"""
Crypto Dip Bot Strategy

24/7 cryptocurrency dip buying strategy using Alpaca's crypto API.
Monitors major cryptocurrencies for significant dips and executes
buy orders with configurable parameters.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from .alpaca_crypto_client import (
    AlpacaCryptoClient,
    CryptoAsset,
    CryptoQuote,
    CryptoBar,
    CryptoOrder,
    CryptoPosition,
)

logger = logging.getLogger(__name__)


class DipSeverity(Enum):
    """Severity levels for price dips."""
    MINOR = "minor"  # 3-5%
    MODERATE = "moderate"  # 5-10%
    MAJOR = "major"  # 10-15%
    SEVERE = "severe"  # 15%+


@dataclass
class DipSignal:
    """Signal for a detected dip."""
    symbol: str
    current_price: Decimal
    reference_price: Decimal  # Recent high
    dip_percentage: float
    severity: DipSeverity
    volume_confirmation: bool
    timestamp: datetime
    timeframe: str  # "1h", "4h", "24h"

    @property
    def is_actionable(self) -> bool:
        """Check if signal meets action criteria."""
        return self.dip_percentage >= 5.0 and self.severity in (
            DipSeverity.MODERATE,
            DipSeverity.MAJOR,
            DipSeverity.SEVERE,
        )


@dataclass
class CryptoDipBotConfig:
    """Configuration for the crypto dip bot."""
    # Assets to monitor
    watch_list: List[str] = field(default_factory=lambda: [
        "BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD"
    ])

    # Dip detection parameters
    min_dip_percentage: float = 5.0  # Minimum dip to consider
    severe_dip_percentage: float = 15.0  # Severe dip threshold
    lookback_hours: int = 24  # Hours to look back for reference price

    # Position sizing
    max_position_pct: float = 0.10  # Max 10% of portfolio per position
    min_order_size: Decimal = Decimal("10.00")  # Minimum order in USD
    max_order_size: Decimal = Decimal("1000.00")  # Maximum order in USD
    scale_with_severity: bool = True  # Larger position for bigger dips

    # Risk management
    max_open_positions: int = 5
    max_daily_trades: int = 10
    take_profit_pct: float = 10.0  # Take profit at 10% gain
    stop_loss_pct: float = 8.0  # Stop loss at 8% loss
    trailing_stop_pct: Optional[float] = None  # Optional trailing stop

    # Execution
    use_limit_orders: bool = True
    limit_offset_pct: float = 0.1  # Offset from current price
    order_timeout_seconds: int = 300  # Cancel unfilled orders after 5 min

    # Cooldowns
    asset_cooldown_minutes: int = 60  # Wait before buying same asset again
    global_cooldown_minutes: int = 5  # Wait between any trades


@dataclass
class DipBotState:
    """Current state of the dip bot."""
    is_running: bool = False
    last_scan_time: Optional[datetime] = None
    daily_trades_count: int = 0
    last_trade_time: Optional[datetime] = None
    asset_cooldowns: Dict[str, datetime] = field(default_factory=dict)
    active_signals: List[DipSignal] = field(default_factory=list)
    pending_orders: List[str] = field(default_factory=list)
    error_count: int = 0


class CryptoDipBot:
    """
    Cryptocurrency Dip Buying Bot.

    Monitors crypto markets 24/7 for significant price dips and
    automatically executes buy orders when conditions are met.
    """

    def __init__(
        self,
        crypto_client: AlpacaCryptoClient,
        config: Optional[CryptoDipBotConfig] = None,
    ):
        """
        Initialize the crypto dip bot.

        Args:
            crypto_client: Alpaca crypto client
            config: Bot configuration
        """
        self.client = crypto_client
        self.config = config or CryptoDipBotConfig()
        self.state = DipBotState()

        # Price history cache
        self._price_cache: Dict[str, List[CryptoBar]] = {}
        self._cache_updated: Dict[str, datetime] = {}

    async def start(self) -> None:
        """Start the dip bot monitoring loop."""
        if self.state.is_running:
            logger.warning("Dip bot is already running")
            return

        self.state.is_running = True
        self.state.daily_trades_count = 0
        logger.info("Starting crypto dip bot")

        try:
            while self.state.is_running:
                await self._scan_cycle()
                await asyncio.sleep(60)  # Scan every minute
        except Exception as e:
            logger.error(f"Dip bot error: {e}")
            self.state.is_running = False
            raise

    async def stop(self) -> None:
        """Stop the dip bot."""
        self.state.is_running = False
        logger.info("Stopping crypto dip bot")

    async def _scan_cycle(self) -> None:
        """Execute one scan cycle."""
        self.state.last_scan_time = datetime.now()

        try:
            # Reset daily counter at midnight
            if self._is_new_day():
                self.state.daily_trades_count = 0

            # Scan for dips
            signals = await self._detect_dips()
            self.state.active_signals = signals

            # Process actionable signals
            for signal in signals:
                if signal.is_actionable:
                    await self._process_signal(signal)

            # Check pending orders
            await self._check_pending_orders()

            # Manage existing positions (take profit / stop loss)
            await self._manage_positions()

        except Exception as e:
            logger.error(f"Scan cycle error: {e}")
            self.state.error_count += 1

    async def _detect_dips(self) -> List[DipSignal]:
        """Detect dips across all watched assets."""
        signals = []

        for symbol in self.config.watch_list:
            try:
                signal = await self._check_asset_for_dip(symbol)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error checking {symbol} for dip: {e}")

        return signals

    async def _check_asset_for_dip(self, symbol: str) -> Optional[DipSignal]:
        """Check a single asset for dip condition."""
        # Get current price
        current_price = await self.client.get_price(symbol)
        if not current_price:
            return None

        # Get historical bars
        bars = await self._get_cached_bars(symbol)
        if not bars:
            return None

        # Calculate reference price (recent high)
        reference_price = max(bar.high for bar in bars)

        # Calculate dip percentage
        dip_pct = float((reference_price - current_price) / reference_price * 100)

        if dip_pct < self.config.min_dip_percentage:
            return None

        # Determine severity
        if dip_pct >= self.config.severe_dip_percentage:
            severity = DipSeverity.SEVERE
        elif dip_pct >= 10.0:
            severity = DipSeverity.MAJOR
        elif dip_pct >= 5.0:
            severity = DipSeverity.MODERATE
        else:
            severity = DipSeverity.MINOR

        # Check volume confirmation
        volume_confirmation = self._check_volume_confirmation(bars)

        return DipSignal(
            symbol=symbol,
            current_price=current_price,
            reference_price=reference_price,
            dip_percentage=dip_pct,
            severity=severity,
            volume_confirmation=volume_confirmation,
            timestamp=datetime.now(),
            timeframe=f"{self.config.lookback_hours}h",
        )

    async def _get_cached_bars(self, symbol: str) -> List[CryptoBar]:
        """Get bars with caching."""
        cache_age = timedelta(minutes=5)

        if (symbol in self._cache_updated and
                datetime.now() - self._cache_updated[symbol] < cache_age):
            return self._price_cache.get(symbol, [])

        # Fetch fresh data
        start = datetime.now() - timedelta(hours=self.config.lookback_hours)
        bars = await self.client.get_historical_bars(
            symbol,
            timeframe="1Hour",
            start=start,
            limit=self.config.lookback_hours,
        )

        self._price_cache[symbol] = bars
        self._cache_updated[symbol] = datetime.now()

        return bars

    def _check_volume_confirmation(self, bars: List[CryptoBar]) -> bool:
        """Check if recent volume confirms the dip."""
        if len(bars) < 5:
            return False

        # Compare recent volume to average
        recent_volume = sum(float(b.volume) for b in bars[-3:]) / 3
        avg_volume = sum(float(b.volume) for b in bars) / len(bars)

        # Higher volume on dip is confirmation
        return recent_volume > avg_volume * 1.5

    async def _process_signal(self, signal: DipSignal) -> None:
        """Process an actionable dip signal."""
        # Check cooldowns
        if not self._can_trade(signal.symbol):
            logger.debug(f"Skipping {signal.symbol} due to cooldown")
            return

        # Check limits
        if self.state.daily_trades_count >= self.config.max_daily_trades:
            logger.info("Daily trade limit reached")
            return

        # Check max positions
        positions = await self.client.get_positions()
        if len(positions) >= self.config.max_open_positions:
            logger.info("Maximum positions reached")
            return

        # Already have position?
        if any(p.symbol == signal.symbol for p in positions):
            logger.debug(f"Already have position in {signal.symbol}")
            return

        # Calculate position size
        order_size = self._calculate_order_size(signal)

        # Execute order
        order = await self._execute_buy(signal, order_size)

        if order:
            self._update_state_after_trade(signal.symbol)
            logger.info(
                f"Executed buy for {signal.symbol}: "
                f"${order_size} at dip of {signal.dip_percentage:.1f}%"
            )

    def _can_trade(self, symbol: str) -> bool:
        """Check if trading is allowed."""
        now = datetime.now()

        # Global cooldown
        if self.state.last_trade_time:
            cooldown = timedelta(minutes=self.config.global_cooldown_minutes)
            if now - self.state.last_trade_time < cooldown:
                return False

        # Asset cooldown
        if symbol in self.state.asset_cooldowns:
            cooldown = timedelta(minutes=self.config.asset_cooldown_minutes)
            if now - self.state.asset_cooldowns[symbol] < cooldown:
                return False

        return True

    def _calculate_order_size(self, signal: DipSignal) -> Decimal:
        """Calculate order size based on signal and config."""
        base_size = self.config.min_order_size

        if self.config.scale_with_severity:
            # Scale up for bigger dips
            if signal.severity == DipSeverity.SEVERE:
                multiplier = Decimal("3.0")
            elif signal.severity == DipSeverity.MAJOR:
                multiplier = Decimal("2.0")
            elif signal.severity == DipSeverity.MODERATE:
                multiplier = Decimal("1.5")
            else:
                multiplier = Decimal("1.0")

            order_size = base_size * multiplier
        else:
            order_size = base_size

        # Clamp to max
        return min(order_size, self.config.max_order_size)

    async def _execute_buy(
        self,
        signal: DipSignal,
        order_size: Decimal,
    ) -> Optional[CryptoOrder]:
        """Execute a buy order."""
        try:
            if self.config.use_limit_orders:
                # Set limit price slightly above current
                offset = signal.current_price * Decimal(str(self.config.limit_offset_pct / 100))
                limit_price = signal.current_price + offset

                order = await self.client.buy(
                    symbol=signal.symbol,
                    notional=order_size,
                    limit_price=limit_price,
                )
            else:
                order = await self.client.buy(
                    symbol=signal.symbol,
                    notional=order_size,
                )

            if order:
                self.state.pending_orders.append(order.id)

            return order

        except Exception as e:
            logger.error(f"Error executing buy for {signal.symbol}: {e}")
            return None

    def _update_state_after_trade(self, symbol: str) -> None:
        """Update state after a trade."""
        now = datetime.now()
        self.state.last_trade_time = now
        self.state.asset_cooldowns[symbol] = now
        self.state.daily_trades_count += 1

    async def _check_pending_orders(self) -> None:
        """Check and manage pending orders."""
        if not self.state.pending_orders:
            return

        orders = await self.client.get_orders()
        order_ids = {o.id for o in orders}

        # Remove filled/cancelled orders from pending
        self.state.pending_orders = [
            oid for oid in self.state.pending_orders
            if oid in order_ids
        ]

        # Cancel stale orders
        for order in orders:
            if order.id in self.state.pending_orders:
                if order.submitted_at:
                    age = (datetime.now() - order.submitted_at).total_seconds()
                    if age > self.config.order_timeout_seconds:
                        await self.client.cancel_order(order.id)
                        logger.info(f"Cancelled stale order {order.id}")

    async def _manage_positions(self) -> None:
        """Manage existing positions (take profit / stop loss)."""
        positions = await self.client.get_positions()

        for position in positions:
            if position.symbol not in [s for s in self.config.watch_list]:
                continue

            # Check take profit
            if position.unrealized_plpc >= self.config.take_profit_pct:
                logger.info(
                    f"Taking profit on {position.symbol}: "
                    f"{position.unrealized_plpc:.1f}%"
                )
                await self.client.close_position(position.symbol)
                continue

            # Check stop loss
            if position.unrealized_plpc <= -self.config.stop_loss_pct:
                logger.info(
                    f"Stop loss triggered on {position.symbol}: "
                    f"{position.unrealized_plpc:.1f}%"
                )
                await self.client.close_position(position.symbol)
                continue

    def _is_new_day(self) -> bool:
        """Check if it's a new trading day."""
        if not self.state.last_scan_time:
            return True
        return datetime.now().date() > self.state.last_scan_time.date()

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            "is_running": self.state.is_running,
            "last_scan": self.state.last_scan_time.isoformat() if self.state.last_scan_time else None,
            "daily_trades": self.state.daily_trades_count,
            "active_signals": len(self.state.active_signals),
            "pending_orders": len(self.state.pending_orders),
            "error_count": self.state.error_count,
            "watch_list": self.config.watch_list,
        }

    async def get_signals(self) -> List[Dict[str, Any]]:
        """Get current active signals."""
        return [
            {
                "symbol": s.symbol,
                "current_price": float(s.current_price),
                "reference_price": float(s.reference_price),
                "dip_percentage": s.dip_percentage,
                "severity": s.severity.value,
                "volume_confirmation": s.volume_confirmation,
                "is_actionable": s.is_actionable,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in self.state.active_signals
        ]


async def create_dip_bot(
    api_key: str,
    secret_key: str,
    config: Optional[CryptoDipBotConfig] = None,
    paper_trading: bool = True,
) -> CryptoDipBot:
    """Factory function to create a crypto dip bot."""
    client = AlpacaCryptoClient(api_key, secret_key, paper_trading)
    return CryptoDipBot(client, config)
