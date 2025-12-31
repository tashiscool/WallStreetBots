"""
Alpaca Crypto Trading Client

Native crypto trading via Alpaca API for 24/7 cryptocurrency markets.
Supports BTC, ETH, SOL, and other major cryptocurrencies.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

# Alpaca imports
try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import (
        CryptoBarsRequest,
        CryptoLatestQuoteRequest,
        CryptoLatestTradeRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopLimitOrderRequest,
    )
    ALPACA_CRYPTO_AVAILABLE = True
except ImportError:
    ALPACA_CRYPTO_AVAILABLE = False
    CryptoHistoricalDataClient = None
    CryptoBarsRequest = None
    CryptoLatestQuoteRequest = None
    CryptoLatestTradeRequest = None
    TimeFrame = None
    TradingClient = None
    OrderSide = None
    OrderType = None
    TimeInForce = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    StopLimitOrderRequest = None

logger = logging.getLogger(__name__)


class CryptoAsset(Enum):
    """Supported crypto assets on Alpaca."""
    BTC_USD = "BTC/USD"
    ETH_USD = "ETH/USD"
    SOL_USD = "SOL/USD"
    AVAX_USD = "AVAX/USD"
    LINK_USD = "LINK/USD"
    DOGE_USD = "DOGE/USD"
    SHIB_USD = "SHIB/USD"
    UNI_USD = "UNI/USD"
    AAVE_USD = "AAVE/USD"
    LTC_USD = "LTC/USD"
    BCH_USD = "BCH/USD"
    DOT_USD = "DOT/USD"
    MATIC_USD = "MATIC/USD"
    ATOM_USD = "ATOM/USD"
    XLM_USD = "XLM/USD"
    ALGO_USD = "ALGO/USD"
    XTZ_USD = "XTZ/USD"
    MKR_USD = "MKR/USD"
    COMP_USD = "COMP/USD"
    GRT_USD = "GRT/USD"
    SUSHI_USD = "SUSHI/USD"
    YFI_USD = "YFI/USD"
    CRV_USD = "CRV/USD"


@dataclass
class CryptoQuote:
    """Real-time crypto quote."""
    symbol: str
    bid: Decimal
    ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp: datetime

    @property
    def mid_price(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return float(self.spread / self.mid_price * 100)
        return 0.0


@dataclass
class CryptoTrade:
    """Crypto trade data."""
    symbol: str
    price: Decimal
    size: Decimal
    timestamp: datetime
    exchange: Optional[str] = None


@dataclass
class CryptoBar:
    """OHLCV bar for crypto."""
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    vwap: Optional[Decimal] = None


@dataclass
class CryptoPosition:
    """Crypto position."""
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    market_value: Decimal
    current_price: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: float

    @property
    def cost_basis(self) -> Decimal:
        return self.qty * self.avg_entry_price


@dataclass
class CryptoOrder:
    """Crypto order."""
    id: str
    symbol: str
    side: str
    qty: Decimal
    filled_qty: Decimal
    order_type: str
    status: str
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    filled_avg_price: Optional[Decimal] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


@dataclass
class CryptoMarketHours:
    """Crypto market hours (24/7)."""
    is_open: bool = True  # Always open
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None

    @staticmethod
    def is_market_open() -> bool:
        """Crypto markets are always open."""
        return True


class AlpacaCryptoClient:
    """
    Alpaca Crypto Trading Client.

    Provides crypto trading functionality through Alpaca's API:
    - Real-time quotes and trades
    - Historical OHLCV data
    - Market and limit orders
    - Position management
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper_trading: bool = True,
    ):
        """
        Initialize the Alpaca Crypto client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper_trading: Use paper trading (default True)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading

        if not ALPACA_CRYPTO_AVAILABLE:
            logger.warning("Alpaca crypto SDK not available")
            self.trading_client = None
            self.data_client = None
            return

        try:
            # Initialize trading client
            self.trading_client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper_trading,
            )

            # Initialize crypto data client (no auth needed for crypto data)
            self.data_client = CryptoHistoricalDataClient()

            logger.info(f"Alpaca Crypto client initialized (paper={paper_trading})")

        except Exception as e:
            logger.error(f"Failed to initialize Alpaca Crypto client: {e}")
            self.trading_client = None
            self.data_client = None

    def is_available(self) -> bool:
        """Check if client is available."""
        return self.trading_client is not None and self.data_client is not None

    async def get_quote(self, symbol: str) -> Optional[CryptoQuote]:
        """
        Get real-time quote for a crypto asset.

        Args:
            symbol: Crypto symbol (e.g., "BTC/USD" or "BTCUSD")

        Returns:
            CryptoQuote or None
        """
        if not self.data_client:
            return None

        try:
            # Normalize symbol
            symbol = self._normalize_symbol(symbol)

            request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_crypto_latest_quote(request)

            if symbol in quotes:
                q = quotes[symbol]
                return CryptoQuote(
                    symbol=symbol,
                    bid=Decimal(str(q.bid_price)),
                    ask=Decimal(str(q.ask_price)),
                    bid_size=Decimal(str(q.bid_size)),
                    ask_size=Decimal(str(q.ask_size)),
                    timestamp=q.timestamp,
                )
            return None

        except Exception as e:
            logger.error(f"Error getting crypto quote for {symbol}: {e}")
            return None

    async def get_latest_trade(self, symbol: str) -> Optional[CryptoTrade]:
        """
        Get latest trade for a crypto asset.

        Args:
            symbol: Crypto symbol

        Returns:
            CryptoTrade or None
        """
        if not self.data_client:
            return None

        try:
            symbol = self._normalize_symbol(symbol)
            request = CryptoLatestTradeRequest(symbol_or_symbols=[symbol])
            trades = self.data_client.get_crypto_latest_trade(request)

            if symbol in trades:
                t = trades[symbol]
                return CryptoTrade(
                    symbol=symbol,
                    price=Decimal(str(t.price)),
                    size=Decimal(str(t.size)),
                    timestamp=t.timestamp,
                    exchange=getattr(t, 'exchange', None),
                )
            return None

        except Exception as e:
            logger.error(f"Error getting crypto trade for {symbol}: {e}")
            return None

    async def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a crypto asset.

        Args:
            symbol: Crypto symbol

        Returns:
            Current price or None
        """
        trade = await self.get_latest_trade(symbol)
        if trade:
            return trade.price

        quote = await self.get_quote(symbol)
        if quote:
            return quote.mid_price

        return None

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Hour",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CryptoBar]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Crypto symbol
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars

        Returns:
            List of CryptoBar
        """
        if not self.data_client:
            return []

        try:
            symbol = self._normalize_symbol(symbol)

            # Parse timeframe
            tf_map = {
                "1min": TimeFrame.Minute,
                "5min": TimeFrame(5, "Min"),
                "15min": TimeFrame(15, "Min"),
                "1hour": TimeFrame.Hour,
                "1day": TimeFrame.Day,
            }
            tf = tf_map.get(timeframe.lower(), TimeFrame.Hour)

            if start is None:
                start = datetime.now() - timedelta(days=7)
            if end is None:
                end = datetime.now()

            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars_response = self.data_client.get_crypto_bars(request)

            bars = []
            if symbol in bars_response.data:
                for bar in bars_response.data[symbol]:
                    bars.append(CryptoBar(
                        symbol=symbol,
                        open=Decimal(str(bar.open)),
                        high=Decimal(str(bar.high)),
                        low=Decimal(str(bar.low)),
                        close=Decimal(str(bar.close)),
                        volume=Decimal(str(bar.volume)),
                        timestamp=bar.timestamp,
                        vwap=Decimal(str(bar.vwap)) if hasattr(bar, 'vwap') else None,
                    ))
            return bars

        except Exception as e:
            logger.error(f"Error getting crypto bars for {symbol}: {e}")
            return []

    async def get_positions(self) -> List[CryptoPosition]:
        """
        Get all crypto positions.

        Returns:
            List of CryptoPosition
        """
        if not self.trading_client:
            return []

        try:
            positions = self.trading_client.get_all_positions()

            crypto_positions = []
            for pos in positions:
                # Filter for crypto positions (symbol contains '/')
                if '/' in pos.symbol or 'USD' in pos.symbol:
                    crypto_positions.append(CryptoPosition(
                        symbol=pos.symbol,
                        qty=Decimal(str(pos.qty)),
                        avg_entry_price=Decimal(str(pos.avg_entry_price)),
                        market_value=Decimal(str(pos.market_value)),
                        current_price=Decimal(str(pos.current_price)),
                        unrealized_pl=Decimal(str(pos.unrealized_pl)),
                        unrealized_plpc=float(pos.unrealized_plpc),
                    ))
            return crypto_positions

        except Exception as e:
            logger.error(f"Error getting crypto positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[CryptoPosition]:
        """
        Get position for a specific crypto asset.

        Args:
            symbol: Crypto symbol

        Returns:
            CryptoPosition or None
        """
        symbol = self._normalize_symbol(symbol)
        positions = await self.get_positions()

        for pos in positions:
            if pos.symbol == symbol or pos.symbol.replace("/", "") == symbol.replace("/", ""):
                return pos
        return None

    async def buy(
        self,
        symbol: str,
        qty: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
    ) -> Optional[CryptoOrder]:
        """
        Buy crypto.

        Args:
            symbol: Crypto symbol
            qty: Quantity to buy (fractional allowed)
            notional: Dollar amount to buy (alternative to qty)
            limit_price: Limit price (optional, market order if None)

        Returns:
            CryptoOrder or None
        """
        if not self.trading_client:
            return None

        try:
            symbol = self._normalize_symbol(symbol)

            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=float(qty) if qty else None,
                    notional=float(notional) if notional else None,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    limit_price=float(limit_price),
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=float(qty) if qty else None,
                    notional=float(notional) if notional else None,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )

            order = self.trading_client.submit_order(order_request)

            return CryptoOrder(
                id=str(order.id),
                symbol=order.symbol,
                side="buy",
                qty=Decimal(str(order.qty)) if order.qty else Decimal("0"),
                filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
                order_type=str(order.order_type),
                status=str(order.status),
                limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
                filled_avg_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
            )

        except Exception as e:
            logger.error(f"Error buying {symbol}: {e}")
            return None

    async def sell(
        self,
        symbol: str,
        qty: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
    ) -> Optional[CryptoOrder]:
        """
        Sell crypto.

        Args:
            symbol: Crypto symbol
            qty: Quantity to sell
            notional: Dollar amount to sell
            limit_price: Limit price (optional)

        Returns:
            CryptoOrder or None
        """
        if not self.trading_client:
            return None

        try:
            symbol = self._normalize_symbol(symbol)

            if limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=float(qty) if qty else None,
                    notional=float(notional) if notional else None,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    limit_price=float(limit_price),
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=float(qty) if qty else None,
                    notional=float(notional) if notional else None,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )

            order = self.trading_client.submit_order(order_request)

            return CryptoOrder(
                id=str(order.id),
                symbol=order.symbol,
                side="sell",
                qty=Decimal(str(order.qty)) if order.qty else Decimal("0"),
                filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
                order_type=str(order.order_type),
                status=str(order.status),
                limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
                filled_avg_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
                submitted_at=order.submitted_at,
                filled_at=order.filled_at,
            )

        except Exception as e:
            logger.error(f"Error selling {symbol}: {e}")
            return None

    async def close_position(self, symbol: str) -> Optional[CryptoOrder]:
        """
        Close entire position for a crypto asset.

        Args:
            symbol: Crypto symbol

        Returns:
            CryptoOrder or None
        """
        if not self.trading_client:
            return None

        try:
            symbol = self._normalize_symbol(symbol)

            # Get current position
            position = await self.get_position(symbol)
            if not position or position.qty == 0:
                logger.warning(f"No position to close for {symbol}")
                return None

            # Sell entire position
            return await self.sell(symbol, qty=position.qty)

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None

    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: str = "open",
    ) -> List[CryptoOrder]:
        """
        Get crypto orders.

        Args:
            symbol: Filter by symbol (optional)
            status: "open", "closed", or "all"

        Returns:
            List of CryptoOrder
        """
        if not self.trading_client:
            return []

        try:
            orders = self.trading_client.get_orders()

            crypto_orders = []
            for order in orders:
                # Filter for crypto orders
                if '/' not in order.symbol and 'USD' not in order.symbol:
                    continue

                if symbol and order.symbol != self._normalize_symbol(symbol):
                    continue

                crypto_orders.append(CryptoOrder(
                    id=str(order.id),
                    symbol=order.symbol,
                    side=str(order.side),
                    qty=Decimal(str(order.qty)) if order.qty else Decimal("0"),
                    filled_qty=Decimal(str(order.filled_qty)) if order.filled_qty else Decimal("0"),
                    order_type=str(order.order_type),
                    status=str(order.status),
                    limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
                    filled_avg_price=Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None,
                    submitted_at=order.submitted_at,
                    filled_at=order.filled_at,
                ))
            return crypto_orders

        except Exception as e:
            logger.error(f"Error getting crypto orders: {e}")
            return []

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if not self.trading_client:
            return False

        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def get_account_crypto_info(self) -> Dict[str, Any]:
        """
        Get account info relevant for crypto trading.

        Returns:
            Dictionary with account info
        """
        if not self.trading_client:
            return {}

        try:
            account = self.trading_client.get_account()

            return {
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "crypto_status": getattr(account, 'crypto_status', 'active'),
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol format."""
        # Remove any slashes and spaces
        symbol = symbol.upper().replace(" ", "")

        # Convert BTCUSD to BTC/USD if needed
        if "/" not in symbol and symbol.endswith("USD"):
            base = symbol[:-3]
            return f"{base}/USD"

        return symbol

    @staticmethod
    def get_supported_assets() -> List[str]:
        """Get list of supported crypto assets."""
        return [asset.value for asset in CryptoAsset]


def create_crypto_client(
    api_key: str,
    secret_key: str,
    paper_trading: bool = True,
) -> AlpacaCryptoClient:
    """Factory function to create crypto client."""
    return AlpacaCryptoClient(api_key, secret_key, paper_trading)
