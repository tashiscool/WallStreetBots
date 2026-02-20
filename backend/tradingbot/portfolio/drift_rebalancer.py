"""
Drift Rebalancer - Automatic Portfolio Rebalancing.

Ported from Lumibot's drift_rebalancer_logic.py, adapted for
stock/options trading with WallStreetBots.

Automatically rebalances portfolio when asset weights drift
from target allocations beyond specified thresholds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Type of drift measurement."""
    ABSOLUTE = "absolute"     # Drift in percentage points (e.g., 5% = 0.05)
    RELATIVE = "relative"     # Drift as % of target (e.g., 10% of 60% target)


class RebalanceMethod(Enum):
    """Rebalancing method."""
    FULL = "full"             # Rebalance all positions to exact targets
    THRESHOLD = "threshold"   # Only rebalance positions exceeding threshold
    PARTIAL = "partial"       # Partially rebalance towards target


@dataclass
class DriftConfig:
    """Configuration for drift rebalancing."""
    # Drift thresholds
    drift_type: DriftType = DriftType.ABSOLUTE
    drift_threshold: float = 0.05  # 5% absolute drift triggers rebalance

    # Rebalancing settings
    rebalance_method: RebalanceMethod = RebalanceMethod.THRESHOLD
    partial_rebalance_pct: float = 0.5  # For partial, move 50% towards target

    # Timing
    min_rebalance_interval: timedelta = timedelta(days=1)
    rebalance_on_days: Optional[List[int]] = None  # 0=Mon, 6=Sun

    # Trading constraints
    min_order_value: Decimal = Decimal("100")  # Minimum order to execute
    avoid_wash_sales: bool = True  # Don't sell recently bought positions at loss
    wash_sale_days: int = 30

    # Cash management
    maintain_cash_pct: float = 0.02  # Keep 2% in cash
    use_limit_orders: bool = True
    limit_order_buffer: float = 0.001  # 0.1% buffer for limit orders


@dataclass
class RebalanceOrder:
    """Represents an order needed for rebalancing."""
    symbol: str
    action: str  # "buy" or "sell"
    quantity: int
    current_weight: float
    target_weight: float
    drift: float
    estimated_value: Decimal
    order_type: str = "market"
    limit_price: Optional[Decimal] = None

    @property
    def is_buy(self) -> bool:
        return self.action == "buy"

    @property
    def is_sell(self) -> bool:
        return self.action == "sell"


@dataclass
class RebalanceResult:
    """Result of a rebalancing operation."""
    orders: List[RebalanceOrder]
    executed_orders: List[RebalanceOrder] = field(default_factory=list)
    failed_orders: List[RebalanceOrder] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    total_drift_before: float = 0.0
    total_drift_after: float = 0.0
    portfolio_value: Decimal = Decimal("0")
    cash_balance: Decimal = Decimal("0")
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.failed_orders) == 0 and len(self.errors) == 0

    @property
    def orders_needed(self) -> bool:
        return len(self.orders) > 0


class DriftRebalancer:
    """
    Automatic portfolio rebalancer based on weight drift.

    Monitors portfolio positions and generates rebalancing orders
    when weights drift from target allocations.

    Usage:
        # Define target allocations
        targets = {
            "SPY": 0.60,   # 60% S&P 500
            "BND": 0.30,   # 30% Bonds
            "GLD": 0.10,   # 10% Gold
        }

        rebalancer = DriftRebalancer(
            target_allocations=targets,
            config=DriftConfig(drift_threshold=0.05)
        )

        # Check if rebalancing is needed
        result = await rebalancer.check_and_rebalance(broker)

        if result.orders_needed:
            print(f"Rebalancing with {len(result.orders)} orders")
    """

    def __init__(
        self,
        target_allocations: Dict[str, float],
        config: Optional[DriftConfig] = None,
        broker=None,
    ):
        """
        Initialize the drift rebalancer.

        Args:
            target_allocations: Dict of symbol -> target weight (0-1)
            config: DriftConfig for rebalancing settings
            broker: Broker client for fetching positions and executing orders
        """
        self.target_allocations = target_allocations
        self.config = config or DriftConfig()
        self.broker = broker

        # Validate allocations sum to ~1.0
        total = sum(target_allocations.values())
        if not (0.98 <= total <= 1.02):
            logger.warning(
                f"Target allocations sum to {total:.2%}, expected ~100%"
            )

        self._last_rebalance: Optional[datetime] = None
        self._purchase_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}

        # Callbacks
        self._on_rebalance_start: List[Callable] = []
        self._on_rebalance_complete: List[Callable] = []

    def set_targets(self, allocations: Dict[str, float]) -> None:
        """Update target allocations."""
        self.target_allocations = allocations
        logger.info(f"Updated target allocations: {allocations}")

    def add_target(self, symbol: str, weight: float) -> None:
        """Add or update a single target allocation."""
        self.target_allocations[symbol] = weight

    def remove_target(self, symbol: str) -> None:
        """Remove a symbol from target allocations."""
        self.target_allocations.pop(symbol, None)

    def calculate_drift(
        self,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate drift for each position.

        Args:
            current_weights: Dict of symbol -> current weight

        Returns:
            Dict of symbol -> drift (positive = overweight, negative = underweight)
        """
        drifts = {}

        # Check all target symbols
        for symbol, target in self.target_allocations.items():
            current = current_weights.get(symbol, 0.0)

            if self.config.drift_type == DriftType.ABSOLUTE:
                drift = current - target
            else:  # RELATIVE
                if target > 0:
                    drift = (current - target) / target
                else:
                    drift = current  # Any position is infinite drift from 0

            drifts[symbol] = drift

        # Check for positions not in targets (should be sold)
        for symbol, current in current_weights.items():
            if symbol not in self.target_allocations and current > 0:
                drifts[symbol] = current  # Full current weight is drift

        return drifts

    def get_total_drift(self, drifts: Dict[str, float]) -> float:
        """
        Calculate total portfolio drift.

        Returns sum of absolute drifts.
        """
        return sum(abs(d) for d in drifts.values())

    def needs_rebalancing(
        self,
        current_weights: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if portfolio needs rebalancing.

        Args:
            current_weights: Current position weights

        Returns:
            Tuple of (needs_rebalance: bool, drifts: Dict)
        """
        # Check timing constraints
        if self._last_rebalance is not None:
            time_since = datetime.now() - self._last_rebalance
            if time_since < self.config.min_rebalance_interval:
                return False, {}

        # Check day of week
        if self.config.rebalance_on_days is not None:
            if datetime.now().weekday() not in self.config.rebalance_on_days:
                return False, {}

        # Calculate drifts
        drifts = self.calculate_drift(current_weights)

        # Check if any position exceeds threshold
        for drift in drifts.values():
            if abs(drift) >= self.config.drift_threshold:
                return True, drifts

        return False, drifts

    def generate_rebalance_orders(
        self,
        current_positions: Dict[str, Dict[str, Any]],
        portfolio_value: Decimal,
        prices: Dict[str, Decimal],
    ) -> List[RebalanceOrder]:
        """
        Generate orders needed to rebalance portfolio.

        Args:
            current_positions: Dict of symbol -> position info
            portfolio_value: Total portfolio value
            prices: Current prices for each symbol

        Returns:
            List of RebalanceOrder objects
        """
        orders = []

        # Calculate current weights
        current_weights = {}
        for symbol, position in current_positions.items():
            pos_value = Decimal(str(position.get("market_value", 0)))
            if portfolio_value > 0:
                current_weights[symbol] = float(pos_value / portfolio_value)
            else:
                current_weights[symbol] = 0.0

        # Calculate drifts
        drifts = self.calculate_drift(current_weights)

        # Calculate target dollar amounts
        for symbol, target_weight in self.target_allocations.items():
            current_weight = current_weights.get(symbol, 0.0)
            drift = drifts.get(symbol, 0.0)

            # Skip if within threshold
            if abs(drift) < self.config.drift_threshold:
                continue

            # Calculate target position value
            target_value = portfolio_value * Decimal(str(target_weight))

            # Get current position value
            current_pos = current_positions.get(symbol, {})
            current_value = Decimal(str(current_pos.get("market_value", 0)))
            current_qty = current_pos.get("quantity", 0)

            # Calculate needed change
            value_change = target_value - current_value

            # Apply partial rebalancing if configured
            if self.config.rebalance_method == RebalanceMethod.PARTIAL:
                value_change = value_change * Decimal(str(self.config.partial_rebalance_pct))

            # Skip small orders
            if abs(value_change) < self.config.min_order_value:
                continue

            # Calculate shares to trade
            price = prices.get(symbol, Decimal("0"))
            if price <= 0:
                logger.warning(f"No price available for {symbol}")
                continue

            shares = int(value_change / price)
            if shares == 0:
                continue

            # Check wash sale rule
            if shares < 0 and self.config.avoid_wash_sales:
                if self._would_trigger_wash_sale(symbol):
                    logger.info(f"Skipping {symbol} sale to avoid wash sale")
                    continue

            # Determine order type and price
            order_type = "limit" if self.config.use_limit_orders else "market"
            limit_price = None
            if order_type == "limit":
                buffer = Decimal(str(self.config.limit_order_buffer))
                if shares > 0:  # Buy
                    limit_price = price * (1 + buffer)
                else:  # Sell
                    limit_price = price * (1 - buffer)

            orders.append(RebalanceOrder(
                symbol=symbol,
                action="buy" if shares > 0 else "sell",
                quantity=abs(shares),
                current_weight=current_weight,
                target_weight=target_weight,
                drift=drift,
                estimated_value=abs(value_change),
                order_type=order_type,
                limit_price=limit_price,
            ))

        # Handle positions not in targets (sell all)
        for symbol, position in current_positions.items():
            if symbol not in self.target_allocations:
                qty = position.get("quantity", 0)
                if qty > 0:
                    value = Decimal(str(position.get("market_value", 0)))
                    orders.append(RebalanceOrder(
                        symbol=symbol,
                        action="sell",
                        quantity=qty,
                        current_weight=current_weights.get(symbol, 0),
                        target_weight=0.0,
                        drift=current_weights.get(symbol, 0),
                        estimated_value=value,
                        order_type="market",
                    ))

        # Sort: sells first (to free up cash), then buys
        orders.sort(key=lambda o: (o.is_buy, -o.estimated_value))

        return orders

    def _would_trigger_wash_sale(self, symbol: str) -> bool:
        """Check if selling would trigger wash sale rule."""
        purchases = self._purchase_history.get(symbol, [])
        if not purchases:
            return False

        cutoff = datetime.now() - timedelta(days=self.config.wash_sale_days)
        recent_purchases = [p for p in purchases if p[0] > cutoff]
        return len(recent_purchases) > 0

    def record_purchase(self, symbol: str, price: Decimal) -> None:
        """Record a purchase for wash sale tracking."""
        if symbol not in self._purchase_history:
            self._purchase_history[symbol] = []
        self._purchase_history[symbol].append((datetime.now(), price))

        # Clean up old history
        cutoff = datetime.now() - timedelta(days=self.config.wash_sale_days + 1)
        self._purchase_history[symbol] = [
            p for p in self._purchase_history[symbol]
            if p[0] > cutoff
        ]

    async def check_and_rebalance(
        self,
        broker=None,
        execute: bool = True,
    ) -> RebalanceResult:
        """
        Check portfolio and execute rebalancing if needed.

        Args:
            broker: Broker client (uses self.broker if not provided)
            execute: Whether to actually execute orders (False = dry run)

        Returns:
            RebalanceResult with orders and execution status
        """
        broker = broker or self.broker
        if broker is None:
            raise ValueError("No broker provided")

        result = RebalanceResult()

        try:
            # Get current positions
            positions = await broker.get_positions()
            current_positions = {
                p["symbol"]: p for p in positions
            }

            # Get account info
            account = await broker.get_account()
            portfolio_value = Decimal(str(account.get("portfolio_value", 0)))
            cash_balance = Decimal(str(account.get("cash", 0)))

            result.portfolio_value = portfolio_value
            result.cash_balance = cash_balance

            if portfolio_value <= 0:
                result.errors.append("Portfolio value is zero")
                return result

            # Calculate current weights
            current_weights = {}
            for symbol, pos in current_positions.items():
                pos_value = Decimal(str(pos.get("market_value", 0)))
                current_weights[symbol] = float(pos_value / portfolio_value)

            # Check if rebalancing is needed
            needs_rebalance, drifts = self.needs_rebalancing(current_weights)
            result.total_drift_before = self.get_total_drift(drifts)

            if not needs_rebalance:
                logger.info("No rebalancing needed")
                return result

            # Get current prices
            prices = {}
            all_symbols = set(self.target_allocations.keys()) | set(current_positions.keys())
            for symbol in all_symbols:
                try:
                    quote = await broker.get_quote(symbol)
                    prices[symbol] = Decimal(str(quote.get("price", 0)))
                except Exception as e:
                    logger.warning(f"Failed to get price for {symbol}: {e}")

            # Generate rebalancing orders
            orders = self.generate_rebalance_orders(
                current_positions, portfolio_value, prices
            )
            result.orders = orders

            if not orders:
                logger.info("No orders needed for rebalancing")
                return result

            # Fire callbacks
            for callback in self._on_rebalance_start:
                await self._call_callback(callback, result)

            # Execute orders
            if execute:
                for order in orders:
                    try:
                        if order.order_type == "limit":
                            await broker.submit_order(
                                symbol=order.symbol,
                                qty=order.quantity,
                                side=order.action,
                                type="limit",
                                limit_price=float(order.limit_price),
                            )
                        else:
                            await broker.submit_order(
                                symbol=order.symbol,
                                qty=order.quantity,
                                side=order.action,
                                type="market",
                            )

                        result.executed_orders.append(order)

                        # Track purchases for wash sale
                        if order.is_buy:
                            price = prices.get(order.symbol, Decimal("0"))
                            self.record_purchase(order.symbol, price)

                    except Exception as e:
                        logger.error(f"Failed to execute order {order}: {e}")
                        result.failed_orders.append(order)
                        result.errors.append(str(e))

            # Update state
            self._last_rebalance = datetime.now()

            # Calculate final drift (estimated)
            if result.success:
                result.total_drift_after = 0.0  # Should be near zero after rebalance
            else:
                result.total_drift_after = result.total_drift_before

            # Fire callbacks
            for callback in self._on_rebalance_complete:
                await self._call_callback(callback, result)

            logger.info(
                f"Rebalancing complete: {len(result.executed_orders)} orders executed, "
                f"{len(result.failed_orders)} failed"
            )

        except Exception as e:
            logger.error(f"Rebalancing error: {e}")
            result.errors.append(str(e))

        return result

    async def _call_callback(self, callback: Callable, *args) -> None:
        """Call a callback, handling both sync and async."""
        import asyncio
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def on_rebalance_start(self, callback: Callable) -> "DriftRebalancer":
        """Register callback for when rebalancing starts."""
        self._on_rebalance_start.append(callback)
        return self

    def on_rebalance_complete(self, callback: Callable) -> "DriftRebalancer":
        """Register callback for when rebalancing completes."""
        self._on_rebalance_complete.append(callback)
        return self

    def get_status(self, current_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Get current rebalancing status.

        Args:
            current_weights: Current position weights

        Returns:
            Status dict with drift info
        """
        drifts = self.calculate_drift(current_weights)
        needs_rebalance, _ = self.needs_rebalancing(current_weights)

        return {
            "needs_rebalancing": needs_rebalance,
            "total_drift": self.get_total_drift(drifts),
            "drift_threshold": self.config.drift_threshold,
            "drifts_by_symbol": drifts,
            "target_allocations": self.target_allocations,
            "current_weights": current_weights,
            "last_rebalance": self._last_rebalance,
        }


# Preset portfolio allocations
class PresetAllocations:
    """Common portfolio allocation presets."""

    @staticmethod
    def classic_60_40() -> Dict[str, float]:
        """Classic 60/40 stocks/bonds allocation."""
        return {
            "SPY": 0.60,   # S&P 500
            "BND": 0.40,   # Total Bond
        }

    @staticmethod
    def three_fund() -> Dict[str, float]:
        """Three-fund portfolio (US, International, Bonds)."""
        return {
            "VTI": 0.50,   # US Total Market
            "VXUS": 0.30,  # International
            "BND": 0.20,   # Bonds
        }

    @staticmethod
    def all_weather() -> Dict[str, float]:
        """Ray Dalio's All Weather portfolio."""
        return {
            "VTI": 0.30,   # US Stocks
            "TLT": 0.40,   # Long-term Treasuries
            "IEF": 0.15,   # Intermediate Treasuries
            "GLD": 0.075,  # Gold
            "DBC": 0.075,  # Commodities
        }

    @staticmethod
    def permanent_portfolio() -> Dict[str, float]:
        """Harry Browne's Permanent Portfolio."""
        return {
            "VTI": 0.25,   # Stocks
            "TLT": 0.25,   # Long-term Bonds
            "SHY": 0.25,   # Cash/Short-term
            "GLD": 0.25,   # Gold
        }

    @staticmethod
    def aggressive_growth() -> Dict[str, float]:
        """Aggressive growth allocation."""
        return {
            "QQQ": 0.40,   # Nasdaq 100
            "VGT": 0.30,   # Tech sector
            "ARKK": 0.20,  # Innovation
            "VWO": 0.10,   # Emerging Markets
        }

    @staticmethod
    def dividend_income() -> Dict[str, float]:
        """Dividend-focused allocation."""
        return {
            "VYM": 0.35,   # High Dividend Yield
            "SCHD": 0.35,  # Dividend Growth
            "VIGI": 0.15,  # International Dividend
            "VNQ": 0.15,   # REITs
        }

    @staticmethod
    def sector_rotation() -> Dict[str, float]:
        """Sector-balanced allocation."""
        return {
            "XLK": 0.15,   # Technology
            "XLV": 0.12,   # Healthcare
            "XLF": 0.12,   # Financials
            "XLY": 0.10,   # Consumer Discretionary
            "XLP": 0.10,   # Consumer Staples
            "XLI": 0.10,   # Industrials
            "XLE": 0.08,   # Energy
            "XLB": 0.08,   # Materials
            "XLU": 0.08,   # Utilities
            "XLRE": 0.07,  # Real Estate
        }
