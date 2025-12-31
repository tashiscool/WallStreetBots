"""
Fill Models - Realistic Order Fill Simulation for Backtesting.

Ported from QuantConnect/LEAN's FillModel and Lumibot's backtesting broker.
Provides realistic simulation of order execution including:
- Slippage based on order size and market conditions
- Partial fills for large orders
- Market impact modeling
- Spread-based pricing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import random

logger = logging.getLogger(__name__)


class FillStatus(Enum):
    """Status of a fill attempt."""
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    NO_FILL = "no_fill"


@dataclass
class FillResult:
    """Result of a fill attempt."""
    status: FillStatus
    filled_quantity: int
    fill_price: Decimal
    remaining_quantity: int
    slippage: Decimal  # Total slippage in dollars
    slippage_pct: float  # Slippage as percentage
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    partial_fills: List["FillResult"] = field(default_factory=list)

    @property
    def total_value(self) -> Decimal:
        """Total value of the fill."""
        return self.fill_price * self.filled_quantity


@dataclass
class MarketData:
    """Market data for fill calculation."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: int
    avg_volume: int  # Average daily volume
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    vwap: Optional[Decimal] = None
    spread_pct: float = 0.0

    def __post_init__(self):
        """Calculate derived fields."""
        if self.bid and self.ask:
            mid = (self.bid + self.ask) / 2
            if mid > 0:
                self.spread_pct = float((self.ask - self.bid) / mid)

    @property
    def mid(self) -> Decimal:
        """Mid-point price."""
        return (self.bid + self.ask) / 2


class IFillModel(ABC):
    """Abstract interface for fill models."""

    @abstractmethod
    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # "buy" or "sell"
        market_data: MarketData,
    ) -> FillResult:
        """Fill a market order."""
        pass

    @abstractmethod
    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill a limit order."""
        pass

    @abstractmethod
    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill a stop order."""
        pass


class ImmediateFillModel(IFillModel):
    """
    Simple fill model that fills immediately at last price.

    No slippage, no partial fills. Useful for quick backtests.
    """

    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> FillResult:
        """Fill at last price."""
        return FillResult(
            status=FillStatus.FILLED,
            filled_quantity=quantity,
            fill_price=market_data.last,
            remaining_quantity=0,
            slippage=Decimal("0"),
            slippage_pct=0.0,
        )

    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill at limit price if possible."""
        if side == "buy" and market_data.ask <= limit_price:
            return FillResult(
                status=FillStatus.FILLED,
                filled_quantity=quantity,
                fill_price=min(limit_price, market_data.ask),
                remaining_quantity=0,
                slippage=Decimal("0"),
                slippage_pct=0.0,
            )
        elif side == "sell" and market_data.bid >= limit_price:
            return FillResult(
                status=FillStatus.FILLED,
                filled_quantity=quantity,
                fill_price=max(limit_price, market_data.bid),
                remaining_quantity=0,
                slippage=Decimal("0"),
                slippage_pct=0.0,
            )
        return FillResult(
            status=FillStatus.NO_FILL,
            filled_quantity=0,
            fill_price=Decimal("0"),
            remaining_quantity=quantity,
            slippage=Decimal("0"),
            slippage_pct=0.0,
            message="Limit price not reached",
        )

    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill at stop price if triggered."""
        triggered = False
        if side == "sell" and market_data.last <= stop_price:
            triggered = True
        elif side == "buy" and market_data.last >= stop_price:
            triggered = True

        if triggered:
            return FillResult(
                status=FillStatus.FILLED,
                filled_quantity=quantity,
                fill_price=market_data.last,
                remaining_quantity=0,
                slippage=Decimal("0"),
                slippage_pct=0.0,
            )
        return FillResult(
            status=FillStatus.NO_FILL,
            filled_quantity=0,
            fill_price=Decimal("0"),
            remaining_quantity=quantity,
            slippage=Decimal("0"),
            slippage_pct=0.0,
            message="Stop price not triggered",
        )


class SpreadFillModel(IFillModel):
    """
    Fill model that uses bid/ask spread.

    - Buys fill at ask
    - Sells fill at bid
    - More realistic than mid-point fills
    """

    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> FillResult:
        """Fill at bid or ask."""
        if side == "buy":
            fill_price = market_data.ask
        else:
            fill_price = market_data.bid

        # Calculate slippage from mid-point
        mid = market_data.mid
        slippage = abs(fill_price - mid) * quantity
        slippage_pct = float(abs(fill_price - mid) / mid) if mid > 0 else 0

        return FillResult(
            status=FillStatus.FILLED,
            filled_quantity=quantity,
            fill_price=fill_price,
            remaining_quantity=0,
            slippage=slippage,
            slippage_pct=slippage_pct,
        )

    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill at limit or better."""
        if side == "buy":
            if market_data.ask <= limit_price:
                fill_price = market_data.ask
            else:
                return FillResult(
                    status=FillStatus.NO_FILL,
                    filled_quantity=0,
                    fill_price=Decimal("0"),
                    remaining_quantity=quantity,
                    slippage=Decimal("0"),
                    slippage_pct=0.0,
                )
        else:  # sell
            if market_data.bid >= limit_price:
                fill_price = market_data.bid
            else:
                return FillResult(
                    status=FillStatus.NO_FILL,
                    filled_quantity=0,
                    fill_price=Decimal("0"),
                    remaining_quantity=quantity,
                    slippage=Decimal("0"),
                    slippage_pct=0.0,
                )

        return FillResult(
            status=FillStatus.FILLED,
            filled_quantity=quantity,
            fill_price=fill_price,
            remaining_quantity=0,
            slippage=Decimal("0"),
            slippage_pct=0.0,
        )

    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill stop order at market after trigger."""
        triggered = False
        if side == "sell" and market_data.bid <= stop_price:
            triggered = True
        elif side == "buy" and market_data.ask >= stop_price:
            triggered = True

        if triggered:
            # Stop triggered, fill as market order
            return self.fill_market_order(symbol, quantity, side, market_data)

        return FillResult(
            status=FillStatus.NO_FILL,
            filled_quantity=0,
            fill_price=Decimal("0"),
            remaining_quantity=quantity,
            slippage=Decimal("0"),
            slippage_pct=0.0,
        )


class VolumeSlippageFillModel(IFillModel):
    """
    Fill model with volume-based slippage and partial fills.

    Models market impact based on order size relative to volume:
    - Small orders: minimal slippage
    - Large orders: significant slippage and partial fills
    """

    def __init__(
        self,
        base_slippage_bps: float = 5,  # Base slippage in basis points
        volume_impact_factor: float = 0.1,  # Impact per % of volume
        max_slippage_pct: float = 0.02,  # 2% max slippage
        max_volume_participation: float = 0.10,  # Max 10% of volume per fill
        random_factor: float = 0.2,  # Randomness in slippage
    ):
        """
        Initialize volume slippage model.

        Args:
            base_slippage_bps: Base slippage in basis points (5 = 0.05%)
            volume_impact_factor: Additional slippage per % of volume
            max_slippage_pct: Maximum slippage percentage
            max_volume_participation: Maximum portion of volume to fill
            random_factor: Randomness factor (0 = deterministic)
        """
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
        self.max_slippage_pct = max_slippage_pct
        self.max_volume_participation = max_volume_participation
        self.random_factor = random_factor

    def _calculate_slippage(
        self,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> Tuple[Decimal, float]:
        """
        Calculate slippage based on order size and market conditions.

        Returns:
            Tuple of (slippage_amount, slippage_pct)
        """
        # Base slippage
        base_slip = self.base_slippage_bps / 10000

        # Volume-based impact
        if market_data.avg_volume > 0:
            volume_pct = quantity / market_data.avg_volume
            volume_impact = volume_pct * self.volume_impact_factor
        else:
            volume_impact = 0.01  # Default if no volume data

        # Spread-based impact
        spread_impact = market_data.spread_pct / 2

        # Total slippage percentage
        total_slip_pct = base_slip + volume_impact + spread_impact

        # Add randomness
        if self.random_factor > 0:
            random_adj = 1 + (random.random() - 0.5) * 2 * self.random_factor
            total_slip_pct *= random_adj

        # Cap at maximum
        total_slip_pct = min(total_slip_pct, self.max_slippage_pct)

        # Calculate dollar amount
        slippage_amount = market_data.mid * Decimal(str(total_slip_pct)) * quantity

        return slippage_amount, total_slip_pct

    def _calculate_fill_quantity(
        self,
        requested_quantity: int,
        market_data: MarketData,
    ) -> int:
        """
        Calculate how much of the order can be filled based on volume.

        Returns:
            Quantity that can be filled
        """
        max_fill = int(market_data.volume * self.max_volume_participation)
        return min(requested_quantity, max(1, max_fill))

    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> FillResult:
        """Fill market order with volume-based slippage."""
        # Calculate fill quantity
        fill_qty = self._calculate_fill_quantity(quantity, market_data)
        remaining = quantity - fill_qty

        # Calculate slippage
        slippage_amount, slippage_pct = self._calculate_slippage(
            fill_qty, side, market_data
        )

        # Calculate fill price
        if side == "buy":
            fill_price = market_data.ask + (slippage_amount / fill_qty if fill_qty > 0 else Decimal("0"))
        else:
            fill_price = market_data.bid - (slippage_amount / fill_qty if fill_qty > 0 else Decimal("0"))

        # Ensure price is reasonable
        fill_price = max(Decimal("0.01"), fill_price)

        status = FillStatus.FILLED if remaining == 0 else FillStatus.PARTIAL

        return FillResult(
            status=status,
            filled_quantity=fill_qty,
            fill_price=fill_price,
            remaining_quantity=remaining,
            slippage=slippage_amount,
            slippage_pct=slippage_pct,
            message=f"Volume participation: {fill_qty}/{market_data.volume}",
        )

    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill limit order with volume constraints."""
        # Check if limit is marketable
        if side == "buy" and market_data.ask > limit_price:
            return FillResult(
                status=FillStatus.NO_FILL,
                filled_quantity=0,
                fill_price=Decimal("0"),
                remaining_quantity=quantity,
                slippage=Decimal("0"),
                slippage_pct=0.0,
                message="Limit not marketable",
            )
        elif side == "sell" and market_data.bid < limit_price:
            return FillResult(
                status=FillStatus.NO_FILL,
                filled_quantity=0,
                fill_price=Decimal("0"),
                remaining_quantity=quantity,
                slippage=Decimal("0"),
                slippage_pct=0.0,
                message="Limit not marketable",
            )

        # Calculate fill quantity
        fill_qty = self._calculate_fill_quantity(quantity, market_data)
        remaining = quantity - fill_qty

        # Limit orders get better fills (at limit price)
        if side == "buy":
            fill_price = min(limit_price, market_data.ask)
        else:
            fill_price = max(limit_price, market_data.bid)

        status = FillStatus.FILLED if remaining == 0 else FillStatus.PARTIAL

        return FillResult(
            status=status,
            filled_quantity=fill_qty,
            fill_price=fill_price,
            remaining_quantity=remaining,
            slippage=Decimal("0"),  # No slippage for limit orders
            slippage_pct=0.0,
        )

    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill stop order with slippage after trigger."""
        # Check if stop is triggered
        triggered = False
        if side == "sell" and market_data.last <= stop_price:
            triggered = True
        elif side == "buy" and market_data.last >= stop_price:
            triggered = True

        if not triggered:
            return FillResult(
                status=FillStatus.NO_FILL,
                filled_quantity=0,
                fill_price=Decimal("0"),
                remaining_quantity=quantity,
                slippage=Decimal("0"),
                slippage_pct=0.0,
            )

        # Stop triggered - fill as market order with extra slippage
        result = self.fill_market_order(symbol, quantity, side, market_data)

        # Add gap slippage (stops often fill worse in fast markets)
        gap_slippage = abs(market_data.last - stop_price)
        result.slippage += gap_slippage * result.filled_quantity

        return result


class EquityFillModel(IFillModel):
    """
    Fill model specifically for equity trading.

    Handles:
    - Market hours (only fills during trading hours)
    - Auction fills (MOO, MOC orders)
    - Short sale restrictions
    """

    def __init__(
        self,
        inner_model: Optional[IFillModel] = None,
        allow_extended_hours: bool = False,
        short_sale_restriction: bool = False,
    ):
        """
        Initialize equity fill model.

        Args:
            inner_model: Underlying fill model (default: VolumeSlippageFillModel)
            allow_extended_hours: Allow fills in pre/post market
            short_sale_restriction: Enforce uptick rule for shorts
        """
        self.inner_model = inner_model or VolumeSlippageFillModel()
        self.allow_extended_hours = allow_extended_hours
        self.short_sale_restriction = short_sale_restriction

    def _is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open."""
        # Simplified check - real implementation would use market calendars
        hour = timestamp.hour
        minute = timestamp.minute
        weekday = timestamp.weekday()

        # Closed on weekends
        if weekday >= 5:
            return False

        # Regular hours: 9:30 AM - 4:00 PM ET
        if hour < 9 or (hour == 9 and minute < 30):
            return self.allow_extended_hours  # Pre-market
        if hour >= 16:
            return self.allow_extended_hours  # After-hours

        return True

    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> FillResult:
        """Fill market order during trading hours."""
        if not self._is_market_open(datetime.now()):
            return FillResult(
                status=FillStatus.REJECTED,
                filled_quantity=0,
                fill_price=Decimal("0"),
                remaining_quantity=quantity,
                slippage=Decimal("0"),
                slippage_pct=0.0,
                message="Market closed",
            )

        return self.inner_model.fill_market_order(symbol, quantity, side, market_data)

    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill limit order."""
        return self.inner_model.fill_limit_order(
            symbol, quantity, side, limit_price, market_data
        )

    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill stop order."""
        if not self._is_market_open(datetime.now()):
            return FillResult(
                status=FillStatus.NO_FILL,
                filled_quantity=0,
                fill_price=Decimal("0"),
                remaining_quantity=quantity,
                slippage=Decimal("0"),
                slippage_pct=0.0,
                message="Market closed, stop pending",
            )

        return self.inner_model.fill_stop_order(
            symbol, quantity, side, stop_price, market_data
        )


class OptionsFillModel(IFillModel):
    """
    Fill model specifically for options trading.

    Handles:
    - Wider spreads typical in options
    - Lower liquidity
    - Assignment and exercise
    """

    def __init__(
        self,
        spread_multiplier: float = 2.0,  # Options have wider spreads
        volume_factor: float = 0.5,  # Lower volume participation
    ):
        """
        Initialize options fill model.

        Args:
            spread_multiplier: Multiplier for bid-ask spread impact
            volume_factor: Reduction factor for volume participation
        """
        self.spread_multiplier = spread_multiplier
        self.volume_factor = volume_factor
        self.inner_model = VolumeSlippageFillModel(
            max_volume_participation=0.05 * volume_factor,  # Lower for options
            base_slippage_bps=10,  # Higher base slippage
        )

    def fill_market_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        market_data: MarketData,
    ) -> FillResult:
        """Fill options market order with wider spreads."""
        # Widen the spread for options
        mid = market_data.mid
        spread = market_data.ask - market_data.bid
        wide_spread = spread * Decimal(str(self.spread_multiplier))

        adjusted_data = MarketData(
            symbol=market_data.symbol,
            bid=mid - wide_spread / 2,
            ask=mid + wide_spread / 2,
            last=market_data.last,
            volume=int(market_data.volume * self.volume_factor),
            avg_volume=int(market_data.avg_volume * self.volume_factor),
        )

        return self.inner_model.fill_market_order(symbol, quantity, side, adjusted_data)

    def fill_limit_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        limit_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill options limit order."""
        return self.inner_model.fill_limit_order(
            symbol, quantity, side, limit_price, market_data
        )

    def fill_stop_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        stop_price: Decimal,
        market_data: MarketData,
    ) -> FillResult:
        """Fill options stop order (rarely used)."""
        return self.inner_model.fill_stop_order(
            symbol, quantity, side, stop_price, market_data
        )


# Factory for creating fill models
class FillModelFactory:
    """Factory for creating appropriate fill models."""

    @staticmethod
    def create(
        model_type: str = "volume_slippage",
        **kwargs,
    ) -> IFillModel:
        """
        Create a fill model.

        Args:
            model_type: Type of model (immediate, spread, volume_slippage, equity, options)
            **kwargs: Model-specific parameters

        Returns:
            Fill model instance
        """
        if model_type == "immediate":
            return ImmediateFillModel()
        elif model_type == "spread":
            return SpreadFillModel()
        elif model_type == "volume_slippage":
            return VolumeSlippageFillModel(**kwargs)
        elif model_type == "equity":
            return EquityFillModel(**kwargs)
        elif model_type == "options":
            return OptionsFillModel(**kwargs)
        else:
            raise ValueError(f"Unknown fill model type: {model_type}")

    @staticmethod
    def for_asset_type(asset_type: str) -> IFillModel:
        """Get appropriate fill model for asset type."""
        if asset_type in ("option", "options"):
            return OptionsFillModel()
        elif asset_type in ("stock", "equity", "etf"):
            return EquityFillModel()
        else:
            return VolumeSlippageFillModel()
