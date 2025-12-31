"""
Technical Analysis-based Stock Screeners.

Screen stocks based on technical indicators and price action.
"""

from decimal import Decimal
from typing import List, Optional

from .base import IScreener, StockData


class TechnicalScreener(IScreener):
    """
    Multi-criteria technical screener.

    Combines multiple technical conditions.
    """

    def __init__(
        self,
        require_above_sma_20: bool = False,
        require_above_sma_50: bool = False,
        require_above_sma_200: bool = False,
        require_below_sma_20: bool = False,
        require_below_sma_50: bool = False,
        require_below_sma_200: bool = False,
        min_rsi: Optional[float] = None,
        max_rsi: Optional[float] = None,
        min_atr: Optional[float] = None,
        max_atr: Optional[float] = None,
        name: str = "TechnicalScreener",
    ):
        """
        Initialize technical screener.

        Args:
            require_above_sma_*: Require price above SMA
            require_below_sma_*: Require price below SMA
            min_rsi/max_rsi: RSI range filter
            min_atr/max_atr: ATR range filter
        """
        super().__init__(name=name)
        self.require_above_sma_20 = require_above_sma_20
        self.require_above_sma_50 = require_above_sma_50
        self.require_above_sma_200 = require_above_sma_200
        self.require_below_sma_20 = require_below_sma_20
        self.require_below_sma_50 = require_below_sma_50
        self.require_below_sma_200 = require_below_sma_200
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.min_atr = min_atr
        self.max_atr = max_atr

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by technical criteria."""
        filtered = []

        for stock in stocks:
            # SMA filters
            if self.require_above_sma_20 and not stock.above_sma_20:
                continue
            if self.require_above_sma_50 and not stock.above_sma_50:
                continue
            if self.require_above_sma_200 and not stock.above_sma_200:
                continue

            if self.require_below_sma_20:
                if stock.sma_20 is None or float(stock.price) >= stock.sma_20:
                    continue
            if self.require_below_sma_50:
                if stock.sma_50 is None or float(stock.price) >= stock.sma_50:
                    continue
            if self.require_below_sma_200:
                if stock.sma_200 is None or float(stock.price) >= stock.sma_200:
                    continue

            # RSI filter
            if stock.rsi is not None:
                if self.min_rsi is not None and stock.rsi < self.min_rsi:
                    continue
                if self.max_rsi is not None and stock.rsi > self.max_rsi:
                    continue

            # ATR filter
            if stock.atr is not None:
                if self.min_atr is not None and stock.atr < self.min_atr:
                    continue
                if self.max_atr is not None and stock.atr > self.max_atr:
                    continue

            filtered.append(stock)

        return filtered


class RSIScreener(IScreener):
    """
    Filter stocks by RSI (Relative Strength Index).

    Find overbought or oversold stocks.
    """

    def __init__(
        self,
        min_rsi: Optional[float] = None,
        max_rsi: Optional[float] = None,
        condition: Optional[str] = None,  # 'overbought', 'oversold', 'neutral'
        name: str = "RSIScreener",
    ):
        """
        Initialize RSI screener.

        Args:
            min_rsi: Minimum RSI value
            max_rsi: Maximum RSI value
            condition: Preset condition (overrides min/max)
        """
        super().__init__(name=name)

        # Apply preset conditions
        if condition == "overbought":
            self.min_rsi = 70.0
            self.max_rsi = None
        elif condition == "oversold":
            self.min_rsi = None
            self.max_rsi = 30.0
        elif condition == "neutral":
            self.min_rsi = 40.0
            self.max_rsi = 60.0
        else:
            self.min_rsi = min_rsi
            self.max_rsi = max_rsi

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by RSI."""
        filtered = []

        for stock in stocks:
            if stock.rsi is None:
                continue

            if self.min_rsi is not None and stock.rsi < self.min_rsi:
                continue

            if self.max_rsi is not None and stock.rsi > self.max_rsi:
                continue

            filtered.append(stock)

        return filtered


class MovingAverageScreener(IScreener):
    """
    Filter stocks by moving average relationships.

    Find stocks in uptrends or downtrends based on SMA stack.
    """

    def __init__(
        self,
        trend: str = "bullish",  # 'bullish', 'bearish', or 'crossover'
        use_sma_20: bool = True,
        use_sma_50: bool = True,
        use_sma_200: bool = True,
        name: str = "MovingAverageScreener",
    ):
        """
        Initialize moving average screener.

        Args:
            trend: Trend to filter for
            use_sma_*: Which SMAs to use in analysis
        """
        super().__init__(name=name)
        self.trend = trend
        self.use_sma_20 = use_sma_20
        self.use_sma_50 = use_sma_50
        self.use_sma_200 = use_sma_200

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by moving average trend."""
        filtered = []

        for stock in stocks:
            price = float(stock.price)

            if self.trend == "bullish":
                # Bullish: Price > SMA20 > SMA50 > SMA200
                conditions = []
                if self.use_sma_20 and stock.sma_20:
                    conditions.append(price > stock.sma_20)
                if self.use_sma_50 and stock.sma_50:
                    conditions.append(price > stock.sma_50)
                if self.use_sma_200 and stock.sma_200:
                    conditions.append(price > stock.sma_200)

                # Also check SMA stack order
                if self.use_sma_20 and self.use_sma_50:
                    if stock.sma_20 and stock.sma_50:
                        conditions.append(stock.sma_20 > stock.sma_50)
                if self.use_sma_50 and self.use_sma_200:
                    if stock.sma_50 and stock.sma_200:
                        conditions.append(stock.sma_50 > stock.sma_200)

                if conditions and all(conditions):
                    filtered.append(stock)

            elif self.trend == "bearish":
                # Bearish: Price < SMA20 < SMA50 < SMA200
                conditions = []
                if self.use_sma_20 and stock.sma_20:
                    conditions.append(price < stock.sma_20)
                if self.use_sma_50 and stock.sma_50:
                    conditions.append(price < stock.sma_50)
                if self.use_sma_200 and stock.sma_200:
                    conditions.append(price < stock.sma_200)

                if self.use_sma_20 and self.use_sma_50:
                    if stock.sma_20 and stock.sma_50:
                        conditions.append(stock.sma_20 < stock.sma_50)
                if self.use_sma_50 and self.use_sma_200:
                    if stock.sma_50 and stock.sma_200:
                        conditions.append(stock.sma_50 < stock.sma_200)

                if conditions and all(conditions):
                    filtered.append(stock)

            elif self.trend == "crossover":
                # Golden/Death cross detection
                if stock.sma_50 and stock.sma_200:
                    # Recent crossover (SMA50 near SMA200)
                    diff_pct = abs(stock.sma_50 - stock.sma_200) / stock.sma_200 * 100
                    if diff_pct < 2.0:  # Within 2%
                        filtered.append(stock)

        return filtered


class VolatilityScreener(IScreener):
    """
    Filter stocks by volatility metrics.

    Find high or low volatility stocks.
    """

    def __init__(
        self,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        min_atr_pct: Optional[float] = None,  # ATR as % of price
        max_atr_pct: Optional[float] = None,
        name: str = "VolatilityScreener",
    ):
        """
        Initialize volatility screener.

        Args:
            min_volatility/max_volatility: Volatility range
            min_atr_pct/max_atr_pct: ATR as percentage of price
        """
        super().__init__(name=name)
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by volatility."""
        filtered = []

        for stock in stocks:
            # Volatility filter
            if stock.volatility is not None:
                if self.min_volatility and stock.volatility < self.min_volatility:
                    continue
                if self.max_volatility and stock.volatility > self.max_volatility:
                    continue

            # ATR percentage filter
            if stock.atr is not None and float(stock.price) > 0:
                atr_pct = (stock.atr / float(stock.price)) * 100

                if self.min_atr_pct and atr_pct < self.min_atr_pct:
                    continue
                if self.max_atr_pct and atr_pct > self.max_atr_pct:
                    continue

            filtered.append(stock)

        return filtered


class SupportResistanceScreener(IScreener):
    """
    Filter stocks near support or resistance levels.

    Uses daily high/low as intraday S/R levels.
    """

    def __init__(
        self,
        near_high_pct: float = 2.0,  # Within 2% of high
        near_low_pct: float = 2.0,   # Within 2% of low
        level_type: str = "both",    # 'resistance', 'support', or 'both'
        name: str = "SupportResistanceScreener",
    ):
        """
        Initialize S/R screener.

        Args:
            near_high_pct: Percentage threshold for near high
            near_low_pct: Percentage threshold for near low
            level_type: Which levels to filter for
        """
        super().__init__(name=name)
        self.near_high_pct = near_high_pct
        self.near_low_pct = near_low_pct
        self.level_type = level_type

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by proximity to S/R levels."""
        filtered = []

        for stock in stocks:
            if stock.high is None or stock.low is None:
                continue

            price = float(stock.price)
            high = float(stock.high)
            low = float(stock.low)

            near_high = ((high - price) / high * 100) <= self.near_high_pct
            near_low = ((price - low) / low * 100) <= self.near_low_pct

            if self.level_type == "resistance" and near_high:
                filtered.append(stock)
            elif self.level_type == "support" and near_low:
                filtered.append(stock)
            elif self.level_type == "both" and (near_high or near_low):
                filtered.append(stock)

        return filtered


class PriceRangeScreener(IScreener):
    """
    Filter stocks by absolute price range.

    Find penny stocks, mid-price, or high-price stocks.
    """

    def __init__(
        self,
        min_price: Optional[Decimal] = None,
        max_price: Optional[Decimal] = None,
        name: str = "PriceRangeScreener",
    ):
        """
        Initialize price range screener.

        Args:
            min_price: Minimum stock price
            max_price: Maximum stock price
        """
        super().__init__(name=name)
        self.min_price = min_price
        self.max_price = max_price

    async def screen(
        self,
        stocks: List[StockData],
    ) -> List[StockData]:
        """Filter by price range."""
        filtered = []

        for stock in stocks:
            if self.min_price is not None and stock.price < self.min_price:
                continue
            if self.max_price is not None and stock.price > self.max_price:
                continue

            filtered.append(stock)

        return filtered

