"""
Risk Management Models - Ported from QuantConnect LEAN.

Provides risk controls and position management to protect portfolios.

Original: https://github.com/QuantConnect/Lean/tree/master/Algorithm.Framework/Risk
License: Apache 2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from enum import Enum


@dataclass
class RiskManagementResult:
    """Result from risk management evaluation."""
    symbol: str
    should_liquidate: bool = False
    target_quantity: Optional[float] = None
    reason: str = ""
    risk_level: float = 0.0  # 0-1 scale

    def __repr__(self) -> str:
        status = "LIQUIDATE" if self.should_liquidate else "OK"
        return f"RiskResult({self.symbol}: {status}, risk={self.risk_level:.2%})"


class RiskManagementModel(ABC):
    """
    Abstract base class for Risk Management Models.

    Risk Management Models monitor portfolio risk and can:
    - Reduce or liquidate positions that exceed risk limits
    - Apply trailing stops and maximum drawdown limits
    - Enforce position size constraints
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """
        Evaluate and manage portfolio risk.

        Args:
            holdings: Current holdings {symbol: quantity}
            prices: Current prices {symbol: price}
            portfolio_value: Total portfolio value

        Returns:
            List of RiskManagementResults with actions to take
        """
        pass

    def on_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Process new market data for risk tracking."""
        pass

    def __repr__(self) -> str:
        return f"{self._name}"


class MaximumDrawdownModel(RiskManagementModel):
    """
    Maximum Drawdown Risk Management Model.

    Liquidates positions when portfolio drawdown exceeds threshold.

    Ported from LEAN's MaximumDrawdownPercentPortfolio.cs
    """

    def __init__(self, max_drawdown: float = 0.10,
                 is_trailing: bool = False,
                 name: Optional[str] = None):
        """
        Args:
            max_drawdown: Maximum allowed drawdown (10%)
            is_trailing: Whether to use trailing high water mark
        """
        name = name or f"MaxDrawdown({max_drawdown*100:.0f}%)"
        super().__init__(name)
        self._max_drawdown = max_drawdown
        self._is_trailing = is_trailing
        self._high_water_mark: float = 0.0
        self._initial_value: Optional[float] = None
        self._is_liquidating: bool = False

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check portfolio drawdown and liquidate if exceeded."""
        results = []

        # Initialize high water mark
        if self._initial_value is None:
            self._initial_value = portfolio_value
            self._high_water_mark = portfolio_value
            return results

        # Update high water mark if trailing
        if self._is_trailing:
            self._high_water_mark = max(self._high_water_mark, portfolio_value)
        else:
            self._high_water_mark = self._initial_value

        # Calculate drawdown
        if self._high_water_mark > 0:
            drawdown = (self._high_water_mark - portfolio_value) / self._high_water_mark
        else:
            drawdown = 0

        # Check if we should liquidate
        if drawdown > self._max_drawdown and not self._is_liquidating:
            self._is_liquidating = True
            for symbol, quantity in holdings.items():
                if quantity != 0:
                    results.append(RiskManagementResult(
                        symbol=symbol,
                        should_liquidate=True,
                        target_quantity=0,
                        reason=f"Max drawdown exceeded: {drawdown*100:.1f}% > {self._max_drawdown*100:.0f}%",
                        risk_level=min(drawdown / self._max_drawdown, 1.0),
                    ))

        return results

    def reset(self) -> None:
        """Reset high water mark."""
        self._high_water_mark = 0.0
        self._initial_value = None
        self._is_liquidating = False


class MaximumDrawdownPerSecurityModel(RiskManagementModel):
    """
    Maximum Drawdown Per Security Risk Management Model.

    Liquidates individual positions when their drawdown exceeds threshold.

    Ported from LEAN's MaximumDrawdownPercentPerSecurity.cs
    """

    def __init__(self, max_drawdown: float = 0.05,
                 name: Optional[str] = None):
        """
        Args:
            max_drawdown: Maximum allowed drawdown per security (5%)
        """
        name = name or f"MaxDrawdownPerSecurity({max_drawdown*100:.0f}%)"
        super().__init__(name)
        self._max_drawdown = max_drawdown
        self._high_prices: Dict[str, float] = {}
        self._liquidated: Set[str] = set()

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check per-security drawdown."""
        results = []

        for symbol, quantity in holdings.items():
            if quantity == 0 or symbol in self._liquidated:
                continue

            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue

            # Update high water mark
            if symbol not in self._high_prices:
                self._high_prices[symbol] = current_price
            else:
                self._high_prices[symbol] = max(self._high_prices[symbol], current_price)

            # Calculate drawdown from high
            high = self._high_prices[symbol]
            if high > 0:
                drawdown = (high - current_price) / high
            else:
                drawdown = 0

            # Check if exceeded
            if drawdown > self._max_drawdown:
                self._liquidated.add(symbol)
                results.append(RiskManagementResult(
                    symbol=symbol,
                    should_liquidate=True,
                    target_quantity=0,
                    reason=f"Security drawdown: {drawdown*100:.1f}% > {self._max_drawdown*100:.0f}%",
                    risk_level=min(drawdown / self._max_drawdown, 1.0),
                ))

        return results

    def on_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update high price tracking."""
        if symbol in self._high_prices:
            self._high_prices[symbol] = max(self._high_prices[symbol], price)
        else:
            self._high_prices[symbol] = price


class TrailingStopModel(RiskManagementModel):
    """
    Trailing Stop Risk Management Model.

    Applies trailing stops to protect profits on winning positions.

    Ported from LEAN's TrailingStopRiskManagementModel.cs
    """

    def __init__(self, trailing_percent: float = 0.05,
                 name: Optional[str] = None):
        """
        Args:
            trailing_percent: Trailing stop distance (5%)
        """
        name = name or f"TrailingStop({trailing_percent*100:.0f}%)"
        super().__init__(name)
        self._trailing_pct = trailing_percent
        self._high_prices: Dict[str, float] = {}
        self._low_prices: Dict[str, float] = {}
        self._entry_prices: Dict[str, float] = {}
        self._triggered: Set[str] = set()

    def set_entry_price(self, symbol: str, price: float) -> None:
        """Set entry price for a position."""
        self._entry_prices[symbol] = price
        self._high_prices[symbol] = price
        self._low_prices[symbol] = price

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check trailing stops."""
        results = []

        for symbol, quantity in holdings.items():
            if quantity == 0 or symbol in self._triggered:
                continue

            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue

            # Initialize tracking if needed
            if symbol not in self._high_prices:
                self._high_prices[symbol] = current_price
                self._low_prices[symbol] = current_price

            if quantity > 0:  # Long position
                # Update trailing high
                self._high_prices[symbol] = max(self._high_prices[symbol], current_price)
                high = self._high_prices[symbol]

                # Check stop
                stop_price = high * (1 - self._trailing_pct)
                if current_price <= stop_price:
                    self._triggered.add(symbol)
                    pnl_pct = (current_price - self._entry_prices.get(symbol, high)) / \
                              self._entry_prices.get(symbol, high) * 100
                    results.append(RiskManagementResult(
                        symbol=symbol,
                        should_liquidate=True,
                        target_quantity=0,
                        reason=f"Trailing stop hit at {current_price:.2f} (stop: {stop_price:.2f}, P&L: {pnl_pct:.1f}%)",
                        risk_level=1.0,
                    ))

            else:  # Short position
                # Update trailing low
                self._low_prices[symbol] = min(self._low_prices[symbol], current_price)
                low = self._low_prices[symbol]

                # Check stop
                stop_price = low * (1 + self._trailing_pct)
                if current_price >= stop_price:
                    self._triggered.add(symbol)
                    results.append(RiskManagementResult(
                        symbol=symbol,
                        should_liquidate=True,
                        target_quantity=0,
                        reason=f"Short trailing stop hit at {current_price:.2f} (stop: {stop_price:.2f})",
                        risk_level=1.0,
                    ))

        return results

    def on_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update price tracking."""
        if symbol in self._high_prices:
            self._high_prices[symbol] = max(self._high_prices[symbol], price)
        if symbol in self._low_prices:
            self._low_prices[symbol] = min(self._low_prices[symbol], price)

    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset tracking for symbol or all."""
        if symbol:
            self._high_prices.pop(symbol, None)
            self._low_prices.pop(symbol, None)
            self._entry_prices.pop(symbol, None)
            self._triggered.discard(symbol)
        else:
            self._high_prices.clear()
            self._low_prices.clear()
            self._entry_prices.clear()
            self._triggered.clear()


class MaximumPositionSizeModel(RiskManagementModel):
    """
    Maximum Position Size Risk Management Model.

    Limits individual position sizes to prevent concentration risk.

    Ported from LEAN's MaximumSectorExposureRiskManagementModel approach
    """

    def __init__(self, max_position_percent: float = 0.10,
                 max_total_exposure: float = 1.0,
                 name: Optional[str] = None):
        """
        Args:
            max_position_percent: Max position as % of portfolio (10%)
            max_total_exposure: Max total exposure (100% = fully invested)
        """
        name = name or f"MaxPositionSize({max_position_percent*100:.0f}%)"
        super().__init__(name)
        self._max_position_pct = max_position_percent
        self._max_exposure = max_total_exposure

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check position sizes and reduce if necessary."""
        results = []

        if portfolio_value <= 0:
            return results

        total_exposure = 0.0

        for symbol, quantity in holdings.items():
            current_price = prices.get(symbol, 0)
            if current_price <= 0:
                continue

            position_value = abs(quantity * current_price)
            position_pct = position_value / portfolio_value
            total_exposure += position_pct

            # Check individual position limit
            if position_pct > self._max_position_pct:
                # Calculate new target quantity
                max_value = portfolio_value * self._max_position_pct
                target_qty = max_value / current_price
                if quantity < 0:
                    target_qty = -target_qty

                results.append(RiskManagementResult(
                    symbol=symbol,
                    should_liquidate=False,
                    target_quantity=target_qty,
                    reason=f"Position too large: {position_pct*100:.1f}% > {self._max_position_pct*100:.0f}%",
                    risk_level=min(position_pct / self._max_position_pct, 1.0),
                ))

        # Check total exposure
        if total_exposure > self._max_exposure:
            scale_factor = self._max_exposure / total_exposure
            for symbol, quantity in holdings.items():
                if quantity != 0:
                    # Only add if not already flagged for reduction
                    existing = next((r for r in results if r.symbol == symbol), None)
                    if existing is None:
                        target_qty = quantity * scale_factor
                        results.append(RiskManagementResult(
                            symbol=symbol,
                            should_liquidate=False,
                            target_quantity=target_qty,
                            reason=f"Total exposure too high: {total_exposure*100:.1f}% > {self._max_exposure*100:.0f}%",
                            risk_level=min(total_exposure / self._max_exposure, 1.0),
                        ))

        return results


class MaximumUnrealizedProfitModel(RiskManagementModel):
    """
    Maximum Unrealized Profit Risk Management Model.

    Takes profits when unrealized gains exceed threshold.
    """

    def __init__(self, profit_target: float = 0.10,
                 name: Optional[str] = None):
        """
        Args:
            profit_target: Take profit at this return (10%)
        """
        name = name or f"ProfitTarget({profit_target*100:.0f}%)"
        super().__init__(name)
        self._profit_target = profit_target
        self._entry_prices: Dict[str, float] = {}

    def set_entry_price(self, symbol: str, price: float) -> None:
        """Set entry price for profit calculation."""
        self._entry_prices[symbol] = price

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check for profit targets."""
        results = []

        for symbol, quantity in holdings.items():
            if quantity == 0:
                continue

            current_price = prices.get(symbol, 0)
            entry_price = self._entry_prices.get(symbol)

            if current_price <= 0 or entry_price is None:
                continue

            # Calculate unrealized P&L
            if quantity > 0:  # Long
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # Short
                pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct >= self._profit_target:
                results.append(RiskManagementResult(
                    symbol=symbol,
                    should_liquidate=True,
                    target_quantity=0,
                    reason=f"Profit target hit: {pnl_pct*100:.1f}% >= {self._profit_target*100:.0f}%",
                    risk_level=0.0,  # Not a risk, it's a win!
                ))

        return results


class SectorExposureModel(RiskManagementModel):
    """
    Sector Exposure Risk Management Model.

    Limits exposure to any single sector.

    Ported from LEAN's MaximumSectorExposureRiskManagementModel.cs
    """

    def __init__(self, max_sector_exposure: float = 0.30,
                 name: Optional[str] = None):
        """
        Args:
            max_sector_exposure: Max exposure to any sector (30%)
        """
        name = name or f"SectorExposure({max_sector_exposure*100:.0f}%)"
        super().__init__(name)
        self._max_sector_exposure = max_sector_exposure
        self._symbol_sectors: Dict[str, str] = {}

    def set_symbol_sector(self, symbol: str, sector: str) -> None:
        """Assign a symbol to a sector."""
        self._symbol_sectors[symbol] = sector

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Check sector exposures."""
        results = []

        if portfolio_value <= 0:
            return results

        # Calculate sector exposures
        sector_exposure: Dict[str, float] = {}
        sector_holdings: Dict[str, List[str]] = {}

        for symbol, quantity in holdings.items():
            if quantity == 0:
                continue

            price = prices.get(symbol, 0)
            if price <= 0:
                continue

            sector = self._symbol_sectors.get(symbol, "Unknown")
            position_value = abs(quantity * price)
            position_pct = position_value / portfolio_value

            if sector not in sector_exposure:
                sector_exposure[sector] = 0
                sector_holdings[sector] = []

            sector_exposure[sector] += position_pct
            sector_holdings[sector].append(symbol)

        # Check sector limits
        for sector, exposure in sector_exposure.items():
            if exposure > self._max_sector_exposure:
                # Need to reduce this sector
                scale_factor = self._max_sector_exposure / exposure

                for symbol in sector_holdings[sector]:
                    quantity = holdings.get(symbol, 0)
                    if quantity != 0:
                        results.append(RiskManagementResult(
                            symbol=symbol,
                            should_liquidate=False,
                            target_quantity=quantity * scale_factor,
                            reason=f"Sector {sector} exposure: {exposure*100:.1f}% > {self._max_sector_exposure*100:.0f}%",
                            risk_level=min(exposure / self._max_sector_exposure, 1.0),
                        ))

        return results


class CompositeRiskModel(RiskManagementModel):
    """
    Composite Risk Management Model.

    Combines multiple risk models. Most restrictive action wins.
    """

    def __init__(self, models: List[RiskManagementModel],
                 name: Optional[str] = None):
        name = name or "CompositeRisk"
        super().__init__(name)
        self._models = models

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Run all risk models and combine results."""
        all_results: Dict[str, RiskManagementResult] = {}

        for model in self._models:
            results = model.manage_risk(holdings, prices, portfolio_value)

            for result in results:
                symbol = result.symbol

                if symbol not in all_results:
                    all_results[symbol] = result
                else:
                    existing = all_results[symbol]

                    # Liquidation takes priority
                    if result.should_liquidate:
                        all_results[symbol] = result
                    # Otherwise, most restrictive target quantity
                    elif result.target_quantity is not None:
                        if existing.target_quantity is None:
                            all_results[symbol] = result
                        elif abs(result.target_quantity) < abs(existing.target_quantity):
                            all_results[symbol] = result

        return list(all_results.values())

    def on_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Forward data to all models."""
        for model in self._models:
            model.on_data(symbol, price, timestamp)


class NullRiskModel(RiskManagementModel):
    """
    Null Risk Management Model.

    No risk management. Use when risk is managed externally.
    """

    def __init__(self):
        super().__init__("Null")

    def manage_risk(self, holdings: Dict[str, float],
                   prices: Dict[str, float],
                   portfolio_value: float) -> List[RiskManagementResult]:
        """Return no risk actions."""
        return []
