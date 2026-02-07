"""Almgren-Chriss (2000) Optimal Execution with Market Impact.

Computes an optimal liquidation trajectory that balances urgency
(timing risk from price volatility) against market impact costs.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..execution_model import ExecutionModel, Order, OrderType, OrderSide, TimeInForce
from ..portfolio_target import PortfolioTarget


class LiquidityBucket(Enum):
    """Asset liquidity classification for impact parameter selection."""
    MEGA_CAP = "mega_cap"       # AAPL, MSFT, top-10 by ADV
    LARGE_CAP = "large_cap"     # S&P 500 typical
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    MICRO_CAP = "micro_cap"
    CRYPTO_MAJOR = "crypto_major"  # BTC, ETH
    CRYPTO_ALT = "crypto_alt"


# Empirically calibrated impact parameters per liquidity bucket
# gamma = permanent impact, eta = temporary impact
IMPACT_PARAMS: Dict[LiquidityBucket, Dict[str, float]] = {
    LiquidityBucket.MEGA_CAP:     {'permanent_impact': 0.01,  'temporary_impact': 0.001},
    LiquidityBucket.LARGE_CAP:    {'permanent_impact': 0.05,  'temporary_impact': 0.005},
    LiquidityBucket.MID_CAP:      {'permanent_impact': 0.10,  'temporary_impact': 0.01},
    LiquidityBucket.SMALL_CAP:    {'permanent_impact': 0.30,  'temporary_impact': 0.03},
    LiquidityBucket.MICRO_CAP:    {'permanent_impact': 0.80,  'temporary_impact': 0.10},
    LiquidityBucket.CRYPTO_MAJOR: {'permanent_impact': 0.05,  'temporary_impact': 0.005},
    LiquidityBucket.CRYPTO_ALT:   {'permanent_impact': 0.50,  'temporary_impact': 0.05},
}


def get_impact_params(bucket: LiquidityBucket) -> Dict[str, float]:
    """Return calibrated (gamma, eta) for a liquidity bucket."""
    return IMPACT_PARAMS[bucket].copy()


@dataclass
class AlmgrenChrissConfig:
    """Configuration for Almgren-Chriss optimal execution."""
    total_shares: Decimal
    total_time: float           # execution horizon in minutes
    num_slices: int = 10
    volatility: float = 0.02   # daily volatility (sigma)
    daily_volume: float = 1e6  # average daily volume (ADV)
    permanent_impact: float = 0.1    # gamma — permanent impact coefficient
    temporary_impact: float = 0.01   # eta — temporary impact coefficient
    risk_aversion: float = 1e-6      # lambda — urgency parameter

    def validate(self) -> List[str]:
        """Validate config for unit consistency. Returns list of warnings."""
        warnings = []
        # Volatility sanity: daily vol should be 0.001–0.20 for equities
        if self.volatility > 1.0:
            warnings.append(
                f"volatility={self.volatility} looks like percentage, not decimal "
                f"(expected daily vol ~0.01–0.05 for equities)"
            )
        if self.volatility < 0:
            warnings.append("volatility must be non-negative")
        # Time horizon sanity
        if self.total_time > 390:
            warnings.append(
                f"total_time={self.total_time} minutes exceeds full trading day (390 min)"
            )
        # Participation rate check
        if self.daily_volume > 0:
            shares_per_min = float(self.total_shares) / max(self.total_time, 1)
            vol_per_min = self.daily_volume / 390.0
            participation = shares_per_min / vol_per_min if vol_per_min > 0 else 0
            if participation > 0.25:
                warnings.append(
                    f"Participation rate {participation:.1%} exceeds 25% of volume — "
                    f"market impact model may underestimate costs"
                )
        return warnings


class AlmgrenChrissModel:
    """Almgren-Chriss optimal trajectory calculator.

    Given risk aversion λ, volatility σ, and impact parameters (γ, η),
    the optimal trajectory minimises:

        E[cost] + λ · Var[cost]

    yielding an analytical closed-form solution involving hyperbolic
    functions.
    """

    def compute_optimal_trajectory(self, config: AlmgrenChrissConfig) -> List[Decimal]:
        """Return the number of shares to trade in each slice.

        The trajectory is a list of length *num_slices* whose values sum
        to *config.total_shares*.
        """
        n = config.num_slices
        if n <= 0:
            return []
        if n == 1:
            return [config.total_shares]

        total = float(config.total_shares)
        tau = config.total_time / n  # time per slice in minutes

        kappa = self._kappa(config)

        # Optimal holdings at each time step k = 0 … n
        # x_k = total * sinh(kappa * (n - k) * tau) / sinh(kappa * n * tau)
        denom = math.sinh(kappa * n * tau)
        if abs(denom) < 1e-15:
            # Degenerate — equal slices (TWAP)
            qty = total / n
            return [Decimal(str(round(qty, 6))) for _ in range(n)]

        holdings = []
        for k in range(n + 1):
            x_k = total * math.sinh(kappa * (n - k) * tau) / denom
            holdings.append(x_k)

        # Trade list: n_k = x_{k-1} - x_k
        trades = []
        for k in range(1, n + 1):
            trade = holdings[k - 1] - holdings[k]
            trades.append(Decimal(str(round(trade, 6))))

        # Ensure trades sum exactly to total_shares via residual adjustment
        residual = config.total_shares - sum(trades)
        trades[-1] += residual

        return trades

    def estimate_execution_cost(self, config: AlmgrenChrissConfig) -> Dict[str, float]:
        """Estimate execution cost components.

        Returns
        -------
        dict with keys:
            expected_cost : total expected cost in price units
            variance : variance of execution cost
            is_cost : implementation shortfall (expected + risk penalty)
            timing_risk : sqrt(variance), a.k.a. timing risk
        """
        total = float(config.total_shares)
        n = config.num_slices
        tau = config.total_time / n if n > 0 else config.total_time

        sigma = config.volatility
        gamma = config.permanent_impact
        eta = config.temporary_impact
        lam = config.risk_aversion

        kappa = self._kappa(config)

        # Expected cost (permanent + temporary)
        permanent_cost = 0.5 * gamma * total ** 2
        # Temporary impact cost depends on trajectory
        trajectory = self.compute_optimal_trajectory(config)
        temp_cost = 0.0
        for trade_dec in trajectory:
            trade = float(trade_dec)
            rate = trade / tau if tau > 0 else 0.0
            temp_cost += eta * rate * trade  # eta * (n_k/tau) * n_k

        expected_cost = permanent_cost + temp_cost

        # Variance of cost — proportional to sigma^2 * sum of x_k^2 * tau
        denom = math.sinh(kappa * n * tau)
        variance = 0.0
        if abs(denom) > 1e-15:
            for k in range(n + 1):
                x_k = total * math.sinh(kappa * (n - k) * tau) / denom
                variance += sigma ** 2 * x_k ** 2 * tau

        timing_risk = math.sqrt(max(variance, 0.0))
        is_cost = expected_cost + lam * variance

        return {
            'expected_cost': expected_cost,
            'variance': variance,
            'is_cost': is_cost,
            'timing_risk': timing_risk,
        }

    @staticmethod
    def _kappa(config: AlmgrenChrissConfig) -> float:
        r"""Compute κ = sqrt(λσ² / η).

        κ controls the shape of the optimal trajectory:
        - Large κ → front-loaded (eager) execution
        - Small κ → uniform (TWAP-like) execution
        """
        lam = config.risk_aversion
        sigma = config.volatility
        eta = config.temporary_impact
        if eta <= 0:
            return 0.0
        return math.sqrt(lam * sigma ** 2 / eta)


class AlmgrenChrissExecutionModel(ExecutionModel):
    """Execution model that uses Almgren-Chriss optimal trajectories.

    Slices orders according to the analytically optimal schedule that
    minimises expected cost + risk-aversion-weighted variance.
    """

    def __init__(
        self,
        duration_minutes: float = 30.0,
        num_slices: int = 10,
        volatility: float = 0.02,
        daily_volume: float = 1e6,
        permanent_impact: float = 0.1,
        temporary_impact: float = 0.01,
        risk_aversion: float = 1e-6,
        name: str = "AlmgrenChrissExecution",
    ):
        super().__init__(name)
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
        self.volatility = volatility
        self.daily_volume = daily_volume
        self.permanent_impact = permanent_impact
        self.temporary_impact = temporary_impact
        self.risk_aversion = risk_aversion
        self._current_positions: Dict[str, Decimal] = {}
        self._model = AlmgrenChrissModel()

    def set_current_positions(self, positions: Dict[str, Decimal]) -> None:
        """Update current positions."""
        self._current_positions = positions

    def execute(
        self,
        targets: List[PortfolioTarget],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> List[Order]:
        """Generate optimally-sliced orders using Almgren-Chriss."""
        orders = []

        for target in targets:
            current_qty = self._current_positions.get(target.symbol, Decimal("0"))
            total_qty = self.calculate_order_quantity(target, current_qty)

            if total_qty == 0:
                continue

            side = self.calculate_order_side(target, current_qty)

            # Override volume/volatility from market data if available
            vol = self.volatility
            adv = self.daily_volume
            if market_data and target.symbol in market_data:
                sym_data = market_data[target.symbol]
                vol = sym_data.get('volatility', vol)
                adv = sym_data.get('daily_volume', adv)

            config = AlmgrenChrissConfig(
                total_shares=total_qty,
                total_time=self.duration_minutes,
                num_slices=self.num_slices,
                volatility=vol,
                daily_volume=adv,
                permanent_impact=self.permanent_impact,
                temporary_impact=self.temporary_impact,
                risk_aversion=self.risk_aversion,
            )

            trajectory = self._model.compute_optimal_trajectory(config)

            for i, slice_qty in enumerate(trajectory):
                if slice_qty <= 0:
                    continue

                order = Order(
                    symbol=target.symbol,
                    side=side,
                    quantity=slice_qty.quantize(Decimal("1")),
                    order_type=OrderType.MARKET,
                    source_target_id=target.id,
                    metadata={
                        'slice_index': i,
                        'total_slices': self.num_slices,
                        'algorithm': 'AlmgrenChriss',
                    },
                )
                orders.append(order)

        return orders
