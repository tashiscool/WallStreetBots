"""Tail Risk Hedging via Protective Puts.

Provides a rules-based tail-hedge manager that recommends when and how
much to hedge based on VIX levels, portfolio value, and cost constraints.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import math


@dataclass
class TailHedgeConfig:
    """Configuration for tail-hedge programme."""
    vix_threshold: float = 25.0        # activate when VIX ≥ this
    hedge_ratio: float = 0.05          # 5 % of portfolio value
    put_delta: float = -0.20           # ~ 20-delta puts
    dte_target: int = 30               # ~ 30 DTE
    rebalance_days: int = 7            # rebalance weekly
    max_hedge_cost_pct: float = 0.02   # max 2 % of portfolio / year


class TailHedgeManager:
    """Rules-based tail-risk hedging via protective puts.

    Uses parametric approximations (Black-Scholes-style) rather than
    live option chains so the module has no external data dependency.
    """

    def __init__(self, config: Optional[TailHedgeConfig] = None):
        self.config = config or TailHedgeConfig()
        self._realized_spend: float = 0.0  # cumulative hedge cost this year
        self._spend_reset_year: Optional[int] = None

    def record_hedge_cost(self, cost: float) -> None:
        """Record realized hedge spend and enforce annual cap."""
        now = datetime.now()
        if self._spend_reset_year != now.year:
            self._realized_spend = 0.0
            self._spend_reset_year = now.year
        self._realized_spend += cost

    def get_realized_spend(self) -> float:
        """Return cumulative realized hedge spend for current year."""
        now = datetime.now()
        if self._spend_reset_year != now.year:
            return 0.0
        return self._realized_spend

    def remaining_budget(self, portfolio_value: float) -> float:
        """Return remaining hedge budget for current year."""
        max_annual = portfolio_value * self.config.max_hedge_cost_pct
        return max(0.0, max_annual - self.get_realized_spend())

    def should_roll(self, dte_remaining: int, current_delta: float = -0.20) -> bool:
        """Deterministic roll rule: roll when DTE or delta breach thresholds.

        Parameters
        ----------
        dte_remaining : int
            Days to expiry of current hedge.
        current_delta : float
            Current delta of the put (negative).

        Returns
        -------
        bool : True if hedge should be rolled.
        """
        # Roll if DTE within rebalance window
        if dte_remaining <= self.config.rebalance_days:
            return True
        # Roll if delta has drifted too far from target (>50% deviation)
        target_delta = abs(self.config.put_delta)
        current_abs = abs(current_delta)
        if target_delta > 0 and abs(current_abs - target_delta) / target_delta > 0.5:
            return True
        return False

    def should_hedge(self, vix: float, portfolio_value: float) -> bool:
        """Return True if hedging should be active."""
        if portfolio_value <= 0:
            return False
        return vix >= self.config.vix_threshold

    def calculate_hedge_size(self, portfolio_value: float, vix: float) -> Dict:
        """Calculate hedge notional and estimated cost.

        Returns
        -------
        dict with keys: notional, num_contracts, estimated_cost, cost_pct
        """
        notional = portfolio_value * self.config.hedge_ratio

        # Approximate put premium using simplified BS relationship
        # Premium ~ S * N(-d1) * sigma * sqrt(T)  (at-the-money approximation scaled by delta)
        dte_years = self.config.dte_target / 365.0
        implied_vol = vix / 100.0  # VIX is annualised vol in %
        premium_pct = implied_vol * math.sqrt(dte_years) * abs(self.config.put_delta)
        estimated_cost = notional * premium_pct

        # Cap cost
        max_cost = portfolio_value * self.config.max_hedge_cost_pct * (self.config.dte_target / 365.0)
        if estimated_cost > max_cost and max_cost > 0:
            # Scale notional down to meet cost cap
            scale = max_cost / estimated_cost
            notional *= scale
            estimated_cost = max_cost

        # Options contracts (1 contract = 100 shares)
        # Use approximate price = portfolio_value / num_shares approximation
        # For index/ETF hedges, assume ~$400 reference price
        ref_price = 400.0
        num_contracts = max(1, int(notional / (ref_price * 100)))

        cost_pct = estimated_cost / portfolio_value if portfolio_value > 0 else 0.0

        return {
            'notional': notional,
            'num_contracts': num_contracts,
            'estimated_cost': estimated_cost,
            'cost_pct': cost_pct,
        }

    def evaluate_hedge_effectiveness(
        self,
        hedge_pnl: float,
        portfolio_pnl: float,
    ) -> Dict:
        """Evaluate how effective a hedge was.

        Parameters
        ----------
        hedge_pnl : float
            P&L of the hedge leg (positive = hedge paid off).
        portfolio_pnl : float
            P&L of the portfolio *excluding* the hedge.

        Returns
        -------
        dict with keys: protection_ratio, cost_of_carry, net_benefit
        """
        if abs(portfolio_pnl) < 1e-10:
            protection_ratio = 0.0
        else:
            protection_ratio = -hedge_pnl / portfolio_pnl if portfolio_pnl < 0 else 0.0

        net_benefit = hedge_pnl + portfolio_pnl
        cost_of_carry = -hedge_pnl if hedge_pnl < 0 else 0.0

        return {
            'protection_ratio': protection_ratio,
            'cost_of_carry': cost_of_carry,
            'net_benefit': net_benefit,
        }

    def get_hedge_recommendation(
        self,
        vix: float,
        portfolio_value: float,
        current_hedges: Optional[List[Dict]] = None,
    ) -> Dict:
        """Return an actionable hedge recommendation.

        Parameters
        ----------
        vix : float
            Current VIX level.
        portfolio_value : float
            Current portfolio value.
        current_hedges : list of dict, optional
            Existing hedge positions, each with keys: ``notional``,
            ``dte_remaining``, ``pnl``.

        Returns
        -------
        dict with keys: action (ADD/HOLD/REMOVE/ROLL), details
        """
        current_hedges = current_hedges or []
        has_hedges = len(current_hedges) > 0

        if not self.should_hedge(vix, portfolio_value):
            if has_hedges:
                return {
                    'action': 'REMOVE',
                    'details': f'VIX ({vix:.1f}) below threshold ({self.config.vix_threshold}); close hedges',
                }
            return {
                'action': 'HOLD',
                'details': f'VIX ({vix:.1f}) below threshold ({self.config.vix_threshold}); no action',
            }

        # VIX above threshold — should be hedged
        if not has_hedges:
            size = self.calculate_hedge_size(portfolio_value, vix)
            return {
                'action': 'ADD',
                'details': (
                    f'VIX ({vix:.1f}) above threshold; buy {size["num_contracts"]} put contracts, '
                    f'est. cost ${size["estimated_cost"]:.0f}'
                ),
                'hedge_size': size,
            }

        # Already hedged — check if roll needed
        needs_roll = any(h.get('dte_remaining', 999) <= self.config.rebalance_days for h in current_hedges)
        if needs_roll:
            size = self.calculate_hedge_size(portfolio_value, vix)
            return {
                'action': 'ROLL',
                'details': (
                    f'Existing hedge near expiry; roll to {self.config.dte_target} DTE, '
                    f'{size["num_contracts"]} contracts'
                ),
                'hedge_size': size,
            }

        return {
            'action': 'HOLD',
            'details': 'Existing hedges adequate; maintain position',
        }
