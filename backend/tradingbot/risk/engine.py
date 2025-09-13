# backend/tradingbot/risk/engine.py
from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

log = logging.getLogger("wsb.risk")


@dataclass
class RiskLimits:
    max_total_risk: float  # e.g. 0.30
    max_position_size: float  # e.g. 0.10
    max_drawdown: float = 0.30
    kill_switch_dd: float = 0.35


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._peak_equity = None
        self._kill_switch_active = False

    def update_peak(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity or equity, equity)

    def drawdown(self, equity: float) -> float:
        if not self._peak_equity:
            self._peak_equity = equity
        return 1.0 - (equity / self._peak_equity if self._peak_equity > 0 else 1.0)

    def var_cvar(
        self, pnl_samples: Sequence[float], alpha: float = 0.95
    ) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        arr = np.asarray(pnl_samples, dtype=float)
        if arr.size == 0:
            return {"var": 0.0, "cvar": 0.0}
        q = np.quantile(arr, 1 - alpha)
        cvar = arr[arr <= q].mean() if np.any(arr <= q) else q
        return {"var": float(-q), "cvar": float(-cvar)}

    def pretrade_check(self, current_exposure: float, new_position_risk: float) -> bool:
        """Pre-trade risk check - returns True if trade is allowed"""
        if self._kill_switch_active:
            log.warning("Kill switch active - blocking all trades")
            return False

        if current_exposure + new_position_risk > self.limits.max_total_risk:
            log.warning(
                f"Total risk limit exceeded: {current_exposure + new_position_risk:.2%} > {self.limits.max_total_risk:.2%}"
            )
            return False

        if new_position_risk > self.limits.max_position_size:
            log.warning(
                f"Position size limit exceeded: {new_position_risk:.2%} > {self.limits.max_position_size:.2%}"
            )
            return False

        return True

    def posttrade_check(self, equity: float) -> bool:
        """Post-trade risk check - returns True if system should continue"""
        dd = self.drawdown(equity)

        if dd >= self.limits.kill_switch_dd:
            log.critical(
                f"Kill switch triggered! Drawdown {dd:.2%} >= {self.limits.kill_switch_dd:.2%}"
            )
            self._kill_switch_active = True
            return False

        if dd >= self.limits.max_drawdown:
            log.error(
                f"Maximum drawdown exceeded: {dd:.2%} >= {self.limits.max_drawdown:.2%}"
            )
            return False

        return True

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def reset_kill_switch(self) -> None:
        """Manual kill switch reset - use with extreme caution"""
        log.warning("Kill switch manually reset")
        self._kill_switch_active = False
