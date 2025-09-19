"""Capital Efficiency / Kelly Analysis with Stable + Safe Implementation."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """Kelly sizing analysis results."""
    kelly_fraction: float
    conservative_kelly: float
    recommended_position_size: float
    win_rate: float
    win_loss_ratio: float
    expected_return: float


def _max_dd(r: pd.Series) -> float:
    """Calculate maximum drawdown safely."""
    eq = (1 + r.dropna()).cumprod()
    dd = eq / eq.cummax() - 1
    return float(dd.min() if len(dd) else 0.0)


def _sharpe(r: pd.Series) -> float:
    """Calculate Sharpe ratio safely."""
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(252) * r.mean() / r.std(ddof=1))


class CapitalEfficiencyAnalyzer:
    """Analyzes capital efficiency and optimal leverage."""
    
    def __init__(self, daily_margin_rate: float = 0.00008):  # ~2%/yr
        self.daily_margin_rate = daily_margin_rate

    def analyze_leverage_efficiency(self, strategy, capital_levels: List[float]) -> Dict[str, Any]:
        """Analyze leverage efficiency across different capital levels."""
        results: Dict[Tuple[int, float], Dict] = {}
        
        for cap in capital_levels:
            for lev in [1.0, 1.5, 2.0]:
                try:
                    bt = strategy.backtest_with_capital(cap * lev)
                    r = bt.returns.copy()
                    margin_cost = (lev - 1.0) * self.daily_margin_rate
                    r_adj = r - margin_cost
                    
                    results[(cap, lev)] = {
                        'sharpe_ratio': _sharpe(r_adj),
                        'return_on_actual_capital': float((1 + r_adj).prod() - 1),
                        'max_drawdown': _max_dd(r_adj),
                        'margin_calls': getattr(bt, 'margin_calls', 0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to analyze leverage {lev}x for capital {cap}: {e}")
                    continue

        optimal = {}
        for cap in capital_levels:
            candidates = {lev: v for (c, lev), v in results.items() if c == cap}
            if candidates:
                best = max(candidates, key=lambda k: candidates[k]['return_on_actual_capital'])
                optimal[cap] = {
                    'optimal_leverage': best, 
                    'expected_return_on_capital': candidates[best]['return_on_actual_capital']
                }

        return {'detailed_results': results, 'optimal_setups': optimal}

    def kelly_sizing_analysis(self, trade_returns: pd.Series, cap_at: float = 0.25) -> KellyResult:
        """Perform Kelly sizing analysis with stability caps."""
        r = trade_returns.dropna()
        if r.empty: 
            return KellyResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        wins = r[r > 0]
        losses = r[r <= 0]
        win_rate = float((r > 0).mean())
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
        
        if win_rate == 0.0:
            # All losses
            expected_return = -avg_loss
            return KellyResult(0.0, 0.0, 0.0, win_rate, 0.0, expected_return)

        if avg_loss == 0 and win_rate == 1.0:
            # All wins - use conservative approach since no loss data
            expected_return = avg_win
            return KellyResult(1.0, 0.5, min(0.5, cap_at), win_rate, float('inf'), expected_return)

        if avg_loss == 0:
            # Mixed wins and zeros - treat as conservative
            expected_return = win_rate * avg_win
            return KellyResult(0.0, 0.0, 0.0, win_rate, 0.0, expected_return)
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        # Kelly formula: f = (bp - q) / b
        k = (b * p - q) / max(b, 1e-9)
        k = max(min(k, 1.0), 0.0)  # clamp to [0,1]
        
        # Conservative Kelly (half Kelly)
        ck = 0.5 * k
        
        # Expected return
        expected_return = p * avg_win - (1 - p) * avg_loss
        
        return KellyResult(
            kelly_fraction=k,
            conservative_kelly=ck,
            recommended_position_size=float(min(ck, cap_at)),
            win_rate=win_rate,
            win_loss_ratio=b,
            expected_return=expected_return
        )

    def analyze_capital_allocation(self, strategies: Dict[str, Any], 
                                 total_capital: float) -> Dict[str, Any]:
        """Analyze optimal capital allocation across strategies."""
        allocations = {}
        
        for name, strategy_data in strategies.items():
            if 'trade_returns' in strategy_data:
                kelly_result = self.kelly_sizing_analysis(strategy_data['trade_returns'])
                
                # Calculate suggested allocation
                suggested_allocation = kelly_result.recommended_position_size * total_capital
                
                allocations[name] = {
                    'kelly_result': kelly_result,
                    'suggested_allocation': suggested_allocation,
                    'allocation_percentage': suggested_allocation / max(total_capital, 1e-9),
                    'risk_adjusted_return': kelly_result.expected_return / max(kelly_result.recommended_position_size, 1e-9)
                }
        
        # Normalize allocations if they exceed total capital
        total_suggested = sum(a['suggested_allocation'] for a in allocations.values())
        if total_suggested > total_capital:
            scale_factor = total_capital / total_suggested
            for allocation in allocations.values():
                allocation['suggested_allocation'] *= scale_factor
                allocation['allocation_percentage'] *= scale_factor
        
        total_allocated = sum(a['suggested_allocation'] for a in allocations.values())
        return {
            'strategy_allocations': allocations,
            'total_capital': total_capital,
            'total_allocated': total_allocated,
            'capital_utilization': total_allocated / max(total_capital, 1e-9) if total_capital > 0 else 0.0
        }