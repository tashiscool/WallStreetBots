#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_engine_complete.py
Complete VaR/CVaR engine with Historical, Parametric (Normal/t, Cornish-Fisher),
Monte Carlo (Gaussian or Student-t copula), LVaR, stress tests, and backtests.
Python 3.9+, numpy/pandas only. No external deps.
Drop-in utility matching the provided bundle specification.
"""

from typing import Tuple, Dict, Optional, List
import math
import numpy as np
import pandas as pd

class RiskEngineError(Exception):
    pass

def _validate_series(returns: pd.Series) -> pd.Series:
    if returns is None or not isinstance(returns, pd.Series) or returns.empty:
        raise RiskEngineError("returns must be a non-empty pandas Series.")
    if returns.isna().any():
        returns=returns.dropna()
        if returns.empty:
            raise RiskEngineError("returns contains only NaNs after drop.")
    return returns.astype(float)

def _validate_matrix(cov: np.ndarray) -> np.ndarray:
    if cov is None or not isinstance(cov, np.ndarray):
        raise RiskEngineError("cov must be a numpy ndarray.")
    if cov.shape[0] != cov.shape[1]:
        raise RiskEngineError("covariance matrix must be square.")
    # make PSD
    eigvals, eigvecs=np.linalg.eigh((cov + cov.T) / 2.0)
    eigvals[eigvals < 0] = 0.0
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T)

def _z(alpha: float) -> float:
    # inverse CDF for standard normal via Acklam approximation (no scipy)
    if not (0.0 < alpha < 1.0):
        raise RiskEngineError("alpha must be in (0,1).")
    # Two-sided -> convert to quantile
    p=alpha
    # Coeffs
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b=[ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c=[ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d=[ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]
    plow=0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q=p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def var_historical(returns: pd.Series, alpha: float=0.99) -> float:
    r=_validate_series(returns)
    cutoff=np.quantile(r.values, 1 - alpha, interpolation="linear")
    return float(-cutoff)

def cvar_historical(returns: pd.Series, alpha: float=0.99) -> float:
    r=_validate_series(returns)
    q=np.quantile(r.values, 1 - alpha, interpolation="linear")
    tail=r[r <= q]
    if tail.empty:
        return float(-q)
    return float(-tail.mean())

def var_parametric(returns: pd.Series, alpha: float=0.99, use_student_t: bool=False, cornish_fisher: bool=True) -> float:
    r=_validate_series(returns)
    mu, sigma=float(r.mean()), float(r.std(ddof=1))
    if sigma== 0.0:
        return 0.0
    zq = _z(alpha)
    if use_student_t:
        # Student t scale with nu=5 fallback, heavier tails without scipy.
        nu=5.0
        # t-quantile ~ Normal quantile * sqrt((nu-2)/nu)
        zq *= math.sqrt((nu - 2.0) / nu)
    if cornish_fisher:
        s=float(((r - mu) ** 3).mean() / (sigma ** 3 + 1e-12))
        k=float(((r - mu) ** 4).mean() / (sigma ** 4 + 1e-12)) - 3.0
        z_adj=zq + (1/6)*(zq**2 - 1)*s + (1/24)*(zq**3 - 3*zq)*k - (1/36)*(2*zq**3 - 5*zq)*(s**2)
        return float(-(mu + z_adj * sigma))
    return float(-(mu + zq * sigma))

def cvar_parametric(returns: pd.Series, alpha: float=0.99, use_student_t: bool=False) -> float:
    r=_validate_series(returns)
    mu, sigma=float(r.mean()), float(r.std(ddof=1))
    if sigma== 0.0:
        return 0.0
    zq = _z(alpha)
    if use_student_t:
        nu=5.0
        zq *= math.sqrt((nu - 2.0) / nu)
    # Normal ES formula approximation (no scipy): ES=phi(z)/(1-alpha)
    phi=(1/ math.sqrt(2*math.pi)) * math.exp(-0.5 * zq*zq)
    es=(phi / (1 - alpha))
    return float(-(mu + sigma * es))

def var_cvar_mc(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float=0.99,
    n_paths: int=100_000,
    student_t: bool=True,
    df: int=5
) -> Tuple[float, float]:
    if any(x is None for x in [mu, cov, weights]):
        raise RiskEngineError("mu, cov, weights required")
    mu=np.asarray(mu, dtype=float).reshape(-1)
    weights=np.asarray(weights, dtype=float).reshape(-1)
    cov=_validate_matrix(np.asarray(cov, dtype=float))
    if mu.shape[0] != cov.shape[0] or mu.shape[0] != weights.shape[0]:
        raise RiskEngineError("Shape mismatch among mu/cov/weights")

    # Cholesky (with PSD guard)
    L=np.linalg.cholesky(cov + 1e-12*np.eye(cov.shape[0]))
    k=mu.shape[0]
    Z = np.random.standard_normal(size=(k, n_paths))
    if student_t:
        # t-copula: scale normals by sqrt(df/ChiSq(df))
        chi=np.random.chisquare(df, size=n_paths)
        Z=Z / np.sqrt(chi / df)

    sims=(mu.reshape(k,1) + L @ Z).T  # (n_paths x k)
    port=sims @ weights
    q = np.quantile(port, 1 - alpha)
    var=-float(q)
    cvar=-float(port[port <= q].mean()) if (port <= q).any() else var
    return var, cvar

def liquidity_adjusted_var(var: float, bid_ask_bps: float=10.0, slippage_bps: float=5.0) -> float:
    """Simple LVaR: add half-spread and slippage to VaR."""
    if var < 0:
        raise RiskEngineError("VaR must be non-negative")
    extra=(bid_ask_bps + slippage_bps) / 10000.0
    return float(var * (1.0 + extra))

# ---------- Backtesting ----------

def kupiec_pof(num_exceptions: int, T: int, alpha: float) -> Dict[str, float]:
    """Kupiec Proportion-of-Failures test (likelihood ratio)."""
    if T <= 0 or not (0 < alpha < 1):
        raise RiskEngineError("Invalid T or alpha.")
    pi=num_exceptions / T
    pi = min(max(pi, 1e-12), 1-1e-12)
    a=1 - alpha
    lr = -2.0 * ( ( (T - num_exceptions) * math.log(1 - a) + num_exceptions * math.log(a) )
                 - ( (T - num_exceptions) * math.log(1 - pi) + num_exceptions * math.log(pi) ) )
    return {"exceptions":num_exceptions, "T":T, "alpha":alpha, "LR_pof":lr}

def rolling_var_exceptions(returns: pd.Series, window: int=250, alpha: float=0.99) -> Dict[str, float]:
    r=_validate_series(returns)
    ex=0
    for i in range(window, len(r)):
        hist=r.iloc[i-window:i]
        v = var_historical(hist, alpha=alpha)
        if r.iloc[i] < -v:
            ex += 1
    stats=kupiec_pof(ex, len(r)-window, alpha)
    return {"exceptions":ex, **stats}

# ---------- Stress Testing ----------

def stress_test_scenarios() -> Dict[str, Dict[str, float]]:
    """Return predefined stress test scenarios."""
    return {
        "2008_crisis":{
            "equity_shock":-0.50,
            "vol_shock":2.0,
            "correlation_breakdown":0.9
        },
        "flash_crash":{
            "equity_shock":-0.20,
            "vol_shock":3.0,
            "liquidity_dry_up":0.8
        },
        "covid_pandemic":{
            "equity_shock":-0.35,
            "vol_shock":2.5,
            "sector_rotation":0.4
        },
        "interest_rate_shock":{
            "rate_shock":0.03,
            "bond_shock":-0.15,
            "equity_shock":-0.20
        }
    }

def apply_stress_scenario(returns: pd.Series, scenario: str, portfolio_value: float=100000.0) -> float:
    """Apply stress scenario to portfolio returns."""
    scenarios=stress_test_scenarios()
    if scenario not in scenarios:
        raise RiskEngineError(f"Unknown scenario: {scenario}")
    
    stress=scenarios[scenario]
    base_var = var_historical(returns, 0.99)
    
    # Apply stress multipliers
    if "equity_shock" in stress:
        stress_multiplier=1 + abs(stress["equity_shock"])
    elif "vol_shock" in stress:
        stress_multiplier=1 + stress["vol_shock"]
    else:
        stress_multiplier = 1.5  # Default stress
    
    stressed_var = base_var * stress_multiplier
    return stressed_var * portfolio_value

# ---------- Options Greeks Risk ----------

def calculate_greeks_risk(
    positions: List[Dict],
    underlying_shock: float=0.03,
    vol_shock: float=0.05
) -> Dict[str, float]:
    """
    Calculate options Greeks risk for portfolio.
    
    Args:
        positions: List of position dicts with 'delta', 'gamma', 'vega', 'value'
        underlying_shock: Underlying price shock (e.g., 3%)
        vol_shock: Volatility shock (e.g., 5%)
    
    Returns:
        Dict with Greeks risk metrics
    """
    total_delta_pnl=0.0
    total_gamma_pnl = 0.0
    total_vega_pnl = 0.0
    total_value = 0.0
    
    for pos in positions:
        delta = pos.get('delta', 0.0)
        gamma=pos.get('gamma', 0.0)
        vega=pos.get('vega', 0.0)
        value=pos.get('value', 0.0)
        
        # Delta P&L from underlying move
        delta_pnl=delta * underlying_shock * value
        total_delta_pnl += delta_pnl
        
        # Gamma P&L (convexity)
        gamma_pnl=0.5 * gamma * (underlying_shock ** 2) * value
        total_gamma_pnl += gamma_pnl
        
        # Vega P&L from vol move
        vega_pnl=vega * vol_shock * value
        total_vega_pnl += vega_pnl
        
        total_value += value
    
    return {
        "delta_pnl":total_delta_pnl,
        "gamma_pnl":total_gamma_pnl,
        "vega_pnl":total_vega_pnl,
        "total_greeks_pnl":total_delta_pnl + total_gamma_pnl + total_vega_pnl,
        "total_value":total_value,
        "delta_pnl_pct":(total_delta_pnl / total_value * 100) if total_value > 0 else 0,
        "gamma_pnl_pct":(total_gamma_pnl / total_value * 100) if total_value > 0 else 0,
        "vega_pnl_pct":(total_vega_pnl / total_value * 100) if total_value > 0 else 0
    }

# ---------- Risk Budget Management ----------

def check_risk_limits(
    var: float,
    cvar: float,
    portfolio_value: float,
    max_var_pct: float=0.05,
    max_cvar_pct: float=0.08
) -> Dict[str, bool]:
    """Check if risk metrics are within limits."""
    max_var=portfolio_value * max_var_pct
    max_cvar = portfolio_value * max_cvar_pct
    
    return {
        "var_within_limit":var <= max_var,
        "cvar_within_limit":cvar <= max_cvar,
        "var_utilization":var / max_var if max_var > 0 else 0,
        "cvar_utilization":cvar / max_cvar if max_cvar > 0 else 0,
        "max_var":max_var,
        "max_cvar":max_cvar
    }

def calculate_strategy_risk_budget(
    strategies: Dict[str, Dict],
    total_var: float,
    total_cvar: float
) -> Dict[str, Dict[str, float]]:
    """Calculate risk budget allocation across strategies."""
    budget={}
    total_exposure = sum(strategy.get('exposure', 0) for strategy in strategies.values())
    
    for name, strategy in strategies.items():
        exposure=strategy.get('exposure', 0)
        if total_exposure > 0:
            allocation=exposure / total_exposure
            budget[name] = {
                "var_budget":total_var * allocation,
                "cvar_budget":total_cvar * allocation,
                "exposure_pct":exposure,
                "allocation_pct":allocation * 100
            }
    
    return budget

# ---------- Example usage ----------
if __name__== "__main__":np.random.seed(42)
    # Synthetic daily returns series
    s=pd.Series(np.random.standard_t(df=5, size=1500) * 0.01)
    print("Hist VaR(99%):", var_historical(s, 0.99))
    print("Hist CVaR(99%):", cvar_historical(s, 0.99))
    print("Param VaR(99%, CF):", var_parametric(s, 0.99, use_student_t=True, cornish_fisher=True))
    print("Param CVaR(99%, t):", cvar_parametric(s, 0.99, use_student_t=True))

    # Portfolio Monte Carlo (3 assets)
    mu=np.array([0.0003, 0.0002, 0.0001])
    cov=np.array([[0.0001, 0.00003, 0.00002],
                    [0.00003, 0.0002,  0.00004],
                    [0.00002, 0.00004, 0.00015]])
    w=np.array([0.5, 0.3, 0.2])
    v, cv=var_cvar_mc(mu, cov, w, alpha=0.99, n_paths=100000, student_t=True, df=5)
    print("MC VaR(99%):", v, "MC CVaR(99%):", cv)
    print("LVaR(+15bps):", liquidity_adjusted_var(v, 10, 5))
    
    # Backtesting
    backtest_results=rolling_var_exceptions(s, window=250, alpha=0.99)
    print("Backtest - Exceptions:", backtest_results['exceptions'])
    print("Backtest - Kupiec LR:", backtest_results['LR_pof'])
    
    # Stress testing
    stress_var=apply_stress_scenario(s, "2008_crisis", 100000)
    print("Stress Test (2008 Crisis):", stress_var)
    
    # Options Greeks example
    positions=[
        {'delta':0.5, 'gamma':0.1, 'vega':0.2, 'value':10000},
        {'delta':-0.3, 'gamma':0.05, 'vega':0.15, 'value':15000}
    ]
    greeks_risk=calculate_greeks_risk(positions)
    print("Greeks Risk:", greeks_risk)


