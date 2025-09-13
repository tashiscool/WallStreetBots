"""Options assignment risk management.

This module provides tools to assess and manage assignment risks for options positions,
including auto-exercise detection, early assignment risk around ex-dividend dates,
and pin risk near expiration.
"""
from __future__ import annotations
from dataclasses import dataclass
import datetime as dt
import math

@dataclass
class OptionContract:
    symbol: str   # e.g. "AAPL 2025-10-17 180 C"
    underlying: str
    strike: float
    right: str    # 'C' or 'P'
    expiry: dt.date

@dataclass
class UnderlyingState:
    price: float
    borrow_bps: float = 0.0
    next_ex_div_date: dt.date | None = None
    div_amount: float = 0.0

def auto_exercise_likely(oc: OptionContract, u: UnderlyingState, threshold: float = 0.01) -> bool:
    """Simple OCC auto-exercise heuristic: >= $0.01 ITM at expiration."""
    if dt.date.today() != oc.expiry: return False
    intrinsic = (u.price - oc.strike) if oc.right == "C" else (oc.strike - u.price)
    return intrinsic >= threshold

def early_assignment_risk(oc: OptionContract, u: UnderlyingState, r_annual=0.03) -> bool:
    """
    Heuristic: Short calls are at risk the day before ex-dividend if:
      (1) deep ITM; (2) dividend > remaining extrinsic value.
    Here we approximate extrinsic ~ max(0, mid - intrinsic). If unknown, use threshold on moneyness.
    """
    today = dt.date.today()
    if oc.right != "C": return False
    if not u.next_ex_div_date or (u.next_ex_div_date - today).days not in (0,1):
        return False
    # Without full IV data, use a blunt instrument: deep ITM by >2% and positive dividend
    moneyness = u.price / oc.strike
    return (moneyness > 1.02) and (u.div_amount > 0.0)

def pin_risk(oc: OptionContract, u: UnderlyingState, band_bps: float = 10.0) -> bool:
    """If spot is within +/- band_bps of strike at expiry close, you're exposed to random assignment."""
    if dt.date.today() != oc.expiry: return False
    band = oc.strike * (band_bps / 10_000.0)
    return abs(u.price - oc.strike) <= band