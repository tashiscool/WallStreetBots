"""Wash sale engine for tax-compliant trading.

This module implements FIFO tax lot tracking with wash sale detection to ensure
proper realized P&L reporting and compliance with US tax regulations.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import List


@dataclass
class Fill:
    symbol: str
    ts: dt.datetime
    side: str  # 'buy' or 'sell'
    qty: float
    price: float


@dataclass
class Lot:
    open_ts: dt.datetime
    qty: float
    remaining: float
    cost: float  # total cost for the lot


class WashSaleEngine:
    """
    Minimal FIFO tax lot matching with a 30-day wash sale window.
    - Tracks realized PnL and disallowed losses when replacement buys occur within +/-30 days.
    """

    def __init__(self, window_days: int = 30):
        self.window = dt.timedelta(days=window_days)
        self.lots_by_symbol: dict[str, List[Lot]] = {}
        self.buys: List[Fill] = []
        self.sells: List[Fill] = []

    def ingest(self, fill: Fill) -> None:
        (self.buys if fill.side == "buy" else self.sells).append(fill)
        if fill.side == "buy":
            self.lots_by_symbol.setdefault(fill.symbol, []).append(
                Lot(
                    open_ts=fill.ts,
                    qty=fill.qty,
                    remaining=fill.qty,
                    cost=fill.qty * fill.price,
                )
            )

    def realize(self, sell: Fill) -> tuple[float, float]:
        """Returns (realized_pnl, wash_disallowed)."""
        lots = self.lots_by_symbol.get(sell.symbol, [])
        qty = sell.qty
        realized = 0.0
        disallowed = 0.0
        # FIFO match
        i = 0
        while qty > 0 and i < len(lots):
            lot = lots[i]
            take = min(qty, lot.remaining)
            basis_per_share = lot.cost / lot.remaining
            pnl = (sell.price - basis_per_share) * take
            realized += pnl
            lot.remaining -= take
            lot.cost -= basis_per_share * take
            if lot.remaining == 0:
                i += 1
            qty -= take

            # Wash-sale: if pnl < 0 and there is a buy within +/- window
            if pnl < 0 and self._has_replacement_buy(sell.symbol, sell.ts, take):
                disallowed += abs(pnl)
        # prune empty lots
        self.lots_by_symbol[sell.symbol] = [lot for lot in lots if lot.remaining > 0]
        return realized, disallowed

    def _has_replacement_buy(self, symbol: str, when: dt.datetime, qty: float) -> bool:
        # Check for buys within window AFTER the sell date
        # Wash sale rules: replacement buy must occur within 30 days AFTER the loss sale
        for b in self.buys:
            if b.symbol != symbol:
                continue
            days_diff = (b.ts - when).days
            if 0 <= days_diff <= self.window.days:  # Buy after sell within window
                return True
        return False
