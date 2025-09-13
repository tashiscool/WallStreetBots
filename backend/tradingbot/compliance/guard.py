"""Compliance guard for US equity trading regulations.

This module implements critical compliance checks required for live US equity trading,
including Pattern Day Trading rules, Short Sale Restrictions, trading halts, and
session management.
"""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Literal, List, Dict, Optional

Side = Literal["buy", "sell", "short", "cover"]

class ComplianceError(Exception): ...
class PDTViolation(ComplianceError): ...
class SSRViolation(ComplianceError): ...
class HaltViolation(ComplianceError): ...
class SessionViolation(ComplianceError): ...

@dataclass(frozen=True)
class DayTradeEvent:
    symbol: str
    open_ts: dt.datetime
    close_ts: dt.datetime

class SessionCalendar:
    """NYSE regular + pre/post with simple buffers. Replace with exchange calendar if you have one."""
    def __init__(self, tz=dt.timezone.utc, open_t=(13,30), close_t=(20,0), pre_open=(9,0), post_close=(22,0)):
        self.tz = tz
        self.open_t = open_t
        self.close_t = close_t
        self.pre_open = pre_open
        self.post_close = post_close

    def _tod(self, t: dt.time) -> dt.time: return t

    def session(self, when: dt.datetime) -> Literal["pre","regular","post","closed"]:
        # naive UTC logic; swap with pandas_market_calendars in prod
        t = when.astimezone(self.tz).time()
        o = dt.time(*self.open_t); c = dt.time(*self.close_t)
        pre = dt.time(*self.pre_open); post = dt.time(*self.post_close)
        if pre <= t < o: return "pre"
        if o <= t < c: return "regular"
        if c <= t < post: return "post"
        return "closed"

@dataclass
class SSRState:
    """Track SSR (Rule 201) per symbol for the trading day."""
    symbols_on_ssr: Dict[str, dt.date]

    def is_on_ssr(self, symbol: str, on_date: dt.date) -> bool:
        d = self.symbols_on_ssr.get(symbol.upper())
        return bool(d and d == on_date)

class ComplianceGuard:
    """Blocks orders that would violate broker/exchange constraints."""
    def __init__(
        self,
        min_equity_for_day_trading: float = 25_000.0,
        day_trade_window_days: int = 5,
        session_calendar: Optional[SessionCalendar] = None,
    ):
        self.min_equity_for_day_trading = min_equity_for_day_trading
        self.day_trade_window_days = day_trade_window_days
        self.session_calendar = session_calendar or SessionCalendar()
        self.day_trades: List[DayTradeEvent] = []
        self.halted_symbols: Dict[str, str] = {}  # symbol -> reason
        self.luld_limits: Dict[str, Dict[str, float]] = {}  # symbol -> {"lower":x,"upper":y}
        self.ssr = SSRState(symbols_on_ssr={})

    # ---- Admin feeds (call from your market data / broker hooks) ----
    def set_halt(self, symbol: str, reason: str) -> None:
        self.halted_symbols[symbol.upper()] = reason

    def clear_halt(self, symbol: str) -> None:
        self.halted_symbols.pop(symbol.upper(), None)

    def set_luld(self, symbol: str, lower: float, upper: float) -> None:
        self.luld_limits[symbol.upper()] = {"lower": lower, "upper": upper}

    def set_ssr(self, symbol: str, active_date: dt.date) -> None:
        self.ssr.symbols_on_ssr[symbol.upper()] = active_date

    def record_day_trade(self, symbol: str, opened: dt.datetime, closed: dt.datetime) -> None:
        self.day_trades.append(DayTradeEvent(symbol.upper(), opened, closed))
        # prune old
        cutoff = closed.date() - dt.timedelta(days=self.day_trade_window_days + 2)
        self.day_trades = [e for e in self.day_trades if e.close_ts.date() >= cutoff]

    # ---- Checks (call before placing an order) ----
    def check_pdt(self, account_equity: float, pending_day_trades_count: int, now: dt.datetime) -> None:
        """If equity < $25k and >= 4 day-trades in 5 bdays => violation."""
        if account_equity >= self.min_equity_for_day_trading:
            return
        # naive: use recorded events + pending signals
        if pending_day_trades_count >= 4:
            raise PDTViolation("Pattern Day Trader rule: equity < $25k and >= 4 day trades in 5 days.")

    def check_session(self, now: dt.datetime, allow_pre: bool, allow_post: bool) -> None:
        s = self.session_calendar.session(now)
        if s == "regular": return
        if s == "pre" and allow_pre: return
        if s == "post" and allow_post: return
        raise SessionViolation(f"Orders not allowed in current session: {s}")

    def check_halt(self, symbol: str) -> None:
        reason = self.halted_symbols.get(symbol.upper())
        if reason:
            raise HaltViolation(f"{symbol} halted: {reason}")

    def check_luld(self, symbol: str, limit_price: Optional[float]) -> None:
        if not limit_price: return
        band = self.luld_limits.get(symbol.upper())
        if band:
            if not (band["lower"] <= limit_price <= band["upper"]):
                raise HaltViolation(f"{symbol} limit {limit_price} breaches LULD [{band['lower']}, {band['upper']}]")

    def check_ssr(self, symbol: str, side: Side, now: dt.datetime) -> None:
        if side != "short": return
        if self.ssr.is_on_ssr(symbol, now.date()):
            # Most brokers will only accept shorts if order is on an uptick; retail APIs often reject entirely.
            raise SSRViolation(f"{symbol} is under SSR today; shorting is restricted.")