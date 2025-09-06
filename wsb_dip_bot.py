#!/usr/bin/env python3
"""
WSB Dip-After-Run Scanner & Exact-Clone Options Planner/Monitor

Replicates the common r/WSB winning tactic:
- Buy ~5% OTM calls with ~30 DTE on a HARD DIP DAY that follows a BIG RUN.
- Single strike/expiry, big size allowed (configurable risk%).
- Exit same/next day when:
    • option ≈ 3x entry, OR
    • goes ITM / Δ ≥ 0.60 (approx), OR
    • time stop / loss stop triggers.

Notes
- Data: yfinance (Yahoo). This is best-effort retail-grade data.
- Trading is NOT executed by this script—only scanning, planning, and monitoring.
- Use at your own risk.

Install deps:
    pip install -r requirements.txt

Examples:
    # End-of-day scan across default mega-caps; propose an options line per hit
    python wsb_dip_bot.py scan-eod --account-size 450000 --risk-pct 1.0 --use-options-chain
    
    # Intraday polling every 2 minutes for 30 minutes
    python wsb_dip_bot.py scan-intraday --poll-seconds 120 --max-minutes 30 \
        --account-size 450000 --risk-pct 1.0 --use-options-chain

    # Build a plan for one ticker now (e.g., GOOG at spot 207)
    python wsb_dip_bot.py plan --ticker GOOG --spot 207 --account-size 450000 --risk-pct 1.0 --use-options-chain

    # Monitor a built plan (target 3x or Δ≥0.60), polling every 60s, stop after market close
    python wsb_dip_bot.py monitor --ticker GOOG --expiry 2025-10-17 --strike 220 --entry-prem 4.70 \
        --target-mult 3.0 --delta-target 0.60 --poll-seconds 60 --max-minutes 360

Author: ChatGPT
License: MIT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from utils.yfinance_hardening import safe_mid, fetch_last_and_prior_close
from utils.error_handling import retry


# -------------------------- Config Defaults --------------------------

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AVGO", "TSLA",
    "AMZN", "NFLX", "CRM", "COST", "ADBE", "V", "MA", "LIN"
]

RUN_LOOKBACK = 10     # sessions to consider as the "run" window
RUN_PCT = 0.10        # >= +10% over run window
DIP_PCT = -0.03       # hard dip day threshold (≤ -3% vs prior close)

TARGET_DTE_DAYS = 30
OTM_PCT = 0.05

RISK_PCT_DEFAULT = 0.10  # 10% of account per signal by default (can set 1.0 to mirror "all-in")


# -------------------------- Helpers --------------------------

def now_ny() -> datetime:
    return datetime.now(pytz.timezone("America/New_York"))

def round_to_increment(x: float, inc: float = 1.0) -> float:
    return round(x / inc) * inc

def nearest_expiry(expiries: List[str], target_days: int) -> Optional[str]:
    if not expiries:
        return None
    today = date.today()
    best = None
    best_diff = 10**9
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        diff = abs((d - today).days - target_days)
        if diff < best_diff:
            best_diff = diff
            best = e
    return best

def pct(a: float) -> str:
    return f"{a*100:.2f}%"


# -------------------------- Black-Scholes & Greeks --------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_d1(spot: float, strike: float, t: float, r: float, q: float, iv: float) -> float:
    return (math.log(spot/strike) + (r - q + 0.5*iv*iv)*t) / (iv*math.sqrt(t))

def bs_call(spot: float, strike: float, t: float, r: float, q: float, iv: float) -> float:
    if min(spot, strike, t, iv) <= 0: raise ValueError("positive inputs required")
    d1 = bs_d1(spot, strike, t, r, q, iv)
    d2 = d1 - iv*math.sqrt(t)
    return spot*math.exp(-q*t)*_norm_cdf(d1) - strike*math.exp(-r*t)*_norm_cdf(d2)

def bs_delta_call(spot: float, strike: float, t: float, r: float, q: float, iv: float) -> float:
    d1 = bs_d1(spot, strike, t, r, q, iv)
    return math.exp(-q*t) * _norm_cdf(d1)

def implied_vol_call(market_px: float, spot: float, strike: float, t: float, r: float = 0.04, q: float = 0.0,
                     iv_init: float = 0.3, tol: float = 1e-4, max_iter: int = 60) -> Optional[float]:
    """ Solve for IV using Newton-Raphson; returns None if fails. """
    if market_px <= 0 or min(spot, strike, t) <= 0:
        return None
    iv = iv_init
    for _ in range(max_iter):
        try:
            d1 = bs_d1(spot, strike, t, r, q, iv)
            d2 = d1 - iv*math.sqrt(t)
            model = spot*math.exp(-q*t)*_norm_cdf(d1) - strike*math.exp(-r*t)*_norm_cdf(d2)
            vega = spot*math.exp(-q*t)*math.sqrt(t)*_norm_pdf(d1)
            diff = model - market_px
            if abs(diff) < tol:
                return max(iv, 1e-6)
            if vega <= 1e-8:
                break
            iv -= diff / vega
            if iv <= 0 or iv > 10:
                iv = max(min(iv, 10.0), 1e-6)
        except Exception:
            break
    return None


# -------------------------- Data Access --------------------------

def fetch_daily_history(ticker: str, period: str = "120d") -> pd.DataFrame:
    tkr = yf.Ticker(ticker)
    df = tkr.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No daily history for {ticker}")
    return df.dropna()

def fetch_intraday_last_and_prior_close(ticker: str) -> Optional[Dict[str, float]]:
    tkr = yf.Ticker(ticker)
    dailies = tkr.history(period="7d", interval="1d")
    if dailies is None or len(dailies) < 2:
        return None
    prior_close = float(dailies["Close"].iloc[-2])
    intraday = tkr.history(period="2d", interval="5m")
    if intraday is None or intraday.empty:
        return None
    last = float(intraday["Close"].iloc[-1])
    return {"last": last, "prior_close": prior_close}

@retry(tries=3, delay=1.0, exceptions=(Exception,))
def get_option_mid_for_nearest_5pct_otm(ticker: str, expiry: str, desired_strike: float) -> Optional[Dict[str, float]]:
    tkr = yf.Ticker(ticker)
    chain = tkr.option_chain(expiry)
    calls = chain.calls.copy()
    if calls is None or calls.empty:
        return None
    calls["absdiff"] = (calls["strike"] - desired_strike).abs()
    row = calls.sort_values(["absdiff", "strike"]).head(1)
    if row.empty:
        return None
    strike = float(row["strike"].iloc[0])
    bid = float(row["bid"].iloc[0]) if pd.notna(row["bid"].iloc[0]) else 0.0
    ask = float(row["ask"].iloc[0]) if pd.notna(row["ask"].iloc[0]) else 0.0
    last = float(row["lastPrice"].iloc[0]) if pd.notna(row["lastPrice"].iloc[0]) else 0.0
    mid = safe_mid(bid, ask, last)
    return {"strike": strike, "mid": mid, "bid": bid, "ask": ask, "last": last}


# -------------------------- Signals & Plans --------------------------

@dataclass
class DipSignal:
    ticker: str
    ts_ny: str
    spot: float
    prior_close: float
    intraday_pct: float
    run_lookback: int
    run_return: float

@dataclass
class OptionPlan:
    ticker: str
    ts_ny: str
    spot: float
    expiry: str
    strike: float
    otm_pct: float
    dte_days: int
    premium_est_per_contract: float  # $
    contracts: int
    total_cost: float                # $
    breakeven_at_expiry: float       # $
    notes: str

def detect_eod_signal(ticker: str, run_lookback: int, run_pct: float, dip_pct: float) -> Optional[DipSignal]:
    df = fetch_daily_history(ticker)
    if len(df) < run_lookback + 2:
        return None
    today = df.iloc[-1]
    yday = df.iloc[-2]
    if yday["Close"] <= 0:
        return None
    day_chg = (today["Close"] / yday["Close"]) - 1.0
    if day_chg > dip_pct:
        return None
    window = df["Close"].iloc[-(run_lookback+1):-1]
    run_ret = (window.iloc[-1] / window.iloc[0]) - 1.0
    if run_ret < run_pct:
        return None
    return DipSignal(
        ticker=ticker,
        ts_ny=now_ny().isoformat(),
        spot=float(today["Close"]),
        prior_close=float(yday["Close"]),
        intraday_pct=float(day_chg),
        run_lookback=run_lookback,
        run_return=float(run_ret)
    )

def detect_intraday_signal(ticker: str, run_lookback: int, run_pct: float, dip_pct: float) -> Optional[DipSignal]:
    # Run check on dailies
    df = fetch_daily_history(ticker, period="180d")
    if len(df) < run_lookback + 2:
        return None
    window = df["Close"].iloc[-(run_lookback+1):-1]
    run_ret = (window.iloc[-1] / window.iloc[0]) - 1.0
    if run_ret < run_pct:
        return None
    # Live dip vs prior close
    live = fetch_intraday_last_and_prior_close(ticker)
    if not live or live["prior_close"] <= 0:
        return None
    live_pct = (live["last"] / live["prior_close"]) - 1.0
    if live_pct > dip_pct:
        return None
    return DipSignal(
        ticker=ticker,
        ts_ny=now_ny().isoformat(),
        spot=float(live["last"]),
        prior_close=float(live["prior_close"]),
        intraday_pct=float(live_pct),
        run_lookback=run_lookback,
        run_return=float(run_ret)
    )

def build_exact_plan(ticker: str, spot: float, account_size: float, risk_pct: float,
                     target_dte_days: int, otm_pct: float, use_chain: bool) -> OptionPlan:
    tkr = yf.Ticker(ticker)
    # Choose expiry closest to target DTE
    try:
        expiries = list(tkr.options)
    except Exception:
        expiries = []
    if expiries:
        expiry = nearest_expiry(expiries, target_dte_days)
    else:
        expiry = (date.today() + timedelta(days=target_dte_days)).isoformat()
    # Desired strike ≈ 5% OTM
    desired_strike = round_to_increment(spot * (1.0 + otm_pct), 1.0)
    # Price via chain or fallback via BS
    if use_chain and expiries:
        row = get_option_mid_for_nearest_5pct_otm(ticker, expiry, desired_strike)
        if row:
            strike = float(row["strike"])
            mid = float(row["mid"]) * 100.0  # contract premium $
            prem = max(mid, 0.01)
        else:
            strike = desired_strike
            prem = None
    else:
        strike = desired_strike
        prem = None
    if prem is None:
        # Fallback estimate via BS with IV=30%, r=4%
        try:
            dte = max((datetime.fromisoformat(expiry).date() - date.today()).days, 1)
        except Exception:
            dte = target_dte_days
        t = dte / 365.0
        px = bs_call(spot, strike, t, r=0.04, q=0.0, iv=0.30)
        prem = max(px * 100.0, 0.01)
    contracts = int((account_size * risk_pct) // prem)
    total_cost = contracts * prem
    try:
        dte_days = (datetime.fromisoformat(expiry).date() - date.today()).days
    except Exception:
        dte_days = target_dte_days
    breakeven = strike + prem / 100.0
    return OptionPlan(
        ticker=ticker,
        ts_ny=now_ny().isoformat(),
        spot=float(spot),
        expiry=str(expiry),
        strike=float(strike),
        otm_pct=float(otm_pct),
        dte_days=int(dte_days),
        premium_est_per_contract=float(round(prem, 2)),
        contracts=int(contracts),
        total_cost=float(round(total_cost, 2)),
        breakeven_at_expiry=float(round(breakeven, 2)),
        notes="Buy the line; aim for 3x or Δ≥0.60 within 1–2 sessions; strict time/stop exits."
    )


# -------------------------- Monitor Logic --------------------------

def monitor_plan(ticker: str, expiry: str, strike: float, entry_prem: float,
                 target_mult: float = 3.0, delta_target: float = 0.60, loss_stop_mult: float = 0.5,
                 poll_seconds: int = 60, max_minutes: int = 360) -> None:
    """
    Polls the option mid every poll_seconds. Alerts when:
    - price >= target_mult * entry_prem
    - approx delta >= delta_target (via BS using IV implied from current mid if possible)
    - price <= loss_stop_mult * entry_prem (stop)
    """
    end_time = now_ny() + timedelta(minutes=max_minutes) if max_minutes > 0 else None
    tkr = yf.Ticker(ticker)
    target_px = target_mult * entry_prem
    stop_px = loss_stop_mult * entry_prem

    print(f"[MONITOR] {ticker} {expiry} C{strike} | entry ${entry_prem:.2f} | target ${target_px:.2f} | stop ${stop_px:.2f}")
    while True:
        if end_time and now_ny() >= end_time:
            print("[MONITOR] Time limit reached.")
            return
        try:
            chain = tkr.option_chain(expiry)
            calls = chain.calls
            calls["absdiff"] = (calls["strike"] - strike).abs()
            row = calls.sort_values(["absdiff", "strike"]).head(1)
            if row.empty:
                print("[WARN] Contract not found; retrying...")
                time.sleep(poll_seconds)
                continue
            bid = float(row["bid"].iloc[0]) if pd.notna(row["bid"].iloc[0]) else 0.0
            ask = float(row["ask"].iloc[0]) if pd.notna(row["ask"].iloc[0]) else 0.0
            last = float(row["lastPrice"].iloc[0]) if pd.notna(row["lastPrice"].iloc[0]) else 0.0
            mid_per_share = safe_mid(bid, ask, last)
            mid = mid_per_share * 100.0

            # Spot and DTE
            live = fetch_last_and_prior_close(ticker)
            if not live:
                # handle market-closed or API hiccup
                continue
            spot = live["last"]
            dte_days = max((datetime.fromisoformat(expiry).date() - date.today()).days, 1)
            t = dte_days / 365.0

            # Try to estimate IV from mid; then delta
            iv = implied_vol_call(market_px=mid_per_share, spot=spot, strike=strike, t=t, r=0.04, q=0.0) or 0.30
            delta = bs_delta_call(spot, strike, t, r=0.04, q=0.0, iv=iv)

            timestamp = now_ny().strftime('%H:%M:%S')
            print(f"[{timestamp}] spot={spot:.2f} mid={mid:.2f} (bid={bid*100:.2f}/ask={ask*100:.2f}) IV≈{iv:.2%} Δ≈{delta:.2f}")

            if mid >= target_px:
                print(f"[TAKE-PROFIT] Target hit: ${mid:.2f} ≥ ${target_px:.2f}")
                return
            if delta >= delta_target:
                print(f"[DELTA EXIT] Delta target hit: Δ≈{delta:.2f} ≥ {delta_target:.2f}")
                return
            if mid <= stop_px:
                print(f"[STOP] Loss stop hit: ${mid:.2f} ≤ ${stop_px:.2f}")
                return
        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped by user.")
            return
        except Exception as e:
            print(f"[WARN] monitor error: {e}")
        time.sleep(max(5, poll_seconds))


# -------------------------- CSV/JSON Writers --------------------------

def write_outputs(signals: List[DipSignal], plans: List[OptionPlan], out_prefix: str) -> None:
    if signals:
        df_s = pd.DataFrame([asdict(s) for s in signals])
        df_s.to_csv(f"{out_prefix}_signals.csv", index=False)
        with open(f"{out_prefix}_signals.json", "w") as f:
            json.dump(df_s.to_dict(orient="records"), f, indent=2)
    if plans:
        df_p = pd.DataFrame([asdict(p) for p in plans])
        df_p.to_csv(f"{out_prefix}_plans.csv", index=False)
        with open(f"{out_prefix}_plans.json", "w") as f:
            json.dump(df_p.to_dict(orient="records"), f, indent=2)


# -------------------------- Subcommand Runners --------------------------

def run_scan_eod(universe: List[str], account_size: float, risk_pct: float, use_chain: bool,
                 run_lookback: int, run_pct: float, dip_pct: float, out_prefix: str) -> None:
    hits, plans = [], []
    for t in universe:
        try:
            sig = detect_eod_signal(t, run_lookback, run_pct, dip_pct)
            if sig:
                print(f"[EOD] {t}: dip {pct(sig.intraday_pct)} after {sig.run_lookback}d run {pct(sig.run_return)} | spot={sig.spot:.2f}")
                plan = build_exact_plan(t, sig.spot, account_size, risk_pct, TARGET_DTE_DAYS, OTM_PCT, use_chain)
                print("      Plan: " + str(t) + " " + str(plan.expiry) + " C" + str(plan.strike) + " ~5%%OTM | est prem $" + str(plan.premium_est_per_contract) + " | "
                      f"contracts {plan.contracts} | cost ${plan.total_cost:.2f} | BE {plan.breakeven_at_expiry:.2f}")
                hits.append(sig)
                plans.append(plan)
        except Exception as e:
            print(f"[WARN] {t}: {e}", file=sys.stderr)
    write_outputs(hits, plans, out_prefix)
    if not hits:
        print("No EOD signals.")

def run_scan_intraday(universe: List[str], account_size: float, risk_pct: float, use_chain: bool,
                      run_lookback: int, run_pct: float, dip_pct: float,
                      poll_seconds: int, max_minutes: int, out_prefix: str) -> None:
    end_time = now_ny() + timedelta(minutes=max_minutes) if max_minutes > 0 else None
    seen: set[str] = set()
    hits_all, plans_all = [], []
    while True:
        if end_time and now_ny() >= end_time:
            break
        for t in universe:
            if t in seen: 
                continue
            try:
                sig = detect_intraday_signal(t, run_lookback, run_pct, dip_pct)
                if sig:
                    print(f"[INTRADAY] {t}: dip {pct(sig.intraday_pct)} after {sig.run_lookback}d run {pct(sig.run_return)} | spot={sig.spot:.2f}")
                    plan = build_exact_plan(t, sig.spot, account_size, risk_pct, TARGET_DTE_DAYS, OTM_PCT, use_chain)
                    print("          Plan: " + str(t) + " " + str(plan.expiry) + " C" + str(plan.strike) + " ~5%%OTM | est prem $" + str(plan.premium_est_per_contract) + " | "
                          f"contracts {plan.contracts} | cost ${plan.total_cost:.2f} | BE {plan.breakeven_at_expiry:.2f}")
                    hits_all.append(sig)
                    plans_all.append(plan)
                    seen.add(t)
            except Exception as e:
                print(f"[WARN] {t}: {e}", file=sys.stderr)
        time.sleep(max(5, poll_seconds))
    write_outputs(hits_all, plans_all, out_prefix)
    if not hits_all:
        print("No intraday signals recorded.")

def run_plan_one(ticker: str, spot: float, account_size: float, risk_pct: float, use_chain: bool) -> None:
    plan = build_exact_plan(ticker, spot, account_size, risk_pct, TARGET_DTE_DAYS, OTM_PCT, use_chain)
    print(json.dumps(asdict(plan), indent=2))

def run_monitor_one(**kwargs) -> None:
    monitor_plan(**kwargs)


# -------------------------- CLI --------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSB Dip-after-Run Scanner/Planner/Monitor")
    sub = p.add_subparsers(dest="cmd", required=True)

    # scan-eod
    se = sub.add_parser("scan-eod", help="End-of-day scan across universe")
    se.add_argument("--universe", type=str, default=",".join(DEFAULT_UNIVERSE))
    se.add_argument("--account-size", type=float, default=500_000.0)
    se.add_argument("--risk-pct", type=float, default=RISK_PCT_DEFAULT)
    se.add_argument("--use-options-chain", action="store_true")
    se.add_argument("--run-lookback", type=int, default=RUN_LOOKBACK)
    se.add_argument("--run-pct", type=float, default=RUN_PCT)
    se.add_argument("--dip-pct", type=float, default=DIP_PCT)
    se.add_argument("--out-prefix", type=str, default="wsb_dip_eod")

    # scan-intraday
    si = sub.add_parser("scan-intraday", help="Intraday scan across universe (polling)")
    si.add_argument("--universe", type=str, default=",".join(DEFAULT_UNIVERSE))
    si.add_argument("--account-size", type=float, default=500_000.0)
    si.add_argument("--risk-pct", type=float, default=RISK_PCT_DEFAULT)
    si.add_argument("--use-options-chain", action="store_true")
    si.add_argument("--run-lookback", type=int, default=RUN_LOOKBACK)
    si.add_argument("--run-pct", type=float, default=RUN_PCT)
    si.add_argument("--dip-pct", type=float, default=DIP_PCT)
    si.add_argument("--poll-seconds", type=int, default=90)
    si.add_argument("--max-minutes", type=int, default=0)
    si.add_argument("--out-prefix", type=str, default="wsb_dip_intraday")

    # plan (single ticker)
    pl = sub.add_parser("plan", help="Plan the exact ~5% OTM ~30DTE line for one ticker")
    pl.add_argument("--ticker", required=True, type=str)
    pl.add_argument("--spot", required=True, type=float, help="Current spot you see")
    pl.add_argument("--account-size", type=float, default=500_000.0)
    pl.add_argument("--risk-pct", type=float, default=RISK_PCT_DEFAULT)
    pl.add_argument("--use-options-chain", action="store_true")

    # monitor (single contract line)
    mo = sub.add_parser("monitor", help="Monitor a chosen contract for TP/Δ/SL exits")
    mo.add_argument("--ticker", required=True, type=str)
    mo.add_argument("--expiry", required=True, type=str, help="YYYY-MM-DD")
    mo.add_argument("--strike", required=True, type=float)
    mo.add_argument("--entry-prem", required=True, type=float, help="Entry premium per contract ($)")
    mo.add_argument("--target-mult", type=float, default=3.0)
    mo.add_argument("--delta-target", type=float, default=0.60)
    mo.add_argument("--loss-stop-mult", type=float, default=0.50)
    mo.add_argument("--poll-seconds", type=int, default=60)
    mo.add_argument("--max-minutes", type=int, default=360)

    return p.parse_args()

def main() -> None:
    args = parse_args()

    if args.cmd == "scan-eod":
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
        if not (0 < args.risk_pct <= 1.0):
            print("risk-pct must be in (0,1].", file=sys.stderr)
            sys.exit(2)
        run_scan_eod(universe, args.account_size, args.risk_pct, args.use_options_chain,
                     args.run_lookback, args.run_pct, args.dip_pct, args.out_prefix)

    elif args.cmd == "scan-intraday":
        universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
        if not (0 < args.risk_pct <= 1.0):
            print("risk-pct must be in (0,1].", file=sys.stderr)
            sys.exit(2)
        run_scan_intraday(universe, args.account_size, args.risk_pct, args.use_options_chain,
                          args.run_lookback, args.run_pct, args.dip_pct,
                          args.poll_seconds, args.max_minutes, args.out_prefix)

    elif args.cmd == "plan":
        if not (0 < args.risk_pct <= 1.0):
            print("risk-pct must be in (0,1].", file=sys.stderr)
            sys.exit(2)
        run_plan_one(args.ticker.upper(), args.spot, args.account_size, args.risk_pct, args.use_options_chain)

    elif args.cmd == "monitor":
        run_monitor_one(
            ticker=args.ticker.upper(),
            expiry=args.expiry,
            strike=args.strike,
            entry_prem=args.entry_prem,
            target_mult=args.target_mult,
            delta_target=args.delta_target,
            loss_stop_mult=args.loss_stop_mult,
            poll_seconds=args.poll_seconds,
            max_minutes=args.max_minutes
        )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)