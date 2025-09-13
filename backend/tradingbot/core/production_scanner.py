#!/usr / bin / env python3
"""
Production Hard Dip Scanner - Integrated Best of Both Worlds

Combines the practical real - data approach from the provided scanner with
the exact clone positioning logic. Detects "hard dip after big run" and
generates concrete options trade plans.

Key Features: 
- Real market data via yfinance
- Actual options chains with real strikes / premiums
- Both EOD and intraday scanning modes
- Exact clone position sizing (70 - 100% deployment)
- Concrete executable trade plans
- CSV / JSON output for further processing

Author: Integrated from both implementations
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf


# ---------- Configuration ----------
DEFAULT_UNIVERSE = [
    # Liquid mega - caps with tight spreads & huge OI (from original)
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AVGO", "TSLA",
    "AMZN", "GOOG", "NFLX", "CRM", "COST"
]

# Signal detection thresholds
RUN_LOOKBACK = 10        # Days to look back for "big run"
RUN_PCT = 0.10           # +10% minimum run to qualify
DIP_PCT = -0.03          # -3% minimum dip to trigger

# Exact clone options parameters
TARGET_DTE_DAYS = 30     # ~30 days to expiry
OTM_PCT = 0.05           # 5% out of the money
DEPLOY_PCT_DEFAULT = 0.90  # 90% all - in deployment (exact clone style)

# Black - Scholes fallback parameters
DEFAULT_IV = 0.28        # 28% IV assumption
DEFAULT_RATE = 0.04      # 4% risk - free rate
DEFAULT_DIV_YIELD = 0.0  # 0% dividend yield


# ---------- Utilities ----------
def now_ny()->datetime: 
    """Get current time in NY timezone"""
    return datetime.now(pytz.timezone("America / New_York"))

def to_pct(value: float)->str:
    """Format as percentage"""
    return f"{value * 100: .2f}%"

def round_to_increment(x: float, inc: float=1.0)->float:
    """Round to nearest increment (for strikes)"""
    return round(x / inc) * inc

def nearest_expiry(expiries: List[str], target_days: int)->Optional[str]:
    """Find expiry closest to target DTE"""
    if not expiries: 
        return None

    today = date.today()
    best_expiry = None
    best_diff = float('inf')

    for expiry_str in expiries: 
        try: 
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            days_diff = abs((expiry_date-today).days - target_days)
            if days_diff  <  best_diff: 
                best_diff = days_diff
                best_expiry = expiry_str
        except ValueError: 
            continue

    return best_expiry

# Black - Scholes implementation (fallback)
def _norm_cdf(x: float)->float:
    """Standard normal CDF using error function"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_call_price(spot: float, strike: float, t_years: float,
                  r: float, q: float, iv: float)->float:
    """Black - Scholes call price per share"""
    if any(val  <=  0 for val in [spot, strike, t_years, iv]): 
        raise ValueError("Invalid BS parameters")

    d1 = (math.log(spot / strike) + (r - q + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
    d2 = d1 - iv * math.sqrt(t_years)

    call_value = (spot * math.exp(-q * t_years) * _norm_cdf(d1) -
                  strike * math.exp(-r * t_years) * _norm_cdf(d2))

    return max(call_value, 0.0)


# ---------- Data Classes ----------
@dataclass
class DipSignal: 
    """Signal for hard dip after big run"""
    ticker: str
    timestamp_ny: str
    spot_price: float
    prior_close: float
    dip_percentage: float
    run_lookback_days: int
    run_return: float
    signal_type: str  # "eod" or "intraday"

@dataclass
class ExactClonePlan: 
    """Exact clone options trade plan"""
    ticker: str
    timestamp_ny: str

    # Market data
    spot_price: float

    # Options details
    expiry_date: str
    strike: float
    dte_days: int
    otm_percentage: float

    # Pricing
    premium_per_contract: float  # Dollars
    bid: float
    ask: float
    mid: float
    pricing_source: str  # "options_chain" or "black_scholes"

    # Position sizing (exact clone style)
    deploy_percentage: float
    contracts: int
    total_cost: float
    breakeven_at_expiry: float

    # Risk metrics
    effective_leverage: float
    ruin_risk_percentage: float

    # Exit targets (from original trade)
    exit_3x_target: float
    exit_4x_target: float

    # Notes
    strategy_notes: str

@dataclass
class ScanResults: 
    """Complete scan results"""
    scan_mode: str
    scan_timestamp: str
    universe_scanned: List[str]
    signals_found: int
    signals: List[DipSignal]
    trade_plans: List[ExactClonePlan]


# ---------- Market Data Functions ----------
def fetch_daily_history(ticker: str, period: str="90d")->pd.DataFrame:
    """Fetch daily price history with error handling"""
    try: 
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period, interval="1d", auto_adjust=False)

        if df.empty: 
            raise ValueError(f"No data returned for {ticker}")

        # Clean data
        df = df.dropna()

        if len(df)  <  20:  # Need sufficient history
            raise ValueError(f"Insufficient history for {ticker}")

        return df

    except Exception as e: 
        raise RuntimeError(f"Failed to fetch daily history for {ticker}: {e}")

def fetch_current_price(ticker: str)->Optional[Dict[str, float]]: 
    """Get current price and prior close for intraday scanning"""
    try: 
        ticker_obj = yf.Ticker(ticker)

        # Get prior close from daily data
        daily_hist = ticker_obj.history(period="5d", interval="1d")
        if len(daily_hist)  <  2: 
            return None
        prior_close = float(daily_hist["Close"].iloc[-2])

        # Get recent price from 5 - minute data
        intraday_hist = ticker_obj.history(period="2d", interval="5m")
        if intraday_hist.empty: 
            return None
        current_price = float(intraday_hist["Close"].iloc[-1])

        return {
            "current_price": current_price,
            "prior_close": prior_close
        }

    except Exception as e: 
        return None

def get_options_chain_data(ticker: str, expiry: str, target_strike: float)->Optional[Dict]:
    """Get real options chain data for specific expiry and strike"""
    try: 
        ticker_obj = yf.Ticker(ticker)
        options_chain = ticker_obj.option_chain(expiry)

        if options_chain.calls.empty: 
            return None

        # Find closest strike to target
        calls_df = options_chain.calls.copy()
        calls_df['strike_diff'] = abs(calls_df['strike'] - target_strike)
        closest_option = calls_df.loc[calls_df['strike_diff'].idxmin()]

        return {
            'strike': float(closest_option['strike']),
            'bid': float(closest_option['bid']) if not pd.isna(closest_option['bid']) else 0.0,
            'ask': float(closest_option['ask']) if not pd.isna(closest_option['ask']) else 0.0,
            'last_price': float(closest_option['lastPrice']) if not pd.isna(closest_option['lastPrice']) else 0.0,
            'volume': int(closest_option['volume']) if not pd.isna(closest_option['volume']) else 0,
            'open_interest': int(closest_option['openInterest']) if not pd.isna(closest_option['openInterest']) else 0
        }

    except Exception as e: 
        return None


# ---------- Signal Detection ----------
def detect_eod_signal(ticker: str,
                      run_lookback: int=RUN_LOOKBACK,
                      run_pct: float=RUN_PCT,
                      dip_pct: float=DIP_PCT)->Optional[DipSignal]:
    """Detect end - of - day hard dip signal after big run"""

    try: 
        df = fetch_daily_history(ticker)

        if len(df)  <  run_lookback + 2: 
            return None

        # Get today and yesterday
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        if yesterday["Close"]  <=  0: 
            return None

        # Calculate today's performance vs yesterday close
        day_change = (today["Close"] / yesterday["Close"]) - 1.0

        # Check if it's a hard dip
        if day_change  >  dip_pct:  # dip_pct is negative, so this checks if decline is big enough
            return None

        # Check for big run in prior period (ending yesterday)
        run_window = df["Close"].iloc[-(run_lookback + 1): -1]  # Exclude today
        run_return = (run_window.iloc[-1] / run_window.iloc[0]) - 1.0

        if run_return  <  run_pct: 
            return None

        return DipSignal(
            ticker=ticker,
            timestamp_ny = now_ny().isoformat(),
            spot_price = float(today["Close"]),
            prior_close = float(yesterday["Close"]),
            dip_percentage = float(day_change),
            run_lookback_days=run_lookback,
            run_return = float(run_return),
            signal_type = "eod"
        )

    except Exception as e: 
        return None

def detect_intraday_signal(ticker: str,
                          run_lookback: int=RUN_LOOKBACK,
                          run_pct: float=RUN_PCT,
                          dip_pct: float=DIP_PCT)->Optional[DipSignal]:
    """Detect intraday hard dip signal after big run"""

    try: 
        # First check for big run using daily data
        df = fetch_daily_history(ticker)

        if len(df)  <  run_lookback + 1: 
            return None

        # Check run in prior period (ending yesterday)
        run_window = df["Close"].iloc[-(run_lookback + 1): -1]
        run_return = (run_window.iloc[-1] / run_window.iloc[0]) - 1.0

        if run_return  <  run_pct: 
            return None

        # Get current live price vs prior close
        price_data = fetch_current_price(ticker)
        if not price_data: 
            return None

        current_price = price_data["current_price"]
        prior_close = price_data["prior_close"]

        if prior_close  <=  0: 
            return None

        # Calculate intraday dip
        intraday_change = (current_price / prior_close) - 1.0

        if intraday_change  >  dip_pct:  # Not enough of a dip
            return None

        return DipSignal(
            ticker=ticker,
            timestamp_ny = now_ny().isoformat(),
            spot_price = float(current_price),
            prior_close = float(prior_close),
            dip_percentage = float(intraday_change),
            run_lookback_days=run_lookback,
            run_return = float(run_return),
            signal_type = "intraday"
        )

    except Exception as e: 
        return None


# ---------- Options Plan Generation ----------
def create_exact_clone_plan(signal: DipSignal,
                           account_size: float,
                           deploy_pct: float=DEPLOY_PCT_DEFAULT,
                           target_dte: int=TARGET_DTE_DAYS,
                           otm_pct: float=OTM_PCT,
                           use_options_chain: bool=True)->ExactClonePlan:
    """Create exact clone options trade plan"""

    ticker = signal.ticker
    spot = signal.spot_price

    # Calculate target strike (5% OTM, rounded to whole dollars)
    raw_strike = spot * (1.0 + otm_pct)
    target_strike = round_to_increment(raw_strike, 1.0)

    # Find optimal expiry
    expiry_date = None
    actual_dte = target_dte

    try: 
        ticker_obj = yf.Ticker(ticker)
        available_expiries = ticker_obj.options
        expiry_date = nearest_expiry(available_expiries, target_dte)

        if expiry_date: 
            actual_dte = (datetime.strptime(expiry_date, "%Y-%m-%d").date() - date.today()).days
    except Exception: 
        # Fallback to synthetic expiry
        expiry_date = (date.today() + timedelta(days=target_dte)).isoformat()
        actual_dte = target_dte

    # Get option pricing
    bid, ask, mid=0.0, 0.0, 0.0
    premium_per_contract = 0.0
    pricing_source = "black_scholes"
    final_strike = target_strike

    if use_options_chain and expiry_date: 
        # Try to get real options chain data
        chain_data = get_options_chain_data(ticker, expiry_date, target_strike)

        if chain_data: 
            final_strike = chain_data['strike']
            bid = chain_data['bid']
            ask = chain_data['ask']

            # Calculate mid price
            if bid  >  0 and ask  >  0: 
                mid = (bid + ask) / 2.0
            else: 
                mid = chain_data['last_price']

            premium_per_contract = mid * 100.0  # Convert to per - contract
            pricing_source = "options_chain"

    # Fallback to Black - Scholes if no chain data
    if premium_per_contract  <=  0: 
        try: 
            t_years = max(actual_dte, 1) / 365.0
            bs_price_per_share = bs_call_price(
                spot=spot,
                strike=final_strike,
                t_years=t_years,
                r=DEFAULT_RATE,
                q=DEFAULT_DIV_YIELD,
                iv = DEFAULT_IV)
            premium_per_contract = bs_price_per_share * 100.0
            mid = premium_per_contract / 100.0
            pricing_source = "black_scholes"
        except Exception: 
            premium_per_contract = 100.0  # Conservative fallback
            mid = 1.0

    # Position sizing (exact clone style-high deployment)
    deploy_capital = account_size * deploy_pct
    contracts = int(deploy_capital / premium_per_contract) if premium_per_contract  >  0 else 0
    total_cost = contracts * premium_per_contract

    # Risk calculations
    actual_deploy_pct = (total_cost / account_size) if account_size  >  0 else 0
    notional_exposure = contracts * 100 * spot
    effective_leverage = notional_exposure / total_cost if total_cost  >  0 else 0

    # Breakeven and exit targets
    breakeven = final_strike+(premium_per_contract / 100.0)
    exit_3x = premium_per_contract * 3.0
    exit_4x = premium_per_contract * 4.0

    return ExactClonePlan(
        ticker=ticker,
        timestamp_ny = signal.timestamp_ny,
        spot_price=spot,
        expiry_date = expiry_date or "Unknown",
        strike=final_strike,
        dte_days=actual_dte,
        otm_percentage=otm_pct,
        premium_per_contract = round(premium_per_contract, 2),
        bid=bid,
        ask=ask,
        mid=mid,
        pricing_source=pricing_source,
        deploy_percentage=actual_deploy_pct,
        contracts=contracts,
        total_cost = round(total_cost, 2),
        breakeven_at_expiry = round(breakeven, 2),
        effective_leverage = round(effective_leverage, 1),
        ruin_risk_percentage = round(actual_deploy_pct * 100, 1),
        exit_3x_target = round(exit_3x, 2),
        exit_4x_target = round(exit_4x, 2),
        strategy_notes = "Exact clone: Buy dip, hold 1 - 2 days, exit at 3x - 4x or ITM"
    )


# ---------- Scanner Functions ----------
def run_eod_scan(universe: List[str],
                 account_size: float,
                 deploy_pct: float,
                 use_options_chain: bool)->ScanResults:
    """Run end - of - day scan"""

    signals = []
    trade_plans = []

    print(f"🔍 EOD Scan starting at {now_ny().strftime('%Y-%m-%d %H: %M:%S')} NY")
    print(f"Universe: {', '.join(universe)}")
    print(f"Account size: ${account_size:,.0f}")
    print(f"Deploy percentage: {deploy_pct:.1%}")

    for ticker in universe: 
        try: 
            signal = detect_eod_signal(ticker)

            if signal: 
                print(f"\n🚨 SIGNAL: {ticker}")
                print(f"   Dip: {to_pct(signal.dip_percentage)} (${signal.spot_price: .2f} from ${signal.prior_close: .2f})")
                print(f"   Prior run: {to_pct(signal.run_return)} over {signal.run_lookback_days} days")

                # Generate trade plan
                plan = create_exact_clone_plan(
                    signal=signal,
                    account_size=account_size,
                    deploy_pct=deploy_pct,
                    use_options_chain = use_options_chain)

                print("   📋 TRADE PLAN: ")
                print(f"      Strike: ${plan.strike} ({plan.otm_percentage: .1%} OTM)")
                print(f"      Expiry: {plan.expiry_date} ({plan.dte_days} DTE)")
                print(f"      Premium: ${plan.premium_per_contract:.2f} per contract ({plan.pricing_source})")
                print(f"      Position: {plan.contracts:,} contracts=${plan.total_cost: ,.0f}")
                print(f"      Risk: {plan.ruin_risk_percentage:.1f}% of account")
                print(f"      Leverage: {plan.effective_leverage:.1f}x")
                print(f"      Breakeven: ${plan.breakeven_at_expiry:.2f}")
                print(f"      Exit targets: ${plan.exit_3x_target:.2f} (3x) | ${plan.exit_4x_target: .2f} (4x)")

                signals.append(signal)
                trade_plans.append(plan)

        except Exception as e: 
            print(f"⚠️  Error scanning {ticker}: {e}")

    return ScanResults(
        scan_mode = "eod",
        scan_timestamp = now_ny().isoformat(),
        universe_scanned=universe,
        signals_found = len(signals),
        signals=signals,
        trade_plans = trade_plans)

def run_intraday_scan(universe: List[str],
                     account_size: float,
                     deploy_pct: float,
                     use_options_chain: bool,
                     poll_seconds: int,
                     max_minutes: int)->ScanResults:
    """Run intraday scanning loop"""

    end_time = now_ny() + timedelta(minutes=max_minutes) if max_minutes  >  0 else None
    alerted_tickers = set()
    all_signals = []
    all_plans = []

    print(f"🔍 Intraday scan starting at {now_ny().strftime('%Y-%m-%d %H: %M:%S')} NY")
    print(f"Universe: {', '.join(universe)}")
    print(f"Poll interval: {poll_seconds}s")
    if max_minutes  >  0: 
        print(f"Max duration: {max_minutes} minutes")
    print(f"Account size: ${account_size:,.0f}")
    print(f"Deploy percentage: {deploy_pct:.1%}")

    scan_count = 0

    try: 
        while True: 
            if end_time and now_ny()  >=  end_time: 
                print("⏰ Time limit reached")
                break

            scan_count += 1
            print(f"\n--- Scan #{scan_count} at {now_ny().strftime('%H: %M:%S')} ---")

            for ticker in universe: 
                if ticker in alerted_tickers: 
                    continue  # Only alert once per day per ticker

                try: 
                    signal = detect_intraday_signal(ticker)

                    if signal: 
                        print(f"\n🚨 INTRADAY SIGNAL: {ticker}")
                        print(f"   Live dip: {to_pct(signal.dip_percentage)} (${signal.spot_price: .2f} from ${signal.prior_close: .2f})")
                        print(f"   Prior run: {to_pct(signal.run_return)} over {signal.run_lookback_days} days")

                        # Generate trade plan
                        plan = create_exact_clone_plan(
                            signal=signal,
                            account_size=account_size,
                            deploy_pct=deploy_pct,
                            use_options_chain = use_options_chain)

                        print("   📋 TRADE PLAN: ")
                        print(f"      Strike: ${plan.strike} ({plan.otm_percentage: .1%} OTM)")
                        print(f"      Expiry: {plan.expiry_date} ({plan.dte_days} DTE)")
                        print(f"      Premium: ${plan.premium_per_contract:.2f} per contract ({plan.pricing_source})")
                        print(f"      Position: {plan.contracts:,} contracts=${plan.total_cost: ,.0f}")
                        print(f"      Risk: {plan.ruin_risk_percentage:.1f}% of account")
                        print(f"      Leverage: {plan.effective_leverage:.1f}x")
                        print(f"      Exit targets: ${plan.exit_3x_target:.2f} (3x) | ${plan.exit_4x_target: .2f} (4x)")

                        all_signals.append(signal)
                        all_plans.append(plan)
                        alerted_tickers.add(ticker)

                except Exception as e: 
                    print(f"⚠️  Error scanning {ticker}: {e}")

            if not any(t not in alerted_tickers for t in universe): 
                print("✅ All tickers alerted - stopping scan")
                break

            print("😴 Sleeping...")
            time.sleep(max(5, poll_seconds))

    except KeyboardInterrupt: 
        print("\n⛔ Scan stopped by user")

    return ScanResults(
        scan_mode = "intraday",
        scan_timestamp = now_ny().isoformat(),
        universe_scanned=universe,
        signals_found = len(all_signals),
        signals=all_signals,
        trade_plans = all_plans)


# ---------- Output Functions ----------
def write_results(results: ScanResults, output_prefix: str)->None:
    """Write scan results to files"""

    # Write signals
    if results.signals: 
        signals_df = pd.DataFrame([asdict(s) for s in results.signals])
        signals_df.to_csv(f"{output_prefix}_signals.csv", index=False)

        with open(f"{output_prefix}_signals.json", "w") as f: 
            json.dump([asdict(s) for s in results.signals], f, indent=2)

    # Write trade plans
    if results.trade_plans: 
        plans_df = pd.DataFrame([asdict(p) for p in results.trade_plans])
        plans_df.to_csv(f"{output_prefix}_plans.csv", index=False)

        with open(f"{output_prefix}_plans.json", "w") as f: 
            json.dump([asdict(p) for p in results.trade_plans], f, indent=2)

    # Write summary
    summary = {
        "scan_summary": asdict(results),
        "total_signals": len(results.signals),
        "total_capital_at_risk": sum(p.total_cost for p in results.trade_plans),
        "average_risk_per_trade": sum(p.ruin_risk_percentage for p in results.trade_plans) / len(results.trade_plans) if results.trade_plans else 0,
        "average_leverage": sum(p.effective_leverage for p in results.trade_plans) / len(results.trade_plans) if results.trade_plans else 0
    }

    with open(f"{output_prefix}_summary.json", "w") as f: 
        json.dump(summary, f, indent=2)

    print(f"\n📁 Results written to {output_prefix}_*.csv / json")


# ---------- Main Function ----------
def main(): 
    parser = argparse.ArgumentParser(
        description = "Production Hard Dip Scanner - Exact Clone Implementation",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
Examples: 
  # EOD scan with 90% deployment
  python production_scanner.py --mode eod --account - size 450000 --deploy - pct 0.90 --use-options - chain

  # Intraday scan for 2 hours, checking every 2 minutes
  python production_scanner.py --mode intraday --poll - seconds 120 --max - minutes 120 \\
    --account - size 450000 --deploy - pct 1.0 --use-options - chain

  # Paper trading with smaller size
  python production_scanner.py --mode eod --account - size 50000 --deploy - pct 0.10
        """
    )

    # Basic options
    parser.add_argument("--mode", choices=["eod", "intraday"], default="eod",
                       help = "Scan mode: eod (end of day) or intraday (live)")
    parser.add_argument("--universe", type=str, default=",".join(DEFAULT_UNIVERSE),
                       help = "Comma - separated list of tickers to scan")

    # Account sizing
    parser.add_argument("--account - size", type=float, default=500000.0,
                       help = "Total account size in dollars")
    parser.add_argument("--deploy - pct", type=float, default=DEPLOY_PCT_DEFAULT,
                       help = "Percentage of account to deploy per trade (0.0 - 1.0)")

    # Options
    parser.add_argument("--use-options - chain", action="store_true",
                       help = "Use real options chain data (vs Black - Scholes)")

    # Intraday options
    parser.add_argument("--poll - seconds", type=int, default=90,
                       help = "Polling interval for intraday mode (seconds)")
    parser.add_argument("--max - minutes", type=int, default=0,
                       help = "Maximum scan duration for intraday mode (0=unlimited)")

    # Signal parameters
    parser.add_argument("--run - lookback", type=int, default=RUN_LOOKBACK,
                       help = "Days to look back for big run")
    parser.add_argument("--run - pct", type=float, default=RUN_PCT,
                       help = "Minimum run percentage to qualify")
    parser.add_argument("--dip - pct", type=float, default=DIP_PCT,
                       help = "Minimum dip percentage (negative)")

    # Output
    parser.add_argument("--output - prefix", type=str, default="hard_dip_scan",
                       help = "Prefix for output files")

    args = parser.parse_args()

    # Validation
    universe = [ticker.strip().upper() for ticker in args.universe.split(",")]
    universe = [t for t in universe if t]  # Remove empty strings

    if not universe: 
        print("❌ Empty ticker universe", file=sys.stderr)
        sys.exit(1)

    if not (0.0  <  args.deploy_pct  <=  1.0): 
        print("❌ Deploy percentage must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    if args.account_size  <=  0: 
        print("❌ Account size must be positive", file=sys.stderr)
        sys.exit(1)

    # Update global thresholds
    global RUN_LOOKBACK, RUN_PCT, DIP_PCT
    RUN_LOOKBACK = args.run_lookback
    RUN_PCT = args.run_pct
    DIP_PCT = args.dip_pct

    try: 
        # Run scan
        if args.mode  ==  "eod": results = run_eod_scan(
                universe=universe,
                account_size = args.account_size,
                deploy_pct = args.deploy_pct,
                use_options_chain = args.use_options_chain
            )
        else: 
            results = run_intraday_scan(
                universe=universe,
                account_size = args.account_size,
                deploy_pct = args.deploy_pct,
                use_options_chain = args.use_options_chain,
                poll_seconds = args.poll_seconds,
                max_minutes = args.max_minutes
            )

        # Write results
        write_results(results, args.output_prefix)

        # Summary
        print("\n🎯 SCAN COMPLETE")
        print(f"Mode: {results.scan_mode}")
        print(f"Signals found: {results.signals_found}")
        print(f"Universe scanned: {len(results.universe_scanned)} tickers")

        if results.trade_plans: 
            total_cost = sum(p.total_cost for p in results.trade_plans)
            avg_risk = sum(p.ruin_risk_percentage for p in results.trade_plans) / len(results.trade_plans)
            avg_leverage = sum(p.effective_leverage for p in results.trade_plans) / len(results.trade_plans)

            print(f"Total capital at risk: ${total_cost:,.0f}")
            print(f"Average risk per trade: {avg_risk:.1f}%")
            print(f"Average leverage: {avg_leverage:.1f}x")
        else: 
            print("No signals detected")

    except KeyboardInterrupt: 
        print("\n⛔ Stopped by user")
    except Exception as e: 
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ ==  "__main__": main()
