#!/usr/bin/env python
"""Integration examples for WallStreetBots domain-specific modules.

This file demonstrates how to integrate the production-ready domain modules
for live US equities/options trading with proper compliance, risk management,
and operational safety.
"""

import datetime as dt
from typing import Dict, List
import pandas as pd
import numpy as np

# Import all the domain modules
from backend.tradingbot.compliance import (
    ComplianceGuard, ComplianceError, PDTViolation, SSRViolation, HaltViolation, SessionViolation
)
from backend.tradingbot.options import (
    OptionContract, UnderlyingState, auto_exercise_likely, early_assignment_risk, pin_risk
)
from backend.tradingbot.borrow import BorrowClient, LocateQuote, guard_can_short
from backend.tradingbot.accounting import WashSaleEngine, Fill
from backend.tradingbot.universe import UniverseProvider, Membership
from backend.tradingbot.risk.portfolio_rules import sector_cap_check, simple_corr_guard
from backend.tradingbot.infra.runtime_safety import Journal, assert_ntp_ok, ClockDriftError
from backend.tradingbot.data.corporate_actions import CorporateAction, CorporateActionsAdjuster

# Mock execution interfaces for examples
from backend.tradingbot.execution import OrderRequest, ExecutionClient
from backend.tradingbot.risk.engine import RiskEngine

def example_pre_trade_pipeline():
    """Example: Complete pre-trade compliance and risk pipeline."""
    print("=== PRE-TRADE COMPLIANCE AND RISK PIPELINE ===")

    # 1. Initialize compliance guard
    guard = ComplianceGuard(min_equity_for_day_trading=25_000.0)

    # 2. Set up market state (normally fed from market data)
    guard.set_halt("MSFT", "News pending")  # Example halt
    guard.set_luld("SPY", lower=400.0, upper=500.0)  # LULD bands
    guard.set_ssr("TSLA", dt.date.today())  # Short sale restriction

    # 3. Initialize borrow client for shorting
    borrow_client = BorrowClient()

    # 4. Sample order request
    order = OrderRequest(
        client_order_id="test_order_001",
        symbol="SPY",
        qty=100,
        side="buy",
        type="limit",
        time_in_force="day",
        limit_price=450.0
    )

    # 5. Run compliance checks
    now = dt.datetime.utcnow()
    account_equity = 50_000.0  # Sample account equity
    pending_day_trades = 2     # Sample day trade count

    try:
        # Session check (allow pre/post market if needed)
        guard.check_session(now, allow_pre=True, allow_post=False)
        print("✓ Session check passed")

        # Halt check
        guard.check_halt(order.symbol)
        print(f"✓ Halt check passed for {order.symbol}")

        # LULD check
        guard.check_luld(order.symbol, order.limit_price)
        print(f"✓ LULD check passed for {order.symbol} at ${order.limit_price}")

        # SSR check (for shorts)
        guard.check_ssr(order.symbol, order.side, now)
        print(f"✓ SSR check passed for {order.side} {order.symbol}")

        # PDT check
        guard.check_pdt(account_equity, pending_day_trades, now)
        print(f"✓ PDT check passed (equity: ${account_equity:,.0f}, day trades: {pending_day_trades})")

        # Borrow check (if shorting)
        if order.side == "short":
            borrow_bps = guard_can_short(borrow_client, order.symbol, order.qty)
            print(f"✓ Short locate available at {borrow_bps} bps")

        print(f"✓ All pre-trade checks passed for {order.side} {order.qty} {order.symbol}")

    except ComplianceError as e:
        print(f"✗ Compliance violation: {e}")
        return False

    return True

def example_options_risk_management():
    """Example: Options assignment risk management."""
    print("\n=== OPTIONS ASSIGNMENT RISK MANAGEMENT ===")

    # Sample option contracts
    call_option = OptionContract(
        symbol="AAPL 2025-01-17 180 C",
        underlying="AAPL",
        strike=180.0,
        right="C",
        expiry=dt.date.today()  # Expiring today
    )

    # Underlying state with dividend coming
    underlying = UnderlyingState(
        price=185.0,
        borrow_bps=25.0,
        next_ex_div_date=dt.date.today() + dt.timedelta(days=1),
        div_amount=2.0
    )

    # Check assignment risks
    auto_exercise = auto_exercise_likely(call_option, underlying)
    early_assignment = early_assignment_risk(call_option, underlying)
    pin_risk_flag = pin_risk(call_option, underlying, band_bps=20.0)

    print(f"Option: {call_option.symbol}")
    print(f"Underlying price: ${underlying.price}")
    print(f"Auto-exercise likely: {auto_exercise}")
    print(f"Early assignment risk: {early_assignment}")
    print(f"Pin risk: {pin_risk_flag}")

    # Risk management actions
    if auto_exercise and call_option.expiry == dt.date.today():
        print("⚠️  ACTION: ITM option will be auto-exercised at expiry")

    if early_assignment:
        print("⚠️  ACTION: Consider closing short calls before ex-dividend")

    if pin_risk_flag:
        print("⚠️  ACTION: Consider flattening deltas or closing positions")

def example_wash_sale_tracking():
    """Example: Tax lot tracking with wash sale detection."""
    print("\n=== WASH SALE TRACKING ===")

    # Initialize wash sale engine
    wash_engine = WashSaleEngine(window_days=30)

    # Sample trading activity
    fills = [
        Fill("AAPL", dt.datetime(2025, 1, 1), "buy", 100, 150.0),
        Fill("AAPL", dt.datetime(2025, 1, 10), "sell", 100, 140.0),  # Loss
        Fill("AAPL", dt.datetime(2025, 1, 15), "buy", 50, 145.0),   # Replacement within window
    ]

    for fill in fills:
        if fill.side == "buy":
            wash_engine.ingest(fill)
            print(f"Opened position: {fill.side} {fill.qty} {fill.symbol} @ ${fill.price}")
        else:
            realized, disallowed = wash_engine.realize(fill)
            print(f"Closed position: {fill.side} {fill.qty} {fill.symbol} @ ${fill.price}")
            print(f"  Realized P&L: ${realized:.2f}")
            if disallowed > 0:
                print(f"  Wash sale disallowed: ${disallowed:.2f}")
            else:
                print("  No wash sale detected")

def example_corporate_actions():
    """Example: Historical data adjustment for backtesting."""
    print("\n=== CORPORATE ACTIONS ADJUSTMENT ===")

    # Sample price data
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    bars = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        'low': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
    }, index=dates)

    print("Original price data (last 3 days):")
    print(bars[['close']].tail(3))

    # Define corporate actions
    actions = [
        CorporateAction("split", pd.Timestamp('2025-01-05'), factor=2.0, amount=0.0),  # 2:1 split
        CorporateAction("div", pd.Timestamp('2025-01-07'), factor=0.0, amount=1.0),    # $1 dividend
    ]

    # Apply adjustments
    adjuster = CorporateActionsAdjuster(actions)
    adjusted = adjuster.adjust(bars)

    print("\nAdjusted price data (last 3 days):")
    print(adjusted[['close', 'tr_close', 'split_adj_factor']].tail(3))

def example_universe_management():
    """Example: Point-in-time universe membership."""
    print("\n=== POINT-IN-TIME UNIVERSE MANAGEMENT ===")

    # Define historical S&P 500 membership changes (simplified example)
    sp500_members = [
        Membership("AAPL", dt.date(2020, 1, 1), dt.date(2025, 12, 31)),
        Membership("MSFT", dt.date(2020, 1, 1), dt.date(2025, 12, 31)),
        Membership("TSLA", dt.date(2020, 12, 21), dt.date(2025, 12, 31)),  # Added Dec 2020
        Membership("FB", dt.date(2020, 1, 1), dt.date(2021, 10, 28)),      # Became META
        Membership("META", dt.date(2021, 10, 29), dt.date(2025, 12, 31)),   # Name change
    ]

    universe = UniverseProvider({"SP500": sp500_members})

    # Check membership at different dates
    test_dates = [
        dt.date(2020, 6, 1),   # Before Tesla inclusion
        dt.date(2021, 6, 1),   # After Tesla, before Meta rename
        dt.date(2022, 1, 1),   # After Meta rename
    ]

    for test_date in test_dates:
        members = universe.members("SP500", test_date)
        print(f"S&P 500 members as of {test_date}: {len(members)} stocks")
        tech_stocks = [s for s in members if s in ["AAPL", "MSFT", "TSLA", "FB", "META"]]
        print(f"  Tech stocks: {tech_stocks}")

def example_portfolio_risk_rules():
    """Example: Portfolio construction risk rules."""
    print("\n=== PORTFOLIO CONSTRUCTION RISK RULES ===")

    # Sample portfolio weights
    weights = {
        "AAPL": 0.15,    # Technology
        "MSFT": 0.12,    # Technology
        "GOOGL": 0.08,   # Technology
        "JPM": 0.10,     # Financial
        "BAC": 0.08,     # Financial
        "JNJ": 0.12,     # Healthcare
        "PFE": 0.09,     # Healthcare
        "XOM": 0.11,     # Energy
        "CVX": 0.15,     # Energy
    }

    # Sector mapping
    sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
        "JPM": "Financial", "BAC": "Financial",
        "JNJ": "Healthcare", "PFE": "Healthcare",
        "XOM": "Energy", "CVX": "Energy"
    }

    # Check sector concentration (max 35%)
    sector_ok = sector_cap_check(weights, sectors, cap=0.35)
    print(f"Sector concentration check (≤35%): {'✓ PASS' if sector_ok else '✗ FAIL'}")

    # Calculate sector exposures
    sector_weights = {}
    total_weight = sum(abs(w) for w in weights.values())
    for sym, weight in weights.items():
        sector = sectors[sym]
        sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)

    print("Sector exposures:")
    for sector, weight in sector_weights.items():
        pct = (weight / total_weight) * 100
        print(f"  {sector}: {pct:.1f}%")

    # Mock correlation check with random returns
    np.random.seed(42)  # For reproducible example
    returns = np.random.randn(252, len(weights)) * 0.02  # Daily returns
    corr_ok = simple_corr_guard(returns, threshold=0.9)
    print(f"Correlation check (≤90%): {'✓ PASS' if corr_ok else '✗ FAIL'}")

def example_operational_safety():
    """Example: Runtime safety with journaling and clock checks."""
    print("\n=== OPERATIONAL SAFETY ===")

    # Check clock drift
    try:
        assert_ntp_ok(max_drift_ms=100)
        print("✓ Clock drift check passed")
    except ClockDriftError as e:
        print(f"✗ Clock drift detected: {e}")

    # Initialize journal
    journal = Journal(path="./.state/example_journal.jsonl")

    # Log trading decisions
    decision_id = journal.append({
        "timestamp": dt.datetime.utcnow().isoformat(),
        "event": "trade_signal",
        "symbol": "SPY",
        "action": "buy",
        "quantity": 100,
        "reason": "momentum_breakout",
        "confidence": 0.75
    })

    print(f"✓ Logged trading decision: {decision_id}")

    # Simulate restart and replay
    print("Simulating system restart and journal replay...")
    replay_records = journal.replay_from_cursor()

    for record in replay_records:
        print(f"  Replaying: {record['event']} for {record.get('symbol', 'N/A')}")

def main():
    """Run all integration examples."""
    print("WallStreetBots Domain Integration Examples")
    print("=" * 50)

    # Run all examples
    try:
        example_pre_trade_pipeline()
        example_options_risk_management()
        example_wash_sale_tracking()
        example_corporate_actions()
        example_universe_management()
        example_portfolio_risk_rules()
        example_operational_safety()

        print("\n" + "=" * 50)
        print("✓ All integration examples completed successfully!")

    except Exception as e:
        print(f"\n✗ Integration example failed: {e}")
        raise

if __name__ == "__main__":
    main()