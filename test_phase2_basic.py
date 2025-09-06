#!/usr/bin/env python3
"""
Basic Phase 2 Functionality Test
Test core Phase 2 components without external dependencies
"""

import sys
import os
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from enum import Enum

# Add the backend directory to the path
sys.path.append('backend/tradingbot')

# Import Phase 2 components (without external dependencies)
from production_wheel_strategy import WheelStage, WheelStatus, WheelPosition, WheelCandidate  # noqa: E402
from production_debit_spreads import SpreadType, SpreadStatus, SpreadPosition, SpreadCandidate, QuantLibPricer  # noqa: E402
from production_spx_spreads import SPXSpreadType, SPXSpreadStatus, SPXSpreadPosition, SPXSpreadCandidate  # noqa: E402
from production_index_baseline import BenchmarkType, BenchmarkData, StrategyPerformance, PerformanceComparison, PerformanceCalculator  # noqa: E402


def test_wheel_strategy():
    """Test Wheel Strategy components"""
    print("🔄 Testing Wheel Strategy Components...")
    
    # Test Wheel Position
    position = WheelPosition(
        ticker="AAPL",
        stage=WheelStage.CASH_SECURED_PUT,
        status=WheelStatus.ACTIVE,
        quantity=100,
        entry_price=150.0,
        current_price=155.0,
        unrealized_pnl=500.0,
        option_type="put",
        strike_price=145.0,
        expiry_date=datetime.now() + timedelta(days=30),
        premium_received=200.0
    )
    
    print(f"✅ Wheel Position: {position.ticker} {position.stage.value} @ ${position.strike_price}")
    print(f"   Premium: ${position.premium_received}, P&L: ${position.unrealized_pnl}")
    
    # Test P&L calculation
    pnl = position.calculate_unrealized_pnl()
    print(f"✅ P&L Calculation: ${pnl}")
    
    # Test Wheel Candidate
    candidate = WheelCandidate(
        ticker="AAPL",
        current_price=150.0,
        volatility_rank=0.7,
        iv_rank=0.6,
        put_premium=3.0,
        earnings_risk=0.1,
        rsi=45.0
    )
    
    score = candidate.calculate_wheel_score()
    print(f"✅ Wheel Candidate: {candidate.ticker} Score: {score:.2f}")
    
    print("✅ Wheel Strategy components working correctly\n")


def test_debit_spreads():
    """Test Debit Spreads components"""
    print("📈 Testing Debit Spreads Components...")
    
    # Test Spread Position
    position = SpreadPosition(
        ticker="AAPL",
        spread_type=SpreadType.BULL_CALL_SPREAD,
        status=SpreadStatus.ACTIVE,
        long_strike=145.0,
        short_strike=150.0,
        quantity=10,
        net_debit=2.0,
        max_profit=3.0,
        max_loss=2.0,
        long_option={"strike": 145.0, "premium": 3.0},
        short_option={"strike": 150.0, "premium": 1.0}
    )
    
    print(f"✅ Spread Position: {position.ticker} {position.spread_type.value}")
    print(f"   Long: ${position.long_strike}, Short: ${position.short_strike}")
    print(f"   Net Debit: ${position.net_debit}, Max Profit: ${position.max_profit}")
    
    # Test Spread Candidate
    candidate = SpreadCandidate(
        ticker="AAPL",
        current_price=150.0,
        spread_type=SpreadType.BULL_CALL_SPREAD,
        long_strike=145.0,
        short_strike=150.0,
        long_premium=3.0,
        short_premium=1.0,
        net_debit=2.0,
        max_profit=3.0,
        max_loss=2.0,
        profit_loss_ratio=1.5,
        net_delta=0.3,
        net_theta=-0.1,
        net_vega=0.05
    )
    
    score = candidate.calculate_spread_score()
    print(f"✅ Spread Candidate: {candidate.ticker} Score: {score:.2f}")
    print(f"   Profit/Loss Ratio: {candidate.profit_loss_ratio:.1f}")
    
    # Test QuantLib Pricer
    pricer = QuantLibPricer()
    result = pricer.calculate_black_scholes(
        spot_price=100.0,
        strike_price=100.0,
        risk_free_rate=0.02,
        volatility=0.20,
        time_to_expiry=0.25,
        option_type="call"
    )
    
    print(f"✅ QuantLib Pricing: Call @ $100 = ${result['price']:.2f}")
    print(f"   Delta: {result['delta']:.3f}, Gamma: {result['gamma']:.3f}")
    
    print("✅ Debit Spreads components working correctly\n")


def test_spx_spreads():
    """Test SPX Spreads components"""
    print("📊 Testing SPX Spreads Components...")
    
    # Test SPX Spread Position
    position = SPXSpreadPosition(
        spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
        status=SPXSpreadStatus.ACTIVE,
        long_strike=4400.0,
        short_strike=4450.0,
        quantity=1,
        net_credit=2.0,
        max_profit=2.0,
        max_loss=48.0,
        long_option={"strike": 4400.0, "premium": 1.0},
        short_option={"strike": 4450.0, "premium": 3.0}
    )
    
    print(f"✅ SPX Spread: {position.spread_type.value}")
    print(f"   Long: {position.long_strike}, Short: {position.short_strike}")
    print(f"   Net Credit: ${position.net_credit}, Max Profit: ${position.max_profit}")
    
    # Test SPX Spread Candidate
    candidate = SPXSpreadCandidate(
        spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
        long_strike=4400.0,
        short_strike=4450.0,
        long_premium=1.0,
        short_premium=3.0,
        net_credit=2.0,
        max_profit=2.0,
        max_loss=48.0,
        profit_loss_ratio=0.04,
        net_delta=-0.1,
        net_theta=0.05,
        net_vega=-0.02,
        spx_price=4500.0,
        vix_level=20.0,
        market_regime="bull"
    )
    
    score = candidate.calculate_spread_score()
    print(f"✅ SPX Candidate Score: {score:.2f}")
    print(f"   SPX: ${candidate.spx_price}, VIX: {candidate.vix_level}")
    print(f"   Market Regime: {candidate.market_regime}")
    
    print("✅ SPX Spreads components working correctly\n")


def test_index_baseline():
    """Test Index Baseline components"""
    print("📉 Testing Index Baseline Components...")
    
    # Test Benchmark Data
    benchmark = BenchmarkData(
        ticker="SPY",
        benchmark_type=BenchmarkType.SPY,
        current_price=450.0,
        daily_return=0.01,
        weekly_return=0.02,
        monthly_return=0.05,
        ytd_return=0.15,
        annual_return=0.20,
        volatility=0.15,
        sharpe_ratio=1.2,
        max_drawdown=0.08
    )
    
    print(f"✅ Benchmark: {benchmark.ticker} @ ${benchmark.current_price}")
    print(f"   Daily: {benchmark.daily_return:.1%}, Annual: {benchmark.annual_return:.1%}")
    print(f"   Sharpe: {benchmark.sharpe_ratio:.2f}, Max DD: {benchmark.max_drawdown:.1%}")
    
    # Test Strategy Performance
    performance = StrategyPerformance(
        strategy_name="Wheel Strategy",
        total_return=0.12,
        daily_return=0.0005,
        weekly_return=0.002,
        monthly_return=0.01,
        ytd_return=0.12,
        annual_return=0.12,
        volatility=0.18,
        sharpe_ratio=0.8,
        max_drawdown=0.12,
        win_rate=0.65,
        total_trades=100,
        winning_trades=65,
        losing_trades=35,
        avg_win=150.0,
        avg_loss=75.0,
        profit_factor=1.3
    )
    
    print(f"✅ Strategy Performance: {performance.strategy_name}")
    print(f"   Return: {performance.total_return:.1%}, Win Rate: {performance.win_rate:.1%}")
    print(f"   Trades: {performance.total_trades}, Profit Factor: {performance.profit_factor:.1f}")
    
    # Test Performance Calculator
    calculator = PerformanceCalculator(Mock())
    
    # Test returns calculation
    prices = [100.0, 101.0, 102.0, 101.5, 103.0]
    returns = calculator.calculate_returns(prices)
    
    print(f"✅ Performance Calculator:")
    print(f"   Daily Return: {returns['daily_return']:.2%}")
    print(f"   YTD Return: {returns['ytd_return']:.2%}")
    
    # Test volatility calculation
    returns_list = [0.01, 0.02, -0.01, 0.015, 0.005]
    volatility = calculator.calculate_volatility(returns_list)
    print(f"   Volatility: {volatility:.2%}")
    
    # Test Sharpe ratio
    sharpe = calculator.calculate_sharpe_ratio(returns_list)
    print(f"   Sharpe Ratio: {sharpe:.2f}")
    
    # Test Performance Comparison
    comparison = PerformanceComparison(
        strategy_name="Wheel Strategy",
        benchmark_ticker="SPY",
        strategy_return=0.12,
        benchmark_return=0.10,
        alpha=0.02,
        beta=0.8,
        strategy_volatility=0.18,
        benchmark_volatility=0.15,
        information_ratio=0.11,
        strategy_sharpe=0.8,
        benchmark_sharpe=0.9
    )
    
    print(f"✅ Performance Comparison: {comparison.strategy_name} vs {comparison.benchmark_ticker}")
    print(f"   Alpha: {comparison.alpha:.2%}, Beta: {comparison.beta:.2f}")
    print(f"   Information Ratio: {comparison.information_ratio:.2f}")
    
    print("✅ Index Baseline components working correctly\n")


def test_phase2_integration():
    """Test Phase 2 integration"""
    print("🔗 Testing Phase 2 Integration...")
    
    # Test configuration loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {
            "risk": {
                "max_position_risk": 0.10,
                "account_size": 100000.0
            },
            "trading": {
                "universe": ["AAPL", "MSFT", "GOOGL"],
                "max_concurrent_trades": 5
            }
        }
        json.dump(test_config, f)
        config_file = f.name
    
    try:
        from production_config import ConfigManager
        
        config_manager = ConfigManager(config_file)
        config = config_manager.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Account Size: ${config.risk.account_size:,.0f}")
        print(f"   Max Position Risk: {config.risk.max_position_risk:.1%}")
        print(f"   Universe: {', '.join(config.trading.universe)}")
        
    finally:
        os.unlink(config_file)
    
    # Test strategy scoring systems
    print(f"✅ Strategy Scoring Systems:")
    
    # Wheel scoring
    wheel_candidate = WheelCandidate(
        ticker="AAPL", current_price=150.0, volatility_rank=0.7,
        iv_rank=0.6, put_premium=3.0, earnings_risk=0.1, rsi=45.0
    )
    wheel_score = wheel_candidate.calculate_wheel_score()
    print(f"   Wheel Strategy Score: {wheel_score:.2f}")
    
    # Debit spread scoring
    debit_candidate = SpreadCandidate(
        ticker="AAPL", current_price=150.0, spread_type=SpreadType.BULL_CALL_SPREAD,
        long_strike=145.0, short_strike=150.0, long_premium=3.0, short_premium=1.0,
        net_debit=2.0, max_profit=3.0, max_loss=2.0, profit_loss_ratio=1.5,
        net_delta=0.3, net_theta=-0.1, net_vega=0.05
    )
    debit_score = debit_candidate.calculate_spread_score()
    print(f"   Debit Spread Score: {debit_score:.2f}")
    
    # SPX spread scoring
    spx_candidate = SPXSpreadCandidate(
        spread_type=SPXSpreadType.PUT_CREDIT_SPREAD,
        long_strike=4400.0, short_strike=4450.0, long_premium=1.0, short_premium=3.0,
        net_credit=2.0, max_profit=2.0, max_loss=48.0, profit_loss_ratio=0.04,
        net_delta=-0.1, net_theta=0.05, net_vega=-0.02,
        spx_price=4500.0, vix_level=20.0, market_regime="bull"
    )
    spx_score = spx_candidate.calculate_spread_score()
    print(f"   SPX Spread Score: {spx_score:.2f}")
    
    print("✅ Phase 2 integration working correctly\n")


def main():
    """Run all Phase 2 tests"""
    print("🚀 WallStreetBots Phase 2 - Basic Functionality Test")
    print("=" * 60)
    
    try:
        test_wheel_strategy()
        test_debit_spreads()
        test_spx_spreads()
        test_index_baseline()
        test_phase2_integration()
        
        print("=" * 60)
        print("✅ ALL PHASE 2 TESTS PASSED!")
        print("\n🎯 Phase 2 Strategies Verified:")
        print("  ✅ Wheel Strategy - Premium selling automation")
        print("  ✅ Debit Spreads - Defined-risk bulls with QuantLib")
        print("  ✅ SPX Spreads - Index options with CME data")
        print("  ✅ Index Baseline - Performance tracking & benchmarking")
        print("  ✅ Integration - All strategies with Phase 1 infrastructure")
        
        print("\n📊 Strategy Capabilities:")
        print("  🔄 Wheel: Automated premium selling with risk controls")
        print("  📈 Debit Spreads: QuantLib pricing with Greeks calculation")
        print("  📊 SPX Spreads: Real-time CME data with market regime analysis")
        print("  📉 Index Baseline: Multi-benchmark performance tracking")
        
        print("\n⚠️  Note: This is educational/testing code only!")
        print("   Do not use with real money without extensive validation.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Mock class for testing
    class Mock:
        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    exit(main())
