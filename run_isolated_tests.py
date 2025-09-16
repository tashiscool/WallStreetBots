#!/usr/bin/env python3
"""Test runner for isolated tests."""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_isolated_test(test_path: str, verbose: bool = False) -> bool:
    """Run a single test in isolation."""
    cmd = [
        sys.executable, '-m', 'pytest',
        test_path,
        '--tb=short',
        '--isolated',  # Custom marker for isolated tests
    ]
    
    if verbose:
        cmd.append('-v')
    
    # Set environment variables for isolation
    env = os.environ.copy()
    env.update({
        'TEST_ISOLATION': 'true',
        'PYTHONPATH': os.pathsep.join(sys.path),
        'PYTEST_DISABLE_PLUGIN_AUTOLOAD': 'true',
    })
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running isolated test {test_path}: {e}")
        return False


def run_problematic_tests() -> None:
    """Run known problematic tests in isolation."""
    problematic_tests = [
        'tests/phases/test_phase1_integration.py::TestHealthChecker::test_health_check_execution',
        'tests/phases/test_phase2_comprehensive.py::TestWheelStrategy::test_wheel_strategy_scan_opportunities',
        'tests/phases/test_phase2_comprehensive.py::TestPhase2Integration::test_phase2_strategy_manager',
        'tests/phases/test_phase4_comprehensive.py::TestPhase4Backtesting::test_backtest_execution',
        'tests/phases/test_phase4_comprehensive.py::TestPhase4Deployment::test_deployment_process',
        'tests/production/test_production_data_integration_real_api.py::TestReliableDataProvider::test_get_options_chain_mocked',
        'tests/production/test_production_data_integration_real_api.py::TestReliableDataProvider::test_get_earnings_calendar_synthetic',
        'tests/strategies/test_earnings_protection_real_api.py::TestEarningsProtectionScanner::test_create_deep_itm_strategy_mocked',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_scan_strategy_performance_mocked',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_batch_strategy_comparison',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_real_time_comparison_simulation',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_performance_attribution',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_risk_adjusted_metrics',
        'tests/strategies/test_index_baseline_real_api.py::TestIndexBaselineScanner::test_market_regime_analysis',
        'tests/strategies/test_spx_credit_spreads_real_api.py::TestSPXCreditSpreadsScanner::test_fetch_options_data_mocked',
        'tests/strategies/test_swing_trading_real_api.py::TestSwingTradingScanner::test_detect_breakout_mocked',
        'tests/strategies/test_wsb_dip_bot.py::TestWSBDipBot::test_detect_eod_signal_function',
        'tests/test_options_calculator_real_api.py::TestOptionsTradeCalculatorRealAPI::test_multiple_symbols_trade_calculation',
        'tests/test_options_calculator_real_api.py::TestOptionsStrategySetupRealAPI::test_mega_cap_tickers_validation',
        'tests/test_options_calculator_real_api.py::TestIntegrationScenariosRealAPI::test_complete_trading_workflow_real_data',
        'tests/test_wheel_strategy_comprehensive.py::TestDataFetching::test_get_quality_score_mocked',
        'tests/test_wheel_strategy_comprehensive.py::TestDataFetching::test_get_quality_score_error_handling',
        'tests/test_wheel_strategy_comprehensive.py::TestPositionUpdates::test_update_positions_mocked',
        'tests/test_wsb_dip_bot_comprehensive.py::TestDataFetching::test_fetch_daily_history_mocked',
        'tests/test_wsb_dip_bot_comprehensive.py::TestDataFetching::test_fetch_intraday_last_and_prior_close_mocked',
        'tests/test_wsb_dip_bot_comprehensive.py::TestOptionsDataFetching::test_get_option_mid_for_nearest_5pct_otm_mocked',
    ]
    
    print("Running problematic tests in isolation...")
    passed = 0
    failed = 0
    
    for test in problematic_tests:
        print(f"\nRunning isolated test: {test}")
        if run_isolated_test(test, verbose=True):
            print(f"✅ PASSED: {test}")
            passed += 1
        else:
            print(f"❌ FAILED: {test}")
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run isolated tests')
    parser.add_argument('--test', help='Run specific test')
    parser.add_argument('--problematic', action='store_true', help='Run known problematic tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_isolated_test(args.test, args.verbose)
        sys.exit(0 if success else 1)
    elif args.problematic:
        success = run_problematic_tests()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
