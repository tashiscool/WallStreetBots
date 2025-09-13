#!/usr / bin/env python3
"""
Smoke Tests for Trading Strategies
Simple tests to ensure strategies don't crash and basic functionality works
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock yfinance and pandas for strategy scripts
sys.modules['yfinance'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()

from backend.tradingbot.strategies.momentum_weeklies import MomentumWeekliesScanner, MomentumSignal  # noqa: E402
from backend.tradingbot.strategies.debit_spreads import DebitSpreadScanner, SpreadOpportunity  # noqa: E402
from backend.tradingbot.strategies.leaps_tracker import LEAPSTracker, LEAPSPosition  # noqa: E402
from backend.tradingbot.strategies.lotto_scanner import LottoScanner, LottoPlay  # noqa: E402
from backend.tradingbot.strategies.wheel_strategy import WheelStrategy, WheelPosition  # noqa: E402
from backend.tradingbot.strategies.wsb_dip_bot import DipSignal  # noqa: E402


class TestStrategySmokeTests(unittest.TestCase): 
    """Basic smoke tests to ensure strategies don't crash"""
    
    def test_momentum_weeklies_initialization(self): 
        """Test momentum weeklies scanner initializes"""
        scanner = MomentumWeekliesScanner()
        self.assertIsInstance(scanner, MomentumWeekliesScanner)
        self.assertIsInstance(scanner.mega_caps, list)
        self.assertGreater(len(scanner.mega_caps), 0)
        
    def test_momentum_weeklies_expiry_calculation(self): 
        """Test weekly expiry calculation"""
        scanner = MomentumWeekliesScanner()
        expiry = scanner.get_next_weekly_expiry()
        self.assertIsInstance(expiry, str)
        self.assertEqual(len(expiry), 10)  # YYYY - MM-DD format
        
    def test_momentum_signal_creation(self): 
        """Test MomentumSignal dataclass"""
        signal = MomentumSignal(
            ticker = "AAPL",
            signal_time = datetime.now(),
            current_price = 150.0,
            reversal_type = "bullish_reversal",
            volume_spike = 2.0,
            price_momentum = 5.0,
            weekly_expiry = "2024 - 01-19",
            target_strike = 155,
            premium_estimate = 2.50,
            risk_level = "medium",
            exit_target = 160.0,
            stop_loss = 145.0
        )
        self.assertEqual(signal.ticker, "AAPL")
        self.assertEqual(signal.current_price, 150.0)
        
    def test_debit_spreads_initialization(self): 
        """Test debit spreads scanner initializes"""
        scanner = DebitSpreadScanner()
        self.assertIsInstance(scanner, DebitSpreadScanner)
        
    def test_debit_spreads_black_scholes(self): 
        """Test Black - Scholes calculation"""
        scanner = DebitSpreadScanner()
        price, delta = scanner.black_scholes_call(100.0, 105.0, 0.25, 0.05, 0.20)
        self.assertIsInstance(price, float)
        self.assertIsInstance(delta, float)
        self.assertGreater(price, 0)
        self.assertGreater(delta, 0)
        
    def test_spread_opportunity_creation(self): 
        """Test SpreadOpportunity dataclass"""
        from datetime import date
        opportunity = SpreadOpportunity(
            ticker = "SPY",
            scan_date = date.today(),
            spot_price = 400.0,
            trend_strength = 0.8,
            expiry_date = "2024 - 01-19",
            days_to_expiry = 30,
            long_strike = 395,
            short_strike = 400,
            spread_width = 5,
            long_premium = 2.50,
            short_premium = 1.50,
            net_debit = 1.00,
            max_profit = 4.00,
            max_profit_pct = 4.0,
            breakeven = 396.0,
            prob_profit = 0.65,
            risk_reward = 4.0,
            iv_rank = 45.0,
            volume_score = 1.2
        )
        self.assertEqual(opportunity.ticker, "SPY")
        self.assertEqual(opportunity.trend_strength, 0.8)
        
    def test_leaps_tracker_initialization(self): 
        """Test LEAPS tracker initializes"""
        tracker = LEAPSTracker()
        self.assertIsInstance(tracker, LEAPSTracker)
        
    def test_leaps_position_creation(self): 
        """Test LEAPSPosition dataclass"""
        position = LEAPSPosition(
            ticker = "AAPL",
            theme = "AI Revolution",
            entry_date = date(2024, 1, 15),
            expiry_date = "2025 - 01-17",
            strike = 150,
            entry_premium = 15.0,
            current_premium = 16.50,
            spot_at_entry = 145.0,
            current_spot = 150.0,
            contracts = 10,
            cost_basis = 15000.0,
            current_value = 16500.0,
            unrealized_pnl = 1500.0,
            unrealized_pct = 10.0,
            days_held = 100,
            days_to_expiry = 200,
            delta = 0.65,
            profit_target_hit = False,
            stop_loss_hit = False,
            scale_out_level = 0
        )
        self.assertEqual(position.ticker, "AAPL")
        self.assertEqual(position.strike, 150)
        
    def test_lotto_scanner_initialization(self): 
        """Test lotto scanner initializes"""
        scanner = LottoScanner()
        self.assertIsInstance(scanner, LottoScanner)
        
    def test_lotto_play_creation(self): 
        """Test LottoPlay dataclass"""
        play = LottoPlay(
            ticker = "TSLA",
            play_type = "0dte",
            expiry_date = "2024 - 01-19",
            days_to_expiry = 0,
            strike = 200,
            option_type = "call",
            current_premium = 1.50,
            breakeven = 201.50,
            current_spot = 200.0,
            catalyst_event = "earnings",
            expected_move = 0.08,
            max_position_size = 1000.0,
            max_contracts = 6,
            risk_level = "extreme",
            win_probability = 0.15,
            potential_return = 2.0,
            stop_loss_price = 0.75,
            profit_target_price = 4.50
        )
        self.assertEqual(play.ticker, "TSLA")
        self.assertEqual(play.play_type, "0dte")
        
    def test_wheel_strategy_initialization(self): 
        """Test wheel strategy initializes"""
        strategy = WheelStrategy()
        self.assertIsInstance(strategy, WheelStrategy)
        
    def test_wheel_position_creation(self): 
        """Test WheelPosition dataclass"""
        position = WheelPosition(
            ticker = "AAPL",
            position_type = "cash_secured_put",
            shares = 0,
            avg_cost = 0.0,
            strike = 150,
            expiry = "2024 - 01-19",
            premium_collected = 2.50,
            days_to_expiry = 5,
            current_price = 149.0,
            unrealized_pnl = 125.0,
            total_premium_collected = 7.50,
            assignment_risk = 0.75,
            annualized_return = 0.12,
            status = "active"
        )
        self.assertEqual(position.ticker, "AAPL")
        self.assertEqual(position.strike, 150)
        
    def test_dip_signal_creation(self): 
        """Test DipSignal class"""
        # Use the correct DipSignal fields from wsb_dip_bot.py
        signal = DipSignal(
            ticker = "AAPL",
            ts_ny = "2024 - 01-19 15: 30: 00",
            spot = 150.0,
            prior_close = 155.0,
            intraday_pct = -3.2,
            run_lookback = 10,
            run_return = 25.0
        )
        self.assertEqual(signal.ticker, "AAPL")
        self.assertEqual(signal.spot, 150.0)


class TestStrategyBasicFunctionality(unittest.TestCase): 
    """Test basic functionality of strategies"""
    
    def test_momentum_weeklies_volume_detection(self): 
        """Test volume spike detection"""
        scanner = MomentumWeekliesScanner()
        
        with patch('backend.tradingbot.strategies.momentum_weeklies.yf.Ticker') as mock_ticker: 
            mock_stock = Mock()
            mock_stock.history.return_value = pd.DataFrame({
                'Volume': [1000000, 1200000, 1100000, 1300000, 1000000, 2000000]
            })
            mock_ticker.return_value = mock_stock
            
            result = scanner.detect_volume_spike("AAPL")
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            
    def test_momentum_weeklies_reversal_detection(self): 
        """Test reversal pattern detection"""
        scanner = MomentumWeekliesScanner()
        
        with patch('backend.tradingbot.strategies.momentum_weeklies.yf.Ticker') as mock_ticker: 
            mock_stock = Mock()
            mock_stock.history.return_value = pd.DataFrame({
                'Close': [100, 98, 96, 94, 92, 90, 88, 90, 92, 94, 96, 98, 100],
                'Volume': [1000000] * 13
            })
            mock_ticker.return_value = mock_stock
            
            result = scanner.detect_reversal_pattern("AAPL")
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            
    def test_debit_spreads_iv_calculation(self): 
        """Test IV rank calculation"""
        scanner = DebitSpreadScanner()
        
        with patch('backend.tradingbot.strategies.debit_spreads.yf.Ticker') as mock_ticker: 
            mock_stock = Mock()
            mock_stock.history.return_value = pd.DataFrame({
                'Close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
            })
            mock_ticker.return_value = mock_stock
            
            iv_rank = scanner.calculate_iv_rank("SPY", 0.20)
            self.assertIsInstance(iv_rank, (int, float))
            self.assertGreaterEqual(iv_rank, 0.0)
            self.assertLessEqual(iv_rank, 100.0)
            
    def test_debit_spreads_trend_assessment(self): 
        """Test trend strength assessment"""
        scanner = DebitSpreadScanner()
        
        with patch('backend.tradingbot.strategies.debit_spreads.yf.Ticker') as mock_ticker: 
            mock_stock = Mock()
            mock_stock.history.return_value = pd.DataFrame({
                'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
            })
            mock_ticker.return_value = mock_stock
            
            trend_strength = scanner.assess_trend_strength("SPY")
            self.assertIsInstance(trend_strength, float)
            self.assertGreaterEqual(trend_strength, -1.0)
            self.assertLessEqual(trend_strength, 1.0)


class TestStrategyErrorHandling(unittest.TestCase): 
    """Test error handling in strategies"""
    
    def test_momentum_weeklies_error_handling(self): 
        """Test error handling in momentum weeklies"""
        scanner = MomentumWeekliesScanner()
        
        with patch('backend.tradingbot.strategies.momentum_weeklies.yf.Ticker') as mock_ticker: 
            mock_ticker.side_effect = Exception("Network error")
            
            # Should handle gracefully without crashing
            result = scanner.detect_volume_spike("INVALID")
            self.assertIsInstance(result, tuple)
            
    def test_debit_spreads_error_handling(self): 
        """Test error handling in debit spreads"""
        scanner = DebitSpreadScanner()
        
        with patch('backend.tradingbot.strategies.debit_spreads.yf.Ticker') as mock_ticker: 
            mock_ticker.side_effect = Exception("Network error")
            
            # Should handle gracefully without crashing
            result = scanner.calculate_iv_rank("INVALID", 0.20)
            self.assertIsInstance(result, float)


def run_smoke_tests(): 
    """Run all smoke tests"""
    print(" = " * 60)
    print("TRADING STRATEGY SMOKE TESTS")
    print(" = " * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestStrategySmokeTests,
        TestStrategyBasicFunctionality,
        TestStrategyErrorHandling
    ]
    
    for test_class in test_classes: 
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + " = " * 60)
    print("SMOKE TEST SUMMARY")
    print(" = " * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if result.failures: 
        print("\nFAILURES: ")
        for test, traceback in result.failures: 
            print(f"  - {test}")
            print(f"    {traceback.split('AssertionError: ')[-1].strip()}")
    
    if result.errors: 
        print("\nERRORS: ")
        for test, traceback in result.errors: 
            print(f"  - {test}")
            print(f"    {str(traceback).split('Exception: ')[-1].strip()}")
    
    return result


if __name__ ==  "__main__": run_smoke_tests()
