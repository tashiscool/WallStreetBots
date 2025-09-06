#!/usr/bin/env python3
"""
Comprehensive Test Suite for Swing Trading WSB Strategy Module
Tests all components of the swing trading scanner and strategy logic
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.tradingbot.strategies.swing_trading import (  # noqa: E402
    SwingSignal, SwingTradingScanner, ActiveSwingTrade
)


class TestSwingTradingScanner(unittest.TestCase):
    """Test the swing trading scanner functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.scanner = SwingTradingScanner()
        
        # Create mock price data
        self.mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        })

    def test_scanner_initialization(self):
        """Test that scanner initializes properly"""
        self.assertIsInstance(self.scanner, SwingTradingScanner)
        self.assertIsInstance(self.scanner.swing_tickers, list)
        self.assertGreater(len(self.scanner.swing_tickers), 0)

    @patch('backend.tradingbot.strategies.swing_trading.yf.Ticker')
    def test_detect_breakout(self, mock_ticker):
        """Test breakout detection"""
        mock_stock = Mock()
        mock_stock.history.return_value = self.mock_data
        mock_ticker.return_value = mock_stock
        
        result = self.scanner.detect_breakout("AAPL")
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        is_breakout, strength, volume_ratio = result
        self.assertIsInstance(is_breakout, bool)
        self.assertIsInstance(strength, float)
        self.assertIsInstance(volume_ratio, float)

    @patch('backend.tradingbot.strategies.swing_trading.yf.download')
    def test_detect_momentum_continuation(self, mock_download):
        """Test momentum continuation detection"""
        mock_download.return_value = self.mock_data
        
        is_momentum, strength = self.scanner.detect_momentum_continuation("AAPL")
        
        self.assertIsInstance(is_momentum, bool)
        self.assertIsInstance(strength, float)

    @patch('backend.tradingbot.strategies.swing_trading.yf.download')
    def test_detect_reversal_setup(self, mock_download):
        """Test reversal setup detection"""
        mock_download.return_value = self.mock_data
        
        is_reversal, setup_type, strength = self.scanner.detect_reversal_setup("AAPL")
        
        self.assertIsInstance(is_reversal, bool)
        self.assertIsInstance(setup_type, str)
        self.assertIsInstance(strength, float)

    def test_get_optimal_expiry(self):
        """Test optimal expiry calculation"""
        expiry = self.scanner.get_optimal_expiry(30)
        
        self.assertIsInstance(expiry, str)
        # Should be a date string in YYYY-MM-DD format
        self.assertEqual(len(expiry), 10)
        self.assertEqual(expiry.count('-'), 2)

    def test_calculate_option_targets(self):
        """Test option target calculations"""
        targets = self.scanner.calculate_option_targets(100.0, 105, 2.50)
        
        self.assertEqual(len(targets), 4)
        profit_25, profit_50, profit_100, stop_loss = targets
        
        self.assertIsInstance(profit_25, float)
        self.assertIsInstance(profit_50, float)
        self.assertIsInstance(profit_100, float)
        self.assertIsInstance(stop_loss, float)
        
        # Profit targets should be positive
        self.assertGreater(profit_25, 0)
        self.assertGreater(profit_50, 0)
        self.assertGreater(profit_100, 0)
        
        # Stop loss should be less than original premium (30% loss)
        self.assertLess(stop_loss, 2.50)

    @patch('backend.tradingbot.strategies.swing_trading.yf.Ticker')
    def test_estimate_swing_premium(self, mock_ticker):
        """Test swing premium estimation"""
        mock_options = Mock()
        mock_options.calls = Mock()
        mock_options.calls.return_value = pd.DataFrame({
            'strike': [105, 110, 115],
            'lastPrice': [2.50, 1.50, 0.80]
        })
        mock_ticker.return_value.options = mock_options
        
        premium = self.scanner.estimate_swing_premium("AAPL", 105, "2024-01-19")
        
        self.assertIsInstance(premium, float)
        self.assertGreater(premium, 0)

    @patch('backend.tradingbot.strategies.swing_trading.yf.download')
    def test_scan_swing_opportunities(self, mock_download):
        """Test scanning for swing opportunities"""
        mock_download.return_value = self.mock_data
        
        opportunities = self.scanner.scan_swing_opportunities()
        
        self.assertIsInstance(opportunities, list)
        # Each opportunity should be a SwingSignal
        for opp in opportunities:
            self.assertIsInstance(opp, SwingSignal)

    def test_format_signals(self):
        """Test signal formatting"""
        signals = [
            SwingSignal(
                ticker="AAPL",
                signal_time=datetime.now(),
                signal_type="breakout",
                entry_price=100.0,
                breakout_level=105.0,
                volume_confirmation=1.5,
                strength_score=75.0,
                target_strike=105,
                target_expiry="2024-01-19",
                option_premium=2.50,
                max_hold_hours=24,
                profit_target_1=0.25,
                profit_target_2=0.50,
                profit_target_3=1.00,
                stop_loss=-0.20,
                risk_level="medium"
            )
        ]
        
        formatted = self.scanner.format_signals(signals)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("AAPL", formatted)
        self.assertIn("breakout", formatted)


class TestSwingSignal(unittest.TestCase):
    """Test the SwingSignal dataclass"""

    def test_swing_signal_creation(self):
        """Test creating a swing signal"""
        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=100.0,
            breakout_level=105.0,
            volume_confirmation=1.5,
            strength_score=75.0,
            target_strike=105,
            target_expiry="2024-01-19",
            option_premium=2.50,
            max_hold_hours=24,
            profit_target_1=0.25,
            profit_target_2=0.50,
            profit_target_3=1.00,
            stop_loss=-0.20,
            risk_level="medium"
        )
        
        self.assertEqual(signal.ticker, "AAPL")
        self.assertEqual(signal.signal_type, "breakout")
        self.assertEqual(signal.entry_price, 100.0)
        self.assertEqual(signal.target_strike, 105)
        self.assertEqual(signal.risk_level, "medium")


class TestActiveSwingTrade(unittest.TestCase):
    """Test the ActiveSwingTrade dataclass"""

    def test_active_trade_creation(self):
        """Test creating an active swing trade"""
        signal = SwingSignal(
            ticker="AAPL",
            signal_time=datetime.now(),
            signal_type="breakout",
            entry_price=100.0,
            breakout_level=105.0,
            volume_confirmation=1.5,
            strength_score=75.0,
            target_strike=105,
            target_expiry="2024-01-19",
            option_premium=2.50,
            max_hold_hours=24,
            profit_target_1=0.25,
            profit_target_2=0.50,
            profit_target_3=1.00,
            stop_loss=-0.20,
            risk_level="medium"
        )
        
        trade = ActiveSwingTrade(
            signal=signal,
            entry_time=datetime.now(),
            entry_premium=2.50,
            current_premium=2.75,
            unrealized_pnl=0.25,
            unrealized_pct=0.10,
            hours_held=2.0,
            hit_profit_target=0,
            should_exit=False,
            exit_reason=""
        )
        
        self.assertEqual(trade.signal.ticker, "AAPL")
        self.assertEqual(trade.entry_premium, 2.50)
        self.assertEqual(trade.unrealized_pnl, 0.25)


class TestSwingTradingIntegration(unittest.TestCase):
    """Test integration scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.scanner = SwingTradingScanner()

    @patch('backend.tradingbot.strategies.swing_trading.yf.download')
    def test_error_handling_bad_ticker(self, mock_download):
        """Test error handling with bad ticker"""
        mock_download.side_effect = Exception("Ticker not found")
        
        # Should not crash, should handle gracefully
        is_breakout, strength, volume_ratio = self.scanner.detect_breakout("INVALID")
        
        self.assertFalse(is_breakout)
        self.assertEqual(strength, 0.0)
        self.assertEqual(volume_ratio, 0.0)

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        from backend.tradingbot.strategies.swing_trading import main
        self.assertTrue(callable(main))


if __name__ == '__main__':
    unittest.main()