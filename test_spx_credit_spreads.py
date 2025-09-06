#!/usr/bin/env python3
"""
Comprehensive Test Suite for SPX Credit Spreads WSB Strategy Module
Tests all components of the 0DTE credit spreads scanner and strategy logic
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

from spx_  # noqa: E402  # noqa: E402credit_spreads import (  # noqa: E402
    CreditSpreadOpportunity, SPXCreditSpreadsScanner
)


class TestSPXCreditSpreadsScanner(unittest.TestCase):
    """Test the SPX/SPY credit spread scanner functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.scanner = SPXCreditSpreadsScanner()

    def test_scanner_initialization(self):
        """Test that scanner initializes properly"""
        self.assertIsInstance(self.scanner, SPXCreditSpreadsScanner)
        self.assertIsInstance(self.scanner.credit_tickers, list)
        self.assertGreater(len(self.scanner.credit_tickers), 0)

    def test_norm_cdf(self):
        """Test normal CDF calculation"""
        result = self.scanner.norm_cdf(0.0)
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_black_scholes_put(self):
        """Test Black-Scholes put calculation"""
        price, delta = self.scanner.black_scholes_put(100.0, 95.0, 0.25, 0.05, 0.20)
        
        self.assertIsInstance(price, float)
        self.assertIsInstance(delta, float)
        self.assertGreater(price, 0)
        self.assertLess(delta, 0)  # Put delta should be negative

    def test_black_scholes_call(self):
        """Test Black-Scholes call calculation"""
        price, delta = self.scanner.black_scholes_call(100.0, 105.0, 0.25, 0.05, 0.20)
        
        self.assertIsInstance(price, float)
        self.assertIsInstance(delta, float)
        self.assertGreater(price, 0)
        self.assertGreater(delta, 0)  # Call delta should be positive

    def test_get_0dte_expiry(self):
        """Test 0DTE expiry calculation"""
        expiry = self.scanner.get_0dte_expiry()
        
        if expiry:  # Only test if market is open
            self.assertIsInstance(expiry, str)
            self.assertEqual(len(expiry), 10)  # YYYY-MM-DD format

    @patch('spx_credit_spreads.yf.Ticker')
    def test_estimate_iv_from_expected_move(self, mock_ticker):
        """Test IV estimation from expected move"""
        mock_stock = Mock()
        mock_stock.info = {'regularMarketPrice': 100.0}
        mock_ticker.return_value = mock_stock
        
        iv = self.scanner.estimate_iv_from_expected_move("SPY", 2.0)
        
        self.assertIsInstance(iv, float)
        self.assertGreater(iv, 0)

    @patch('spx_credit_spreads.yf.Ticker')
    def test_get_expected_move(self, mock_ticker):
        """Test expected move calculation"""
        mock_stock = Mock()
        mock_stock.info = {'regularMarketPrice': 100.0}
        mock_ticker.return_value = mock_stock
        
        move = self.scanner.get_expected_move("SPY")
        
        self.assertIsInstance(move, float)
        self.assertGreater(move, 0)

    def test_calculate_spread_metrics(self):
        """Test spread metrics calculation"""
        net_credit, max_profit, max_loss = self.scanner.calculate_spread_metrics(
            short_strike=4000.0,
            long_strike=3990.0,
            short_premium=0.30,
            long_premium=0.20
        )
        
        self.assertIsInstance(net_credit, float)
        self.assertIsInstance(max_profit, float)
        self.assertIsInstance(max_loss, float)
        
        self.assertAlmostEqual(net_credit, 0.10, places=2)
        self.assertAlmostEqual(max_profit, 0.10, places=2)
        self.assertAlmostEqual(max_loss, 9.90, places=2)

    def test_scan_credit_spreads(self):
        """Test credit spread scanning"""
        spreads = self.scanner.scan_credit_spreads(0)
        
        self.assertIsInstance(spreads, list)
        # Each spread should be a CreditSpreadOpportunity
        for spread in spreads:
            self.assertIsInstance(spread, CreditSpreadOpportunity)

    def test_format_opportunities(self):
        """Test opportunity formatting"""
        opportunities = [
            CreditSpreadOpportunity(
                ticker="SPY",
                strategy_type="put_credit_spread",
                expiry_date="2024-01-19",
                dte=0,
                short_strike=4000.0,
                long_strike=3990.0,
                spread_width=10.0,
                net_credit=0.30,
                max_profit=0.30,
                max_loss=9.70,
                short_delta=0.30,
                prob_profit=0.65,
                profit_target=0.075,
                break_even_lower=3999.70,
                break_even_upper=4000.30,
                iv_rank=45.0,
                underlying_price=4005.0,
                expected_move=2.0,
                volume_score=1.2
            )
        ]
        
        formatted = self.scanner.format_opportunities(opportunities)
        
        self.assertIsInstance(formatted, str)
        self.assertIn("SPY", formatted)
        self.assertIn("PUT SPREAD", formatted)


class TestCreditSpreadOpportunity(unittest.TestCase):
    """Test the CreditSpreadOpportunity dataclass"""

    def test_credit_spread_creation(self):
        """Test creating a credit spread opportunity"""
        spread = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="put_credit_spread",
            expiry_date="2024-01-19",
            dte=0,
            short_strike=4000.0,
            long_strike=3990.0,
            spread_width=10.0,
            net_credit=0.30,
            max_profit=0.30,
            max_loss=9.70,
            short_delta=0.30,
            prob_profit=0.65,
            profit_target=0.075,
            break_even_lower=3999.70,
            break_even_upper=4000.30,
            iv_rank=45.0,
            underlying_price=4005.0,
            expected_move=2.0,
            volume_score=1.2
        )
        
        self.assertEqual(spread.ticker, "SPY")
        self.assertEqual(spread.strategy_type, "put_credit_spread")
        self.assertEqual(spread.short_strike, 4000.0)
        self.assertEqual(spread.long_strike, 3990.0)
        self.assertEqual(spread.net_credit, 0.30)
        self.assertEqual(spread.max_profit, 0.30)
        self.assertEqual(spread.max_loss, 9.70)

    def test_spread_width_calculation(self):
        """Test spread width calculation"""
        spread = CreditSpreadOpportunity(
            ticker="SPY",
            strategy_type="put_credit_spread",
            expiry_date="2024-01-19",
            dte=0,
            short_strike=4000.0,
            long_strike=3990.0,
            spread_width=10.0,
            net_credit=0.30,
            max_profit=0.30,
            max_loss=9.70,
            short_delta=0.30,
            prob_profit=0.65,
            profit_target=0.075,
            break_even_lower=3999.70,
            break_even_upper=4000.30,
            iv_rank=45.0,
            underlying_price=4005.0,
            expected_move=2.0,
            volume_score=1.2
        )
        
        width = spread.short_strike - spread.long_strike
        self.assertEqual(width, 10.0)


class TestSPXCreditSpreadsIntegration(unittest.TestCase):
    """Test integration scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.scanner = SPXCreditSpreadsScanner()

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        from spx_credit_spreads import main
        self.assertTrue(callable(main))


if __name__ == '__main__':
    unittest.main()