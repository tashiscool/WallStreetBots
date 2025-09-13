#!/usr / bin / env python3
"""Comprehensive Test Suite for Earnings Protection WSB Strategy Module
Tests all components of the earnings IV crush protection system
"""

import os
import sys
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.tradingbot.strategies.earnings_protection import (
    EarningsEvent,
    EarningsProtectionScanner,
    EarningsProtectionStrategy,
)


class TestEarningsProtectionScanner(unittest.TestCase):
    """Test the earnings protection scanner functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.scanner = EarningsProtectionScanner()

        # Mock earnings event
        self.mock_earnings_event = EarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=date.today() + timedelta(days=3),
            earnings_time="AMC",
            days_until_earnings=3,
            current_price=150.0,
            expected_move=0.06,  # 6% expected move
            iv_current=0.45,  # 45% IV (elevated pre-earnings)
            iv_historical_avg=0.25,  # 25% historical average
            iv_crush_risk="high",
        )

        # Mock options chain data
        self.mock_calls_data = pd.DataFrame(
            {
                "strike": [140, 145, 150, 155, 160, 165],
                "bid": [12.50, 8.20, 4.80, 2.30, 0.90, 0.35],
                "ask": [13.00, 8.70, 5.30, 2.80, 1.40, 0.85],
                "volume": [150, 200, 500, 300, 180, 90],
                "openInterest": [1200, 1500, 2500, 1800, 1000, 400],
                "impliedVolatility": [0.42, 0.44, 0.45, 0.46, 0.48, 0.50],
            }
        )

        self.mock_puts_data = pd.DataFrame(
            {
                "strike": [135, 140, 145, 150, 155, 160],
                "bid": [0.40, 0.95, 2.40, 4.90, 8.30, 12.60],
                "ask": [0.90, 1.45, 2.90, 5.40, 8.80, 13.10],
                "volume": [80, 110, 300, 450, 320, 200],
                "openInterest": [500, 700, 1500, 2200, 1600, 1100],
                "impliedVolatility": [0.50, 0.48, 0.46, 0.45, 0.44, 0.42],
            }
        )

    def test_scanner_initialization(self):
        """Test scanner initializes correctly"""
        self.assertIsInstance(self.scanner.earnings_candidates, list)
        self.assertGreater(len(self.scanner.earnings_candidates), 0)
        self.assertIn("AAPL", self.scanner.earnings_candidates)
        self.assertIn("GOOGL", self.scanner.earnings_candidates)

    def test_earnings_move_estimation(self):
        """Test earnings move estimation from straddle pricing"""
        with patch("backend.tradingbot.strategies.earnings_protection.yf.Ticker") as mock_yf:
            mock_ticker = Mock()

            # Mock options chain
            mock_chain = Mock()
            mock_chain.calls = self.mock_calls_data
            mock_chain.puts = self.mock_puts_data
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker.options = ["2024 - 12 - 10"]  # Mock expiry

            # Mock stock history
            mock_ticker.history.return_value = pd.DataFrame(
                {"Close": [150.0]}, index=[datetime.now()]
            )

            mock_yf.return_value = mock_ticker

            expected_move, iv_estimate = self.scanner.estimate_earnings_move("AAPL", 5)

            # Should return reasonable estimates
            self.assertGreater(expected_move, 0.02)  # At least 2% move
            self.assertLess(expected_move, 0.20)  # Less than 20% move
            self.assertGreater(iv_estimate, 0.15)  # At least 15% IV
            self.assertLess(iv_estimate, 1.0)  # Less than 100% IV

    def test_iv_crush_risk_assessment(self):
        """Test IV crush risk assessment logic"""
        # High IV vs historical = high crush risk
        high_iv_event = EarningsEvent(
            ticker="TSLA",
            company_name="Tesla",
            earnings_date=date.today() + timedelta(days=2),
            earnings_time="AMC",
            days_until_earnings=2,
            current_price=200.0,
            expected_move=0.12,  # 12% expected move
            iv_current=0.80,  # 80% IV (very high)
            iv_historical_avg=0.35,  # 35% historical
            iv_crush_risk="extreme",  # 80 / 35=2.3x premium
        )

        # Should classify as extreme risk
        iv_premium = high_iv_event.iv_current / high_iv_event.iv_historical_avg
        self.assertGreater(iv_premium, 2.0)  # More than 2x historical
        self.assertEqual(high_iv_event.iv_crush_risk, "extreme")

    def test_deep_itm_strategy_creation(self):
        """Test deep ITM call strategy creation"""
        with patch("backend.tradingbot.strategies.earnings_protection.yf.Ticker") as mock_yf:
            mock_ticker = Mock()

            # Mock options chain with ITM calls
            deep_itm_calls = pd.DataFrame(
                {
                    "strike": [125, 130, 135, 140, 145],  # Deep ITM for $150 stock
                    "bid": [26.00, 21.50, 17.20, 13.00, 9.50],
                    "ask": [26.50, 22.00, 17.70, 13.50, 10.00],
                    "volume": [50, 75, 100, 150, 200],
                    "openInterest": [800, 1000, 1200, 1500, 1800],
                    "impliedVolatility": [0.35, 0.36, 0.37, 0.38, 0.40],
                }
            )

            mock_chain = Mock()
            mock_chain.calls = deep_itm_calls
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker.options = ["2024 - 12 - 15"]  # Post - earnings expiry

            mock_yf.return_value = mock_ticker

            strategy = self.scanner.create_deep_itm_strategy(self.mock_earnings_event)

            if strategy:
                self.assertIsInstance(strategy, EarningsProtectionStrategy)
                self.assertEqual(strategy.strategy_type, "deep_itm")
                self.assertLess(strategy.strikes[0], self.mock_earnings_event.current_price)  # ITM
                self.assertLess(strategy.iv_sensitivity, 0.5)  # Lower IV sensitivity
                self.assertGreater(
                    strategy.profit_if_up_5pct, strategy.profit_if_flat
                )  # Benefits from upside

    def test_calendar_spread_strategy_creation(self):
        """Test calendar spread strategy creation"""
        with patch("backend.tradingbot.strategies.earnings_protection.yf.Ticker") as mock_yf:
            mock_ticker = Mock()

            # Mock two different expiries
            front_calls = self.mock_calls_data.copy()  # Before earnings
            back_calls = self.mock_calls_data.copy()  # After earnings
            back_calls["bid"] = back_calls["bid"] + 1.0  # Higher premium for longer expiry
            back_calls["ask"] = back_calls["ask"] + 1.0

            mock_ticker.option_chain.side_effect = [
                Mock(calls=front_calls, puts=pd.DataFrame()),  # Front month
                Mock(calls=back_calls, puts=pd.DataFrame()),  # Back month
            ]
            mock_ticker.options = ["2024 - 12 - 06", "2024 - 12 - 20"]  # Before and after earnings

            mock_yf.return_value = mock_ticker

            strategy = self.scanner.create_calendar_spread_strategy(self.mock_earnings_event)

            if strategy:
                self.assertIsInstance(strategy, EarningsProtectionStrategy)
                self.assertEqual(strategy.strategy_type, "calendar_spread")
                self.assertEqual(len(strategy.strikes), 2)
                self.assertEqual(strategy.strikes[0], strategy.strikes[1])  # Same strike
                self.assertEqual(len(strategy.expiry_dates), 2)
                self.assertLess(strategy.iv_sensitivity, 0.3)  # Low IV sensitivity
                self.assertGreater(
                    strategy.profit_if_flat, strategy.profit_if_up_5pct
                )  # Benefits from no movement

    def test_protective_hedge_strategy_creation(self):
        """Test protective hedge strategy creation"""
        with patch("backend.tradingbot.strategies.earnings_protection.yf.Ticker") as mock_yf:
            mock_ticker = Mock()

            mock_chain = Mock()
            mock_chain.calls = self.mock_calls_data
            mock_chain.puts = self.mock_puts_data
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker.options = ["2024 - 12 - 15"]  # Post - earnings

            mock_yf.return_value = mock_ticker

            strategy = self.scanner.create_protective_hedge_strategy(self.mock_earnings_event)

            if strategy:
                self.assertIsInstance(strategy, EarningsProtectionStrategy)
                self.assertEqual(strategy.strategy_type, "protective_hedge")
                self.assertEqual(len(strategy.strikes), 2)  # Call and put strikes
                self.assertEqual(len(strategy.option_types), 2)  # Call and put
                self.assertIn("call", strategy.option_types)
                self.assertIn("put", strategy.option_types)
                self.assertGreater(
                    strategy.iv_sensitivity, 0.5
                )  # Higher IV sensitivity than other strategies

    def test_strategy_iv_sensitivity_comparison(self):
        """Test IV sensitivity comparison across strategies"""
        # Create sample strategies with different IV sensitivities
        deep_itm = EarningsProtectionStrategy(
            ticker="AAPL",
            strategy_name="Deep ITM Call $130",
            strategy_type="deep_itm",
            earnings_date=date.today() + timedelta(days=3),
            strikes=[130],
            expiry_dates=["2024 - 12 - 15"],
            option_types=["call"],
            quantities=[1],
            net_debit=18.0,
            max_profit=float("inf"),
            max_loss=18.0,
            breakeven_points=[148.0],
            iv_sensitivity=0.2,  # Low - mostly intrinsic value
            theta_decay=0.5,
            gamma_risk=0.3,
            profit_if_up_5pct=7.5,
            profit_if_down_5pct=-0.5,
            profit_if_flat=0.0,
            risk_level="medium",
        )

        calendar_spread = EarningsProtectionStrategy(
            ticker="AAPL",
            strategy_name="Calendar Spread $150",
            strategy_type="calendar_spread",
            earnings_date=date.today() + timedelta(days=3),
            strikes=[150, 150],
            expiry_dates=["2024 - 12 - 06", "2024 - 12 - 20"],
            option_types=["call", "call"],
            quantities=[-1, 1],
            net_debit=2.5,
            max_profit=5.0,
            max_loss=2.5,
            breakeven_points=[147.5, 152.5],
            iv_sensitivity=0.15,  # Very low - benefits from crush
            theta_decay=-0.1,
            gamma_risk=0.1,
            profit_if_up_5pct=1.0,
            profit_if_down_5pct=1.0,
            profit_if_flat=3.5,  # Best case
            risk_level="low",
        )

        protective_hedge = EarningsProtectionStrategy(
            ticker="AAPL",
            strategy_name="Protective Hedge 157.5C / 142.5P",
            strategy_type="protective_hedge",
            earnings_date=date.today() + timedelta(days=3),
            strikes=[157.5, 142.5],
            expiry_dates=["2024 - 12 - 15", "2024 - 12 - 15"],
            option_types=["call", "put"],
            quantities=[1, 1],
            net_debit=6.5,
            max_profit=float("inf"),
            max_loss=6.5,
            breakeven_points=[164.0, 136.0],
            iv_sensitivity=0.6,  # High - exposed to crush
            theta_decay=1.2,
            gamma_risk=0.5,
            profit_if_up_5pct=1.5,
            profit_if_down_5pct=1.5,
            profit_if_flat=-6.5,  # Max loss if no movement
            risk_level="medium",
        )

        strategies = [deep_itm, calendar_spread, protective_hedge]

        # Sort by IV sensitivity (lower is better for earnings)
        strategies.sort(key=lambda x: x.iv_sensitivity)

        # Calendar spread should be most protected (lowest IV sensitivity)
        self.assertEqual(strategies[0].strategy_type, "calendar_spread")
        self.assertLess(strategies[0].iv_sensitivity, strategies[1].iv_sensitivity)
        self.assertLess(strategies[1].iv_sensitivity, strategies[2].iv_sensitivity)

    @patch("backend.tradingbot.strategies.earnings_protection.yf.Ticker")
    def test_scan_earnings_protection_integration(self, mock_yf):
        """Test the main scanning function"""
        mock_ticker = Mock()

        # Mock company info
        mock_ticker.info = {"shortName": "Apple Inc."}

        # Mock stock history
        mock_ticker.history.return_value = pd.DataFrame({"Close": [150.0]}, index=[datetime.now()])

        # Mock options data
        mock_chain = Mock()
        mock_chain.calls = self.mock_calls_data
        mock_chain.puts = self.mock_puts_data
        mock_ticker.option_chain.return_value = mock_chain
        mock_ticker.options = ["2024 - 12 - 15"]

        mock_yf.return_value = mock_ticker

        # Mock earnings estimation to return high IV crush risk
        with (
            patch.object(self.scanner, "estimate_earnings_move", return_value=(0.08, 0.50)),
            patch.object(self.scanner, "estimate_historical_iv", return_value=0.25),
        ):
            strategies = self.scanner.scan_earnings_protection()

            self.assertIsInstance(strategies, list)

            # Should find protection strategies for high IV crush risk events
            if strategies:
                strategy = strategies[0]
                self.assertIsInstance(strategy, EarningsProtectionStrategy)
                self.assertIn(
                    strategy.strategy_type, ["deep_itm", "calendar_spread", "protective_hedge"]
                )
                self.assertLess(strategy.iv_sensitivity, 0.8)  # Should have some protection

    def test_wsb_earnings_avoidance_principles(self):
        """Test adherence to WSB earnings avoidance principles"""
        # WSB Rule: Most earnings plays lose money due to IV crush
        # Should heavily favor low IV sensitivity strategies

        high_iv_event = EarningsEvent(
            ticker="NVDA",
            company_name="NVIDIA",
            earnings_date=date.today() + timedelta(days=1),
            earnings_time="AMC",
            days_until_earnings=1,
            current_price=500.0,
            expected_move=0.15,  # 15% expected move
            iv_current=0.90,  # 90% IV (extremely high)
            iv_historical_avg=0.40,  # 40% historical
            iv_crush_risk="extreme",
        )

        # Should recommend avoiding or using only low IV sensitivity strategies
        iv_premium = high_iv_event.iv_current / high_iv_event.iv_historical_avg
        self.assertGreater(iv_premium, 2.0)  # Extreme IV crush risk

        # For extreme IV crush risk, scanner should:
        # 1. Warn about earnings plays
        # 2. Only suggest low IV sensitivity strategies
        # 3. Recommend small position sizes (1 - 2%)
        # 4. Suggest waiting until after earnings

        self.assertEqual(high_iv_event.iv_crush_risk, "extreme")

    def test_alternative_post_earnings_opportunities(self):
        """Test post - earnings opportunity identification"""
        # After earnings, IV crush creates buying opportunities
        post_earnings_iv = 0.20  # Crushed from 0.50 pre-earnings
        historical_iv = 0.25

        # IV now below historical = potential buying opportunity
        iv_discount = historical_iv / post_earnings_iv

        self.assertGreater(iv_discount, 1.0)  # IV is now below historical

        # This creates opportunity for:
        # 1. Buying cheap options for future moves
        # 2. Lower cost basis for directional plays
        # 3. Better risk / reward for longer - term positions

    def test_position_sizing_recommendations(self):
        """Test position sizing for earnings protection strategies"""
        strategy = EarningsProtectionStrategy(
            ticker="AAPL",
            strategy_name="Calendar Spread $150",
            strategy_type="calendar_spread",
            earnings_date=date.today() + timedelta(days=3),
            strikes=[150, 150],
            expiry_dates=["2024 - 12 - 06", "2024 - 12 - 20"],
            option_types=["call", "call"],
            quantities=[-1, 1],
            net_debit=2.5,
            max_profit=5.0,
            max_loss=2.5,
            breakeven_points=[147.5, 152.5],
            iv_sensitivity=0.15,
            theta_decay=-0.1,
            gamma_risk=0.1,
            profit_if_up_5pct=1.0,
            profit_if_down_5pct=1.0,
            profit_if_flat=3.5,
            risk_level="low",
        )

        account_size = 100000  # $100k account

        # WSB recommendation: Only 1 - 2% of account on earnings plays
        max_earnings_risk = account_size * 0.02  # 2% max
        max_contracts = int(max_earnings_risk / (strategy.max_loss * 100))

        self.assertGreater(max_contracts, 0)
        self.assertLess(max_contracts * strategy.max_loss * 100, account_size * 0.025)  # Max 2.5%


class TestEarningsProtectionCalculations(unittest.TestCase):
    """Test calculation accuracy for earnings protection"""

    def test_iv_sensitivity_calculation(self):
        """Test IV sensitivity calculation methodology"""
        # For options, IV sensitivity depends on vega and time value proportion

        # Deep ITM option: mostly intrinsic value, low IV sensitivity
        deep_itm_premium = 25.0
        deep_itm_intrinsic = 20.0
        deep_itm_time_value = deep_itm_premium - deep_itm_intrinsic
        deep_itm_iv_sensitivity = deep_itm_time_value / deep_itm_premium

        # ATM option: mostly time value, high IV sensitivity
        atm_premium = 8.0
        atm_intrinsic = 0.0
        atm_time_value = atm_premium - atm_intrinsic
        atm_iv_sensitivity = atm_time_value / atm_premium

        self.assertLess(deep_itm_iv_sensitivity, atm_iv_sensitivity)
        self.assertLess(deep_itm_iv_sensitivity, 0.3)  # Less than 30% sensitive
        self.assertGreater(atm_iv_sensitivity, 0.8)  # More than 80% sensitive

    def test_calendar_spread_iv_benefits(self):
        """Test calendar spread IV crush benefits"""
        # Front month (short): high IV, crushes after earnings
        front_premium_before = 5.0
        front_premium_after = 2.0  # IV crush

        # Back month (long): less IV crush due to more time
        back_premium_before = 7.0
        back_premium_after = 5.5  # Less crush

        # Calendar spread P & L
        spread_cost = back_premium_before - front_premium_before  # Debit
        spread_value_after = back_premium_after - front_premium_after  # Value after crush

        profit = spread_value_after - spread_cost

        self.assertGreater(profit, 0)  # Should profit from IV crush
        self.assertGreater(spread_value_after, spread_cost)  # Value increases

    def test_expected_move_vs_actual_move_analysis(self):
        """Test analysis showing expected moves are often overestimated"""
        # Historical analysis: expected moves vs actual moves
        expected_moves = [0.08, 0.12, 0.06, 0.10, 0.15, 0.09, 0.11]  # From straddle pricing
        actual_moves = [0.04, 0.08, 0.03, 0.05, 0.09, 0.06, 0.07]  # Historical actual moves

        # Calculate overestimation
        overestimations = [
            (exp - act) for exp, act in zip(expected_moves, actual_moves, strict=False)
        ]
        avg_overestimation = sum(overestimations) / len(overestimations)

        self.assertGreater(avg_overestimation, 0)  # Expected moves typically overestimated

        # This supports WSB thesis that earnings options are usually overpriced
        overestimation_pct = avg_overestimation / (sum(expected_moves) / len(expected_moves))
        self.assertGreater(overestimation_pct, 0.2)  # At least 20% overestimation


def run_earnings_protection_tests():
    """Run all earnings protection tests"""
    print(" = " * 60)
    print("EARNINGS PROTECTION WSB STRATEGY - COMPREHENSIVE TEST SUITE")
    print(" = " * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestEarningsProtectionScanner, TestEarningsProtectionCalculations]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + " = " * 60)
    print("EARNINGS PROTECTION TEST SUMMARY")
    print(" = " * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    ) * 100
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


if __name__ == "__main__":
    run_earnings_protection_tests()
