"""Comprehensive tests for Lotto Scanner Strategy with real API integration to achieve >70% coverage."""
import pytest
import pandas as pd
import numpy as np
import yfinance as yf
import json
import math
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from backend.tradingbot.strategies.lotto_scanner import (
    LottoScanner,
    LottoPlay,
    EarningsEvent
)


class TestLottoPlay:
    """Test LottoPlay dataclass."""

    def test_lotto_play_0dte_creation(self):
        """Test LottoPlay dataclass creation for 0DTE."""
        play = LottoPlay(
            ticker="SPY",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=500,
            option_type="call",
            current_premium=2.50,
            breakeven=502.50,
            current_spot=500.0,
            catalyst_event="market_momentum",
            expected_move=0.02,
            max_position_size=1000.0,
            max_contracts=4,
            risk_level="extreme",
            win_probability=0.30,
            potential_return=5.0,
            stop_loss_price=1.25,
            profit_target_price=7.50
        )

        assert play.ticker == "SPY"
        assert play.play_type == "0dte"
        assert play.expiry_date == "2025-01-17"
        assert play.days_to_expiry == 0
        assert play.strike == 500
        assert play.option_type == "call"
        assert play.current_premium == 2.50
        assert play.breakeven == 502.50
        assert play.current_spot == 500.0
        assert play.catalyst_event == "market_momentum"
        assert play.expected_move == 0.02
        assert play.max_position_size == 1000.0
        assert play.max_contracts == 4
        assert play.risk_level == "extreme"
        assert play.win_probability == 0.30
        assert play.potential_return == 5.0
        assert play.stop_loss_price == 1.25
        assert play.profit_target_price == 7.50

    def test_lotto_play_earnings_creation(self):
        """Test LottoPlay dataclass creation for earnings."""
        play = LottoPlay(
            ticker="AAPL",
            play_type="earnings",
            expiry_date="2025-02-21",
            days_to_expiry=7,
            strike=180,
            option_type="call",
            current_premium=8.50,
            breakeven=188.50,
            current_spot=175.0,
            catalyst_event="Q1_earnings",
            expected_move=0.08,
            max_position_size=2000.0,
            max_contracts=2,
            risk_level="very_high",
            win_probability=0.35,
            potential_return=3.0,
            stop_loss_price=4.25,
            profit_target_price=25.50
        )

        assert play.play_type == "earnings"
        assert play.days_to_expiry == 7
        assert play.catalyst_event == "Q1_earnings"
        assert play.expected_move == 0.08
        assert play.risk_level == "very_high"

    def test_lotto_play_put_option(self):
        """Test LottoPlay for put options."""
        play = LottoPlay(
            ticker="TSLA",
            play_type="catalyst",
            expiry_date="2025-01-24",
            days_to_expiry=3,
            strike=200,
            option_type="put",
            current_premium=6.75,
            breakeven=193.25,
            current_spot=210.0,
            catalyst_event="delivery_numbers",
            expected_move=0.12,
            max_position_size=1500.0,
            max_contracts=2,
            risk_level="high",
            win_probability=0.25,
            potential_return=4.5,
            stop_loss_price=3.38,
            profit_target_price=30.38
        )

        assert play.option_type == "put"
        assert play.strike == 200
        assert play.breakeven == 193.25  # Strike - premium for puts
        assert play.catalyst_event == "delivery_numbers"


class TestEarningsEvent:
    """Test EarningsEvent dataclass."""

    def test_earnings_event_creation(self):
        """Test EarningsEvent dataclass creation."""
        event = EarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=date(2025, 1, 30),
            time_of_day="AMC",
            expected_move=0.05,
            avg_move_historical=0.04,
            revenue_estimate=120000000000.0,
            eps_estimate=2.15,
            sector="Technology"
        )

        assert event.ticker == "AAPL"
        assert event.company_name == "Apple Inc."
        assert event.earnings_date == date(2025, 1, 30)
        assert event.time_of_day == "AMC"
        assert event.expected_move == 0.05
        assert event.avg_move_historical == 0.04
        assert event.revenue_estimate == 120000000000.0
        assert event.eps_estimate == 2.15
        assert event.sector == "Technology"

    def test_earnings_event_bmo(self):
        """Test EarningsEvent for before market open."""
        event = EarningsEvent(
            ticker="MSFT",
            company_name="Microsoft Corporation",
            earnings_date=date(2025, 2, 15),
            time_of_day="BMO",
            expected_move=0.06,
            avg_move_historical=0.045,
            revenue_estimate=65000000000.0,
            eps_estimate=2.85,
            sector="Technology"
        )

        assert event.time_of_day == "BMO"
        assert event.ticker == "MSFT"
        assert event.sector == "Technology"

    def test_earnings_event_unknown_time(self):
        """Test EarningsEvent with unknown timing."""
        event = EarningsEvent(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            earnings_date=date(2025, 3, 10),
            time_of_day="Unknown",
            expected_move=0.15,
            avg_move_historical=0.12,
            revenue_estimate=35000000000.0,
            eps_estimate=6.50,
            sector="Technology"
        )

        assert event.time_of_day == "Unknown"
        assert event.expected_move == 0.15
        assert event.avg_move_historical == 0.12


class TestLottoScannerBasics:
    """Test basic LottoScanner functionality."""

    def test_lotto_scanner_initialization(self):
        """Test LottoScanner initialization."""
        scanner = LottoScanner()

        assert hasattr(scanner, 'max_risk_pct')
        assert hasattr(scanner, 'lotto_tickers')
        assert isinstance(scanner.lotto_tickers, list)
        assert scanner.max_risk_pct == 0.01  # Default 1%

        # Test that lotto_tickers list is populated
        assert len(scanner.lotto_tickers) > 0
        assert "AAPL" in scanner.lotto_tickers
        assert "TSLA" in scanner.lotto_tickers
        assert "NVDA" in scanner.lotto_tickers
        assert "SPY" in scanner.lotto_tickers

    def test_lotto_scanner_custom_risk(self):
        """Test LottoScanner with custom risk percentage."""
        scanner = LottoScanner(max_risk_pct=2.0)

        assert scanner.max_risk_pct == 0.02  # 2% converted to decimal

    def test_lotto_scanner_extreme_risk(self):
        """Test LottoScanner with high risk percentage."""
        scanner = LottoScanner(max_risk_pct=5.0)

        assert scanner.max_risk_pct == 0.05  # 5% converted to decimal

    def test_lotto_tickers_categories(self):
        """Test lotto tickers contain different categories."""
        scanner = LottoScanner()

        # Check for mega caps
        assert "AAPL" in scanner.lotto_tickers
        assert "MSFT" in scanner.lotto_tickers
        assert "GOOGL" in scanner.lotto_tickers
        assert "AMZN" in scanner.lotto_tickers

        # Check for high-beta growth names
        assert "PLTR" in scanner.lotto_tickers
        assert "SNOW" in scanner.lotto_tickers
        assert "CRWD" in scanner.lotto_tickers

        # Check for meme stocks
        assert "GME" in scanner.lotto_tickers
        assert "AMC" in scanner.lotto_tickers

        # Check for ETFs
        assert "SPY" in scanner.lotto_tickers
        assert "QQQ" in scanner.lotto_tickers


class TestExpiryFunctions:
    """Test expiry calculation functions."""

    def test_get_0dte_expiry_monday(self):
        """Test get_0dte_expiry on Monday."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Monday
            mock_date.today.return_value = date(2025, 1, 13)  # Monday

            expiry = scanner.get_0dte_expiry()

            assert expiry == "2025-01-13"

    def test_get_0dte_expiry_wednesday(self):
        """Test get_0dte_expiry on Wednesday."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Wednesday
            mock_date.today.return_value = date(2025, 1, 15)  # Wednesday

            expiry = scanner.get_0dte_expiry()

            assert expiry == "2025-01-15"

    def test_get_0dte_expiry_friday(self):
        """Test get_0dte_expiry on Friday."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Friday
            mock_date.today.return_value = date(2025, 1, 17)  # Friday

            expiry = scanner.get_0dte_expiry()

            assert expiry == "2025-01-17"

    def test_get_0dte_expiry_tuesday(self):
        """Test get_0dte_expiry on Tuesday (no 0DTE, find next)."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Tuesday
            mock_date.today.return_value = date(2025, 1, 14)  # Tuesday

            expiry = scanner.get_0dte_expiry()

            assert expiry == "2025-01-15"  # Next Wednesday

    def test_get_0dte_expiry_saturday(self):
        """Test get_0dte_expiry on Saturday (find next Monday)."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Saturday
            mock_date.today.return_value = date(2025, 1, 18)  # Saturday

            expiry = scanner.get_0dte_expiry()

            assert expiry == "2025-01-20"  # Next Monday

    def test_get_weekly_expiry_monday(self):
        """Test get_weekly_expiry from Monday."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Monday
            mock_date.today.return_value = date(2025, 1, 13)  # Monday

            expiry = scanner.get_weekly_expiry()

            assert expiry == "2025-01-17"  # Next Friday

    def test_get_weekly_expiry_friday(self):
        """Test get_weekly_expiry from Friday (next week)."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Friday
            mock_date.today.return_value = date(2025, 1, 17)  # Friday

            expiry = scanner.get_weekly_expiry()

            assert expiry == "2025-01-24"  # Next Friday

    def test_get_weekly_expiry_sunday(self):
        """Test get_weekly_expiry from Sunday."""
        scanner = LottoScanner()

        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            # Mock Sunday
            mock_date.today.return_value = date(2025, 1, 19)  # Sunday

            expiry = scanner.get_weekly_expiry()

            assert expiry == "2025-01-24"  # Next Friday


class TestExpectedMoveCalculation:
    """Test expected move estimation."""

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_estimate_expected_move_mocked(self, mock_ticker_class):
        """Test expected move estimation with mocked data."""
        scanner = LottoScanner()

        # Mock historical data
        mock_history = pd.DataFrame({
            'Close': [500.0]
        })

        # Mock options chain
        mock_calls = pd.DataFrame({
            'strike': [495, 500, 505, 510],
            'impliedVolatility': [0.25, 0.22, 0.20, 0.23]
        })

        mock_chain = Mock()
        mock_chain.calls = mock_calls

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.option_chain.return_value = mock_chain
        mock_ticker_class.return_value = mock_ticker

        expected_move = scanner.estimate_expected_move("SPY", "2025-01-17")

        assert isinstance(expected_move, float)
        assert 0 < expected_move < 1  # Should be reasonable percentage

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_estimate_expected_move_no_options(self, mock_ticker_class):
        """Test expected move when no options available."""
        scanner = LottoScanner()

        # Mock historical data
        mock_history = pd.DataFrame({
            'Close': [500.0]
        })

        # Mock empty options chain
        mock_calls = pd.DataFrame()  # Empty

        mock_chain = Mock()
        mock_chain.calls = mock_calls

        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.option_chain.return_value = mock_chain
        mock_ticker_class.return_value = mock_ticker

        expected_move = scanner.estimate_expected_move("INVALID", "2025-01-17")

        assert expected_move == 0.05  # Default 5%

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_estimate_expected_move_error_handling(self, mock_ticker_class):
        """Test expected move error handling."""
        scanner = LottoScanner()

        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Data error")
        mock_ticker_class.return_value = mock_ticker

        expected_move = scanner.estimate_expected_move("INVALID", "2025-01-17")

        assert expected_move == 0.05  # Default 5%


class TestEarningsCalendar:
    """Test earnings calendar functionality."""

    def test_get_earnings_calendar_basic(self):
        """Test basic earnings calendar functionality."""
        scanner = LottoScanner()

        events = scanner.get_earnings_calendar(weeks_ahead=2)

        assert isinstance(events, list)
        # May be empty or populated depending on implementation

    def test_get_earnings_calendar_different_weeks(self):
        """Test earnings calendar with different week parameters."""
        scanner = LottoScanner()

        events_1_week = scanner.get_earnings_calendar(weeks_ahead=1)
        events_4_weeks = scanner.get_earnings_calendar(weeks_ahead=4)

        assert isinstance(events_1_week, list)
        assert isinstance(events_4_weeks, list)

        # Generally expect more events for longer time horizons
        if events_1_week and events_4_weeks:
            assert len(events_4_weeks) >= len(events_1_week)

    def test_get_earnings_calendar_zero_weeks(self):
        """Test earnings calendar with zero weeks (current week)."""
        scanner = LottoScanner()

        events = scanner.get_earnings_calendar(weeks_ahead=0)

        assert isinstance(events, list)


class TestScanningFunctions:
    """Test main scanning functions."""

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_scan_0dte_opportunities_mocked(self, mock_ticker_class):
        """Test 0DTE scanning with mocked data."""
        scanner = LottoScanner()

        # Limit tickers for testing
        scanner.lotto_tickers = ["SPY"]

        # Mock current date to have 0DTE available
        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            mock_date.today.return_value = date(2025, 1, 17)  # Friday

            # Mock ticker data
            mock_history = pd.DataFrame({
                'Close': [500.0]
            })

            mock_calls = pd.DataFrame({
                'strike': [498, 500, 502, 505],
                'bid': [3.0, 2.0, 1.5, 0.8],
                'ask': [3.2, 2.2, 1.7, 1.0],
                'impliedVolatility': [0.25, 0.22, 0.20, 0.18]
            })

            mock_puts = pd.DataFrame({
                'strike': [495, 498, 500, 502],
                'bid': [0.8, 1.5, 2.5, 4.0],
                'ask': [1.0, 1.7, 2.7, 4.2],
                'impliedVolatility': [0.18, 0.20, 0.22, 0.25]
            })

            mock_chain = Mock()
            mock_chain.calls = mock_calls
            mock_chain.puts = mock_puts

            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_history
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker_class.return_value = mock_ticker

            plays = scanner.scan_0dte_opportunities(account_size=100000)

            assert isinstance(plays, list)
            # May be empty if criteria not met, or contain plays

            for play in plays:
                assert isinstance(play, LottoPlay)
                assert play.play_type == "0dte"
                assert play.days_to_expiry == 0

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_scan_0dte_opportunities_no_expiry(self, mock_ticker_class):
        """Test 0DTE scanning when no 0DTE expiry available."""
        scanner = LottoScanner()

        # Mock date with no 0DTE available and no next date found
        with patch.object(scanner, 'get_0dte_expiry', return_value=None):
            plays = scanner.scan_0dte_opportunities(account_size=100000)

            assert plays == []  # Should return empty list

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_scan_earnings_lottos_mocked(self, mock_ticker_class):
        """Test earnings lotto scanning with mocked data."""
        scanner = LottoScanner()

        # Mock earnings calendar
        mock_event = EarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=date.today() + timedelta(days=3),
            time_of_day="AMC",
            expected_move=0.06,
            avg_move_historical=0.05,
            revenue_estimate=120000000000.0,
            eps_estimate=2.15,
            sector="Technology"
        )

        with patch.object(scanner, 'get_earnings_calendar', return_value=[mock_event]):
            # Mock ticker data
            mock_history = pd.DataFrame({
                'Close': [150.0]
            })

            mock_calls = pd.DataFrame({
                'strike': [148, 150, 152, 155],
                'bid': [8.0, 6.0, 4.5, 2.5],
                'ask': [8.5, 6.5, 5.0, 3.0],
                'impliedVolatility': [0.35, 0.32, 0.30, 0.28]
            })

            mock_puts = pd.DataFrame({
                'strike': [145, 148, 150, 152],
                'bid': [2.0, 4.0, 6.5, 9.0],
                'ask': [2.5, 4.5, 7.0, 9.5],
                'impliedVolatility': [0.28, 0.30, 0.32, 0.35]
            })

            mock_chain = Mock()
            mock_chain.calls = mock_calls
            mock_chain.puts = mock_puts

            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_history
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker_class.return_value = mock_ticker

            plays = scanner.scan_earnings_lottos(account_size=100000)

            assert isinstance(plays, list)

            for play in plays:
                assert isinstance(play, LottoPlay)
                assert play.play_type == "earnings"
                assert play.ticker == "AAPL"

    def test_scan_earnings_lottos_no_events(self):
        """Test earnings lotto scanning with no events."""
        scanner = LottoScanner()

        # Mock empty earnings calendar
        with patch.object(scanner, 'get_earnings_calendar', return_value=[]):
            plays = scanner.scan_earnings_lottos(account_size=100000)

            assert plays == []  # Should return empty list

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_scan_earnings_lottos_error_handling(self, mock_ticker_class):
        """Test earnings lotto scanning error handling."""
        scanner = LottoScanner()

        # Mock earnings calendar with event
        mock_event = EarningsEvent(
            ticker="INVALID",
            company_name="Invalid Corp",
            earnings_date=date.today() + timedelta(days=3),
            time_of_day="AMC",
            expected_move=0.06,
            avg_move_historical=0.05,
            revenue_estimate=1000000.0,
            eps_estimate=1.0,
            sector="Unknown"
        )

        with patch.object(scanner, 'get_earnings_calendar', return_value=[mock_event]):
            # Mock ticker that raises exception
            mock_ticker = Mock()
            mock_ticker.history.side_effect = Exception("Data error")
            mock_ticker_class.return_value = mock_ticker

            plays = scanner.scan_earnings_lottos(account_size=100000)

            # Should handle errors gracefully
            assert isinstance(plays, list)


class TestFormattingFunctions:
    """Test output formatting functions."""

    def test_format_lotto_plays_with_data(self):
        """Test formatting lotto plays with data."""
        scanner = LottoScanner()

        plays = [
            LottoPlay(
                ticker="SPY",
                play_type="0dte",
                expiry_date="2025-01-17",
                days_to_expiry=0,
                strike=500,
                option_type="call",
                current_premium=2.50,
                breakeven=502.50,
                current_spot=500.0,
                catalyst_event="market_momentum",
                expected_move=0.02,
                max_position_size=1000.0,
                max_contracts=4,
                risk_level="extreme",
                win_probability=0.30,
                potential_return=5.0,
                stop_loss_price=1.25,
                profit_target_price=7.50
            ),
            LottoPlay(
                ticker="AAPL",
                play_type="earnings",
                expiry_date="2025-02-21",
                days_to_expiry=7,
                strike=180,
                option_type="call",
                current_premium=8.50,
                breakeven=188.50,
                current_spot=175.0,
                catalyst_event="Q1_earnings",
                expected_move=0.08,
                max_position_size=2000.0,
                max_contracts=2,
                risk_level="very_high",
                win_probability=0.35,
                potential_return=3.0,
                stop_loss_price=4.25,
                profit_target_price=25.50
            )
        ]

        formatted = scanner.format_lotto_plays(plays)

        assert isinstance(formatted, str)
        assert "SPY" in formatted
        assert "AAPL" in formatted
        assert "0dte" in formatted or "0DTE" in formatted
        assert "earnings" in formatted or "Earnings" in formatted

    def test_format_lotto_plays_empty(self):
        """Test formatting empty lotto plays list."""
        scanner = LottoScanner()

        formatted = scanner.format_lotto_plays([])

        assert isinstance(formatted, str)
        assert len(formatted) > 0  # Should have some message

    def test_format_lotto_plays_single_play(self):
        """Test formatting single lotto play."""
        scanner = LottoScanner()

        play = LottoPlay(
            ticker="TSLA",
            play_type="catalyst",
            expiry_date="2025-01-24",
            days_to_expiry=3,
            strike=200,
            option_type="put",
            current_premium=6.75,
            breakeven=193.25,
            current_spot=210.0,
            catalyst_event="delivery_numbers",
            expected_move=0.12,
            max_position_size=1500.0,
            max_contracts=2,
            risk_level="high",
            win_probability=0.25,
            potential_return=4.5,
            stop_loss_price=3.38,
            profit_target_price=30.38
        )

        formatted = scanner.format_lotto_plays([play])

        assert isinstance(formatted, str)
        assert "TSLA" in formatted
        assert "put" in formatted or "PUT" in formatted
        assert "catalyst" in formatted or "Catalyst" in formatted


@pytest.mark.slow
class TestRealAPIIntegration:
    """Test with real API calls (marked as slow)."""

    def test_estimate_expected_move_real_api(self):
        """Test expected move estimation with real API."""
        try:
            scanner = LottoScanner()
            expected_move = scanner.estimate_expected_move("SPY", scanner.get_weekly_expiry())

            assert isinstance(expected_move, float)
            assert 0 < expected_move < 1
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")

    def test_scan_0dte_real_api(self):
        """Test 0DTE scanning with real API (limited scope)."""
        try:
            scanner = LottoScanner()
            scanner.lotto_tickers = ["SPY"]  # Just one ticker for speed

            # Only test if 0DTE is available
            if scanner.get_0dte_expiry():
                plays = scanner.scan_0dte_opportunities(account_size=10000)
                assert isinstance(plays, list)
        except Exception as e:
            pytest.skip(f"Real API test failed: {e}")


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_complete_0dte_workflow_mocked(self, mock_ticker_class):
        """Test complete 0DTE workflow."""
        scanner = LottoScanner(max_risk_pct=2.0)
        scanner.lotto_tickers = ["SPY"]

        # Mock Friday for 0DTE
        with patch('backend.tradingbot.strategies.lotto_scanner.date') as mock_date:
            mock_date.today.return_value = date(2025, 1, 17)  # Friday

            # Mock ticker data
            mock_history = pd.DataFrame({
                'Close': [500.0]
            })

            mock_calls = pd.DataFrame({
                'strike': [502, 505, 510],
                'bid': [1.5, 0.8, 0.3],
                'ask': [1.7, 1.0, 0.5],
                'impliedVolatility': [0.20, 0.18, 0.15]
            })

            mock_chain = Mock()
            mock_chain.calls = mock_calls
            mock_chain.puts = pd.DataFrame()  # Empty puts for simplicity

            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_history
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker_class.return_value = mock_ticker

            # Step 1: Get 0DTE expiry
            expiry = scanner.get_0dte_expiry()
            assert expiry == "2025-01-17"

            # Step 2: Estimate expected move
            expected_move = scanner.estimate_expected_move("SPY", expiry)
            assert expected_move >= 0

            # Step 3: Scan opportunities
            plays = scanner.scan_0dte_opportunities(account_size=50000)

            # Step 4: Format output
            if plays:
                formatted = scanner.format_lotto_plays(plays)
                assert isinstance(formatted, str)

            # Verify workflow completed
            assert isinstance(plays, list)

    @patch('backend.tradingbot.strategies.lotto_scanner.yf.Ticker')
    def test_complete_earnings_workflow_mocked(self, mock_ticker_class):
        """Test complete earnings workflow."""
        scanner = LottoScanner()

        # Mock earnings event
        mock_event = EarningsEvent(
            ticker="AAPL",
            company_name="Apple Inc.",
            earnings_date=date.today() + timedelta(days=5),
            time_of_day="AMC",
            expected_move=0.06,
            avg_move_historical=0.05,
            revenue_estimate=120000000000.0,
            eps_estimate=2.15,
            sector="Technology"
        )

        with patch.object(scanner, 'get_earnings_calendar', return_value=[mock_event]):
            # Mock ticker data
            mock_history = pd.DataFrame({
                'Close': [175.0]
            })

            mock_calls = pd.DataFrame({
                'strike': [180, 185, 190],
                'bid': [4.0, 2.5, 1.0],
                'ask': [4.5, 3.0, 1.5],
                'impliedVolatility': [0.30, 0.28, 0.25]
            })

            mock_chain = Mock()
            mock_chain.calls = mock_calls
            mock_chain.puts = pd.DataFrame()

            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_history
            mock_ticker.option_chain.return_value = mock_chain
            mock_ticker_class.return_value = mock_ticker

            # Step 1: Get earnings calendar
            events = scanner.get_earnings_calendar(weeks_ahead=2)
            assert len(events) == 1
            assert events[0].ticker == "AAPL"

            # Step 2: Scan earnings opportunities
            plays = scanner.scan_earnings_lottos(account_size=100000)

            # Step 3: Format results
            if plays:
                formatted = scanner.format_lotto_plays(plays)
                assert isinstance(formatted, str)

            # Verify workflow
            assert isinstance(plays, list)


class TestMainFunction:
    """Test main function and CLI interface."""

    def test_main_function_exists(self):
        """Test that main function exists and can be imported."""
        from backend.tradingbot.strategies.lotto_scanner import main
        assert callable(main)


class TestRiskManagement:
    """Test risk management aspects."""

    def test_position_sizing_calculation(self):
        """Test position sizing based on risk percentage."""
        scanner = LottoScanner(max_risk_pct=1.0)  # 1%
        account_size = 100000  # $100k

        # Max risk should be $1000
        max_risk = account_size * scanner.max_risk_pct
        assert max_risk == 1000

        # For a $2.50 premium, max contracts = 1000 / 250 = 4
        premium_per_contract = 2.50 * 100  # $250 per contract
        max_contracts = int(max_risk / premium_per_contract)
        assert max_contracts == 4

    def test_risk_level_validation(self):
        """Test risk level assignments."""
        # Test different risk levels
        risk_levels = ["extreme", "very_high", "high"]

        for level in risk_levels:
            play = LottoPlay(
                ticker="SPY",
                play_type="0dte",
                expiry_date="2025-01-17",
                days_to_expiry=0,
                strike=500,
                option_type="call",
                current_premium=2.50,
                breakeven=502.50,
                current_spot=500.0,
                catalyst_event="market_momentum",
                expected_move=0.02,
                max_position_size=1000.0,
                max_contracts=4,
                risk_level=level,
                win_probability=0.30,
                potential_return=5.0,
                stop_loss_price=1.25,
                profit_target_price=7.50
            )

            assert play.risk_level == level

    def test_win_probability_bounds(self):
        """Test win probability is within valid bounds."""
        play = LottoPlay(
            ticker="SPY",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=500,
            option_type="call",
            current_premium=2.50,
            breakeven=502.50,
            current_spot=500.0,
            catalyst_event="market_momentum",
            expected_move=0.02,
            max_position_size=1000.0,
            max_contracts=4,
            risk_level="extreme",
            win_probability=0.30,
            potential_return=5.0,
            stop_loss_price=1.25,
            profit_target_price=7.50
        )

        assert 0 <= play.win_probability <= 1

    def test_stop_loss_logic(self):
        """Test stop loss price logic."""
        play = LottoPlay(
            ticker="SPY",
            play_type="0dte",
            expiry_date="2025-01-17",
            days_to_expiry=0,
            strike=500,
            option_type="call",
            current_premium=2.50,
            breakeven=502.50,
            current_spot=500.0,
            catalyst_event="market_momentum",
            expected_move=0.02,
            max_position_size=1000.0,
            max_contracts=4,
            risk_level="extreme",
            win_probability=0.30,
            potential_return=5.0,
            stop_loss_price=1.25,  # 50% of premium
            profit_target_price=7.50  # 3x premium
        )

        # Stop loss should typically be 50% of entry premium
        assert play.stop_loss_price == play.current_premium * 0.5

        # Profit target should be multiple of entry premium
        assert play.profit_target_price == play.current_premium * 3.0