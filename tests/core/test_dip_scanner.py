#!/usr / bin / env python3
"""Comprehensive Test Suite for Dip Scanner
Tests real - time dip scanning functionality and market hours logic.
"""

import asyncio
import os
import sys
import pytest
from datetime import datetime, time
from unittest.mock import AsyncMock, Mock, patch

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.tradingbot.dip_scanner import LiveDipScanner, MarketHours


class TestMarketHours:
    """Test market hours configuration and logic."""

    def setup_method(self):
        self.market_hours = MarketHours()

    def test_default_market_hours(self):
        """Test default market hours configuration."""
        assert self.market_hours.market_open == time(9, 30)
        assert self.market_hours.market_close == time(16, 0)
        assert self.market_hours.optimal_entry_start == time(10, 0)
        assert self.market_hours.optimal_entry_end == time(15, 0)

    def test_custom_market_hours(self):
        """Test custom market hours configuration."""
        custom_hours = MarketHours(
            market_open=time(9, 0),
            market_close=time(16, 30),
            optimal_entry_start=time(9, 45),
            optimal_entry_end=time(15, 30),
        )

        assert custom_hours.market_open == time(9, 0)
        assert custom_hours.market_close == time(16, 30)
        assert custom_hours.optimal_entry_start == time(9, 45)
        assert custom_hours.optimal_entry_end == time(15, 30)


class TestLiveDipScanner:
    """Test live dip scanner functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_system = Mock()
        self.mock_system.universe = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Create scanner
        self.scanner = LiveDipScanner(self.mock_system)

    def test_scanner_initialization(self):
        """Test scanner initialization."""
        assert self.scanner.system == self.mock_system
        assert self.scanner.dip_detector is not None
        assert isinstance(self.scanner.market_hours, MarketHours)
        assert not self.scanner.is_scanning
        assert self.scanner.scan_interval == 60
        assert self.scanner.opportunities_found_today == 0
        assert self.scanner.trades_executed_today == 0
        assert self.scanner.last_scan_time is None

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_market_open_during_hours(self, mock_datetime):
        """Test market open detection during trading hours."""
        # Mock market hours (11: 00 AM)
        mock_datetime.now.return_value.time.return_value = time(11, 0)

        result = self.scanner.is_market_open()
        assert result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_market_open_before_hours(self, mock_datetime):
        """Test market open detection before trading hours."""
        # Mock pre-market (8: 00 AM)
        mock_datetime.now.return_value.time.return_value = time(8, 0)

        result = self.scanner.is_market_open()
        assert not result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_market_open_after_hours(self, mock_datetime):
        """Test market open detection after trading hours."""
        # Mock after - hours (5: 00 PM)
        mock_datetime.now.return_value.time.return_value = time(17, 0)

        result = self.scanner.is_market_open()
        assert not result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_optimal_entry_time_optimal(self, mock_datetime):
        """Test optimal entry time detection during optimal window."""
        # Mock optimal time (12: 00 PM)
        mock_datetime.now.return_value.time.return_value = time(12, 0)

        result = self.scanner.is_optimal_entry_time()
        assert result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_optimal_entry_time_early(self, mock_datetime):
        """Test optimal entry time detection early in session."""
        # Mock early market (9: 45 AM)
        mock_datetime.now.return_value.time.return_value = time(9, 45)

        result = self.scanner.is_optimal_entry_time()
        assert not result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_is_optimal_entry_time_late(self, mock_datetime):
        """Test optimal entry time detection late in session."""
        # Mock late market (3: 30 PM)
        mock_datetime.now.return_value.time.return_value = time(15, 30)

        result = self.scanner.is_optimal_entry_time()
        assert not result

    def test_should_scan_market_closed(self):
        """Test scan decision when market is closed."""
        with patch.object(self.scanner, "is_market_open", return_value=False):
            result = self.scanner.should_scan()
            assert not result

    def test_should_scan_already_scanning(self):
        """Test scan decision when already scanning."""
        self.scanner.is_scanning = True

        with patch.object(self.scanner, "is_market_open", return_value=True):
            result = self.scanner.should_scan()
            assert not result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_should_scan_too_soon(self, mock_datetime):
        """Test scan decision when scanned too recently."""
        # Set last scan time to 30 seconds ago
        now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        self.scanner.last_scan_time = datetime(2024, 1, 1, 11, 59, 30)

        with patch.object(self.scanner, "is_market_open", return_value=True):
            result = self.scanner.should_scan()
            assert not result

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_should_scan_ready(self, mock_datetime):
        """Test scan decision when ready to scan."""
        # Set last scan time to 90 seconds ago
        now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = now
        self.scanner.last_scan_time = datetime(2024, 1, 1, 11, 58, 30)

        with patch.object(self.scanner, "is_market_open", return_value=True):
            result = self.scanner.should_scan()
            assert result

    @pytest.mark.asyncio
    @patch("backend.tradingbot.dip_scanner.asyncio.sleep", new_callable=AsyncMock)
    async def test_scan_universe_basic(self, mock_sleep):
        """Test basic universe scanning functionality."""
        # Mock dip detector to return signals
        mock_dip_signal = Mock()
        mock_dip_signal.ticker = "AAPL"
        mock_dip_signal.dip_percent = -3.5
        mock_dip_signal.confidence_score = 0.85

        with patch.object(
            self.scanner.system, "scan_for_dip_opportunities", return_value=[mock_dip_signal]
        ):
            signals = await self.scanner.scan_universe()

            assert len(signals) == 1
            assert signals[0].ticker == "AAPL"
            assert signals[0].dip_percent == -3.5

    @pytest.mark.asyncio
    @patch("backend.tradingbot.dip_scanner.asyncio.sleep", new_callable=AsyncMock)
    async def test_scan_universe_no_signals(self, mock_sleep):
        """Test universe scanning with no signals."""
        with patch.object(self.scanner.system, "scan_for_dip_opportunities", return_value=[]):
            signals = await self.scanner.scan_universe()
            assert len(signals) == 0

    @pytest.mark.asyncio
    @patch("backend.tradingbot.dip_scanner.asyncio.sleep", new_callable=AsyncMock)
    async def test_scan_universe_error_handling(self, mock_sleep):
        """Test universe scanning error handling."""
        with patch.object(
            self.scanner.system,
            "scan_for_dip_opportunities",
            side_effect=Exception("Network error"),
        ):
            signals = await self.scanner.scan_universe()
            assert len(signals) == 0  # Should return empty list on error

    def test_process_dip_signals_valid(self):
        """Test processing valid dip signals."""
        # Create mock signals
        strong_signal = Mock()
        strong_signal.ticker = "AAPL"
        strong_signal.confidence_score = 0.85
        strong_signal.dip_percent = -4.2

        weak_signal = Mock()
        weak_signal.ticker = "GOOGL"
        weak_signal.confidence_score = 0.55
        weak_signal.dip_percent = -1.8

        signals = [strong_signal, weak_signal]

        with patch.object(self.scanner, "is_optimal_entry_time", return_value=True):
            processed_signals = self.scanner.process_dip_signals(signals)

            # Should filter out weak signal
            assert len(processed_signals) == 1
            assert processed_signals[0].ticker == "AAPL"

    def test_process_dip_signals_suboptimal_time(self):
        """Test processing signals during suboptimal time."""
        strong_signal = Mock()
        strong_signal.ticker = "AAPL"
        strong_signal.confidence_score = 0.85
        strong_signal.dip_percent = -4.2

        signals = [strong_signal]

        with patch.object(self.scanner, "is_optimal_entry_time", return_value=False):
            processed_signals = self.scanner.process_dip_signals(signals)

            # Should still process but with lower priority
            assert len(processed_signals) == 1

    def test_update_daily_stats(self):
        """Test daily statistics tracking."""
        # Initial state
        assert self.scanner.opportunities_found_today == 0
        assert self.scanner.trades_executed_today == 0

        # Update stats
        self.scanner.update_daily_stats(opportunities_found=3, trades_executed=1)

        assert self.scanner.opportunities_found_today == 3
        assert self.scanner.trades_executed_today == 1

        # Update again
        self.scanner.update_daily_stats(opportunities_found=2, trades_executed=1)

        assert self.scanner.opportunities_found_today == 5
        assert self.scanner.trades_executed_today == 2

    @patch("backend.tradingbot.dip_scanner.datetime")
    def test_reset_daily_stats(self, mock_datetime):
        """Test daily statistics reset."""
        # Set up mock to return a real datetime
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

        # Set some stats
        self.scanner.opportunities_found_today = 10
        self.scanner.trades_executed_today = 5

        # Reset stats
        self.scanner.reset_daily_stats()

        assert self.scanner.opportunities_found_today == 0
        assert self.scanner.trades_executed_today == 0
        assert isinstance(self.scanner.last_reset_date, datetime)

    def test_get_scanner_status(self):
        """Test scanner status reporting."""
        # Set some state
        self.scanner.is_scanning = True
        self.scanner.opportunities_found_today = 5
        self.scanner.trades_executed_today = 2
        self.scanner.last_scan_time = datetime(2024, 1, 1, 12, 0, 0)

        status = self.scanner.get_scanner_status()

        assert isinstance(status, dict)
        assert status["is_scanning"]
        assert status["opportunities_found_today"] == 5
        assert status["trades_executed_today"] == 2
        assert "last_scan_time" in status
        assert "market_open" in status


class TestLiveDipScannerIntegration:
    """Test dip scanner integration scenarios."""

    def setup_method(self):
        self.mock_system = Mock()
        self.mock_system.universe = ["AAPL", "GOOGL", "MSFT"]
        self.scanner = LiveDipScanner(self.mock_system)

    @pytest.mark.asyncio
    @patch("backend.tradingbot.dip_scanner.asyncio.sleep", new_callable=AsyncMock)
    async def test_full_scan_cycle(self, mock_sleep):
        """Test complete scan cycle."""
        # Mock dependencies
        mock_dip_signal = Mock()
        mock_dip_signal.ticker = "AAPL"
        mock_dip_signal.confidence_score = 0.90
        mock_dip_signal.dip_percent = -5.2

        with (
            patch.object(
                self.scanner.system,
                "scan_for_dip_opportunities",
                return_value=[mock_dip_signal],
            ),
            patch.object(self.scanner, "is_market_open", return_value=True),
            patch.object(self.scanner, "is_optimal_entry_time", return_value=True),
            patch.object(self.scanner, "should_scan", return_value=True),
        ):
            # Run single scan cycle
            signals = await self.scanner.scan_universe()
            processed_signals = self.scanner.process_dip_signals(signals)

            # Verify results
            assert len(signals) == 1
            assert len(processed_signals) == 1
            assert processed_signals[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_scanner_main_loop_market_closed(self):
        """Test main scanning loop when market is closed."""
        with (
            patch.object(self.scanner, "is_market_open", return_value=False),
            patch.object(self.scanner, "should_scan", return_value=False),
        ):
            # Mock short sleep to prevent infinite loop
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                # Set up to break loop after one iteration
                mock_sleep.side_effect = [None, asyncio.CancelledError()]

                with pytest.raises(asyncio.CancelledError):
                    await self.scanner.start_scanning()

                # Should have slept but not scanned
                mock_sleep.assert_called()

    def test_error_recovery(self):
        """Test error recovery and logging."""
        with (
            patch.object(
                self.scanner.system,
                "scan_for_dip_opportunities",
                side_effect=Exception("Test error"),
            ),
            patch.object(self.scanner.logger, "error") as mock_logger,
        ):
            # Process should handle error gracefully
            try:
                result = asyncio.run(self.scanner.scan_universe())
                assert result == []  # Should return empty list
            except Exception:
                self.fail("Scanner should handle exceptions gracefully")

            # Should log the error
            mock_logger.assert_called()

    def test_configuration_validation(self):
        """Test scanner configuration validation."""
        # Test invalid scan interval
        self.scanner.scan_interval = 0

        # Should handle gracefully
        with patch.object(self.scanner, "is_market_open", return_value=True):
            result = self.scanner.should_scan()
            # Logic should still work despite invalid interval
            assert isinstance(result, bool)

    def test_performance_tracking(self):
        """Test performance tracking and metrics."""
        initial_time = datetime.now()

        # Simulate scanning with timing
        with patch("backend.tradingbot.dip_scanner.datetime") as mock_datetime:
            mock_datetime.now.return_value = initial_time

            # Reset stats
            self.scanner.reset_daily_stats()

            # Update stats
            self.scanner.update_daily_stats(opportunities_found=3, trades_executed=1)

            # Check metrics
            assert self.scanner.opportunities_found_today == 3
            assert self.scanner.trades_executed_today == 1


def run_dip_scanner_tests():
    """Run all dip scanner tests."""
    print(" = " * 60)
    print("DIP SCANNER - COMPREHENSIVE TEST SUITE")
    print(" = " * 60)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestMarketHours, TestLiveDipScanner, TestLiveDipScannerIntegration]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + " = " * 60)
    print("DIP SCANNER TEST SUMMARY")
    print(" = " * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    success_rate = (
        (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
        )
        * 100
        if result.testsRun > 0
        else 0
    )
    print(f"SUCCESS RATE: {success_rate:.1f}%")

    return result


if __name__ == "__main__":
    run_dip_scanner_tests()
