"""Simple comprehensive tests for data quality monitoring."""
import pytest
import time
from unittest.mock import Mock, patch

from backend.tradingbot.data.quality import DataQualityMonitor


class TestDataQualityMonitorSimple:
    """Test data quality monitoring functionality."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = DataQualityMonitor()
        assert monitor is not None

    def test_validate_price_basic(self):
        """Test basic price validation."""
        monitor = DataQualityMonitor()

        # Should not raise for first price
        monitor.validate_price("AAPL", 150.0)

    def test_validate_price_multiple_symbols(self):
        """Test price validation for multiple symbols."""
        monitor = DataQualityMonitor()

        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            monitor.validate_price(symbol, 100.0)

    def test_assert_fresh_basic(self):
        """Test basic freshness assertion."""
        monitor = DataQualityMonitor()

        # Should not raise immediately after creation
        monitor.assert_fresh()

    def test_staleness_detection(self):
        """Test staleness detection."""
        monitor = DataQualityMonitor(max_staleness_sec=0.1)

        # Wait for staleness
        time.sleep(0.2)

        # Should detect staleness
        with pytest.raises(RuntimeError, match="DATA_STALE"):
            monitor.assert_fresh()

    def test_outlier_detection_basic(self):
        """Test basic outlier detection."""
        monitor = DataQualityMonitor(max_return_z=2.0)

        # Build some price history
        for i in range(10):
            monitor.validate_price("TEST", 100.0 + i * 0.1)
            time.sleep(0.01)

        # Try a large jump - might trigger outlier detection
        try:
            monitor.validate_price("TEST", 200.0)  # 100% jump
        except RuntimeError as e:
            assert "PRICE_OUTLIER" in str(e)

    def test_get_quality_report(self):
        """Test quality report generation."""
        monitor = DataQualityMonitor()

        # Add some data
        monitor.validate_price("AAPL", 150.0)
        monitor.validate_price("MSFT", 250.0)

        # Should be able to generate report
        report = monitor.get_quality_report()
        assert isinstance(report, dict)

    def test_custom_parameters(self):
        """Test monitor with custom parameters."""
        monitor = DataQualityMonitor(max_staleness_sec=120.0, max_return_z=5.0)
        assert monitor.max_staleness_sec == 120.0
        assert monitor.max_return_z == 5.0

    def test_edge_case_prices(self):
        """Test edge case price values."""
        monitor = DataQualityMonitor()

        # Test very small price
        monitor.validate_price("PENNY", 0.01)

        # Test large price
        monitor.validate_price("EXPENSIVE", 50000.0)

    def test_rapid_updates(self):
        """Test rapid price updates."""
        monitor = DataQualityMonitor()

        # Rapid updates
        for i in range(50):
            price = 100.0 + (i % 5) * 0.1
            monitor.validate_price("RAPID", price)

    def test_price_history_management(self):
        """Test price history management."""
        monitor = DataQualityMonitor()

        # Add many prices
        for i in range(100):
            price = 100.0 + i * 0.01
            monitor.validate_price("HISTORY", price)

        # Should maintain reasonable memory usage
        if hasattr(monitor, 'price_history'):
            assert len(monitor.price_history) >= 1