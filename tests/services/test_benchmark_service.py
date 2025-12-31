"""
Comprehensive tests for BenchmarkService.

Tests benchmark data retrieval, comparison calculations, caching,
and all edge cases.
Target: 80%+ coverage.
"""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import time

from django.test import TestCase

from backend.auth0login.services.benchmark import (
    BenchmarkService,
    BenchmarkReturn,
    PortfolioComparison,
    BenchmarkSeries,
)


class TestBenchmarkService(TestCase):
    """Test suite for BenchmarkService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = BenchmarkService()
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 31)

    def test_initialization(self):
        """Test service initialization."""
        service = BenchmarkService()
        self.assertIsNotNone(service)

    def test_default_benchmark(self):
        """Test default benchmark is SPY."""
        self.assertEqual(BenchmarkService.DEFAULT_BENCHMARK, 'SPY')

    def test_supported_benchmarks(self):
        """Test supported benchmarks list."""
        expected = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        for benchmark in expected:
            self.assertIn(benchmark, BenchmarkService.SUPPORTED_BENCHMARKS)

    def test_cache_get_and_set(self):
        """Test cache get and set methods."""
        key = 'test_key'
        value = {'data': 'test_value'}

        self.service._set_cached(key, value)
        cached = self.service._get_cached(key)

        self.assertEqual(cached, value)

    def test_cache_expiration(self):
        """Test cache expiration after TTL."""
        key = 'expiring_key'
        value = {'data': 'test'}

        self.service._set_cached(key, value)

        # Mock time to simulate TTL expiration
        with patch('backend.auth0login.services.benchmark.time') as mock_time:
            mock_time.time.return_value = time.time() + 400  # > 300 seconds TTL

            cached = self.service._get_cached(key)
            self.assertIsNone(cached)

    def test_mock_benchmark_return(self):
        """Test _mock_benchmark_return generates valid data."""
        result = self.service._mock_benchmark_return(
            'SPY',
            self.start_date,
            self.end_date
        )

        self.assertIsInstance(result, BenchmarkReturn)
        self.assertEqual(result.ticker, 'SPY')
        self.assertEqual(result.start_price, 450.00)
        self.assertGreater(result.end_price, 0)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', False)
    def test_get_benchmark_return_without_yfinance(self):
        """Test get_benchmark_return when yfinance not available."""
        result = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, BenchmarkReturn)
        self.assertEqual(result.ticker, 'SPY')

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_return_with_yfinance(self, mock_yf):
        """Test get_benchmark_return with yfinance available."""
        # Mock ticker and history
        mock_ticker = Mock()
        mock_hist = MagicMock()
        mock_hist.__len__.return_value = 20
        mock_hist.index.searchsorted.return_value = 0
        mock_hist['Close'].iloc = Mock()
        mock_hist['Close'].iloc.__getitem__ = Mock(side_effect=[450.0, 460.0, 465.0])

        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker

        result = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, BenchmarkReturn)
        mock_yf.Ticker.assert_called_once_with('SPY')

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_return_insufficient_data(self, mock_yf):
        """Test get_benchmark_return falls back to mock when insufficient data."""
        mock_ticker = Mock()
        mock_hist = MagicMock()
        mock_hist.__len__.return_value = 1  # Insufficient

        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker

        result = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, BenchmarkReturn)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_return_error_handling(self, mock_yf):
        """Test get_benchmark_return handles errors gracefully."""
        mock_yf.Ticker.side_effect = Exception("API Error")

        result = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        # Should fall back to mock data
        self.assertIsInstance(result, BenchmarkReturn)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_return_uses_cache(self, mock_yf):
        """Test get_benchmark_return uses cache for repeated calls."""
        mock_ticker = Mock()
        mock_hist = MagicMock()
        mock_hist.__len__.return_value = 20
        mock_hist.index.searchsorted.return_value = 0
        mock_hist['Close'].iloc = Mock()
        mock_hist['Close'].iloc.__getitem__ = Mock(side_effect=[450.0, 460.0, 465.0])

        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker

        # First call
        result1 = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        # Second call should use cache
        result2 = self.service.get_benchmark_return(
            self.start_date,
            self.end_date,
            'SPY'
        )

        # yfinance should only be called once
        self.assertEqual(mock_yf.Ticker.call_count, 1)
        self.assertEqual(result1.ticker, result2.ticker)

    def test_mock_benchmark_series(self):
        """Test _mock_benchmark_series generates valid series."""
        result = self.service._mock_benchmark_series(
            self.start_date,
            self.end_date
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        for entry in result:
            self.assertIn('date', entry)
            self.assertIn('close', entry)
            self.assertIn('return_pct', entry)
            self.assertIn('normalized', entry)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', False)
    def test_get_benchmark_series_without_yfinance(self):
        """Test get_benchmark_series without yfinance."""
        result = self.service.get_benchmark_series(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_series_with_yfinance(self, mock_yf):
        """Test get_benchmark_series with yfinance."""
        import pandas as pd

        mock_ticker = Mock()
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        mock_hist = pd.DataFrame({
            'Close': [450.0 + i for i in range(len(dates))]
        }, index=dates)

        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker

        result = self.service.get_benchmark_series(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_get_benchmark_series_empty_data(self, mock_yf):
        """Test get_benchmark_series with empty data falls back to mock."""
        import pandas as pd

        mock_ticker = Mock()
        mock_hist = pd.DataFrame()  # Empty

        mock_ticker.history.return_value = mock_hist
        mock_yf.Ticker.return_value = mock_ticker

        result = self.service.get_benchmark_series(
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertIsInstance(result, list)

    def test_compare_portfolio_to_benchmark(self):
        """Test compare_portfolio_to_benchmark calculations."""
        result = self.service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=115000.0,
            portfolio_daily_pnl=500.0,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark='SPY'
        )

        self.assertIsInstance(result, PortfolioComparison)
        self.assertEqual(result.portfolio_return_pct, 15.0)
        self.assertGreater(result.your_value, result.hypothetical_benchmark_value)

    def test_compare_portfolio_to_benchmark_underperformance(self):
        """Test comparison when portfolio underperforms."""
        result = self.service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=102000.0,  # Only 2% gain
            portfolio_daily_pnl=100.0,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark='SPY'
        )

        self.assertIsInstance(result, PortfolioComparison)
        self.assertEqual(result.portfolio_return_pct, 2.0)

    def test_compare_portfolio_to_benchmark_zero_start_value(self):
        """Test comparison with zero start value."""
        result = self.service.compare_portfolio_to_benchmark(
            portfolio_start_value=0.0,
            portfolio_end_value=1000.0,
            portfolio_daily_pnl=50.0,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark='SPY'
        )

        self.assertEqual(result.portfolio_return_pct, 0.0)

    def test_compare_portfolio_to_benchmark_negative_returns(self):
        """Test comparison with negative returns."""
        result = self.service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=95000.0,  # 5% loss
            portfolio_daily_pnl=-200.0,
            start_date=self.start_date,
            end_date=self.end_date,
            benchmark='SPY'
        )

        self.assertIsInstance(result, PortfolioComparison)
        self.assertEqual(result.portfolio_return_pct, -5.0)
        self.assertLess(result.alpha_generated, 0)

    @patch('backend.auth0login.services.benchmark.HAS_NUMPY', False)
    def test_calculate_tracking_metrics_without_numpy(self):
        """Test calculate_tracking_metrics without numpy."""
        portfolio_returns = [1.0, 2.0, -0.5, 1.5, 0.8]
        benchmark_returns = [0.8, 1.5, -0.3, 1.2, 0.6]

        result = self.service.calculate_tracking_metrics(
            portfolio_returns,
            benchmark_returns
        )

        # Should return None values
        self.assertIsNone(result['tracking_error'])
        self.assertIsNone(result['information_ratio'])
        self.assertIsNone(result['correlation'])

    @patch('backend.auth0login.services.benchmark.HAS_NUMPY', True)
    def test_calculate_tracking_metrics_with_numpy(self):
        """Test calculate_tracking_metrics with numpy."""
        portfolio_returns = [1.0, 2.0, -0.5, 1.5, 0.8, 1.2, -0.3, 0.9, 1.1, 0.7]
        benchmark_returns = [0.8, 1.5, -0.3, 1.2, 0.6, 1.0, -0.2, 0.7, 0.9, 0.5]

        result = self.service.calculate_tracking_metrics(
            portfolio_returns,
            benchmark_returns
        )

        self.assertIsNotNone(result['tracking_error'])
        self.assertIsNotNone(result['information_ratio'])
        self.assertIsNotNone(result['correlation'])
        self.assertIsNotNone(result['beta'])
        self.assertIsNotNone(result['alpha'])

    @patch('backend.auth0login.services.benchmark.HAS_NUMPY', True)
    def test_calculate_tracking_metrics_insufficient_data(self):
        """Test calculate_tracking_metrics with insufficient data."""
        portfolio_returns = [1.0, 2.0]  # Less than 5
        benchmark_returns = [0.8, 1.5]

        result = self.service.calculate_tracking_metrics(
            portfolio_returns,
            benchmark_returns
        )

        # Should return None values
        self.assertIsNone(result['tracking_error'])

    @patch('backend.auth0login.services.benchmark.HAS_NUMPY', True)
    def test_calculate_tracking_metrics_zero_tracking_error(self):
        """Test calculate_tracking_metrics with zero tracking error."""
        # Same returns = zero tracking error
        returns = [1.0, 2.0, -0.5, 1.5, 0.8]
        portfolio_returns = returns
        benchmark_returns = returns

        result = self.service.calculate_tracking_metrics(
            portfolio_returns,
            benchmark_returns
        )

        self.assertIsNotNone(result['tracking_error'])
        self.assertEqual(result['information_ratio'], 0)  # Division by zero case

    def test_get_comparison_data_empty_portfolio(self):
        """Test get_comparison_data with empty portfolio values."""
        result = self.service.get_comparison_data(
            [],
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)

    @patch.object(BenchmarkService, 'get_benchmark_series')
    def test_get_comparison_data_success(self, mock_get_series):
        """Test get_comparison_data with valid data."""
        mock_get_series.return_value = [
            {'date': '2024-01-01', 'close': 450.0, 'return_pct': 0.0, 'normalized': 100.0},
            {'date': '2024-01-31', 'close': 460.0, 'return_pct': 2.22, 'normalized': 102.22},
        ]

        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 115000.0},
        ]

        result = self.service.get_comparison_data(
            portfolio_values,
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertEqual(result['status'], 'success')
        self.assertIn('portfolio', result)
        self.assertIn('benchmark_data', result)
        self.assertIn('comparison', result)

    @patch.object(BenchmarkService, 'get_benchmark_series')
    def test_get_comparison_data_normalization(self, mock_get_series):
        """Test get_comparison_data normalizes portfolio values correctly."""
        mock_get_series.return_value = [
            {'date': '2024-01-01', 'return_pct': 0.0},
        ]

        portfolio_values = [
            {'date': '2024-01-01', 'value': 50000.0},
            {'date': '2024-01-15', 'value': 60000.0},
            {'date': '2024-01-31', 'value': 55000.0},
        ]

        result = self.service.get_comparison_data(
            portfolio_values,
            self.start_date,
            self.end_date,
            'SPY'
        )

        # Check normalization
        portfolio_series = result['portfolio']['series']
        self.assertEqual(portfolio_series[0]['normalized'], 100.0)
        self.assertEqual(portfolio_series[1]['normalized'], 120.0)  # 60000/50000 * 100
        self.assertEqual(portfolio_series[2]['normalized'], 110.0)  # 55000/50000 * 100

    @patch.object(BenchmarkService, 'get_benchmark_series')
    def test_get_comparison_data_outperforming(self, mock_get_series):
        """Test get_comparison_data when portfolio outperforms."""
        mock_get_series.return_value = [
            {'date': '2024-01-01', 'return_pct': 0.0},
            {'date': '2024-01-31', 'return_pct': 5.0},
        ]

        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 120000.0},  # 20% vs 5% benchmark
        ]

        result = self.service.get_comparison_data(
            portfolio_values,
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertTrue(result['comparison']['outperforming'])
        self.assertGreater(result['comparison']['excess_return'], 0)

    @patch.object(BenchmarkService, 'get_benchmark_series')
    def test_get_comparison_data_underperforming(self, mock_get_series):
        """Test get_comparison_data when portfolio underperforms."""
        mock_get_series.return_value = [
            {'date': '2024-01-01', 'return_pct': 0.0},
            {'date': '2024-01-31', 'return_pct': 10.0},
        ]

        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 105000.0},  # 5% vs 10% benchmark
        ]

        result = self.service.get_comparison_data(
            portfolio_values,
            self.start_date,
            self.end_date,
            'SPY'
        )

        self.assertFalse(result['comparison']['outperforming'])
        self.assertLess(result['comparison']['excess_return'], 0)

    def test_benchmark_return_dataclass(self):
        """Test BenchmarkReturn dataclass."""
        benchmark = BenchmarkReturn(
            ticker='SPY',
            start_price=450.0,
            end_price=460.0,
            period_return_pct=2.22,
            daily_return_pct=0.5,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        self.assertEqual(benchmark.ticker, 'SPY')
        self.assertEqual(benchmark.start_price, 450.0)
        self.assertEqual(benchmark.period_return_pct, 2.22)

    def test_portfolio_comparison_dataclass(self):
        """Test PortfolioComparison dataclass."""
        comparison = PortfolioComparison(
            portfolio_return_pct=15.0,
            benchmark_return_pct=10.0,
            daily_excess=0.5,
            period_excess=5.0,
            hypothetical_benchmark_value=110000.0,
            your_value=115000.0,
            alpha_generated=5000.0,
            information_ratio=1.5,
            tracking_error=0.02
        )

        self.assertEqual(comparison.portfolio_return_pct, 15.0)
        self.assertEqual(comparison.alpha_generated, 5000.0)

    def test_benchmark_series_dataclass(self):
        """Test BenchmarkSeries dataclass."""
        series = BenchmarkSeries(
            dates=['2024-01-01', '2024-01-02'],
            portfolio_values=[100000.0, 101000.0],
            benchmark_values=[450.0, 451.0],
            portfolio_normalized=[100.0, 101.0],
            benchmark_normalized=[100.0, 100.22]
        )

        self.assertEqual(len(series.dates), 2)
        self.assertEqual(series.portfolio_values[0], 100000.0)


if __name__ == '__main__':
    unittest.main()
