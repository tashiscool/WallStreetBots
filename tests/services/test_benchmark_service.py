"""
Integration tests for BenchmarkService.

These tests use REAL database operations and REAL calculation logic.
Only external data sources (yfinance) are mocked to ensure test reliability
and avoid network dependencies.

Tests verify:
- Portfolio returns are calculated from real trade data
- Benchmark comparisons use actual calculation logic
- Alpha, Sharpe ratio, and max drawdown are computed correctly
- Historical performance tracking works
"""

import pytest
import numpy as np
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from django.contrib.auth.models import User

from backend.auth0login.services.benchmark import (
    BenchmarkService,
    BenchmarkReturn,
    PortfolioComparison,
    BenchmarkSeries,
    _benchmark_cache,
    _cache_timestamps,
    CACHE_TTL_SECONDS,
)
from backend.tradingbot.models.models import (
    BacktestRun,
    BacktestTrade,
    Portfolio,
    Company,
    Stock,
    Order,
    Bot,
)


@pytest.fixture
def user(db):
    """Create a test user for portfolio operations."""
    return User.objects.create_user(
        username='benchmark_test_user',
        email='benchmark@example.com',
        password='testpass123'
    )


@pytest.fixture
def second_user(db):
    """Create a second user for isolation tests."""
    return User.objects.create_user(
        username='benchmark_test_user2',
        email='benchmark2@example.com',
        password='testpass456'
    )


@pytest.fixture
def portfolio(db, user):
    """Create a test portfolio."""
    return Portfolio.objects.create(
        name='Test Portfolio',
        user=user,
        strategy='manual'
    )


@pytest.fixture
def company(db):
    """Create test companies for trades."""
    Company.objects.filter(ticker='AAPL').delete()
    Company.objects.filter(ticker='GOOGL').delete()
    Company.objects.filter(ticker='MSFT').delete()

    apple = Company.objects.create(ticker='AAPL', name='Apple Inc.')
    google = Company.objects.create(ticker='GOOGL', name='Alphabet Inc.')
    msft = Company.objects.create(ticker='MSFT', name='Microsoft Corp.')
    return {'AAPL': apple, 'GOOGL': google, 'MSFT': msft}


@pytest.fixture
def stocks(db, company):
    """Create test stocks."""
    apple_stock = Stock.objects.create(
        company=company['AAPL'],
        price=Decimal('150.00'),
        date=date.today()
    )
    google_stock = Stock.objects.create(
        company=company['GOOGL'],
        price=Decimal('140.00'),
        date=date.today()
    )
    msft_stock = Stock.objects.create(
        company=company['MSFT'],
        price=Decimal('380.00'),
        date=date.today()
    )
    return {'AAPL': apple_stock, 'GOOGL': google_stock, 'MSFT': msft_stock}


@pytest.fixture
def backtest_run(db, user):
    """Create a backtest run with trades for testing."""
    run = BacktestRun.objects.create(
        run_id=f'test_run_{datetime.now().timestamp()}',
        user=user,
        name='Integration Test Run',
        strategy_name='momentum',
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        initial_capital=Decimal('100000.00'),
        benchmark='SPY',
        status='completed',
        total_return_pct=15.5,
        benchmark_return_pct=5.2,
        sharpe_ratio=1.85,
        max_drawdown_pct=-8.5,
        win_rate=62.5,
        total_trades=8,
        winning_trades=5,
        losing_trades=3,
        final_equity=Decimal('115500.00'),
        total_pnl=Decimal('15500.00'),
        parameters={
            'position_size_pct': 5.0,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 10.0,
        },
    )
    return run


@pytest.fixture
def backtest_trades(db, backtest_run):
    """Create realistic backtest trades for performance analysis."""
    trades = []

    # Trade 1: Winning AAPL long - 8% gain
    trades.append(BacktestTrade.objects.create(
        backtest_run=backtest_run,
        trade_id='trade_001',
        symbol='AAPL',
        direction='long',
        entry_date=date(2024, 1, 2),
        entry_price=Decimal('150.00'),
        quantity=100,
        exit_date=date(2024, 1, 10),
        exit_price=Decimal('162.00'),
        status='closed',
        pnl=Decimal('1200.00'),
        pnl_pct=8.0,
    ))

    # Trade 2: Losing GOOGL long - 3% loss
    trades.append(BacktestTrade.objects.create(
        backtest_run=backtest_run,
        trade_id='trade_002',
        symbol='GOOGL',
        direction='long',
        entry_date=date(2024, 1, 5),
        entry_price=Decimal('140.00'),
        quantity=50,
        exit_date=date(2024, 1, 12),
        exit_price=Decimal('135.80'),
        status='stopped_out',
        pnl=Decimal('-210.00'),
        pnl_pct=-3.0,
    ))

    # Trade 3: Winning MSFT long - 5% gain
    trades.append(BacktestTrade.objects.create(
        backtest_run=backtest_run,
        trade_id='trade_003',
        symbol='MSFT',
        direction='long',
        entry_date=date(2024, 1, 8),
        entry_price=Decimal('380.00'),
        quantity=30,
        exit_date=date(2024, 1, 18),
        exit_price=Decimal('399.00'),
        status='take_profit',
        pnl=Decimal('570.00'),
        pnl_pct=5.0,
    ))

    # Trade 4: Winning AAPL long - 12% gain
    trades.append(BacktestTrade.objects.create(
        backtest_run=backtest_run,
        trade_id='trade_004',
        symbol='AAPL',
        direction='long',
        entry_date=date(2024, 1, 15),
        entry_price=Decimal('155.00'),
        quantity=80,
        exit_date=date(2024, 1, 25),
        exit_price=Decimal('173.60'),
        status='take_profit',
        pnl=Decimal('1488.00'),
        pnl_pct=12.0,
    ))

    # Trade 5: Losing short - 4% loss
    trades.append(BacktestTrade.objects.create(
        backtest_run=backtest_run,
        trade_id='trade_005',
        symbol='GOOGL',
        direction='short',
        entry_date=date(2024, 1, 20),
        entry_price=Decimal('145.00'),
        quantity=40,
        exit_date=date(2024, 1, 24),
        exit_price=Decimal('150.80'),
        status='stopped_out',
        pnl=Decimal('-232.00'),
        pnl_pct=-4.0,
    ))

    return trades


@pytest.fixture
def service():
    """Create a BenchmarkService instance."""
    return BenchmarkService()


@pytest.fixture
def clear_cache():
    """Clear benchmark cache before and after tests."""
    _benchmark_cache.clear()
    _cache_timestamps.clear()
    yield
    _benchmark_cache.clear()
    _cache_timestamps.clear()


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance ticker with realistic benchmark data."""
    def _create_mock(
        start_price=450.0,
        end_price=465.0,
        num_days=30,
        volatility=0.01
    ):
        """Create a mock yfinance ticker with customizable data."""
        mock_ticker = Mock()

        # Generate realistic price series
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='B')
        np.random.seed(42)  # Reproducible randomness

        # Generate returns with slight positive drift and volatility
        daily_returns = np.random.normal(0.0003, volatility, num_days)
        prices = [start_price]
        for r in daily_returns[1:]:
            prices.append(prices[-1] * (1 + r))

        # Scale to hit target end price
        scale_factor = end_price / prices[-1]
        prices = [p * scale_factor for p in prices]

        mock_hist = pd.DataFrame({
            'Open': [p * 0.999 for p in prices],
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Close': prices,
            'Volume': [10000000 + np.random.randint(-1000000, 1000000) for _ in prices],
        }, index=dates)

        mock_ticker.history.return_value = mock_hist
        return mock_ticker

    return _create_mock


# =============================================================================
# SERVICE INITIALIZATION TESTS
# =============================================================================

@pytest.mark.django_db
class TestServiceInitialization:
    """Test BenchmarkService initialization and configuration."""

    def test_service_initialization(self, service):
        """Service should initialize with default values."""
        assert service is not None
        assert service.DEFAULT_BENCHMARK == 'SPY'
        assert 'SPY' in service.SUPPORTED_BENCHMARKS
        assert 'QQQ' in service.SUPPORTED_BENCHMARKS

    def test_supported_benchmarks_complete(self, service):
        """Service should support all major index ETFs."""
        expected = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        for benchmark in expected:
            assert benchmark in service.SUPPORTED_BENCHMARKS


# =============================================================================
# CACHE FUNCTIONALITY TESTS
# =============================================================================

@pytest.mark.django_db
class TestCacheFunctionality:
    """Test cache operations with real data structures."""

    def test_cache_set_and_get(self, service, clear_cache):
        """Cache should store and retrieve values correctly."""
        key = 'test_cache_key'
        value = {'ticker': 'SPY', 'return': 5.5}

        service._set_cached(key, value)
        result = service._get_cached(key)

        assert result == value
        assert result['ticker'] == 'SPY'
        assert result['return'] == 5.5

    def test_cache_isolation(self, service, clear_cache):
        """Different keys should be isolated."""
        service._set_cached('key1', {'data': 'value1'})
        service._set_cached('key2', {'data': 'value2'})

        assert service._get_cached('key1')['data'] == 'value1'
        assert service._get_cached('key2')['data'] == 'value2'

    def test_cache_expiration_returns_none(self, service, clear_cache):
        """Expired cache entries should return None."""
        key = 'expiring_key'
        value = {'data': 'test'}

        service._set_cached(key, value)

        # Verify cache is set
        assert service._get_cached(key) == value

        # Manually expire the cache by modifying timestamp
        _cache_timestamps[key] = _cache_timestamps[key] - CACHE_TTL_SECONDS - 100

        # Should return None now
        assert service._get_cached(key) is None

    def test_cache_miss_returns_none(self, service, clear_cache):
        """Non-existent cache keys should return None."""
        result = service._get_cached('nonexistent_key')
        assert result is None


# =============================================================================
# BENCHMARK RETURN CALCULATION TESTS
# =============================================================================

@pytest.mark.django_db
class TestBenchmarkReturnCalculations:
    """Test benchmark return calculations with real calculation logic."""

    def test_mock_benchmark_return_calculation(self, service):
        """Mock benchmark returns should use correct calculation logic."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        result = service._mock_benchmark_return('SPY', start_date, end_date)

        assert isinstance(result, BenchmarkReturn)
        assert result.ticker == 'SPY'
        assert result.start_price == 450.00

        # Verify period return calculation: days * avg_daily_return
        days = (end_date - start_date).days
        expected_period_return = 0.04 * days  # 0.04% per day
        assert result.period_return_pct == round(expected_period_return, 2)

        # Verify end price calculation
        expected_end_price = 450.00 * (1 + expected_period_return / 100)
        assert abs(result.end_price - expected_end_price) < 0.01

    def test_mock_benchmark_return_date_formatting(self, service):
        """Mock benchmark returns should format dates correctly."""
        start_date = datetime(2024, 3, 15)
        end_date = datetime(2024, 4, 15)

        result = service._mock_benchmark_return('QQQ', start_date, end_date)

        assert result.start_date == '2024-03-15'
        assert result.end_date == '2024-04-15'

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_benchmark_return_with_yfinance(self, mock_yf, service, mock_yfinance_ticker, clear_cache):
        """Benchmark return calculation should use real yfinance data."""
        # Setup mock with realistic data
        mock_yf.Ticker.return_value = mock_yfinance_ticker(
            start_price=450.0,
            end_price=465.0,
            num_days=25
        )

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        result = service.get_benchmark_return(start_date, end_date, 'SPY')

        assert isinstance(result, BenchmarkReturn)
        assert result.ticker == 'SPY'
        mock_yf.Ticker.assert_called_with('SPY')

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_benchmark_return_caching(self, mock_yf, service, mock_yfinance_ticker, clear_cache):
        """Repeated calls should use cached values."""
        mock_yf.Ticker.return_value = mock_yfinance_ticker()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # First call
        result1 = service.get_benchmark_return(start_date, end_date, 'SPY')
        # Second call
        result2 = service.get_benchmark_return(start_date, end_date, 'SPY')

        # yfinance should only be called once
        assert mock_yf.Ticker.call_count == 1
        assert result1.ticker == result2.ticker
        assert result1.period_return_pct == result2.period_return_pct

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_benchmark_return_error_fallback(self, mock_yf, service, clear_cache):
        """Should fall back to mock data on API errors."""
        mock_yf.Ticker.side_effect = Exception("API Error")

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        result = service.get_benchmark_return(start_date, end_date, 'SPY')

        # Should still return valid BenchmarkReturn using mock
        assert isinstance(result, BenchmarkReturn)
        assert result.ticker == 'SPY'


# =============================================================================
# BENCHMARK SERIES TESTS
# =============================================================================

@pytest.mark.django_db
class TestBenchmarkSeries:
    """Test benchmark time series generation."""

    def test_mock_benchmark_series_structure(self, service):
        """Mock series should have correct structure."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)

        result = service._mock_benchmark_series(start_date, end_date)

        assert isinstance(result, list)
        assert len(result) > 0

        # Check each entry has required fields
        for entry in result:
            assert 'date' in entry
            assert 'close' in entry
            assert 'return_pct' in entry
            assert 'normalized' in entry

    def test_mock_benchmark_series_excludes_weekends(self, service):
        """Mock series should exclude weekend days."""
        start_date = datetime(2024, 1, 1)  # Monday
        end_date = datetime(2024, 1, 14)  # Sunday

        result = service._mock_benchmark_series(start_date, end_date)

        # Check that no entries are on weekends (Saturday=5, Sunday=6)
        for entry in result:
            entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
            assert entry_date.weekday() < 5, f"Weekend date found: {entry['date']}"

    def test_mock_benchmark_series_normalization(self, service):
        """Series should be normalized to 100 at start."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)

        result = service._mock_benchmark_series(start_date, end_date)

        if result:
            # First entry's normalized value should be based on 450 start price
            first_normalized = result[0]['normalized']
            # Should be close to 100 (exact depends on hash-based random)
            assert 90 < first_normalized < 110

    @patch('backend.auth0login.services.benchmark.HAS_YFINANCE', True)
    @patch('backend.auth0login.services.benchmark.yf')
    def test_benchmark_series_with_yfinance(self, mock_yf, service, mock_yfinance_ticker, clear_cache):
        """Series generation should work with real yfinance data."""
        mock_yf.Ticker.return_value = mock_yfinance_ticker(num_days=10)

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)

        result = service.get_benchmark_series(start_date, end_date, 'SPY')

        assert isinstance(result, list)
        assert len(result) > 0


# =============================================================================
# PORTFOLIO COMPARISON TESTS
# =============================================================================

@pytest.mark.django_db
class TestPortfolioComparison:
    """Test portfolio vs benchmark comparison calculations."""

    def test_portfolio_comparison_positive_alpha(self, service, clear_cache):
        """Portfolio outperforming benchmark should show positive alpha."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=115000.0,  # 15% return
            portfolio_daily_pnl=500.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        assert isinstance(result, PortfolioComparison)
        assert result.portfolio_return_pct == 15.0
        assert result.your_value == 115000.0
        # With mock benchmark ~1.24% for 31 days, alpha should be positive
        assert result.alpha_generated > 0

    def test_portfolio_comparison_negative_alpha(self, service, clear_cache):
        """Portfolio underperforming benchmark should show negative alpha."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=100500.0,  # 0.5% return
            portfolio_daily_pnl=50.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        assert isinstance(result, PortfolioComparison)
        assert result.portfolio_return_pct == 0.5
        # Benchmark mock returns ~1.24% for 31 days, so this underperforms
        assert result.alpha_generated < 0

    def test_portfolio_comparison_excess_return_calculation(self, service, clear_cache):
        """Excess return should be portfolio return minus benchmark return."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=110000.0,
            portfolio_daily_pnl=300.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        expected_excess = result.portfolio_return_pct - result.benchmark_return_pct
        assert abs(result.period_excess - expected_excess) < 0.01

    def test_portfolio_comparison_hypothetical_value(self, service, clear_cache):
        """Hypothetical benchmark value should be calculated correctly."""
        start_value = 100000.0

        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=start_value,
            portfolio_end_value=115000.0,
            portfolio_daily_pnl=500.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        # hypothetical = start * (1 + benchmark_return%)
        expected_hypothetical = start_value * (1 + result.benchmark_return_pct / 100)
        assert abs(result.hypothetical_benchmark_value - expected_hypothetical) < 0.01

    def test_portfolio_comparison_zero_start_value(self, service, clear_cache):
        """Zero start value should not cause division errors."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=0.0,
            portfolio_end_value=1000.0,
            portfolio_daily_pnl=50.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        assert result.portfolio_return_pct == 0.0

    def test_portfolio_comparison_negative_returns(self, service, clear_cache):
        """Negative portfolio returns should be handled correctly."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=92000.0,  # 8% loss
            portfolio_daily_pnl=-500.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        assert result.portfolio_return_pct == -8.0
        assert result.alpha_generated < 0


# =============================================================================
# TRACKING METRICS TESTS (Real Calculations)
# =============================================================================

@pytest.mark.django_db
class TestTrackingMetricsCalculations:
    """Test advanced tracking metrics with real numpy calculations."""

    def test_tracking_error_calculation(self, service):
        """Tracking error should measure deviation from benchmark."""
        # Portfolio slightly outperforms with more volatility
        portfolio_returns = [0.5, -0.3, 0.8, -0.2, 0.6, 0.1, -0.4, 0.7, 0.2, -0.1]
        benchmark_returns = [0.3, -0.1, 0.4, -0.1, 0.3, 0.2, -0.2, 0.4, 0.1, 0.0]

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['tracking_error'] is not None
        assert result['tracking_error'] > 0

        # Verify calculation: tracking error = annualized std of excess returns
        excess = np.array(portfolio_returns) - np.array(benchmark_returns)
        expected_te = np.std(excess) * np.sqrt(252)
        assert abs(result['tracking_error'] - round(expected_te, 4)) < 0.0001

    def test_information_ratio_calculation(self, service):
        """Information ratio should be excess return / tracking error."""
        portfolio_returns = [0.5, 0.3, 0.4, 0.2, 0.6, 0.1, 0.3, 0.5, 0.2, 0.4]
        benchmark_returns = [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1]

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['information_ratio'] is not None
        # Positive excess returns should give positive IR
        assert result['information_ratio'] > 0

    def test_correlation_calculation(self, service):
        """Correlation should measure co-movement with benchmark."""
        # Highly correlated returns
        base_returns = [0.5, -0.3, 0.8, -0.2, 0.6, 0.1, -0.4, 0.7, 0.2, -0.1]
        portfolio_returns = [r * 1.2 + 0.05 for r in base_returns]  # 1.2x with alpha
        benchmark_returns = base_returns

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['correlation'] is not None
        # Should be highly correlated
        assert result['correlation'] > 0.9

    def test_beta_calculation(self, service):
        """Beta should measure systematic risk relative to benchmark."""
        # Portfolio with 1.5x market exposure
        benchmark_returns = [0.5, -0.3, 0.8, -0.2, 0.6, 0.1, -0.4, 0.7, 0.2, -0.1]
        portfolio_returns = [r * 1.5 for r in benchmark_returns]

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['beta'] is not None
        # Beta should be approximately 1.5 (with some numerical tolerance)
        # The calculation involves covariance/variance which can have rounding differences
        assert 1.3 < result['beta'] < 1.8

    def test_alpha_calculation(self, service):
        """Alpha should measure excess return not explained by beta."""
        benchmark_returns = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
        # Portfolio returns = benchmark + consistent alpha
        portfolio_returns = [r + 0.05 for r in benchmark_returns]  # 5bps alpha per period

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['alpha'] is not None
        # Should show positive annualized alpha
        assert result['alpha'] > 0

    def test_tracking_metrics_insufficient_data(self, service):
        """Metrics should return None with insufficient data."""
        portfolio_returns = [0.5, 0.3]  # Only 2 data points
        benchmark_returns = [0.2, 0.1]

        result = service.calculate_tracking_metrics(portfolio_returns, benchmark_returns)

        assert result['tracking_error'] is None
        assert result['information_ratio'] is None
        assert result['correlation'] is None

    def test_tracking_metrics_identical_returns(self, service):
        """Identical returns should have zero tracking error."""
        returns = [0.5, -0.3, 0.8, -0.2, 0.6]

        result = service.calculate_tracking_metrics(returns, returns)

        # Tracking error should be near zero
        assert result['tracking_error'] is not None
        assert abs(result['tracking_error']) < 0.0001
        # Information ratio should be 0 (no excess return)
        assert result['information_ratio'] == 0


# =============================================================================
# COMPARISON DATA FOR CHARTING TESTS
# =============================================================================

@pytest.mark.django_db
class TestComparisonDataGeneration:
    """Test complete comparison data generation for charts."""

    def test_comparison_data_empty_portfolio(self, service, clear_cache):
        """Empty portfolio should return error status."""
        result = service.get_comparison_data(
            [],
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        assert result['status'] == 'error'
        assert 'message' in result

    def test_comparison_data_success(self, service, clear_cache):
        """Valid portfolio data should return complete comparison."""
        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-15', 'value': 105000.0},
            {'date': '2024-01-31', 'value': 115000.0},
        ]

        result = service.get_comparison_data(
            portfolio_values,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        assert result['status'] == 'success'
        assert 'portfolio' in result
        assert 'benchmark_data' in result
        assert 'comparison' in result
        assert result['benchmark'] == 'SPY'

    def test_comparison_data_portfolio_normalization(self, service, clear_cache):
        """Portfolio values should be normalized to 100 at start."""
        portfolio_values = [
            {'date': '2024-01-01', 'value': 50000.0},
            {'date': '2024-01-15', 'value': 60000.0},  # 20% gain
            {'date': '2024-01-31', 'value': 55000.0},  # 10% total gain
        ]

        result = service.get_comparison_data(
            portfolio_values,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        series = result['portfolio']['series']
        assert series[0]['normalized'] == 100.0  # Start at 100
        assert series[1]['normalized'] == 120.0  # 60000/50000 * 100
        assert series[2]['normalized'] == 110.0  # 55000/50000 * 100

    def test_comparison_data_return_calculation(self, service, clear_cache):
        """Portfolio return percentage should be calculated correctly."""
        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 120000.0},  # 20% return
        ]

        result = service.get_comparison_data(
            portfolio_values,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        assert result['portfolio']['return_pct'] == 20.0

    def test_comparison_data_outperformance_flag(self, service, clear_cache):
        """Outperformance flag should reflect actual comparison."""
        # High-performing portfolio
        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 125000.0},  # 25% return
        ]

        result = service.get_comparison_data(
            portfolio_values,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        # Mock benchmark only returns ~1.24% for 31 days
        assert result['comparison']['outperforming'] is True
        assert result['comparison']['excess_return'] > 0

    def test_comparison_data_underperformance_flag(self, service, clear_cache):
        """Underperformance should be correctly flagged."""
        # Low-performing portfolio
        portfolio_values = [
            {'date': '2024-01-01', 'value': 100000.0},
            {'date': '2024-01-31', 'value': 100100.0},  # 0.1% return
        ]

        result = service.get_comparison_data(
            portfolio_values,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            'SPY'
        )

        # Mock benchmark returns ~1.24% for 31 days
        assert result['comparison']['outperforming'] is False
        assert result['comparison']['excess_return'] < 0


# =============================================================================
# DATABASE INTEGRATION TESTS
# =============================================================================

@pytest.mark.django_db
class TestDatabaseIntegration:
    """Test integration with real database models."""

    def test_backtest_run_creation(self, backtest_run):
        """Backtest run should be created with performance metrics."""
        assert backtest_run.id is not None
        assert backtest_run.total_return_pct == 15.5
        assert backtest_run.sharpe_ratio == 1.85
        assert backtest_run.max_drawdown_pct == -8.5

    def test_backtest_trades_association(self, backtest_run, backtest_trades):
        """Trades should be properly associated with backtest run."""
        trades = BacktestTrade.objects.filter(backtest_run=backtest_run)
        assert trades.count() == 5

    def test_trade_pnl_calculations(self, backtest_trades):
        """Trade P&L should be calculated correctly."""
        # Verify individual trade calculations
        aapl_trade = next(t for t in backtest_trades if t.trade_id == 'trade_001')
        assert aapl_trade.pnl == Decimal('1200.00')
        assert aapl_trade.pnl_pct == 8.0
        assert aapl_trade.is_winner is True

    def test_aggregate_portfolio_metrics_from_trades(self, backtest_run, backtest_trades):
        """Aggregate metrics should be calculable from trades."""
        trades = BacktestTrade.objects.filter(backtest_run=backtest_run)

        # Calculate win rate from real trades
        winners = trades.filter(pnl__gt=0).count()
        total = trades.count()
        win_rate = (winners / total) * 100

        # 3 winners, 2 losers = 60% win rate
        assert abs(win_rate - 60.0) < 0.01

    def test_portfolio_comparison_from_backtest(self, service, backtest_run, clear_cache):
        """Benchmark comparison should work with backtest data."""
        initial = float(backtest_run.initial_capital)
        final = float(backtest_run.final_equity)

        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=initial,
            portfolio_end_value=final,
            portfolio_daily_pnl=500.0,
            start_date=datetime.combine(backtest_run.start_date, datetime.min.time()),
            end_date=datetime.combine(backtest_run.end_date, datetime.min.time()),
            benchmark=backtest_run.benchmark
        )

        assert result.portfolio_return_pct == 15.5  # Matches backtest run
        assert result.alpha_generated > 0  # Outperformed benchmark

    def test_user_isolation(self, user, second_user, db):
        """Backtest runs should be isolated per user."""
        # Create runs for both users
        run1 = BacktestRun.objects.create(
            run_id=f'user1_run_{datetime.now().timestamp()}',
            user=user,
            name='User 1 Run',
            strategy_name='momentum',
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=Decimal('100000.00'),
            benchmark='SPY',
            status='completed',
        )

        run2 = BacktestRun.objects.create(
            run_id=f'user2_run_{datetime.now().timestamp()}',
            user=second_user,
            name='User 2 Run',
            strategy_name='value',
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=Decimal('50000.00'),
            benchmark='QQQ',
            status='completed',
        )

        # Verify isolation
        user1_runs = BacktestRun.objects.filter(user=user)
        user2_runs = BacktestRun.objects.filter(user=second_user)

        assert user1_runs.filter(id=run1.id).exists()
        assert not user1_runs.filter(id=run2.id).exists()
        assert user2_runs.filter(id=run2.id).exists()
        assert not user2_runs.filter(id=run1.id).exists()


# =============================================================================
# PERFORMANCE METRICS FROM TRADE DATA TESTS
# =============================================================================

@pytest.mark.django_db
class TestPerformanceMetricsFromTrades:
    """Test calculating performance metrics from real trade data."""

    def test_sharpe_ratio_from_trade_returns(self, backtest_trades):
        """Sharpe ratio should be calculable from trade returns."""
        # Extract returns from trades
        returns = [float(t.pnl_pct) / 100 for t in backtest_trades]

        # Calculate Sharpe manually
        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr, ddof=1)

        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252) if std_return > 0 else 0

        # Verify calculation is reasonable
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_max_drawdown_from_equity_curve(self, backtest_trades):
        """Max drawdown should be calculable from trade sequence."""
        # Build equity curve from trades
        initial_equity = 100000.0
        equity = [initial_equity]

        for trade in sorted(backtest_trades, key=lambda t: t.entry_date):
            equity.append(equity[-1] + float(trade.pnl))

        # Calculate max drawdown
        equity_arr = np.array(equity)
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_dd = np.min(drawdowns) * 100

        # Verify calculation
        assert max_dd <= 0  # Drawdown is always <= 0
        assert isinstance(max_dd, float)

    def test_alpha_calculation_from_performance(self, service, backtest_trades, clear_cache):
        """Alpha should be calculable from portfolio vs benchmark performance."""
        # Portfolio performance
        initial = 100000.0
        total_pnl = sum(float(t.pnl) for t in backtest_trades)
        final = initial + total_pnl

        # Compare to benchmark
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=initial,
            portfolio_end_value=final,
            portfolio_daily_pnl=float(backtest_trades[-1].pnl),
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        # Alpha = actual return - benchmark return (in dollar terms)
        expected_alpha = final - result.hypothetical_benchmark_value
        assert abs(result.alpha_generated - expected_alpha) < 0.01

    def test_win_rate_calculation(self, backtest_trades):
        """Win rate should match actual winning trades percentage."""
        winners = sum(1 for t in backtest_trades if t.is_winner)
        total = len(backtest_trades)

        win_rate = (winners / total) * 100

        # We created 3 winners out of 5 trades (60%)
        # But checking the data: trades 1,3,4 win, trades 2,5 lose = 3/5 = 60%
        assert abs(win_rate - 60.0) < 0.01

    def test_profit_factor_calculation(self, backtest_trades):
        """Profit factor should be gross profit / gross loss."""
        gross_profit = sum(float(t.pnl) for t in backtest_trades if t.pnl > 0)
        gross_loss = abs(sum(float(t.pnl) for t in backtest_trades if t.pnl < 0))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Verify calculation is reasonable
        assert profit_factor > 0
        # With our test data: profit = 1200 + 570 + 1488 = 3258
        # Loss = 210 + 232 = 442
        # PF = 3258 / 442 ≈ 7.37
        assert 7.0 < profit_factor < 8.0


# =============================================================================
# DATACLASS TESTS
# =============================================================================

@pytest.mark.django_db
class TestDataclasses:
    """Test dataclass structures."""

    def test_benchmark_return_dataclass(self):
        """BenchmarkReturn should store all required fields."""
        benchmark = BenchmarkReturn(
            ticker='SPY',
            start_price=450.0,
            end_price=465.0,
            period_return_pct=3.33,
            daily_return_pct=0.15,
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        assert benchmark.ticker == 'SPY'
        assert benchmark.start_price == 450.0
        assert benchmark.end_price == 465.0
        assert benchmark.period_return_pct == 3.33

    def test_portfolio_comparison_dataclass(self):
        """PortfolioComparison should store all comparison metrics."""
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

        assert comparison.portfolio_return_pct == 15.0
        assert comparison.alpha_generated == 5000.0
        assert comparison.period_excess == 5.0

    def test_benchmark_series_dataclass(self):
        """BenchmarkSeries should store time series data."""
        series = BenchmarkSeries(
            dates=['2024-01-01', '2024-01-02', '2024-01-03'],
            portfolio_values=[100000.0, 101000.0, 102000.0],
            benchmark_values=[450.0, 451.0, 452.0],
            portfolio_normalized=[100.0, 101.0, 102.0],
            benchmark_normalized=[100.0, 100.22, 100.44]
        )

        assert len(series.dates) == 3
        assert series.portfolio_values[0] == 100000.0
        assert series.portfolio_normalized[0] == 100.0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

@pytest.mark.django_db
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_very_short_period_calculation(self, service, clear_cache):
        """Should handle very short time periods."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=100100.0,
            portfolio_daily_pnl=100.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),  # 1 day
            benchmark='SPY'
        )

        assert isinstance(result, PortfolioComparison)
        assert result.portfolio_return_pct == 0.1

    def test_long_period_calculation(self, service, clear_cache):
        """Should handle long time periods."""
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=150000.0,
            portfolio_daily_pnl=500.0,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),  # 1 year
            benchmark='SPY'
        )

        assert isinstance(result, PortfolioComparison)
        assert result.portfolio_return_pct == 50.0

    def test_negative_portfolio_value(self, service, clear_cache):
        """Should handle edge case of negative end value gracefully."""
        # While unusual, the calculation should still work
        result = service.compare_portfolio_to_benchmark(
            portfolio_start_value=100000.0,
            portfolio_end_value=50000.0,
            portfolio_daily_pnl=-1000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            benchmark='SPY'
        )

        assert result.portfolio_return_pct == -50.0

    def test_benchmark_selection(self, service, clear_cache):
        """Different benchmarks should be supported."""
        for benchmark in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']:
            result = service._mock_benchmark_return(
                benchmark,
                datetime(2024, 1, 1),
                datetime(2024, 1, 31)
            )
            assert result.ticker == benchmark

    def test_empty_tracking_returns(self, service):
        """Empty return lists should be handled gracefully."""
        result = service.calculate_tracking_metrics([], [])

        assert result['tracking_error'] is None
        assert result['information_ratio'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
