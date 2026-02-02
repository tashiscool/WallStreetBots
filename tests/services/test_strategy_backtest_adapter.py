"""
Comprehensive tests for CustomStrategyBacktestAdapter.

Tests conversion of custom strategies to backtest format, data fetching, and execution.
Target: 80%+ coverage.
"""
import asyncio
import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np


class TestCustomStrategyBacktestConfig(unittest.TestCase):
    """Test CustomStrategyBacktestConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestConfig

        mock_strategy = Mock()
        config = CustomStrategyBacktestConfig(
            strategy=mock_strategy,
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        self.assertEqual(config.start_date, date(2023, 1, 1))
        self.assertEqual(config.end_date, date(2024, 1, 1))
        self.assertEqual(config.initial_capital, Decimal("100000"))
        self.assertEqual(config.benchmark, "SPY")

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestConfig

        mock_strategy = Mock()
        config = CustomStrategyBacktestConfig(
            strategy=mock_strategy,
            start_date=date(2022, 6, 1),
            end_date=date(2023, 6, 1),
            initial_capital=Decimal("50000"),
            benchmark="QQQ"
        )

        self.assertEqual(config.initial_capital, Decimal("50000"))
        self.assertEqual(config.benchmark, "QQQ")


class TestUniverseSymbols(unittest.TestCase):
    """Test UNIVERSE_SYMBOLS mapping."""

    def test_universe_symbols_exist(self):
        """Test that all expected universes exist."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        expected_universes = ['sp500', 'nasdaq100', 'dow30', 'russell2000', 'all']

        for universe in expected_universes:
            self.assertIn(universe, CustomStrategyBacktestAdapter.UNIVERSE_SYMBOLS)
            self.assertIsInstance(CustomStrategyBacktestAdapter.UNIVERSE_SYMBOLS[universe], list)
            self.assertGreater(len(CustomStrategyBacktestAdapter.UNIVERSE_SYMBOLS[universe]), 0)


class TestAdapterInitialization(unittest.TestCase):
    """Test CustomStrategyBacktestAdapter initialization."""

    def test_initialization(self):
        """Test adapter can be initialized with a strategy."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30}
            ]
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)

        self.assertEqual(adapter.strategy, mock_strategy)
        self.assertIsNotNone(adapter.runner)
        self.assertEqual(adapter._price_cache, {})


class TestGetSymbols(unittest.TestCase):
    """Test get_symbols method."""

    def test_get_symbols_sp500(self):
        """Test get_symbols for S&P 500 universe."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.universe = 'sp500'
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        symbols = adapter.get_symbols()

        self.assertIsInstance(symbols, list)
        self.assertIn('AAPL', symbols)

    def test_get_symbols_custom(self):
        """Test get_symbols for custom universe."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.universe = 'custom'
        mock_strategy.custom_symbols = ['TSLA', 'NVDA', 'AMD']
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        symbols = adapter.get_symbols()

        self.assertEqual(symbols, ['TSLA', 'NVDA', 'AMD'])

    def test_get_symbols_custom_empty(self):
        """Test get_symbols with empty custom symbols."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.universe = 'custom'
        mock_strategy.custom_symbols = None
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        symbols = adapter.get_symbols()

        self.assertEqual(symbols, ['SPY'])


class TestPositionSizing(unittest.TestCase):
    """Test position sizing methods."""

    def test_get_position_size_pct_default(self):
        """Test default position size percentage."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        size = adapter.get_position_size_pct()

        self.assertEqual(size, 5.0)

    def test_get_position_size_pct_custom(self):
        """Test custom position size percentage."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'position_sizing': {'value': 10.0}
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        size = adapter.get_position_size_pct()

        self.assertEqual(size, 10.0)

    def test_get_max_positions_default(self):
        """Test default max positions."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        max_pos = adapter.get_max_positions()

        self.assertEqual(max_pos, 5)

    def test_get_max_positions_custom(self):
        """Test custom max positions."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'position_sizing': {'max_positions': 10}
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        max_pos = adapter.get_max_positions()

        self.assertEqual(max_pos, 10)


class TestStopLossAndTakeProfit(unittest.TestCase):
    """Test stop loss and take profit extraction."""

    def test_get_stop_loss_pct_exists(self):
        """Test getting stop loss percentage when defined."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 8.0}
            ]
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        stop_loss = adapter.get_stop_loss_pct()

        self.assertEqual(stop_loss, 8.0)

    def test_get_stop_loss_pct_not_defined(self):
        """Test getting stop loss when not defined."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'exit_conditions': []
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        stop_loss = adapter.get_stop_loss_pct()

        self.assertIsNone(stop_loss)

    def test_get_take_profit_pct_exists(self):
        """Test getting take profit percentage when defined."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {
            'exit_conditions': [
                {'type': 'take_profit', 'value': 20.0}
            ]
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        take_profit = adapter.get_take_profit_pct()

        self.assertEqual(take_profit, 20.0)

    def test_get_take_profit_pct_not_defined(self):
        """Test getting take profit when not defined."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        take_profit = adapter.get_take_profit_pct()

        self.assertIsNone(take_profit)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test synthetic data generation."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        data = adapter._generate_synthetic_data(
            ['AAPL', 'MSFT'],
            date(2023, 1, 1),
            date(2023, 3, 1)
        )

        self.assertIn('AAPL', data)
        self.assertIn('MSFT', data)

        for symbol, df in data.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('open', df.columns)
            self.assertIn('high', df.columns)
            self.assertIn('low', df.columns)
            self.assertIn('close', df.columns)
            self.assertIn('volume', df.columns)

    def test_synthetic_data_price_relationships(self):
        """Test that synthetic data has valid price relationships."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)
        data = adapter._generate_synthetic_data(
            ['TEST'],
            date(2023, 1, 1),
            date(2023, 2, 1)
        )

        df = data['TEST']
        # High should be >= close and open
        # Low should be <= close and open
        self.assertTrue((df['high'] >= df['close']).all())
        self.assertTrue((df['low'] <= df['close']).all())


class TestFetchPriceData(unittest.TestCase):
    """Test price data fetching."""

    def test_fetch_price_data_no_yfinance(self):
        """Test fetch_price_data falls back to synthetic when yfinance unavailable."""
        from backend.auth0login.services.strategy_backtest_adapter import CustomStrategyBacktestAdapter

        mock_strategy = Mock()
        mock_strategy.definition = {}

        adapter = CustomStrategyBacktestAdapter(mock_strategy)

        # Patch HAS_YFINANCE to False
        with patch.object(
            __import__('backend.auth0login.services.strategy_backtest_adapter', fromlist=['HAS_YFINANCE']),
            'HAS_YFINANCE',
            False
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                data = loop.run_until_complete(adapter.fetch_price_data(
                    ['AAPL'],
                    date(2023, 1, 1),
                    date(2023, 2, 1)
                ))
            finally:
                loop.close()

            self.assertIn('AAPL', data)


class TestBacktestExecution(unittest.TestCase):
    """Test backtest execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_strategy = Mock()
        self.mock_strategy.id = 1
        self.mock_strategy.name = "Test Strategy"
        self.mock_strategy.universe = 'sp500'
        self.mock_strategy.definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30, 'period': 14}
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
                {'type': 'take_profit', 'value': 15}
            ],
            'position_sizing': {'value': 5, 'max_positions': 5}
        }

    def test_run_backtest_returns_results(self):
        """Test that run_backtest returns BacktestResults."""
        from backend.auth0login.services.strategy_backtest_adapter import (
            CustomStrategyBacktestAdapter,
            CustomStrategyBacktestConfig
        )

        adapter = CustomStrategyBacktestAdapter(self.mock_strategy)

        config = CustomStrategyBacktestConfig(
            strategy=self.mock_strategy,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 3, 1),
            initial_capital=Decimal("100000")
        )

        # Run with synthetic data (no yfinance)
        with patch.object(
            __import__('backend.auth0login.services.strategy_backtest_adapter', fromlist=['HAS_YFINANCE']),
            'HAS_YFINANCE',
            False
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    adapter.run_backtest(config, save_to_db=False)
                )
            finally:
                loop.close()

        # Verify result structure
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.config)
        self.assertIsInstance(results.total_return_pct, float)
        self.assertIsInstance(results.sharpe_ratio, float)
        self.assertIsInstance(results.max_drawdown_pct, float)

    def test_run_backtest_with_progress_callback(self):
        """Test backtest with progress callback."""
        from backend.auth0login.services.strategy_backtest_adapter import (
            CustomStrategyBacktestAdapter,
            CustomStrategyBacktestConfig
        )

        adapter = CustomStrategyBacktestAdapter(self.mock_strategy)

        config = CustomStrategyBacktestConfig(
            strategy=self.mock_strategy,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 1),
        )

        progress_updates = []

        def progress_callback(pct, message):
            progress_updates.append((pct, message))

        with patch.object(
            __import__('backend.auth0login.services.strategy_backtest_adapter', fromlist=['HAS_YFINANCE']),
            'HAS_YFINANCE',
            False
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    adapter.run_backtest(config, progress_callback=progress_callback, save_to_db=False)
                )
            finally:
                loop.close()

        # Verify progress was reported
        self.assertGreater(len(progress_updates), 0)
        # Should end at 100%
        self.assertEqual(progress_updates[-1][0], 100)


class TestMetricsCalculation(unittest.TestCase):
    """Test metrics calculation in backtest results."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio is calculated correctly."""
        from backend.auth0login.services.strategy_backtest_adapter import (
            CustomStrategyBacktestAdapter,
            CustomStrategyBacktestConfig
        )

        mock_strategy = Mock()
        mock_strategy.id = 1
        mock_strategy.name = "Test"
        mock_strategy.universe = 'custom'
        mock_strategy.custom_symbols = ['TEST']
        mock_strategy.definition = {
            'entry_conditions': [{'indicator': 'rsi', 'operator': 'less_than', 'value': 100}],
            'exit_conditions': [{'type': 'stop_loss', 'value': 5}],
        }

        adapter = CustomStrategyBacktestAdapter(mock_strategy)

        config = CustomStrategyBacktestConfig(
            strategy=mock_strategy,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 6, 1),
        )

        with patch.object(
            __import__('backend.auth0login.services.strategy_backtest_adapter', fromlist=['HAS_YFINANCE']),
            'HAS_YFINANCE',
            False
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    adapter.run_backtest(config, save_to_db=False)
                )
            finally:
                loop.close()

        # Sharpe ratio should be a number (can be positive, negative, or zero)
        self.assertIsInstance(results.sharpe_ratio, float)


if __name__ == '__main__':
    unittest.main()
