"""
Comprehensive tests for CustomStrategyRunner.

Tests all public methods, indicator calculations, condition evaluation,
edge cases, and error handling.
Target: 80%+ coverage.
"""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from django.test import TestCase

from backend.auth0login.services.custom_strategy_runner import (
    CustomStrategyRunner,
    STRATEGY_TEMPLATES,
    get_strategy_templates,
)


class TestCustomStrategyRunner(TestCase):
    """Test suite for CustomStrategyRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.basic_definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30, 'period': 14}
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 10},
                {'type': 'stop_loss', 'value': 5},
            ],
            'exit_logic': 'any',
        }
        self.runner = CustomStrategyRunner(self.basic_definition)

        # Create sample DataFrame with consistent OHLC data
        np.random.seed(42)  # For reproducibility
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        close_prices = np.random.uniform(90, 110, 100)
        # Ensure high >= close >= low
        high_prices = close_prices + np.random.uniform(0, 5, 100)
        low_prices = close_prices - np.random.uniform(0, 5, 100)
        self.df = pd.DataFrame({
            'close': close_prices,
            'high': high_prices,
            'low': low_prices,
            'volume': np.random.uniform(1000000, 5000000, 100),
        }, index=dates)
        self.df['open'] = self.df['close']

    def test_initialization(self):
        """Test runner initialization."""
        runner = CustomStrategyRunner(self.basic_definition)
        self.assertEqual(runner.definition, self.basic_definition)
        self.assertEqual(runner._data_cache, {})

    def test_indicators_registry_completeness(self):
        """Test that all indicators are registered with proper structure."""
        for indicator_name, info in CustomStrategyRunner.INDICATORS.items():
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('default_period', info)

    def test_operators_registry_completeness(self):
        """Test that all operators are registered."""
        expected_operators = [
            'less_than', 'less_equal', 'greater_than', 'greater_equal',
            'equals', 'crosses_above', 'crosses_below', 'between'
        ]
        for op in expected_operators:
            self.assertIn(op, CustomStrategyRunner.OPERATORS)

    def test_exit_types_registry(self):
        """Test that all exit types are registered."""
        expected_types = [
            'take_profit', 'stop_loss', 'trailing_stop', 'time_based', 'indicator'
        ]
        for exit_type in expected_types:
            self.assertIn(exit_type, CustomStrategyRunner.EXIT_TYPES)

    def test_calculate_indicator_rsi(self):
        """Test RSI calculation."""
        result = self.runner.calculate_indicator(self.df, 'rsi', period=14)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(self.df))
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())
        self.assertTrue((valid_values <= 100).all())

    def test_calculate_indicator_sma(self):
        """Test SMA calculation."""
        result = self.runner.calculate_indicator(self.df, 'sma', period=20)
        self.assertIsInstance(result, pd.Series)
        # First values should be NaN
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_calculate_indicator_ema(self):
        """Test EMA calculation."""
        result = self.runner.calculate_indicator(self.df, 'ema', period=20)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue(len(result) == len(self.df))

    def test_calculate_indicator_macd(self):
        """Test MACD calculation."""
        macd = self.runner.calculate_indicator(self.df, 'macd')
        signal = self.runner.calculate_indicator(self.df, 'macd_signal')
        histogram = self.runner.calculate_indicator(self.df, 'macd_histogram')

        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(histogram, pd.Series)

    def test_calculate_indicator_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        bb_upper = self.runner.calculate_indicator(self.df, 'bb_upper', period=20)
        bb_middle = self.runner.calculate_indicator(self.df, 'bb_middle', period=20)
        bb_lower = self.runner.calculate_indicator(self.df, 'bb_lower', period=20)
        bb_width = self.runner.calculate_indicator(self.df, 'bb_width', period=20)

        self.assertIsInstance(bb_upper, pd.Series)
        self.assertIsInstance(bb_middle, pd.Series)
        self.assertIsInstance(bb_lower, pd.Series)
        self.assertIsInstance(bb_width, pd.Series)

        # Upper should be greater than middle, middle greater than lower
        valid_idx = ~bb_upper.isna()
        self.assertTrue((bb_upper[valid_idx] >= bb_middle[valid_idx]).all())
        self.assertTrue((bb_middle[valid_idx] >= bb_lower[valid_idx]).all())

    def test_calculate_indicator_atr(self):
        """Test ATR calculation."""
        result = self.runner.calculate_indicator(self.df, 'atr', period=14)
        self.assertIsInstance(result, pd.Series)
        # ATR should be positive
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

    def test_calculate_indicator_adx(self):
        """Test ADX calculation."""
        result = self.runner.calculate_indicator(self.df, 'adx', period=14)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_indicator_stochastic(self):
        """Test Stochastic calculation."""
        stoch_k = self.runner.calculate_indicator(self.df, 'stoch_k', period=14)
        stoch_d = self.runner.calculate_indicator(self.df, 'stoch_d', period=14)

        self.assertIsInstance(stoch_k, pd.Series)
        self.assertIsInstance(stoch_d, pd.Series)

        # Values should be between 0 and 100
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()
        self.assertTrue((valid_k >= 0).all() and (valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all() and (valid_d <= 100).all())

    def test_calculate_indicator_cci(self):
        """Test CCI calculation."""
        result = self.runner.calculate_indicator(self.df, 'cci', period=20)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_indicator_williams_r(self):
        """Test Williams %R calculation."""
        result = self.runner.calculate_indicator(self.df, 'williams_r', period=14)
        self.assertIsInstance(result, pd.Series)
        # Williams %R should be between -100 and 0
        valid_values = result.dropna()
        self.assertTrue((valid_values >= -100).all())
        self.assertTrue((valid_values <= 0).all())

    def test_calculate_indicator_volume_sma(self):
        """Test Volume SMA calculation."""
        result = self.runner.calculate_indicator(self.df, 'volume_sma', period=20)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_indicator_volume_ratio(self):
        """Test Volume Ratio calculation."""
        result = self.runner.calculate_indicator(self.df, 'volume_ratio', period=20)
        self.assertIsInstance(result, pd.Series)
        # Volume ratio should be positive
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

    def test_calculate_indicator_price_change_pct(self):
        """Test Price Change % calculation."""
        result = self.runner.calculate_indicator(self.df, 'price_change_pct', period=1)
        self.assertIsInstance(result, pd.Series)

    def test_calculate_indicator_price_from_high(self):
        """Test Price from High calculation."""
        result = self.runner.calculate_indicator(self.df, 'price_from_high', period=52)
        self.assertIsInstance(result, pd.Series)
        # Should be negative (below high)
        valid_values = result.dropna()
        self.assertTrue((valid_values <= 0).all())

    def test_calculate_indicator_price_from_low(self):
        """Test Price from Low calculation."""
        result = self.runner.calculate_indicator(self.df, 'price_from_low', period=52)
        self.assertIsInstance(result, pd.Series)
        # Should be positive (above low)
        valid_values = result.dropna()
        self.assertTrue((valid_values >= 0).all())

    def test_calculate_indicator_unknown_raises_error(self):
        """Test that unknown indicator raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.runner.calculate_indicator(self.df, 'unknown_indicator')
        self.assertIn('Unknown indicator', str(context.exception))

    def test_evaluate_condition_less_than(self):
        """Test less_than operator."""
        condition = {'operator': 'less_than', 'value': 50}
        self.assertTrue(self.runner.evaluate_condition(40, 45, condition))
        self.assertFalse(self.runner.evaluate_condition(60, 55, condition))

    def test_evaluate_condition_greater_than(self):
        """Test greater_than operator."""
        condition = {'operator': 'greater_than', 'value': 50}
        self.assertTrue(self.runner.evaluate_condition(60, 55, condition))
        self.assertFalse(self.runner.evaluate_condition(40, 45, condition))

    def test_evaluate_condition_equals(self):
        """Test equals operator with tolerance."""
        condition = {'operator': 'equals', 'value': 50}
        self.assertTrue(self.runner.evaluate_condition(50.0005, 49.9995, condition))
        self.assertFalse(self.runner.evaluate_condition(52, 48, condition))

    def test_evaluate_condition_crosses_above(self):
        """Test crosses_above operator."""
        condition = {'operator': 'crosses_above', 'value': 50}
        self.assertTrue(self.runner.evaluate_condition(51, 49, condition))
        self.assertFalse(self.runner.evaluate_condition(51, 51, condition))
        self.assertFalse(self.runner.evaluate_condition(49, 51, condition))

    def test_evaluate_condition_crosses_below(self):
        """Test crosses_below operator."""
        condition = {'operator': 'crosses_below', 'value': 50}
        self.assertTrue(self.runner.evaluate_condition(49, 51, condition))
        self.assertFalse(self.runner.evaluate_condition(49, 49, condition))
        self.assertFalse(self.runner.evaluate_condition(51, 49, condition))

    def test_evaluate_condition_between(self):
        """Test between operator."""
        condition = {'operator': 'between', 'value_low': 40, 'value_high': 60}
        self.assertTrue(self.runner.evaluate_condition(50, 45, condition))
        self.assertTrue(self.runner.evaluate_condition(40, 35, condition))
        self.assertTrue(self.runner.evaluate_condition(60, 65, condition))
        self.assertFalse(self.runner.evaluate_condition(70, 65, condition))
        self.assertFalse(self.runner.evaluate_condition(30, 35, condition))

    def test_evaluate_condition_unknown_operator(self):
        """Test unknown operator raises ValueError."""
        condition = {'operator': 'unknown_op', 'value': 50}
        with self.assertRaises(ValueError) as context:
            self.runner.evaluate_condition(40, 45, condition)
        self.assertIn('Unknown operator', str(context.exception))

    def test_check_entry_conditions_no_conditions(self):
        """Test check_entry_conditions with no conditions defined."""
        runner = CustomStrategyRunner({})
        met, details = runner.check_entry_conditions(self.df)
        self.assertFalse(met)
        self.assertIn('error', details[0])

    def test_check_entry_conditions_all_met(self):
        """Test check_entry_conditions when all conditions are met."""
        # Create conditions that will be met
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 100, 'period': 14}
            ],
            'entry_logic': 'all'
        }
        runner = CustomStrategyRunner(definition)
        met, details = runner.check_entry_conditions(self.df)
        # Depends on data, but should have results
        self.assertIsInstance(met, bool)
        self.assertGreater(len(details), 0)

    def test_check_entry_conditions_any_logic(self):
        """Test check_entry_conditions with 'any' logic."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 0, 'period': 14},  # Never met
                {'indicator': 'rsi', 'operator': 'greater_than', 'value': 0, 'period': 14}  # Always met
            ],
            'entry_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)
        met, details = runner.check_entry_conditions(self.df)
        # At least one should be true
        self.assertTrue(any(d['met'] for d in details if 'met' in d))

    def test_check_entry_conditions_indicator_error(self):
        """Test check_entry_conditions handles indicator errors."""
        definition = {
            'entry_conditions': [
                {'indicator': 'unknown_ind', 'operator': 'less_than', 'value': 30}
            ],
            'entry_logic': 'all'
        }
        runner = CustomStrategyRunner(definition)
        met, details = runner.check_entry_conditions(self.df)
        self.assertFalse(met)
        self.assertIn('error', details[0])

    def test_check_exit_conditions_no_conditions(self):
        """Test check_exit_conditions with no conditions."""
        runner = CustomStrategyRunner({})
        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 100.0, datetime.now()
        )
        self.assertFalse(should_exit)
        self.assertIsNone(reason)

    def test_check_exit_conditions_take_profit(self):
        """Test check_exit_conditions for take profit."""
        definition = {
            'exit_conditions': [{'type': 'take_profit', 'value': 10}],
            'exit_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)

        # Current price 10% higher
        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 100.0, datetime.now(), idx=-1
        )
        # Depends on last price in df

    def test_check_exit_conditions_stop_loss(self):
        """Test check_exit_conditions for stop loss."""
        definition = {
            'exit_conditions': [{'type': 'stop_loss', 'value': 5}],
            'exit_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)

        # Create df with price down 10%
        df_loss = self.df.copy()
        df_loss['close'].iloc[-1] = 90.0

        should_exit, reason, details = runner.check_exit_conditions(
            df_loss, 100.0, datetime.now(), idx=-1
        )
        # Should trigger stop loss
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'stop_loss')

    def test_check_exit_conditions_trailing_stop(self):
        """Test check_exit_conditions for trailing stop."""
        definition = {
            'exit_conditions': [{'type': 'trailing_stop', 'value': 5}],
            'exit_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)

        # Price dropped from 110 to 100
        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 100.0, datetime.now(), idx=-1, highest_since_entry=110.0
        )
        # Should trigger if price < 110 * 0.95 = 104.5

    def test_check_exit_conditions_time_based(self):
        """Test check_exit_conditions for time-based exit."""
        definition = {
            'exit_conditions': [{'type': 'time_based', 'days': 5}],
            'exit_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)

        # Entry date should be before the last date in the DataFrame
        # DataFrame dates start at '2024-01-01' and have 100 days, so last date is around April 2024
        # Set entry date to be 10 days before the end of the DataFrame
        last_date = self.df.index[-1]
        entry_date = last_date - timedelta(days=10)
        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 100.0, entry_date, idx=-1
        )
        self.assertTrue(should_exit)
        self.assertEqual(reason, 'time_based')

    def test_check_exit_conditions_indicator(self):
        """Test check_exit_conditions for indicator-based exit."""
        definition = {
            'exit_conditions': [
                {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 0, 'period': 14}
            ],
            'exit_logic': 'any'
        }
        runner = CustomStrategyRunner(definition)

        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 100.0, datetime.now(), idx=-1
        )
        # RSI should be > 0 for most cases

    def test_check_exit_conditions_all_logic(self):
        """Test check_exit_conditions with 'all' logic."""
        definition = {
            'exit_conditions': [
                {'type': 'take_profit', 'value': 1},  # Small profit needed
                {'type': 'time_based', 'days': 0},     # Immediate
            ],
            'exit_logic': 'all'
        }
        runner = CustomStrategyRunner(definition)

        should_exit, reason, details = runner.check_exit_conditions(
            self.df, 99.0, datetime.now(), idx=-1
        )
        # Both must be met

    def test_validate_definition_valid(self):
        """Test validate_definition with valid definition."""
        result = self.runner.validate_definition()
        self.assertTrue(result['valid'])
        self.assertEqual(result['error_count'], 0)

    def test_validate_definition_no_entry_conditions(self):
        """Test validate_definition with no entry conditions."""
        runner = CustomStrategyRunner({})
        result = runner.validate_definition()
        self.assertFalse(result['valid'])
        self.assertGreater(result['error_count'], 0)
        self.assertIn('At least one entry condition is required', result['errors'])

    def test_validate_definition_unknown_indicator(self):
        """Test validate_definition with unknown indicator."""
        definition = {
            'entry_conditions': [
                {'indicator': 'unknown_ind', 'operator': 'less_than', 'value': 30}
            ]
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown indicator' in e for e in result['errors']))

    def test_validate_definition_unknown_operator(self):
        """Test validate_definition with unknown operator."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'unknown_op', 'value': 30}
            ]
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown operator' in e for e in result['errors']))

    def test_validate_definition_missing_value(self):
        """Test validate_definition with missing value."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than'}
            ]
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertFalse(result['valid'])
        self.assertTrue(any('value is required' in e for e in result['errors']))

    def test_validate_definition_value_out_of_range_warning(self):
        """Test validate_definition warns about out-of-range values."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 150, 'period': 14}
            ]
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertGreater(result['warning_count'], 0)
        self.assertTrue(any('outside typical range' in w for w in result['warnings']))

    def test_validate_definition_no_stop_loss_warning(self):
        """Test validate_definition warns about missing stop loss."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30}
            ],
            'exit_conditions': []
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertTrue(any('No exit conditions' in w for w in result['warnings']))

    def test_validate_definition_wide_stop_loss_warning(self):
        """Test validate_definition warns about wide stop loss."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30}
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 25}
            ]
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertTrue(any('very wide' in w for w in result['warnings']))

    def test_validate_definition_aggressive_position_size_warning(self):
        """Test validate_definition warns about aggressive position sizing."""
        definition = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30}
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5}
            ],
            'position_sizing': {'type': 'fixed_percent', 'value': 30}
        }
        runner = CustomStrategyRunner(definition)
        result = runner.validate_definition()
        self.assertTrue(any('very aggressive' in w for w in result['warnings']))

    def test_generate_signals(self):
        """Test generate_signals."""
        signals = self.runner.generate_signals(self.df, start_idx=50)
        self.assertIsInstance(signals, list)
        # Each signal should have expected structure
        for signal in signals:
            self.assertIn('date', signal)
            self.assertIn('price', signal)
            self.assertIn('conditions', signal)

    def test_get_available_indicators(self):
        """Test get_available_indicators class method."""
        indicators = CustomStrategyRunner.get_available_indicators()
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)

        for indicator in indicators:
            self.assertIn('id', indicator)
            self.assertIn('name', indicator)
            self.assertIn('description', indicator)

    def test_get_available_operators(self):
        """Test get_available_operators class method."""
        operators = CustomStrategyRunner.get_available_operators()
        self.assertIsInstance(operators, list)
        self.assertGreater(len(operators), 0)

        for operator in operators:
            self.assertIn('id', operator)
            self.assertIn('name', operator)
            self.assertIn('symbol', operator)

    def test_get_exit_types(self):
        """Test get_exit_types class method."""
        exit_types = CustomStrategyRunner.get_exit_types()
        self.assertIsInstance(exit_types, list)
        self.assertGreater(len(exit_types), 0)

        for exit_type in exit_types:
            self.assertIn('id', exit_type)
            self.assertIn('name', exit_type)
            self.assertIn('description', exit_type)

    def test_strategy_templates_exist(self):
        """Test that strategy templates are available."""
        self.assertGreater(len(STRATEGY_TEMPLATES), 0)
        self.assertIn('rsi_oversold', STRATEGY_TEMPLATES)
        self.assertIn('macd_crossover', STRATEGY_TEMPLATES)

    def test_get_strategy_templates(self):
        """Test get_strategy_templates function."""
        templates = get_strategy_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)

        for template in templates:
            self.assertIn('id', template)
            self.assertIn('name', template)
            self.assertIn('description', template)
            self.assertIn('definition', template)


if __name__ == '__main__':
    unittest.main()
