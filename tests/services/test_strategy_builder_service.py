"""
Tests for StrategyBuilderService.

Covers config validation (valid/invalid indicators, operators, thresholds,
exit conditions, position sizing), indicator and operator metadata retrieval,
and preset template correctness.
"""
import unittest

from django.test import TestCase

from backend.auth0login.services.strategy_builder_service import (
    StrategyBuilderService,
    INDICATOR_CATEGORIES,
    INDICATOR_DISPLAY_NAMES,
)
from backend.auth0login.services.custom_strategy_runner import CustomStrategyRunner


class TestValidateConfigValid(TestCase):
    """Test validate_config with valid configurations."""

    def setUp(self):
        self.service = StrategyBuilderService()
        self.valid_config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30, 'period': 14},
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 10},
                {'type': 'stop_loss', 'value': 5},
            ],
            'exit_logic': 'any',
            'position_sizing': {
                'type': 'fixed_percent',
                'value': 5,
                'max_positions': 5,
            },
            'filters': {
                'min_price': 10,
                'min_volume': 500000,
            },
        }

    def test_valid_config_returns_valid(self):
        """A fully valid config should pass validation."""
        result = self.service.validate_config(self.valid_config)
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])

    def test_valid_config_with_multiple_entry_conditions(self):
        """Multiple entry conditions should all pass."""
        config = dict(self.valid_config)
        config['entry_conditions'] = [
            {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            {'indicator': 'macd_histogram', 'operator': 'crosses_above', 'value': 0},
            {'indicator': 'adx', 'operator': 'greater_than', 'value': 25},
        ]
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])

    def test_valid_config_with_between_operator(self):
        """The between operator with value_low/value_high should pass."""
        config = dict(self.valid_config)
        config['entry_conditions'] = [
            {
                'indicator': 'rsi',
                'operator': 'between',
                'value_low': 30,
                'value_high': 70,
            },
        ]
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])

    def test_valid_config_with_nested_groups(self):
        """Nested condition groups should pass validation."""
        config = dict(self.valid_config)
        config['entry_conditions'] = {
            'type': 'group',
            'logic': 'AND',
            'conditions': [
                {'type': 'condition', 'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
                {
                    'type': 'group',
                    'logic': 'OR',
                    'conditions': [
                        {'type': 'condition', 'indicator': 'macd', 'operator': 'greater_than', 'value': 0},
                        {'type': 'condition', 'indicator': 'stoch_k', 'operator': 'less_than', 'value': 20},
                    ],
                },
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])

    def test_valid_config_minimal(self):
        """A config with just entry + exit conditions should pass."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])

    def test_valid_config_all_exit_types(self):
        """All exit condition types should be accepted."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'take_profit', 'value': 10},
                {'type': 'stop_loss', 'value': 5},
                {'type': 'trailing_stop', 'value': 7},
                {'type': 'time_based', 'days': 5},
                {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 70},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertEqual(result['errors'], [])

    def test_valid_config_with_crosses_operators(self):
        """crosses_above and crosses_below should work."""
        config = {
            'entry_conditions': [
                {'indicator': 'macd_histogram', 'operator': 'crosses_above', 'value': 0},
            ],
            'exit_conditions': [
                {'type': 'indicator', 'indicator': 'macd_histogram', 'operator': 'crosses_below', 'value': 0},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])

    def test_valid_config_with_builder_alias_indicators(self):
        """Builder alias indicator names should be accepted."""
        config = {
            'entry_conditions': [
                {'indicator': 'bollinger_upper', 'operator': 'less_than', 'value': 110},
                {'indicator': 'stochastic_k', 'operator': 'less_than', 'value': 20},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])


class TestValidateConfigInvalidIndicator(TestCase):
    """Test validate_config with invalid indicator names."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_unknown_indicator_returns_error(self):
        """An unknown indicator name should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'magic_indicator', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown indicator' in e for e in result['errors']))

    def test_missing_indicator_returns_error(self):
        """A condition without an indicator should produce an error."""
        config = {
            'entry_conditions': [
                {'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('indicator is required' in e for e in result['errors']))


class TestValidateConfigInvalidOperator(TestCase):
    """Test validate_config with invalid operator names."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_unknown_operator_returns_error(self):
        """An unknown operator name should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'way_above', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown operator' in e for e in result['errors']))

    def test_missing_operator_returns_error(self):
        """A condition without an operator should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('operator is required' in e for e in result['errors']))


class TestValidateConfigMissingValue(TestCase):
    """Test validate_config with missing or invalid threshold values."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_missing_value_returns_error(self):
        """A condition without a value (for a non-cross operator) should error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than'},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('value is required' in e for e in result['errors']))

    def test_non_numeric_value_returns_error(self):
        """A non-numeric threshold value should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 'thirty'},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('must be numeric' in e for e in result['errors']))

    def test_between_missing_range_returns_error(self):
        """The between operator without value_low/value_high should error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'between', 'value': 50},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('value_low' in e for e in result['errors']))

    def test_between_inverted_range_returns_error(self):
        """value_low >= value_high should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'between', 'value_low': 70, 'value_high': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('less than value_high' in e for e in result['errors']))


class TestValidateConfigExitConditions(TestCase):
    """Test validate_config with various exit condition problems."""

    def setUp(self):
        self.service = StrategyBuilderService()
        self.base_entry = [
            {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
        ]

    def test_invalid_exit_type_returns_error(self):
        """An unknown exit type should produce an error."""
        config = {
            'entry_conditions': self.base_entry,
            'exit_conditions': [
                {'type': 'magic_exit', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown exit type' in e for e in result['errors']))

    def test_missing_exit_type_returns_error(self):
        """An exit condition without a type should error."""
        config = {
            'entry_conditions': self.base_entry,
            'exit_conditions': [
                {'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('type is required' in e for e in result['errors']))

    def test_no_exit_conditions_produces_warning(self):
        """No exit conditions should produce a warning, not an error."""
        config = {
            'entry_conditions': self.base_entry,
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('exit conditions' in w.lower() or 'stop loss' in w.lower() for w in result['warnings']))

    def test_no_stop_loss_produces_warning(self):
        """Exit conditions without a stop loss should produce a warning."""
        config = {
            'entry_conditions': self.base_entry,
            'exit_conditions': [
                {'type': 'take_profit', 'value': 10},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('stop loss' in w.lower() for w in result['warnings']))

    def test_exit_indicator_unknown_returns_error(self):
        """An indicator-based exit with unknown indicator should error."""
        config = {
            'entry_conditions': self.base_entry,
            'exit_conditions': [
                {'type': 'indicator', 'indicator': 'fake_ind', 'operator': 'greater_than', 'value': 70},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('unknown indicator' in e for e in result['errors']))


class TestValidateConfigPositionSizing(TestCase):
    """Test validate_config position sizing validation."""

    def setUp(self):
        self.service = StrategyBuilderService()
        self.base_config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }

    def test_invalid_sizing_type_returns_error(self):
        """An unknown position sizing type should error."""
        config = dict(self.base_config)
        config['position_sizing'] = {'type': 'yolo', 'value': 100}
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('position sizing type' in e.lower() for e in result['errors']))

    def test_aggressive_sizing_produces_warning(self):
        """Position size > 25% should produce a warning."""
        config = dict(self.base_config)
        config['position_sizing'] = {'type': 'fixed_percent', 'value': 30}
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('aggressive' in w for w in result['warnings']))

    def test_negative_sizing_value_returns_error(self):
        """Negative position sizing value should error."""
        config = dict(self.base_config)
        config['position_sizing'] = {'type': 'fixed_percent', 'value': -5}
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('positive' in e for e in result['errors']))


class TestValidateConfigRangeWarnings(TestCase):
    """Test that out-of-range values produce warnings."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_rsi_out_of_range_produces_warning(self):
        """RSI value outside 0-100 should produce a warning."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 150},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])  # out-of-range is warning, not error
        self.assertTrue(any('outside typical range' in w for w in result['warnings']))

    def test_adx_in_range_no_warning(self):
        """ADX value within 0-100 should not produce a range warning."""
        config = {
            'entry_conditions': [
                {'indicator': 'adx', 'operator': 'greater_than', 'value': 25},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        range_warnings = [w for w in result['warnings'] if 'outside typical range' in w]
        self.assertEqual(len(range_warnings), 0)


class TestValidateConfigNoEntryConditions(TestCase):
    """Test validate_config with no entry conditions."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_empty_entry_conditions_returns_error(self):
        """An empty entry_conditions list should produce an error."""
        config = {
            'entry_conditions': [],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('entry condition' in e.lower() for e in result['errors']))

    def test_missing_entry_conditions_returns_error(self):
        """No entry_conditions key should produce an error."""
        config = {
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('entry condition' in e.lower() for e in result['errors']))


class TestGetAvailableIndicators(TestCase):
    """Test get_available_indicators method."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_returns_non_empty_list(self):
        """Should return a list with at least one indicator."""
        indicators = self.service.get_available_indicators()
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)

    def test_has_required_fields(self):
        """Each indicator should have name, display_name, category, description."""
        indicators = self.service.get_available_indicators()
        required_fields = {'name', 'display_name', 'category', 'default_period', 'value_range', 'description'}
        for ind in indicators:
            for field in required_fields:
                self.assertIn(field, ind, f'Indicator {ind.get("name", "?")} missing field "{field}"')

    def test_includes_all_runner_indicators(self):
        """Should include all indicators from CustomStrategyRunner.INDICATORS."""
        indicators = self.service.get_available_indicators()
        indicator_names = {ind['name'] for ind in indicators}
        for key in CustomStrategyRunner.INDICATORS:
            self.assertIn(key, indicator_names, f'Missing runner indicator: {key}')

    def test_categories_are_valid(self):
        """All categories should be from the known set."""
        indicators = self.service.get_available_indicators()
        valid_categories = {'momentum', 'trend', 'volatility', 'volume', 'price', 'other'}
        for ind in indicators:
            self.assertIn(
                ind['category'], valid_categories,
                f'Indicator {ind["name"]} has invalid category "{ind["category"]}"',
            )

    def test_value_range_format(self):
        """value_range should be None or a list of two numbers."""
        indicators = self.service.get_available_indicators()
        for ind in indicators:
            vr = ind['value_range']
            if vr is not None:
                self.assertIsInstance(vr, list)
                self.assertEqual(len(vr), 2)
                self.assertIsInstance(vr[0], (int, float))
                self.assertIsInstance(vr[1], (int, float))

    def test_known_indicators_present(self):
        """Specific well-known indicators should be present."""
        indicators = self.service.get_available_indicators()
        names = {ind['name'] for ind in indicators}
        for expected in ('rsi', 'macd', 'sma', 'ema', 'adx', 'atr', 'cci', 'williams_r', 'volume_sma'):
            self.assertIn(expected, names)


class TestGetAvailableOperators(TestCase):
    """Test get_available_operators method."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_returns_non_empty_list(self):
        """Should return a list with at least one operator."""
        operators = self.service.get_available_operators()
        self.assertIsInstance(operators, list)
        self.assertGreater(len(operators), 0)

    def test_has_required_fields(self):
        """Each operator should have name, display_name, symbol, description."""
        operators = self.service.get_available_operators()
        required_fields = {'name', 'display_name', 'symbol', 'description'}
        for op in operators:
            for field in required_fields:
                self.assertIn(field, op, f'Operator {op.get("name", "?")} missing field "{field}"')

    def test_includes_expected_operators(self):
        """All standard operators should be present."""
        operators = self.service.get_available_operators()
        op_names = {op['name'] for op in operators}
        expected = {
            'less_than', 'greater_than', 'crosses_above', 'crosses_below',
            'between', 'less_equal', 'greater_equal', 'equals',
        }
        for exp in expected:
            self.assertIn(exp, op_names, f'Missing operator: {exp}')

    def test_crossover_operators_require_history(self):
        """crosses_above and crosses_below should have requires_history=True."""
        operators = self.service.get_available_operators()
        for op in operators:
            if op['name'] in ('crosses_above', 'crosses_below'):
                self.assertTrue(op.get('requires_history', False))

    def test_between_requires_range(self):
        """The between operator should have requires_range=True."""
        operators = self.service.get_available_operators()
        between_ops = [op for op in operators if op['name'] == 'between']
        self.assertEqual(len(between_ops), 1)
        self.assertTrue(between_ops[0].get('requires_range', False))


class TestGetPresets(TestCase):
    """Test get_presets method."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_returns_non_empty_list(self):
        """Should return at least one preset."""
        presets = self.service.get_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)

    def test_returns_five_presets(self):
        """Should return exactly five presets as specified."""
        presets = self.service.get_presets()
        self.assertEqual(len(presets), 5)

    def test_preset_has_required_fields(self):
        """Each preset should have id, name, description, and config."""
        presets = self.service.get_presets()
        for preset in presets:
            self.assertIn('id', preset)
            self.assertIn('name', preset)
            self.assertIn('description', preset)
            self.assertIn('config', preset)

    def test_preset_configs_pass_validation(self):
        """Every preset config should pass the service's own validation."""
        presets = self.service.get_presets()
        for preset in presets:
            result = self.service.validate_config(preset['config'])
            self.assertTrue(
                result['valid'],
                f'Preset "{preset["name"]}" failed validation: {result["errors"]}',
            )

    def test_preset_configs_have_entry_conditions(self):
        """Every preset config should have entry_conditions."""
        presets = self.service.get_presets()
        for preset in presets:
            entry = preset['config'].get('entry_conditions', [])
            self.assertTrue(
                len(entry) > 0,
                f'Preset "{preset["name"]}" has no entry conditions',
            )

    def test_preset_configs_have_exit_conditions(self):
        """Every preset config should have exit_conditions."""
        presets = self.service.get_presets()
        for preset in presets:
            exits = preset['config'].get('exit_conditions', [])
            self.assertTrue(
                len(exits) > 0,
                f'Preset "{preset["name"]}" has no exit conditions',
            )

    def test_preset_configs_have_position_sizing(self):
        """Every preset config should have position_sizing."""
        presets = self.service.get_presets()
        for preset in presets:
            sizing = preset['config'].get('position_sizing', {})
            self.assertTrue(
                len(sizing) > 0,
                f'Preset "{preset["name"]}" has no position sizing',
            )

    def test_preset_ids_are_unique(self):
        """All preset ids should be unique."""
        presets = self.service.get_presets()
        ids = [p['id'] for p in presets]
        self.assertEqual(len(ids), len(set(ids)))

    def test_preset_names_match_expected(self):
        """Check that the expected preset names are present."""
        presets = self.service.get_presets()
        names = {p['name'] for p in presets}
        expected_names = {
            'RSI Oversold Bounce',
            'MACD Cross',
            'Bollinger Bounce',
            'Volume Breakout',
            'Mean Reversion',
        }
        self.assertEqual(names, expected_names)

    def test_preset_configs_have_filters(self):
        """Every preset config should have filters."""
        presets = self.service.get_presets()
        for preset in presets:
            filters = preset['config'].get('filters', {})
            self.assertTrue(
                len(filters) > 0,
                f'Preset "{preset["name"]}" has no filters',
            )


class TestValidateConfigEdgeCases(TestCase):
    """Edge-case and boundary tests for validate_config."""

    def setUp(self):
        self.service = StrategyBuilderService()

    def test_empty_config(self):
        """An empty dict should fail validation."""
        result = self.service.validate_config({})
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)

    def test_config_with_only_entry_conditions(self):
        """A config with only entry conditions should be valid but warn."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertGreater(len(result['warnings']), 0)

    def test_wide_stop_loss_produces_warning(self):
        """Stop loss above 20% should produce a warning."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 25},
            ],
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('wide' in w for w in result['warnings']))

    def test_zero_stop_loss_returns_error(self):
        """Stop loss of 0 should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 0},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('positive' in e for e in result['errors']))

    def test_negative_take_profit_returns_error(self):
        """Negative take profit should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'take_profit', 'value': -5},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('positive' in e for e in result['errors']))

    def test_time_based_exit_without_days_returns_error(self):
        """time_based exit without days should produce an error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'time_based'},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('days' in e for e in result['errors']))

    def test_many_max_positions_produces_warning(self):
        """max_positions above 20 should produce a warning."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
            'position_sizing': {
                'type': 'fixed_percent',
                'value': 5,
                'max_positions': 25,
            },
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('too many' in w for w in result['warnings']))

    def test_high_min_price_filter_produces_warning(self):
        """min_price above $100 should produce a warning."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'stop_loss', 'value': 5},
            ],
            'filters': {
                'min_price': 200,
            },
        }
        result = self.service.validate_config(config)
        self.assertTrue(result['valid'])
        self.assertTrue(any('restrictive' in w for w in result['warnings']))

    def test_indicator_exit_missing_operator_returns_error(self):
        """Indicator-type exit without operator should error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'indicator', 'indicator': 'rsi', 'value': 70},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('operator' in e for e in result['errors']))

    def test_indicator_exit_missing_indicator_returns_error(self):
        """Indicator-type exit without indicator name should error."""
        config = {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30},
            ],
            'exit_conditions': [
                {'type': 'indicator', 'operator': 'greater_than', 'value': 70},
                {'type': 'stop_loss', 'value': 5},
            ],
        }
        result = self.service.validate_config(config)
        self.assertFalse(result['valid'])
        self.assertTrue(any('indicator' in e.lower() for e in result['errors']))

    def test_all_runner_indicators_are_valid(self):
        """Every indicator in CustomStrategyRunner.INDICATORS should pass validation."""
        for indicator_key in CustomStrategyRunner.INDICATORS:
            config = {
                'entry_conditions': [
                    {'indicator': indicator_key, 'operator': 'greater_than', 'value': 0},
                ],
                'exit_conditions': [
                    {'type': 'stop_loss', 'value': 5},
                ],
            }
            result = self.service.validate_config(config)
            indicator_errors = [e for e in result['errors'] if 'indicator' in e.lower()]
            self.assertEqual(
                len(indicator_errors), 0,
                f'Runner indicator "{indicator_key}" should be valid but got: {indicator_errors}',
            )

    def test_all_runner_operators_are_valid(self):
        """Every operator in CustomStrategyRunner.OPERATORS should pass validation."""
        for op_key in CustomStrategyRunner.OPERATORS:
            cond = {'indicator': 'rsi', 'operator': op_key}
            if op_key == 'between':
                cond['value_low'] = 20
                cond['value_high'] = 80
            else:
                cond['value'] = 50

            config = {
                'entry_conditions': [cond],
                'exit_conditions': [
                    {'type': 'stop_loss', 'value': 5},
                ],
            }
            result = self.service.validate_config(config)
            operator_errors = [e for e in result['errors'] if 'operator' in e.lower()]
            self.assertEqual(
                len(operator_errors), 0,
                f'Runner operator "{op_key}" should be valid but got: {operator_errors}',
            )
