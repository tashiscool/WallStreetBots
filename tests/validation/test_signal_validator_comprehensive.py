#!/usr/bin/env python3
"""
Comprehensive test suite to achieve >75% coverage for signal validation.

This test file specifically targets all untested methods and edge cases
to maximize test coverage across the signal validation framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
from datetime import datetime, timedelta
import json

# Import signal validation components
from backend.tradingbot.validation.signal_strength_validator import (
    SignalStrengthValidator,
    SignalType,
    SignalQuality,
    SignalValidationResult,
    SignalMetrics,
    BreakoutSignalCalculator,
    MomentumSignalCalculator,
    SignalStrengthCalculator
)

from backend.tradingbot.validation.strategy_signal_integration import (
    StrategySignalConfig,
    StrategySignalMixin
)


class TestSignalValidatorInternalMethods:
    """Test all internal methods of SignalStrengthValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator for testing."""
        return SignalStrengthValidator()

    @pytest.fixture
    def sample_metrics(self):
        """Create sample signal metrics."""
        return SignalMetrics(
            strength_score=75.0,
            confidence=0.8,
            consistency=0.85,
            volume_confirmation=1.0,
            technical_confluence=0.9,
            risk_reward_ratio=2.5,
            time_decay_factor=0.8,
            market_regime_fit=0.7
        )

    def test_get_default_config(self, validator):
        """Test _get_default_config method."""
        config = validator._get_default_config()

        assert isinstance(config, dict)
        assert 'minimum_strength_threshold' in config
        assert 'minimum_confidence_threshold' in config
        assert 'minimum_consistency_threshold' in config
        assert 'volume_confirmation_weight' in config
        assert 'technical_confluence_weight' in config
        assert 'risk_reward_minimum' in config
        assert 'max_time_decay_hours' in config
        assert 'regime_filter_enabled' in config
        assert 'consistency_lookback_days' in config

        # Test default values are reasonable
        assert config['minimum_strength_threshold'] > 0
        assert config['minimum_confidence_threshold'] > 0
        assert config['risk_reward_minimum'] > 0

    def test_create_failed_result(self, validator):
        """Test _create_failed_result method."""
        result = validator._create_failed_result(
            signal_id="TEST_FAIL",
            signal_type=SignalType.BREAKOUT,
            symbol="FAIL_SYMBOL",
            error_msg="Test failure error",
            reason="Test failure reason"
        )

        assert isinstance(result, SignalValidationResult)
        assert result.signal_id == "TEST_FAIL"
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == "FAIL_SYMBOL"
        assert result.quality_grade == SignalQuality.VERY_POOR
        assert result.recommended_action == "reject"
        assert "Reason: Test failure reason" in result.validation_notes

    def test_calculate_normalized_score_edge_cases(self, validator, sample_metrics):
        """Test _calculate_normalized_score with edge cases."""
        # Test with very high metrics
        high_metrics = SignalMetrics(
            strength_score=100.0,
            confidence=1.0,
            consistency=1.0,
            volume_confirmation=1.0,
            technical_confluence=1.0,
            risk_reward_ratio=5.0,
            time_decay_factor=1.0,
            market_regime_fit=1.0
        )

        high_score = validator._calculate_normalized_score(high_metrics)
        assert 0 <= high_score <= 100
        assert high_score > 80  # Should be high

        # Test with very low metrics
        low_metrics = SignalMetrics(
            strength_score=0.0,
            confidence=0.0,
            consistency=0.0,
            volume_confirmation=0.0,
            technical_confluence=0.0,
            risk_reward_ratio=0.5,
            time_decay_factor=0.0,
            market_regime_fit=0.0
        )

        low_score = validator._calculate_normalized_score(low_metrics)
        assert 0 <= low_score <= 100
        assert low_score < 20  # Should be very low

    def test_determine_quality_grade_all_ranges(self, validator):
        """Test _determine_quality_grade for all score ranges."""
        test_cases = [
            (95, SignalQuality.EXCELLENT),
            (85, SignalQuality.GOOD),
            (75, SignalQuality.FAIR),
            (65, SignalQuality.POOR),
            (45, SignalQuality.VERY_POOR),
            (25, SignalQuality.VERY_POOR),
            (0, SignalQuality.VERY_POOR)
        ]

        for score, expected_grade in test_cases:
            grade = validator._determine_quality_grade(score)
            assert grade == expected_grade, f"Score {score} should be {expected_grade}, got {grade}"

    def test_calculate_position_size(self, validator, sample_metrics):
        """Test _calculate_position_size method."""
        # Test with high quality signal
        high_score = 90.0
        position_size = validator._calculate_position_size(sample_metrics, high_score)
        assert isinstance(position_size, float)
        assert 0 <= position_size <= 1.0

        # Test with low quality signal
        low_score = 30.0
        low_position_size = validator._calculate_position_size(sample_metrics, low_score)
        assert low_position_size == 0.0  # Should reject low quality signals

    def test_generate_validation_notes(self, validator, sample_metrics):
        """Test _generate_validation_notes method."""
        result = SignalValidationResult(
            signal_id="TEST_NOTES",
            signal_type=SignalType.BREAKOUT,
            timestamp=datetime.now(),
            symbol="TEST",
            raw_metrics=sample_metrics,
            normalized_score=85.0,
            quality_grade=SignalQuality.GOOD,
            passes_minimum_threshold=True,
            passes_consistency_check=True,
            passes_regime_filter=True,
            passes_risk_check=True,
            recommended_action="execute",
            confidence_level=0.8,
            suggested_position_size=0.02,
            validation_notes=[],
            historical_performance_percentile=None
        )

        notes = validator._generate_validation_notes(result)
        assert isinstance(notes, list)
        assert len(notes) > 0
        assert any("Good quality signal" in note for note in notes)

    def test_calculate_technical_confluence(self, validator):
        """Test _calculate_technical_confluence method."""
        # Create test data with clear patterns
        test_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110],
            'High': [101, 103, 105, 107, 109, 111],
            'Low': [99, 101, 103, 105, 107, 109],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        })

        confluence = validator._calculate_technical_confluence(test_data, SignalType.BREAKOUT)
        assert isinstance(confluence, (int, float))
        assert 0 <= confluence <= 1

    def test_calculate_time_decay_factor(self, validator):
        """Test _calculate_time_decay_factor method."""
        # Test recent signal (should have high factor)
        recent_time = datetime.now() - timedelta(hours=1)
        recent_factor = validator._calculate_time_decay_factor(recent_time)
        assert 0.8 <= recent_factor <= 1.0

        # Test old signal (should have low factor)
        old_time = datetime.now() - timedelta(hours=25)
        old_factor = validator._calculate_time_decay_factor(old_time)
        assert 0 <= old_factor <= 0.5

    def test_calculate_market_regime_fit(self, validator):
        """Test _calculate_market_regime_fit method."""
        test_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })

        regime_fit = validator._calculate_market_regime_fit(test_data, SignalType.MOMENTUM)
        assert isinstance(regime_fit, (int, float))
        assert 0 <= regime_fit <= 1


class TestSignalCalculatorMethods:
    """Test all methods in signal calculator classes."""

    def test_breakout_calculator_comprehensive(self):
        """Test all BreakoutSignalCalculator methods."""
        calculator = BreakoutSignalCalculator()

        # Test initialization
        assert calculator.get_signal_type() == SignalType.BREAKOUT
        assert isinstance(calculator.params, dict)

        # Test with custom parameters
        custom_params = {'lookback_period': 20, 'breakout_threshold': 0.025}
        custom_calc = BreakoutSignalCalculator(custom_params)
        assert custom_calc.params == custom_params

        # Test raw strength calculation with various data patterns
        test_patterns = [
            # Strong breakout pattern
            pd.DataFrame({
                'Close': [100] * 15 + [105, 107, 110, 112],
                'Volume': [1000000] * 15 + [2500000, 3000000, 3500000, 4000000]
            }),
            # Weak breakout pattern
            pd.DataFrame({
                'Close': [100] * 10 + [100.5, 101, 101.5],
                'Volume': [1000000] * 10 + [1100000, 1200000, 1300000]
            }),
            # No breakout pattern
            pd.DataFrame({
                'Close': [100] * 20,
                'Volume': [1000000] * 20
            })
        ]

        for i, pattern in enumerate(test_patterns):
            strength = calculator.calculate_raw_strength(pattern)
            assert isinstance(strength, (int, float))
            assert strength >= 0

            confidence = calculator.calculate_confidence(pattern)
            assert isinstance(confidence, (int, float))
            assert 0 <= confidence <= 1

    def test_momentum_calculator_comprehensive(self):
        """Test all MomentumSignalCalculator methods."""
        calculator = MomentumSignalCalculator()

        # Test initialization
        assert calculator.get_signal_type() == SignalType.MOMENTUM

        # Test with momentum patterns
        test_patterns = [
            # Strong momentum pattern
            pd.DataFrame({
                'Close': [100, 103, 106, 109, 112, 115, 118],
                'Volume': [1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000]
            }),
            # Weak momentum pattern
            pd.DataFrame({
                'Close': [100, 100.5, 101, 101.5, 102],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
            }),
            # No momentum pattern
            pd.DataFrame({
                'Close': [100, 99, 101, 100, 99],
                'Volume': [1000000, 1000000, 1000000, 1000000, 1000000]
            })
        ]

        for pattern in test_patterns:
            strength = calculator.calculate_raw_strength(pattern)
            assert isinstance(strength, (int, float))
            assert strength >= 0

            confidence = calculator.calculate_confidence(pattern)
            assert isinstance(confidence, (int, float))
            assert 0 <= confidence <= 1

    def test_calculator_error_handling(self):
        """Test calculator error handling."""
        calculator = BreakoutSignalCalculator()

        # Test with None data
        strength = calculator.calculate_raw_strength(None)
        assert strength == 0.0

        confidence = calculator.calculate_confidence(None)
        assert confidence == 0.0

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        strength = calculator.calculate_raw_strength(empty_df)
        assert strength == 0.0

        # Test with insufficient data
        small_df = pd.DataFrame({'Close': [100]})
        strength = calculator.calculate_raw_strength(small_df)
        assert strength >= 0.0


class TestSignalValidatorExportMethods:
    """Test export and summary methods."""

    @pytest.fixture
    def validator_with_history(self):
        """Create validator with validation history."""
        validator = SignalStrengthValidator()

        # Add test history
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 105],
            'Volume': [1000000, 1100000, 1200000, 2000000]
        })

        for i in range(5):
            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"TEST_{i}",
                market_data=test_data
            )

        return validator

    def test_get_validation_summary_comprehensive(self, validator_with_history):
        """Test get_validation_summary with comprehensive data."""
        summary = validator_with_history.get_validation_summary()

        assert isinstance(summary, dict)
        assert summary['total_signals_validated'] == 5
        assert 'average_strength_score' in summary
        assert 'signals_by_quality' in summary
        assert 'recommendation_distribution' in summary

        # Test specific summary fields
        assert isinstance(summary['signals_by_quality'], dict)
        assert isinstance(summary['recommendation_distribution'], dict)
        assert isinstance(summary['average_strength_score'], (int, float))

    def test_export_validation_history_all_formats(self, validator_with_history):
        """Test exporting validation history in all formats."""
        # Test CSV format
        csv_export = validator_with_history.export_validation_history(format='csv')
        assert isinstance(csv_export, str)
        assert 'signal_id' in csv_export
        assert 'signal_type' in csv_export
        assert 'normalized_score' in csv_export

        # Test JSON format
        json_export = validator_with_history.export_validation_history(format='json')
        assert isinstance(json_export, str)
        # Verify it's valid JSON
        parsed_json = json.loads(json_export)
        assert isinstance(parsed_json, list)
        assert len(parsed_json) == 5

        # Test DataFrame format
        df_export = validator_with_history.export_validation_history(format='dataframe')
        assert isinstance(df_export, pd.DataFrame)
        assert len(df_export) == 5
        assert 'signal_id' in df_export.columns

        # Test invalid format
        with pytest.raises(ValueError):
            validator_with_history.export_validation_history(format='invalid_format')

    def test_export_with_filters(self, validator_with_history):
        """Test export with different formats."""
        # Test dataframe export
        df_export = validator_with_history.export_validation_history(
            format='dataframe'
        )
        assert isinstance(df_export, pd.DataFrame)

        # Test CSV export
        csv_export = validator_with_history.export_validation_history(
            format='csv'
        )
        assert isinstance(csv_export, str)
        assert 'signal_id' in csv_export

        # Test JSON export
        json_export = validator_with_history.export_validation_history(
            format='json'
        )
        assert isinstance(json_export, str)


class TestStrategySignalIntegrationMethods:
    """Test strategy signal integration methods."""

    def test_strategy_signal_config_post_init(self):
        """Test StrategySignalConfig __post_init__ method."""
        # Test with None custom params
        config = StrategySignalConfig(
            strategy_name="test",
            default_signal_type=SignalType.BREAKOUT,
            custom_validation_params=None
        )
        assert config.custom_validation_params == {}

        # Test with existing custom params
        custom_params = {'param1': 'value1'}
        config_with_params = StrategySignalConfig(
            strategy_name="test",
            default_signal_type=SignalType.BREAKOUT,
            custom_validation_params=custom_params
        )
        assert config_with_params.custom_validation_params == custom_params

    def test_strategy_signal_mixin_complete_initialization(self):
        """Test complete StrategySignalMixin initialization."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self, config=None):
                self.config = config or {}
                self.logger = Mock()
                super().__init__()

        # Test with minimal config
        strategy = TestStrategy()
        assert hasattr(strategy, '_signal_validator')
        assert hasattr(strategy, '_signal_config')
        assert hasattr(strategy, '_signal_history')

        # Test with custom config
        custom_config = {
            'min_strength_score': 80.0,
            'strategy_name': 'custom_strategy'
        }
        strategy_custom = TestStrategy(custom_config)
        assert strategy_custom.config['min_strength_score'] == 80.0

    def test_mixin_validate_signal_edge_cases(self):
        """Test StrategySignalMixin validate_signal edge cases."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation first
        config = StrategySignalConfig(strategy_name="test_strategy", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        # Test with None market data
        result = strategy.validate_signal(
            symbol="NULL_TEST",
            market_data=None,
            signal_type=SignalType.BREAKOUT
        )
        assert result.quality_grade == SignalQuality.VERY_POOR

        # Test with custom signal parameters
        test_data = pd.DataFrame({
            'Close': [100, 105, 110],
            'Volume': [1000000, 2000000, 3000000]
        })

        custom_params = {'risk_reward_ratio': 3.0, 'volume_threshold': 2.5}
        result_custom = strategy.validate_signal(
            symbol="CUSTOM_TEST",
            market_data=test_data,
            signal_type=SignalType.BREAKOUT,
            signal_params=custom_params
        )
        assert isinstance(result_custom, SignalValidationResult)

    def test_mixin_get_strategy_signal_summary_edge_cases(self):
        """Test get_strategy_signal_summary edge cases."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation first
        config = StrategySignalConfig(strategy_name="test_strategy", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        # Test with empty history
        empty_summary = strategy.get_strategy_signal_summary()
        assert isinstance(empty_summary, dict)
        assert 'message' in empty_summary
        assert empty_summary['message'] == "No signal validation history"

        # Test with varied signal types
        test_data = pd.DataFrame({
            'Close': [100, 102, 104],
            'Volume': [1000000, 1200000, 1400000]
        })

        signal_types = [SignalType.BREAKOUT, SignalType.MOMENTUM, SignalType.TREND]
        for signal_type in signal_types:
            try:
                strategy.validate_signal("VARIED_TEST", test_data, signal_type)
            except Exception:
                pass  # Some signal types may not be supported

        summary = strategy.get_strategy_signal_summary()
        assert isinstance(summary, dict)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_validator_with_malformed_config(self):
        """Test validator with malformed configuration."""
        # Test with partially invalid config
        bad_config = {
            'minimum_strength_threshold': -10,  # Invalid negative value
            'minimum_confidence_threshold': 1.5,  # Invalid >1 value
            'invalid_key': 'invalid_value'
        }

        # Should handle gracefully
        validator = SignalStrengthValidator(bad_config)
        assert isinstance(validator.config, dict)

        # Test validation still works
        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })

        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="BAD_CONFIG_TEST",
            market_data=test_data
        )
        assert isinstance(result, SignalValidationResult)

    def test_validator_with_extreme_data(self):
        """Test validator with extreme market data."""
        validator = SignalStrengthValidator()

        # Test with extreme values
        extreme_data = pd.DataFrame({
            'Close': [0.001, 1000000, 0.001, 1000000],  # Extreme price swings
            'Volume': [1, 999999999, 1, 999999999]  # Extreme volume
        })

        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="EXTREME_TEST",
            market_data=extreme_data
        )
        assert isinstance(result, SignalValidationResult)

        # Test with NaN values
        nan_data = pd.DataFrame({
            'Close': [100, np.nan, 102, np.nan],
            'Volume': [np.nan, 1100000, np.nan, 1300000]
        })

        result_nan = validator.validate_signal(
            signal_type=SignalType.MOMENTUM,
            symbol="NAN_TEST",
            market_data=nan_data
        )
        assert isinstance(result_nan, SignalValidationResult)

    def test_calculator_registration_edge_cases(self):
        """Test calculator registration edge cases."""
        validator = SignalStrengthValidator()

        # Test registering calculator for new signal type
        class CustomCalculator(SignalStrengthCalculator):
            def get_signal_type(self):
                return SignalType.TREND

            def calculate_raw_strength(self, market_data, **kwargs):
                return 50.0

            def calculate_confidence(self, market_data, **kwargs):
                return 0.7

        custom_calc = CustomCalculator()
        validator.register_calculator(SignalType.TREND, custom_calc)

        assert SignalType.TREND in validator.calculators
        assert validator.calculators[SignalType.TREND] == custom_calc

        # Test using registered calculator
        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })

        result = validator.validate_signal(
            signal_type=SignalType.TREND,
            symbol="CUSTOM_CALC_TEST",
            market_data=test_data
        )
        assert result.raw_metrics.strength_score == 50.0
        assert result.raw_metrics.confidence == 0.7


if __name__ == '__main__':
    pytest.main([__file__, '-v'])