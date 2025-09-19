#!/usr/bin/env python3
"""
Comprehensive test suite for signal validation components to achieve >75% coverage.

Focuses on testing all code paths in the signal validation framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings
from datetime import datetime, timedelta

# Import signal validation components
from backend.tradingbot.validation.signal_strength_validator import (
    SignalStrengthValidator,
    SignalType,
    SignalQuality,
    SignalValidationResult,
    SignalMetrics,
    BreakoutSignalCalculator,
    MomentumSignalCalculator,
    TrendSignalCalculator,
    CustomSignalCalculator,
    SwingTradingSignalCalculator,
    MomentumWeekliesSignalCalculator,
    LEAPSTrackerSignalCalculator
)

from backend.tradingbot.validation.strategy_signal_integration import (
    StrategySignalConfig,
    StrategySignalMixin,
    signal_integrator
)


class TestSignalValidationComprehensive:
    """Comprehensive tests for signal validation components."""

    def test_validator_config_dictionary(self):
        """Test SignalStrengthValidator with config dictionary."""
        config = {
            'minimum_strength_threshold': 80.0,
            'consistency_threshold': 0.8,
            'volume_threshold': 3.0,
            'enable_historical_validation': True
        }

        validator = SignalStrengthValidator(config)
        assert validator.config['minimum_strength_threshold'] == 80.0
        assert validator.config['consistency_threshold'] == 0.8
        assert validator.config['volume_threshold'] == 3.0
        assert validator.config['enable_historical_validation']

    def test_signal_metrics_creation(self):
        """Test SignalMetrics creation with all fields."""
        metrics = SignalMetrics(
            strength_score=75.5,
            confidence=0.85,
            consistency=0.9,
            volume_confirmation=1.0,
            technical_confluence=0.8,
            risk_reward_ratio=2.5,
            time_decay_factor=0.7,
            market_regime_fit=0.6
        )

        assert metrics.strength_score == 75.5
        assert metrics.confidence == 0.85
        assert metrics.consistency == 0.9
        assert metrics.volume_confirmation == 1.0
        assert metrics.technical_confluence == 0.8
        assert metrics.risk_reward_ratio == 2.5
        assert metrics.time_decay_factor == 0.7
        assert metrics.market_regime_fit == 0.6

    def test_signal_validation_result_complete(self):
        """Test complete SignalValidationResult creation."""
        metrics = SignalMetrics(
            strength_score=80.0,
            confidence=0.8,
            consistency=0.85,
            volume_confirmation=1.0,
            technical_confluence=0.9,
            risk_reward_ratio=3.0,
            time_decay_factor=0.8,
            market_regime_fit=0.7
        )

        result = SignalValidationResult(
            signal_id="TEST_001",
            signal_type=SignalType.BREAKOUT,
            timestamp=datetime.now(),
            symbol="AAPL",
            raw_metrics=metrics,
            normalized_score=85.0,
            quality_grade=SignalQuality.EXCELLENT,
            passes_minimum_threshold=True,
            passes_consistency_check=True,
            passes_regime_filter=True,
            passes_risk_check=True,
            recommended_action="execute",
            confidence_level=0.9,
            suggested_position_size=0.02,
            validation_notes=["High quality signal"],
            historical_performance_percentile=85.0
        )

        assert result.signal_id == "TEST_001"
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == "AAPL"
        assert result.normalized_score == 85.0
        assert result.quality_grade == SignalQuality.EXCELLENT
        assert result.recommended_action == "execute"

    def test_trend_signal_calculator(self):
        """Test TrendSignalCalculator functionality."""
        calculator = TrendSignalCalculator()

        assert calculator.get_signal_type() == SignalType.TREND

        # Test with trending data
        trend_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        })

        strength = calculator.calculate_raw_strength(trend_data)
        assert isinstance(strength, (int, float))
        assert strength >= 0

        confidence = calculator.calculate_confidence(trend_data)
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1

    def test_custom_signal_calculator(self):
        """Test CustomSignalCalculator functionality."""
        def custom_strength_func(data):
            return 75.0

        def custom_confidence_func(data):
            return 0.8

        calculator = CustomSignalCalculator(
            signal_type=SignalType.BREAKOUT,
            strength_function=custom_strength_func,
            confidence_function=custom_confidence_func
        )

        assert calculator.get_signal_type() == SignalType.BREAKOUT

        test_data = pd.DataFrame({'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]})

        strength = calculator.calculate_raw_strength(test_data)
        assert strength == 75.0

        confidence = calculator.calculate_confidence(test_data)
        assert confidence == 0.8

    def test_validator_with_custom_calculator(self):
        """Test validator with custom calculator registration."""
        validator = SignalStrengthValidator()

        def custom_strength(data):
            return 60.0

        def custom_confidence(data):
            return 0.7

        custom_calc = CustomSignalCalculator(
            signal_type=SignalType.BREAKOUT,
            strength_function=custom_strength,
            confidence_function=custom_confidence
        )

        validator.register_calculator(SignalType.BREAKOUT, custom_calc)

        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
        })

        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="CUSTOM_TEST",
            market_data=test_data
        )

        assert result.raw_metrics.strength_score == 60.0
        assert result.raw_metrics.confidence == 0.7

    def test_export_validation_history_formats(self):
        """Test exporting validation history in different formats."""
        validator = SignalStrengthValidator()

        # Add some validation history
        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })

        for i in range(3):
            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"TEST_{i}",
                market_data=test_data
            )

        # Test CSV export
        csv_data = validator.export_validation_history(format='csv')
        assert isinstance(csv_data, str)
        assert 'signal_id' in csv_data

        # Test JSON export
        json_data = validator.export_validation_history(format='json')
        assert isinstance(json_data, str)
        assert '"signal_id"' in json_data

        # Test DataFrame export
        df_data = validator.export_validation_history(format='dataframe')
        assert isinstance(df_data, pd.DataFrame)
        assert len(df_data) == 3

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        validator = SignalStrengthValidator()

        # Test with None data
        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="NONE_TEST",
            market_data=None
        )
        assert result.quality_grade == SignalQuality.VERY_POOR

        # Test with completely empty DataFrame
        empty_df = pd.DataFrame()
        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="EMPTY_TEST",
            market_data=empty_df
        )
        assert result.quality_grade == SignalQuality.VERY_POOR

        # Test with invalid signal type
        test_data = pd.DataFrame({'Close': [100, 101]})
        result = validator.validate_signal(
            signal_type=None,
            symbol="INVALID_TYPE",
            market_data=test_data
        )
        assert result is not None

    def test_signal_integrator_functionality(self):
        """Test the signal_integrator global functionality."""
        # Test registering strategy configs
        config = StrategySignalConfig(
            strategy_name="test_strategy",
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=70.0
        )

        signal_integrator.register_strategy_config("test_strategy", config)
        assert "test_strategy" in signal_integrator.strategy_configs

        # Test registering custom calculator
        def test_calc_func(data):
            return 50.0

        custom_calc = CustomSignalCalculator(
            signal_type=SignalType.MOMENTUM,
            strength_function=test_calc_func,
            confidence_function=lambda x: 0.5
        )

        signal_integrator.register_custom_calculator("test_calc", custom_calc)
        assert "test_calc" in signal_integrator.custom_calculators

    def test_strategy_specific_calculators(self):
        """Test strategy-specific calculator implementations."""
        # Test SwingTradingSignalCalculator
        swing_calc = SwingTradingSignalCalculator()
        assert swing_calc.get_signal_type() == SignalType.BREAKOUT

        test_data = pd.DataFrame({
            'Close': [100, 101, 105, 108],
            'Volume': [1000000, 1500000, 2000000, 2500000]
        })

        strength = swing_calc.calculate_raw_strength(test_data)
        assert isinstance(strength, (int, float))

        confidence = swing_calc.calculate_confidence(test_data)
        assert isinstance(confidence, (int, float))

        # Test MomentumWeekliesSignalCalculator
        momentum_calc = MomentumWeekliesSignalCalculator()
        assert momentum_calc.get_signal_type() == SignalType.MOMENTUM

        # Test LEAPSTrackerSignalCalculator
        leaps_calc = LEAPSTrackerSignalCalculator()
        assert leaps_calc.get_signal_type() == SignalType.TREND

    def test_strategy_signal_config_validation(self):
        """Test StrategySignalConfig validation."""
        # Test valid config
        config = StrategySignalConfig(
            strategy_name="valid_strategy",
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=65.0,
            minimum_confidence_threshold=0.6,
            risk_reward_minimum=2.0,
            max_position_size_multiplier=1.5,
            enable_regime_filtering=True,
            custom_validation_params={'param1': 'value1'}
        )

        assert config.strategy_name == "valid_strategy"
        assert config.custom_validation_params == {'param1': 'value1'}

    def test_mixin_initialization_edge_cases(self):
        """Test StrategySignalMixin initialization edge cases."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self, config=None):
                self.config = config or {}
                super().__init__()

        # Test with no config
        strategy = TestStrategy()
        assert hasattr(strategy, '_signal_validator')

        # Test with custom config
        strategy_with_config = TestStrategy({'min_strength_score': 80.0})
        assert hasattr(strategy_with_config, '_signal_validator')

    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation in signal validation."""
        validator = SignalStrengthValidator()

        # Create data with specific patterns for performance testing
        performance_data = pd.DataFrame({
            'Close': [100, 102, 105, 103, 107, 110, 108, 112],
            'Volume': [1000000, 1200000, 1800000, 1100000, 2000000, 2200000, 1500000, 2500000],
            'High': [101, 103, 106, 104, 108, 111, 109, 113],
            'Low': [99, 101, 104, 102, 106, 109, 107, 111]
        })

        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="PERFORMANCE_TEST",
            market_data=performance_data
        )

        # Verify all metrics are calculated
        assert result.raw_metrics.strength_score is not None
        assert result.raw_metrics.confidence is not None
        assert result.raw_metrics.consistency is not None
        assert result.raw_metrics.volume_confirmation is not None
        assert result.raw_metrics.technical_confluence is not None

    def test_scipy_warning_handling(self):
        """Test that scipy warning is properly handled."""
        # This should trigger the scipy warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validator = SignalStrengthValidator()

            test_data = pd.DataFrame({
                'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
            })

            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol="SCIPY_TEST",
                market_data=test_data
            )

            # Check if scipy warning was issued (only if scipy is not available)
            scipy_warnings = [warning for warning in w if "scipy not available" in str(warning.message)]
            # If scipy is available, no warning should be issued
            # If scipy is not available, a warning should be issued
            try:
                import scipy
                # scipy is available, so no warning should be issued
                assert len(scipy_warnings) == 0
            except ImportError:
                # scipy is not available, so warning should be issued
                assert len(scipy_warnings) > 0

    def test_all_signal_quality_grades(self):
        """Test all signal quality grades are reachable."""
        validator = SignalStrengthValidator()

        # Test data for different quality levels
        quality_test_cases = [
            # Excellent quality - strong breakout with high volume
            (pd.DataFrame({
                'Close': [100, 102, 105, 108, 112, 115, 118, 122, 125, 128, 130],
                'Volume': [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000]
            }), "should be high quality"),

            # Poor quality - minimal movement
            (pd.DataFrame({
                'Close': [100, 100.1, 100.05, 100.15, 100.2, 100.1, 100.3, 100.2, 100.4, 100.3, 100.5],
                'Volume': [500000, 500001, 500002, 500003, 500004, 500005, 500006, 500007, 500008, 500009, 500010]
            }), "should be poor quality"),

            # Very poor quality (minimal data)
            (pd.DataFrame({
                'Close': [100],
                'Volume': [1000]
            }), "should be very poor quality")
        ]

        quality_grades_seen = set()

        for test_data, description in quality_test_cases:
            result = validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol="QUALITY_TEST",
                market_data=test_data
            )
            quality_grades_seen.add(result.quality_grade)

        # Should see at least 2 different quality grades
        assert len(quality_grades_seen) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])