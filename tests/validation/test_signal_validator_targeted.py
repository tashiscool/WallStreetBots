#!/usr/bin/env python3
"""
Targeted test suite to achieve >75% coverage for actual signal validation methods.

This test file targets the real methods found in the signal validation framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import os
import tempfile
from datetime import datetime, timedelta

# Import signal validation components
from backend.tradingbot.validation.signal_strength_validator import (
    SignalStrengthValidator,
    SignalType,
    SignalQuality,
    SignalValidationResult,
    SignalMetrics,
    BreakoutSignalCalculator,
    MomentumSignalCalculator
)


class TestSignalValidatorActualMethods:
    """Test all actual methods in SignalStrengthValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator for testing."""
        return SignalStrengthValidator()

    @pytest.fixture
    def rich_market_data(self):
        """Create rich market data for testing."""
        return pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110, 112, 115, 118, 120],
            'High': [101, 103, 105, 107, 109, 111, 113, 116, 119, 121],
            'Low': [99, 101, 103, 105, 107, 109, 111, 114, 117, 119],
            'Volume': [1000000, 1200000, 1500000, 1800000, 2200000, 2600000, 3000000, 3500000, 4000000, 4500000]
        })

    def test_get_default_config(self, validator):
        """Test _get_default_config method."""
        config = validator._get_default_config()

        expected_keys = [
            'minimum_strength_threshold',
            'minimum_confidence_threshold',
            'minimum_consistency_threshold',
            'volume_confirmation_weight',
            'technical_confluence_weight',
            'risk_reward_minimum',
            'max_time_decay_hours',
            'regime_filter_enabled',
            'consistency_lookback_days'
        ]

        for key in expected_keys:
            assert key in config

        assert config['minimum_strength_threshold'] == 60.0
        assert config['minimum_confidence_threshold'] == 0.6
        assert config['regime_filter_enabled']

    def test_register_calculator(self, validator):
        """Test register_calculator method."""
        # Test registering new calculator
        original_count = len(validator.calculators)

        custom_calc = BreakoutSignalCalculator()
        validator.register_calculator(SignalType.TREND, custom_calc)

        assert len(validator.calculators) == original_count + 1
        assert SignalType.TREND in validator.calculators
        assert validator.calculators[SignalType.TREND] == custom_calc

    def test_calculate_comprehensive_metrics(self, validator, rich_market_data):
        """Test _calculate_comprehensive_metrics method."""
        signal_params = {'risk_reward_ratio': 2.5, 'signal_timestamp': datetime.now()}

        metrics = validator._calculate_comprehensive_metrics(
            raw_strength=75.0,
            confidence=0.8,
            market_data=rich_market_data,
            signal_params=signal_params
        )

        assert isinstance(metrics, SignalMetrics)
        assert metrics.strength_score == 75.0
        assert metrics.confidence == 0.8
        assert 0 <= metrics.volume_confirmation <= 1
        assert 0 <= metrics.technical_confluence <= 1
        assert metrics.risk_reward_ratio == 2.5
        assert 0 <= metrics.time_decay_factor <= 1
        assert 0 <= metrics.market_regime_fit <= 1

    def test_calculate_volume_confirmation(self, validator, rich_market_data):
        """Test _calculate_volume_confirmation method."""
        volume_confirmation = validator._calculate_volume_confirmation(rich_market_data)

        assert isinstance(volume_confirmation, float)
        assert 0 <= volume_confirmation <= 1

        # Test with no volume data
        no_volume_data = rich_market_data.drop('Volume', axis=1)
        no_vol_confirmation = validator._calculate_volume_confirmation(no_volume_data)
        assert no_vol_confirmation == 0.0

    def test_calculate_technical_confluence(self, validator, rich_market_data):
        """Test _calculate_technical_confluence method."""
        confluence = validator._calculate_technical_confluence(rich_market_data)

        assert isinstance(confluence, float)
        assert 0 <= confluence <= 1

        # Test with minimal data
        minimal_data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000000, 1100000]
        })
        minimal_confluence = validator._calculate_technical_confluence(minimal_data)
        assert 0 <= minimal_confluence <= 1

    def test_calculate_time_decay_factor(self, validator):
        """Test _calculate_time_decay_factor method."""
        # Test recent signal
        recent_params = {'signal_timestamp': datetime.now()}
        recent_factor = validator._calculate_time_decay_factor(recent_params)
        assert 0.8 <= recent_factor <= 1.0

        # Test old signal
        old_params = {'signal_timestamp': datetime.now() - timedelta(hours=25)}
        old_factor = validator._calculate_time_decay_factor(old_params)
        assert 0 <= old_factor <= 0.5

        # Test no timestamp
        no_timestamp_params = {}
        default_factor = validator._calculate_time_decay_factor(no_timestamp_params)
        assert 0 <= default_factor <= 1

    def test_calculate_market_regime_fit(self, validator, rich_market_data):
        """Test _calculate_market_regime_fit method."""
        regime_fit = validator._calculate_market_regime_fit(rich_market_data)

        assert isinstance(regime_fit, float)
        assert 0 <= regime_fit <= 1

        # Test with flat market data
        flat_data = pd.DataFrame({
            'Close': [100] * 10,
            'Volume': [1000000] * 10
        })
        flat_regime_fit = validator._calculate_market_regime_fit(flat_data)
        assert 0 <= flat_regime_fit <= 1

    def test_calculate_normalized_score(self, validator):
        """Test _calculate_normalized_score method."""
        # Test high quality metrics
        high_metrics = SignalMetrics(
            strength_score=90.0,
            confidence=0.9,
            consistency=0.85,
            volume_confirmation=1.0,
            technical_confluence=0.9,
            risk_reward_ratio=3.0,
            time_decay_factor=1.0,
            market_regime_fit=0.8
        )

        high_score = validator._calculate_normalized_score(high_metrics)
        assert 80 <= high_score <= 100

        # Test low quality metrics
        low_metrics = SignalMetrics(
            strength_score=20.0,
            confidence=0.3,
            consistency=0.2,
            volume_confirmation=0.1,
            technical_confluence=0.2,
            risk_reward_ratio=1.0,
            time_decay_factor=0.2,
            market_regime_fit=0.1
        )

        low_score = validator._calculate_normalized_score(low_metrics)
        assert 0 <= low_score <= 40

    def test_grade_signal_quality(self, validator):
        """Test _grade_signal_quality method."""
        test_cases = [
            (95, SignalQuality.EXCELLENT),
            (85, SignalQuality.GOOD),
            (75, SignalQuality.FAIR),
            (65, SignalQuality.POOR),
            (35, SignalQuality.VERY_POOR)
        ]

        for score, expected_grade in test_cases:
            grade = validator._grade_signal_quality(score)
            assert grade == expected_grade

    def test_run_validation_checks(self, validator):
        """Test _run_validation_checks method."""
        # Test passing metrics
        good_metrics = SignalMetrics(
            strength_score=80.0,
            confidence=0.8,
            consistency=0.8,
            volume_confirmation=1.0,
            technical_confluence=0.8,
            risk_reward_ratio=2.5,
            time_decay_factor=0.9,
            market_regime_fit=0.7
        )

        checks = validator._run_validation_checks(good_metrics, 85.0)
        assert isinstance(checks, dict)
        assert 'minimum_threshold' in checks
        assert 'consistency_check' in checks
        assert 'regime_filter' in checks
        assert 'risk_check' in checks

        # Test failing metrics
        bad_metrics = SignalMetrics(
            strength_score=30.0,
            confidence=0.3,
            consistency=0.3,
            volume_confirmation=0.1,
            technical_confluence=0.2,
            risk_reward_ratio=1.0,
            time_decay_factor=0.2,
            market_regime_fit=0.2
        )

        bad_checks = validator._run_validation_checks(bad_metrics, 25.0)
        assert not bad_checks['minimum_threshold']

    def test_generate_recommendations(self, validator):
        """Test _generate_recommendations method."""
        # Test high quality signal
        good_metrics = SignalMetrics(
            strength_score=85.0,
            confidence=0.85,
            consistency=0.8,
            volume_confirmation=1.0,
            technical_confluence=0.9,
            risk_reward_ratio=3.0,
            time_decay_factor=0.9,
            market_regime_fit=0.8
        )

        good_checks = {
            'minimum_threshold': True,
            'consistency_check': True,
            'regime_filter': True,
            'risk_check': True
        }

        recommendations = validator._generate_recommendations(good_metrics, 88.0, good_checks)
        assert isinstance(recommendations, dict)
        assert 'action' in recommendations
        assert 'confidence' in recommendations
        assert 'position_size' in recommendations
        assert 'notes' in recommendations

        assert recommendations['action'] in ['trade', 'monitor', 'reject']
        assert 0 <= recommendations['confidence'] <= 1
        assert 0 <= recommendations['position_size'] <= 1

    def test_calculate_historical_percentile(self, validator, rich_market_data):
        """Test _calculate_historical_percentile method."""
        # Add some history first
        for i in range(5):
            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"HIST_{i}",
                market_data=rich_market_data
            )

        percentile = validator._calculate_historical_percentile(75.0)
        if percentile is not None:
            assert 0 <= percentile <= 100

    def test_create_failed_result(self, validator):
        """Test _create_failed_result method."""
        result = validator._create_failed_result(
            signal_id="FAIL_TEST",
            signal_type=SignalType.BREAKOUT,
            symbol="FAIL_SYM",
            error_msg="Test failure error",
            reason="Test failure"
        )

        assert isinstance(result, SignalValidationResult)
        assert result.signal_id == "FAIL_TEST"
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == "FAIL_SYM"
        assert result.quality_grade == SignalQuality.VERY_POOR
        assert result.recommended_action == "reject"
        assert "Reason: Test failure" in result.validation_notes

    def test_get_validation_summary(self, validator, rich_market_data):
        """Test get_validation_summary method."""
        # Test empty summary
        empty_summary = validator.get_validation_summary()
        assert empty_summary['total_signals_validated'] == 0
        assert empty_summary['average_strength_score'] == 0.0

        # Add some history
        for i in range(3):
            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"SUMMARY_{i}",
                market_data=rich_market_data
            )

        summary = validator.get_validation_summary()
        assert summary['total_signals_validated'] == 3
        assert 'average_strength_score' in summary
        assert 'signals_by_quality' in summary
        assert 'recommendation_distribution' in summary

    def test_export_validation_history(self, validator, rich_market_data):
        """Test export_validation_history method."""
        # Add some history
        for i in range(2):
            validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"EXPORT_{i}",
                market_data=rich_market_data
            )

        # Test export to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            validator.export_validation_history(temp_path)
            assert os.path.exists(temp_path)

            # Verify file contains data
            with open(temp_path, 'r') as f:
                content = f.read()
                assert 'signal_id' in content
                assert 'EXPORT_' in content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestBreakoutCalculatorMethods:
    """Test all methods in BreakoutSignalCalculator."""

    def test_breakout_calculator_initialization(self):
        """Test BreakoutSignalCalculator initialization."""
        # Test default initialization
        calc1 = BreakoutSignalCalculator()
        assert calc1.lookback_periods == 20
        assert calc1.volume_weight == 0.3

        # Test custom initialization
        calc2 = BreakoutSignalCalculator(lookback_periods=15, volume_weight=0.4)
        assert calc2.lookback_periods == 15
        assert calc2.volume_weight == 0.4

    def test_breakout_get_signal_type(self):
        """Test BreakoutSignalCalculator get_signal_type."""
        calc = BreakoutSignalCalculator()
        assert calc.get_signal_type() == SignalType.BREAKOUT

    def test_breakout_calculate_raw_strength(self):
        """Test BreakoutSignalCalculator calculate_raw_strength."""
        calc = BreakoutSignalCalculator()

        # Test strong breakout pattern
        strong_breakout = pd.DataFrame({
            'Close': [100] * 15 + [105, 108, 112, 115],
            'Volume': [1000000] * 15 + [2500000, 3000000, 3500000, 4000000]
        })

        strength = calc.calculate_raw_strength(strong_breakout)
        assert isinstance(strength, float)
        assert strength >= 0

        # Test weak pattern
        weak_data = pd.DataFrame({
            'Close': [100] * 20,
            'Volume': [1000000] * 20
        })

        weak_strength = calc.calculate_raw_strength(weak_data)
        assert weak_strength >= 0

        # Test edge case - insufficient data
        small_data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000000, 1100000]
        })

        small_strength = calc.calculate_raw_strength(small_data)
        assert small_strength >= 0

    def test_breakout_calculate_confidence(self):
        """Test BreakoutSignalCalculator calculate_confidence."""
        calc = BreakoutSignalCalculator()

        test_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110],
            'Volume': [1000000, 1200000, 1500000, 1800000, 2200000, 2600000]
        })

        confidence = calc.calculate_confidence(test_data)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

        # Test with no volume data
        no_vol_data = test_data.drop('Volume', axis=1)
        no_vol_confidence = calc.calculate_confidence(no_vol_data)
        assert 0 <= no_vol_confidence <= 1


class TestMomentumCalculatorMethods:
    """Test all methods in MomentumSignalCalculator."""

    def test_momentum_calculator_initialization(self):
        """Test MomentumSignalCalculator initialization."""
        # Test default initialization
        calc1 = MomentumSignalCalculator()
        assert calc1.short_window == 5
        assert calc1.long_window == 20

        # Test custom initialization
        calc2 = MomentumSignalCalculator(short_window=3, long_window=15)
        assert calc2.short_window == 3
        assert calc2.long_window == 15

    def test_momentum_get_signal_type(self):
        """Test MomentumSignalCalculator get_signal_type."""
        calc = MomentumSignalCalculator()
        assert calc.get_signal_type() == SignalType.MOMENTUM

    def test_momentum_calculate_raw_strength(self):
        """Test MomentumSignalCalculator calculate_raw_strength."""
        calc = MomentumSignalCalculator()

        # Test strong momentum pattern
        strong_momentum = pd.DataFrame({
            'Close': [100, 103, 106, 109, 112, 115, 118, 121, 124, 127],
            'Volume': [1000000, 1200000, 1500000, 1800000, 2200000, 2600000, 3000000, 3500000, 4000000, 4500000]
        })

        strength = calc.calculate_raw_strength(strong_momentum)
        assert isinstance(strength, float)
        assert strength >= 0

        # Test sideways pattern
        sideways_data = pd.DataFrame({
            'Close': [100, 99, 101, 100, 99, 101, 100],
            'Volume': [1000000] * 7
        })

        sideways_strength = calc.calculate_raw_strength(sideways_data)
        assert sideways_strength >= 0

    def test_momentum_calculate_confidence(self):
        """Test MomentumSignalCalculator calculate_confidence."""
        calc = MomentumSignalCalculator()

        test_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110, 112, 114],
            'Volume': [1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000]
        })

        confidence = calc.calculate_confidence(test_data)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1


class TestSignalValidatorErrorHandling:
    """Test error handling in signal validation."""

    def test_validate_signal_edge_cases(self):
        """Test validate_signal with edge cases."""
        validator = SignalStrengthValidator()

        # Test with None data
        result_none = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="NONE_TEST",
            market_data=None
        )
        assert result_none.quality_grade == SignalQuality.VERY_POOR

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="EMPTY_TEST",
            market_data=empty_df
        )
        assert result_empty.quality_grade == SignalQuality.VERY_POOR

        # Test with unsupported signal type (should fall back gracefully)
        good_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })

        # This should work with available calculators
        result_trend = validator.validate_signal(
            signal_type=SignalType.TREND,
            symbol="TREND_TEST",
            market_data=good_data
        )
        # Should fail gracefully since TREND calculator isn't registered by default
        assert result_trend.quality_grade == SignalQuality.VERY_POOR

    def test_calculator_error_handling(self):
        """Test calculator error handling."""
        calc = BreakoutSignalCalculator()

        # Test with None
        assert calc.calculate_raw_strength(None) == 0.0
        assert calc.calculate_confidence(None) == 0.0

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        assert calc.calculate_raw_strength(empty_df) == 0.0
        assert calc.calculate_confidence(empty_df) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])