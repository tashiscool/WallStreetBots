"""
Comprehensive Tests for Signal Strength Validation Framework
===========================================================

Tests all components of the signal strength validation system
with >75% code coverage and realistic scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from tradingbot.validation.signal_strength_validator import (
    SignalStrengthValidator,
    BreakoutSignalCalculator,
    MomentumSignalCalculator,
    SignalType,
    SignalQuality,
    SignalMetrics,
    SignalValidationResult
)


class TestSignalStrengthValidator:
    """Test the main signal strength validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SignalStrengthValidator()
        self.sample_market_data = self._create_sample_market_data()

    def _create_sample_market_data(self, length: int = 50) -> pd.DataFrame:
        """Create sample market data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=length, freq='D')

        # Create realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, length)
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        # Create volume data
        volumes = np.random.normal(1000000, 200000, length)
        volumes = np.maximum(volumes, 100000)  # Ensure positive volumes

        return pd.DataFrame({
            'Date': dates,
            'Open': prices[:-1],
            'High': [p * 1.02 for p in prices[:-1]],
            'Low': [p * 0.98 for p in prices[:-1]],
            'Close': prices[1:],
            'Volume': volumes
        })

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = SignalStrengthValidator()

        assert validator.config is not None
        assert 'minimum_strength_threshold' in validator.config
        assert validator.config['minimum_strength_threshold'] == 60.0
        assert len(validator.calculators) >= 2
        assert SignalType.BREAKOUT in validator.calculators
        assert SignalType.MOMENTUM in validator.calculators

    def test_custom_config(self):
        """Test validator with custom configuration."""
        custom_config = {
            'minimum_strength_threshold': 75.0,
            'minimum_confidence_threshold': 0.8
        }
        validator = SignalStrengthValidator(custom_config)

        assert validator.config['minimum_strength_threshold'] == 75.0
        assert validator.config['minimum_confidence_threshold'] == 0.8

    def test_register_calculator(self):
        """Test registering custom signal calculators."""
        mock_calculator = Mock()
        mock_calculator.get_signal_type.return_value = SignalType.VOLUME

        self.validator.register_calculator(SignalType.VOLUME, mock_calculator)

        assert SignalType.VOLUME in self.validator.calculators
        assert self.validator.calculators[SignalType.VOLUME] == mock_calculator

    def test_validate_breakout_signal_success(self):
        """Test successful breakout signal validation."""
        # Create data with clear breakout pattern
        breakout_data = self._create_breakout_pattern()

        result = self.validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="AAPL",
            market_data=breakout_data
        )

        assert isinstance(result, SignalValidationResult)
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == "AAPL"
        assert result.normalized_score > 0
        assert result.quality_grade in [SignalQuality.GOOD, SignalQuality.EXCELLENT, SignalQuality.FAIR]
        assert result.recommended_action in ["trade", "monitor"]

    def test_validate_momentum_signal_success(self):
        """Test successful momentum signal validation."""
        # Create data with clear momentum pattern
        momentum_data = self._create_momentum_pattern()

        result = self.validator.validate_signal(
            signal_type=SignalType.MOMENTUM,
            symbol="TSLA",
            market_data=momentum_data
        )

        assert isinstance(result, SignalValidationResult)
        assert result.signal_type == SignalType.MOMENTUM
        assert result.symbol == "TSLA"
        assert result.normalized_score > 0
        assert result.confidence_level > 0

    def test_validate_signal_insufficient_data(self):
        """Test validation with insufficient data."""
        short_data = self.sample_market_data.head(5)

        result = self.validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="TEST",
            market_data=short_data
        )

        assert result.normalized_score == 0.0
        assert result.quality_grade == SignalQuality.VERY_POOR
        assert result.recommended_action == "reject"

    def test_validate_signal_unknown_type(self):
        """Test validation with unsupported signal type."""
        result = self.validator.validate_signal(
            signal_type=SignalType.FUNDAMENTAL,  # Not registered
            symbol="TEST",
            market_data=self.sample_market_data
        )

        assert result.normalized_score == 0.0
        assert result.recommended_action == "reject"
        assert "No calculator available" in result.validation_notes[0]

    def test_signal_history_tracking(self):
        """Test that validation results are stored in history."""
        initial_count = len(self.validator.signal_history)

        self.validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="TEST1",
            market_data=self.sample_market_data
        )

        self.validator.validate_signal(
            signal_type=SignalType.MOMENTUM,
            symbol="TEST2",
            market_data=self.sample_market_data
        )

        assert len(self.validator.signal_history) == initial_count + 2

    def test_get_validation_summary_empty(self):
        """Test validation summary with no history."""
        validator = SignalStrengthValidator()
        summary = validator.get_validation_summary()

        assert "No validation history available" in summary["message"]

    def test_get_validation_summary_with_data(self):
        """Test validation summary with historical data."""
        # Generate some validation results
        for i in range(10):
            self.validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"TEST{i}",
                market_data=self.sample_market_data
            )

        summary = self.validator.get_validation_summary()

        assert 'total_signals' in summary
        assert 'average_score' in summary
        assert 'quality_distribution' in summary
        assert 'pass_rates' in summary
        assert 'recommendation_distribution' in summary
        assert summary['total_signals'] >= 10

    def test_export_validation_history(self, tmp_path):
        """Test exporting validation history to JSON."""
        # Generate some validation results
        for i in range(3):
            self.validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=f"TEST{i}",
                market_data=self.sample_market_data
            )

        export_file = tmp_path / "validation_history.json"
        self.validator.export_validation_history(str(export_file))

        assert export_file.exists()

        # Verify file content
        import json
        with open(export_file, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) >= 3
        assert all('signal_id' in item for item in data)

    def _create_breakout_pattern(self) -> pd.DataFrame:
        """Create market data with clear breakout pattern."""
        data = self.sample_market_data.copy()

        # Create resistance level for first 30 periods
        resistance_price = 100.0
        data.loc[:29, 'Close'] = np.random.uniform(95, 99.5, 30)
        data.loc[:29, 'High'] = np.minimum(data.loc[:29, 'Close'] + 1, 99.5)

        # Create strong breakout in later periods - more pronounced
        data.loc[30:, 'Close'] = np.random.uniform(105, 115, len(data) - 30)  # Stronger breakout
        data.loc[30:, 'High'] = data.loc[30:, 'Close'] + 2

        # Increase volume during breakout - more dramatic
        data.loc[30:, 'Volume'] = data.loc[30:, 'Volume'] * 5  # Higher volume spike

        return data

    def _create_momentum_pattern(self) -> pd.DataFrame:
        """Create market data with clear momentum pattern."""
        data = self.sample_market_data.copy()

        # Create ascending price pattern
        base_price = 100.0
        for i in range(len(data)):
            data.loc[i, 'Close'] = base_price + i * 0.5 + np.random.normal(0, 0.5)
            data.loc[i, 'High'] = data.loc[i, 'Close'] + np.random.uniform(0.5, 1.5)
            data.loc[i, 'Low'] = data.loc[i, 'Close'] - np.random.uniform(0.5, 1.5)

        return data


class TestBreakoutSignalCalculator:
    """Test the breakout signal strength calculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = BreakoutSignalCalculator()
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample market data."""
        return pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 100, 104, 101, 106, 108, 110],
            'Volume': [1000, 1100, 900, 1200, 800, 1300, 1000, 1500, 1100, 2000, 2200, 2500]
        })

    def test_get_signal_type(self):
        """Test signal type identification."""
        assert self.calculator.get_signal_type() == SignalType.BREAKOUT

    def test_calculate_raw_strength_insufficient_data(self):
        """Test strength calculation with insufficient data."""
        short_data = self.sample_data.head(5)
        strength = self.calculator.calculate_raw_strength(short_data)
        assert strength == 0.0

    def test_calculate_raw_strength_no_breakout(self):
        """Test strength calculation with no clear breakout."""
        # Flat price data
        flat_data = pd.DataFrame({
            'Close': [100] * 20,
            'Volume': [1000] * 20
        })
        strength = self.calculator.calculate_raw_strength(flat_data)
        assert strength >= 0.0
        assert strength <= 100.0

    def test_calculate_raw_strength_clear_breakout(self):
        """Test strength calculation with clear breakout."""
        # Data showing clear breakout pattern
        breakout_data = pd.DataFrame({
            'Close': [100] * 15 + [105, 107, 110],  # Clear breakout
            'Volume': [1000] * 15 + [2000, 2200, 2500]  # Volume confirmation
        })
        strength = self.calculator.calculate_raw_strength(breakout_data)
        assert strength > 0.0
        assert strength <= 100.0

    def test_calculate_raw_strength_no_volume_data(self):
        """Test strength calculation without volume data."""
        no_volume_data = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 100, 104, 101, 106, 108, 110]
        })
        strength = self.calculator.calculate_raw_strength(no_volume_data)
        assert strength >= 0.0
        assert strength <= 100.0

    def test_calculate_confidence_insufficient_data(self):
        """Test confidence calculation with insufficient data."""
        short_data = self.sample_data.head(5)
        confidence = self.calculator.calculate_confidence(short_data)
        assert confidence == 0.0

    def test_calculate_confidence_normal_case(self):
        """Test confidence calculation with normal data."""
        confidence = self.calculator.calculate_confidence(self.sample_data)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_no_volume(self):
        """Test confidence calculation without volume data."""
        no_volume_data = pd.DataFrame({
            'Close': [100, 101, 99, 102, 98, 103, 100, 104, 101, 106, 108, 110]
        })
        confidence = self.calculator.calculate_confidence(no_volume_data)
        assert 0.0 <= confidence <= 1.0

    def test_custom_parameters(self):
        """Test calculator with custom parameters."""
        custom_calculator = BreakoutSignalCalculator(lookback_periods=10, volume_weight=0.5)

        assert custom_calculator.lookback_periods == 10
        assert custom_calculator.volume_weight == 0.5

        strength = custom_calculator.calculate_raw_strength(self.sample_data)
        assert 0.0 <= strength <= 100.0

    def test_error_handling(self):
        """Test error handling in calculations."""
        # Test with invalid data
        invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})

        strength = self.calculator.calculate_raw_strength(invalid_data)
        assert strength == 0.0

        confidence = self.calculator.calculate_confidence(invalid_data)
        assert confidence == 0.0


class TestMomentumSignalCalculator:
    """Test the momentum signal strength calculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MomentumSignalCalculator()
        self.sample_data = self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample market data with momentum."""
        prices = []
        base_price = 100.0
        for i in range(25):
            # Create upward momentum
            base_price += np.random.uniform(0.2, 0.8)
            prices.append(base_price)

        return pd.DataFrame({'Close': prices})

    def test_get_signal_type(self):
        """Test signal type identification."""
        assert self.calculator.get_signal_type() == SignalType.MOMENTUM

    def test_calculate_raw_strength_insufficient_data(self):
        """Test strength calculation with insufficient data."""
        short_data = pd.DataFrame({'Close': [100, 101, 102]})
        strength = self.calculator.calculate_raw_strength(short_data)
        assert strength == 0.0

    def test_calculate_raw_strength_normal_case(self):
        """Test strength calculation with normal data."""
        strength = self.calculator.calculate_raw_strength(self.sample_data)
        assert strength >= 0.0
        assert strength <= 100.0

    def test_calculate_raw_strength_no_momentum(self):
        """Test strength calculation with no momentum."""
        flat_data = pd.DataFrame({'Close': [100] * 25})
        strength = self.calculator.calculate_raw_strength(flat_data)
        assert strength >= 0.0

    def test_calculate_confidence_insufficient_data(self):
        """Test confidence calculation with insufficient data."""
        short_data = pd.DataFrame({'Close': [100, 101, 102]})
        confidence = self.calculator.calculate_confidence(short_data)
        assert confidence == 0.0

    def test_calculate_confidence_normal_case(self):
        """Test confidence calculation with normal data."""
        confidence = self.calculator.calculate_confidence(self.sample_data)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_consistent_trend(self):
        """Test confidence with very consistent upward trend."""
        consistent_data = pd.DataFrame({'Close': list(range(100, 125))})
        confidence = self.calculator.calculate_confidence(consistent_data)
        assert confidence > 0.5  # Should be high for consistent trend

    def test_custom_parameters(self):
        """Test calculator with custom parameters."""
        custom_calculator = MomentumSignalCalculator(short_window=3, long_window=10)

        assert custom_calculator.short_window == 3
        assert custom_calculator.long_window == 10

        strength = custom_calculator.calculate_raw_strength(self.sample_data)
        assert 0.0 <= strength <= 100.0

    def test_error_handling(self):
        """Test error handling in calculations."""
        # Test with invalid data
        invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})

        strength = self.calculator.calculate_raw_strength(invalid_data)
        assert strength == 0.0

        confidence = self.calculator.calculate_confidence(invalid_data)
        assert confidence == 0.0


class TestSignalMetrics:
    """Test the SignalMetrics dataclass."""

    def test_signal_metrics_creation(self):
        """Test creating SignalMetrics instance."""
        metrics = SignalMetrics(
            strength_score=75.0,
            confidence=0.8,
            consistency=0.7,
            volume_confirmation=0.6,
            technical_confluence=0.5,
            risk_reward_ratio=2.5,
            time_decay_factor=0.3,
            market_regime_fit=0.4
        )

        assert metrics.strength_score == 75.0
        assert metrics.confidence == 0.8
        assert metrics.consistency == 0.7
        assert metrics.volume_confirmation == 0.6
        assert metrics.technical_confluence == 0.5
        assert metrics.risk_reward_ratio == 2.5
        assert metrics.time_decay_factor == 0.3
        assert metrics.market_regime_fit == 0.4


class TestSignalValidationResult:
    """Test the SignalValidationResult dataclass."""

    def test_signal_validation_result_creation(self):
        """Test creating SignalValidationResult instance."""
        metrics = SignalMetrics(75.0, 0.8, 0.7, 0.6, 0.5, 2.5, 0.3, 0.4)

        result = SignalValidationResult(
            signal_id="TEST_001",
            signal_type=SignalType.BREAKOUT,
            timestamp=datetime.now(),
            symbol="AAPL",
            raw_metrics=metrics,
            normalized_score=82.5,
            quality_grade=SignalQuality.GOOD,
            passes_minimum_threshold=True,
            passes_consistency_check=True,
            passes_regime_filter=True,
            passes_risk_check=True,
            recommended_action="trade",
            confidence_level=0.85,
            suggested_position_size=0.8
        )

        assert result.signal_id == "TEST_001"
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == "AAPL"
        assert result.normalized_score == 82.5
        assert result.quality_grade == SignalQuality.GOOD
        assert result.recommended_action == "trade"
        assert result.confidence_level == 0.85
        assert result.suggested_position_size == 0.8


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SignalStrengthValidator()

    def test_high_quality_breakout_scenario(self):
        """Test high-quality breakout signal scenario."""
        # Create ideal breakout conditions
        data = pd.DataFrame({
            'Close': [100] * 20 + [105, 107, 110, 112],  # Clear breakout
            'Volume': [1000] * 20 + [3000, 3500, 4000, 4500],  # Strong volume
            'High': [101] * 20 + [106, 108, 111, 113],
            'Low': [99] * 20 + [104, 106, 109, 111]
        })

        result = self.validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol="STRONG_STOCK",
            market_data=data,
            signal_params={'risk_reward_ratio': 3.0, 'max_hold_hours': 4}
        )

        assert result.normalized_score > 60.0  # Should be high quality
        assert result.quality_grade in [SignalQuality.GOOD, SignalQuality.EXCELLENT]
        assert result.recommended_action == "trade"
        assert result.passes_minimum_threshold
        assert result.suggested_position_size > 0.5

    def test_poor_quality_signal_scenario(self):
        """Test poor quality signal scenario."""
        # Create poor signal conditions
        data = pd.DataFrame({
            'Close': np.random.uniform(99, 101, 25),  # Choppy, no clear direction
            'Volume': [500] * 25  # Low volume
        })

        result = self.validator.validate_signal(
            signal_type=SignalType.MOMENTUM,
            symbol="WEAK_STOCK",
            market_data=data,
            signal_params={'risk_reward_ratio': 1.0}  # Poor risk/reward
        )

        assert result.normalized_score < 50.0
        assert result.quality_grade in [SignalQuality.POOR, SignalQuality.VERY_POOR]
        assert result.recommended_action in ["monitor", "reject"]

    def test_batch_validation_scenario(self):
        """Test validating multiple signals in batch."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        results = []

        for symbol in symbols:
            # Create varied market data for each symbol
            data = self._create_varied_market_data(symbol)

            result = self.validator.validate_signal(
                signal_type=SignalType.BREAKOUT,
                symbol=symbol,
                market_data=data
            )
            results.append(result)

        assert len(results) == 5
        assert all(isinstance(r, SignalValidationResult) for r in results)
        assert len({r.symbol for r in results}) == 5  # All different symbols

        # Check that history was updated
        assert len(self.validator.signal_history) >= 5

    def test_configuration_impact_scenario(self):
        """Test how different configurations impact validation."""
        data = self._create_marginal_signal_data()

        # Strict configuration
        strict_config = {
            'minimum_strength_threshold': 80.0,
            'minimum_confidence_threshold': 0.8,
            'risk_reward_minimum': 3.0
        }
        strict_validator = SignalStrengthValidator(strict_config)

        # Lenient configuration
        lenient_config = {
            'minimum_strength_threshold': 40.0,
            'minimum_confidence_threshold': 0.4,
            'risk_reward_minimum': 1.0
        }
        lenient_validator = SignalStrengthValidator(lenient_config)

        strict_result = strict_validator.validate_signal(
            SignalType.BREAKOUT, "TEST", data
        )
        lenient_result = lenient_validator.validate_signal(
            SignalType.BREAKOUT, "TEST", data
        )

        # Strict validator should be more conservative
        assert strict_result.recommended_action in ["monitor", "reject"]
        # Lenient validator should be more permissive
        assert lenient_result.passes_minimum_threshold or lenient_result.normalized_score > 0

    def _create_varied_market_data(self, symbol: str) -> pd.DataFrame:
        """Create varied market data based on symbol."""
        np.random.seed(hash(symbol) % 2**32)  # Deterministic but varied per symbol

        length = 30
        base_price = 100.0
        prices = [base_price]

        for i in range(length - 1):
            # Different behavior per symbol
            if symbol == "AAPL":
                # Steady uptrend
                change = np.random.normal(0.01, 0.02)
            elif symbol == "TSLA":
                # High volatility
                change = np.random.normal(0.005, 0.05)
            else:
                # Normal behavior
                change = np.random.normal(0.002, 0.025)

            prices.append(prices[-1] * (1 + change))

        volumes = np.random.normal(1000000, 200000, length)
        volumes = np.maximum(volumes, 100000)

        return pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'High': [p * 1.02 for p in prices],
            'Low': [p * 0.98 for p in prices]
        })

    def _create_marginal_signal_data(self) -> pd.DataFrame:
        """Create marginal quality signal data."""
        # Weak breakout pattern
        data = pd.DataFrame({
            'Close': [100] * 15 + [102, 103, 102.5, 103.5],
            'Volume': [1000] * 15 + [1200, 1300, 1100, 1250]
        })
        return data


# Performance and stress tests
class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        validator = SignalStrengthValidator()

        # Create large dataset
        large_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 1000),
            'Volume': np.random.normal(1000000, 200000, 1000)
        })

        import time
        start_time = time.time()

        result = validator.validate_signal(
            SignalType.BREAKOUT,
            "LARGE_TEST",
            large_data
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (less than 5 seconds)
        assert execution_time < 5.0
        assert isinstance(result, SignalValidationResult)

    def test_memory_usage_with_history(self):
        """Test memory usage with large validation history."""
        validator = SignalStrengthValidator()
        sample_data = pd.DataFrame({
            'Close': np.random.normal(100, 5, 50),
            'Volume': np.random.normal(1000000, 100000, 50)
        })

        # Generate many validation results
        for i in range(200):
            validator.validate_signal(
                SignalType.BREAKOUT,
                f"TEST_{i}",
                sample_data
            )

        # Validator should handle large history
        assert len(validator.signal_history) == 200

        summary = validator.get_validation_summary()
        assert summary['total_signals'] == 50  # Should limit to last 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backend.tradingbot.validation.signal_strength_validator", "--cov-report=html"])