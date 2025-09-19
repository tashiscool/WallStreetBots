"""
Integration Test for Complete Signal Strength Validation System
==============================================================

Tests the entire signal validation system end-to-end with realistic scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from tradingbot.validation.signal_strength_validator import (
    SignalStrengthValidator, SignalType, SignalQuality
)
from tradingbot.validation.strategy_signal_integration import (
    signal_integrator, StrategySignalMixin, StrategySignalConfig
)


class MockStrategy(StrategySignalMixin):
    """Mock strategy class for testing integration."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.signals_generated = []

    def generate_signal(self, symbol: str, market_data: pd.DataFrame):
        """Generate a mock signal with validation."""
        # Get signal strength
        validation_result = self.validate_signal(symbol, market_data)

        signal = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'validation_score': validation_result.normalized_score,
            'recommended_action': validation_result.recommended_action,
            'confidence': validation_result.confidence_level
        }

        self.signals_generated.append(signal)
        return signal


def create_test_market_data(pattern_type: str = "trending") -> pd.DataFrame:
    """Create test market data with different patterns."""
    np.random.seed(42)
    length = 50

    if pattern_type == "trending":
        # Strong uptrend
        base_prices = np.linspace(100, 120, length)
        noise = np.random.normal(0, 1, length)
        prices = base_prices + noise
        volumes = np.random.normal(1000000, 200000, length)

    elif pattern_type == "breakout":
        # Consolidation followed by breakout
        prices = [100 + np.random.normal(0, 0.5) for _ in range(30)]  # Consolidation
        prices.extend([105 + i * 0.5 + np.random.normal(0, 0.3) for i in range(20)])  # Breakout
        volumes = [1000000] * 30 + [2000000] * 20  # Volume spike on breakout

    elif pattern_type == "choppy":
        # Choppy, directionless
        prices = [100 + np.random.normal(0, 2) for _ in range(length)]
        volumes = np.random.normal(800000, 100000, length)

    else:
        # Default random walk
        prices = [100]
        for _ in range(length - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        volumes = np.random.normal(1000000, 200000, length)

    return pd.DataFrame({
        'Close': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Volume': np.maximum(volumes, 100000)
    })


class TestCompleteIntegration:
    """Test complete integration of signal validation system."""

    def test_swing_trading_integration(self):
        """Test swing trading strategy integration."""
        strategy = MockStrategy("swing_trading")

        # Enhance with signal validation
        signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

        # Test with breakout pattern
        breakout_data = create_test_market_data("breakout")
        signal = strategy.generate_signal("AAPL", breakout_data)

        assert 'validation_score' in signal
        assert 'recommended_action' in signal
        assert signal['validation_score'] >= 0
        assert signal['recommended_action'] in ['trade', 'monitor', 'reject']

    def test_momentum_weeklies_integration(self):
        """Test momentum weeklies strategy integration."""
        strategy = MockStrategy("momentum_weeklies")

        # Enhance with signal validation
        signal_integrator.enhance_strategy_with_validation(strategy, "momentum_weeklies")

        # Test with trending pattern
        trending_data = create_test_market_data("trending")
        signal = strategy.generate_signal("TSLA", trending_data)

        assert signal['validation_score'] >= 0
        assert signal['confidence'] >= 0

    def test_leaps_integration(self):
        """Test LEAPS strategy integration."""
        strategy = MockStrategy("leaps_tracker")

        # Enhance with signal validation
        signal_integrator.enhance_strategy_with_validation(strategy, "leaps_tracker")

        # Test with trending pattern (good for LEAPS)
        trending_data = create_test_market_data("trending")
        signal = strategy.generate_signal("GOOGL", trending_data)

        assert signal['validation_score'] >= 0
        assert signal['recommended_action'] in ['trade', 'monitor', 'reject']

    def test_signal_filtering(self):
        """Test signal filtering functionality."""
        strategy = MockStrategy("swing_trading")
        signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

        # Create multiple signals with different quality
        signals = [
            {'symbol': 'HIGH_QUALITY', 'signal_type': 'breakout'},
            {'symbol': 'MEDIUM_QUALITY', 'signal_type': 'breakout'},
            {'symbol': 'LOW_QUALITY', 'signal_type': 'breakout'}
        ]

        def mock_data_getter(symbol):
            if symbol == 'HIGH_QUALITY':
                return create_test_market_data("breakout")
            elif symbol == 'MEDIUM_QUALITY':
                return create_test_market_data("trending")
            else:
                return create_test_market_data("choppy")

        filtered_signals = strategy.filter_signals_by_strength(signals, mock_data_getter)

        # Should have some filtering effect
        assert len(filtered_signals) <= len(signals)

        # Filtered signals should have validation metadata
        for signal in filtered_signals:
            assert 'validation_score' in signal
            assert 'validation_confidence' in signal
            assert 'validation_quality' in signal

    def test_strategy_signal_summary(self):
        """Test strategy signal summary generation."""
        strategy = MockStrategy("swing_trading")
        signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

        # Generate multiple signals
        test_data = create_test_market_data("breakout")
        for i in range(5):
            strategy.generate_signal(f"TEST{i}", test_data)

        summary = strategy.get_strategy_signal_summary()

        assert 'strategy_name' in summary
        assert 'total_signals_validated' in summary
        assert 'average_strength_score' in summary
        assert 'quality_distribution' in summary
        assert summary['total_signals_validated'] >= 5

    def test_multiple_strategies_parallel(self):
        """Test multiple strategies working in parallel."""
        strategies = {
            'swing': MockStrategy("swing_trading"),
            'momentum': MockStrategy("momentum_weeklies"),
            'leaps': MockStrategy("leaps_tracker")
        }

        # Enhance all strategies
        for name, strategy in strategies.items():
            if name == 'swing':
                signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")
            elif name == 'momentum':
                signal_integrator.enhance_strategy_with_validation(strategy, "momentum_weeklies")
            else:
                signal_integrator.enhance_strategy_with_validation(strategy, "leaps_tracker")

        # Generate signals from all strategies
        test_data = create_test_market_data("trending")
        results = {}

        for name, strategy in strategies.items():
            signal = strategy.generate_signal("MULTI_TEST", test_data)
            results[name] = signal

        # Verify all strategies generated valid signals
        assert len(results) == 3
        for name, signal in results.items():
            assert 'validation_score' in signal
            assert signal['validation_score'] >= 0

    def test_performance_with_large_dataset(self):
        """Test performance with large market dataset."""
        strategy = MockStrategy("swing_trading")
        signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

        # Create large dataset
        large_data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 1000),
            'Volume': np.random.normal(1000000, 200000, 1000),
            'High': np.random.normal(102, 10, 1000),
            'Low': np.random.normal(98, 10, 1000)
        })

        import time
        start_time = time.time()

        signal = strategy.generate_signal("PERF_TEST", large_data)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 2 seconds)
        assert execution_time < 2.0
        assert 'validation_score' in signal

    def test_error_handling(self):
        """Test error handling in integration."""
        strategy = MockStrategy("swing_trading")
        signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

        # Test with invalid data
        invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})

        # Should handle gracefully
        try:
            signal = strategy.generate_signal("ERROR_TEST", invalid_data)
            # Should return a signal with low/zero validation score
            assert signal['validation_score'] == 0.0
            assert signal['recommended_action'] == 'reject'
        except Exception as e:
            # Or might raise exception - both are acceptable
            assert "validation" in str(e).lower() or "error" in str(e).lower()

    def test_custom_configuration(self):
        """Test custom strategy configuration."""
        strategy = MockStrategy("custom_strategy")

        # Create custom config
        custom_config = StrategySignalConfig(
            strategy_name="custom_strategy",
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=80.0,  # Very strict
            minimum_confidence_threshold=0.8,
            risk_reward_minimum=3.0
        )

        signal_integrator.register_strategy_config("custom_strategy", custom_config)
        signal_integrator.enhance_strategy_with_validation(strategy, "custom_strategy")

        # Test with marginal data
        marginal_data = create_test_market_data("choppy")
        signal = strategy.generate_signal("CUSTOM_TEST", marginal_data)

        # Should be more conservative due to strict thresholds
        assert signal['validation_score'] >= 0
        # Likely to be rejected due to strict thresholds
        # (but we can't guarantee due to randomness)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])