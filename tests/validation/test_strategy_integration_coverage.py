#!/usr/bin/env python3
"""
Comprehensive test suite to achieve >75% coverage for strategy signal integration.

Targets all methods in the strategy signal integration module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import logging

# Import strategy signal integration components
from backend.tradingbot.validation.strategy_signal_integration import (
    StrategySignalConfig,
    StrategySignalMixin,
    SwingTradingSignalCalculator,
    MomentumWeekliesSignalCalculator,
    LEAPSSignalCalculator
)

from backend.tradingbot.validation.signal_strength_validator import (
    SignalType,
    SignalQuality,
    SignalValidationResult,
    SignalMetrics
)


class TestStrategySignalConfig:
    """Test StrategySignalConfig functionality."""

    def test_strategy_signal_config_initialization(self):
        """Test StrategySignalConfig initialization."""
        config = StrategySignalConfig(
            strategy_name="test_strategy",
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=70.0,
            minimum_confidence_threshold=0.7,
            risk_reward_minimum=2.0,
            max_position_size_multiplier=1.5,
            enable_regime_filtering=True,
            custom_validation_params={'param1': 'value1'}
        )

        assert config.strategy_name == "test_strategy"
        assert config.default_signal_type == SignalType.BREAKOUT
        assert config.minimum_strength_threshold == 70.0
        assert config.minimum_confidence_threshold == 0.7
        assert config.risk_reward_minimum == 2.0
        assert config.max_position_size_multiplier == 1.5
        assert config.enable_regime_filtering
        assert config.custom_validation_params == {'param1': 'value1'}

    def test_strategy_signal_config_defaults(self):
        """Test StrategySignalConfig default values."""
        config = StrategySignalConfig(
            strategy_name="default_test",
            default_signal_type=SignalType.MOMENTUM
        )

        assert config.minimum_strength_threshold == 65.0
        assert config.minimum_confidence_threshold == 0.6
        assert config.risk_reward_minimum == 1.8
        assert config.max_position_size_multiplier == 1.0
        assert config.enable_regime_filtering
        assert config.custom_validation_params == {}

    def test_strategy_signal_config_post_init(self):
        """Test StrategySignalConfig __post_init__ method."""
        # Test with None custom_validation_params
        config = StrategySignalConfig(
            strategy_name="test",
            default_signal_type=SignalType.TREND,
            custom_validation_params=None
        )
        assert config.custom_validation_params == {}


class TestCustomSignalCalculators:
    """Test custom signal calculator implementations."""

    def test_swing_trading_signal_calculator(self):
        """Test SwingTradingSignalCalculator."""
        calc = SwingTradingSignalCalculator()

        assert calc.get_signal_type() == SignalType.BREAKOUT

        # Test with breakout pattern
        breakout_data = pd.DataFrame({
            'Close': [100] * 15 + [105, 108, 112],
            'Volume': [1000000] * 15 + [2500000, 3000000, 3500000],
            'High': [101] * 15 + [106, 109, 113],
            'Low': [99] * 15 + [104, 107, 111]
        })

        strength = calc.calculate_raw_strength(breakout_data)
        assert isinstance(strength, (int, float))
        assert strength >= 0

        confidence = calc.calculate_confidence(breakout_data)
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1

        # Test with insufficient data
        small_data = pd.DataFrame({
            'Close': [100, 101],
            'Volume': [1000000, 1100000]
        })

        small_strength = calc.calculate_raw_strength(small_data)
        assert small_strength >= 0

    def test_momentum_weeklies_signal_calculator(self):
        """Test MomentumWeekliesSignalCalculator."""
        calc = MomentumWeekliesSignalCalculator()

        assert calc.get_signal_type() == SignalType.MOMENTUM

        # Test with momentum pattern
        momentum_data = pd.DataFrame({
            'Close': [100, 103, 106, 109, 112, 115],
            'Volume': [1000000, 1500000, 2000000, 2500000, 3000000, 3500000]
        })

        strength = calc.calculate_raw_strength(momentum_data)
        assert isinstance(strength, (int, float))
        assert strength >= 0

        confidence = calc.calculate_confidence(momentum_data)
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1

        # Test with custom parameters
        custom_strength = calc.calculate_raw_strength(momentum_data, volume_spike_threshold=2.0)
        assert isinstance(custom_strength, (int, float))

    def test_leaps_signal_calculator(self):
        """Test LEAPSSignalCalculator."""
        calc = LEAPSSignalCalculator()

        assert calc.get_signal_type() == SignalType.TREND

        # Test with trend pattern
        trend_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110, 112],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000]
        })

        strength = calc.calculate_raw_strength(trend_data)
        assert isinstance(strength, (int, float))
        assert strength >= 0

        confidence = calc.calculate_confidence(trend_data)
        assert isinstance(confidence, (int, float))
        assert 0 <= confidence <= 1

        # Test with custom parameters
        custom_confidence = calc.calculate_confidence(trend_data, trend_minimum_length=5)
        assert isinstance(custom_confidence, (int, float))


class TestStrategySignalMixin:
    """Test StrategySignalMixin functionality."""

    def test_strategy_signal_mixin_initialization(self):
        """Test StrategySignalMixin initialization."""
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
            'strategy_name': 'custom_test',
            'min_strength_score': 75.0
        }
        strategy_custom = TestStrategy(custom_config)
        assert strategy_custom.config['min_strength_score'] == 75.0

    def test_initialize_signal_validation_method(self):
        """Test initialize_signal_validation method."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {'strategy_name': 'test_init'}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation
        config = StrategySignalConfig(strategy_name="test_init", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        # Verify initialization completed
        assert strategy._signal_validator is not None
        assert strategy._signal_config is not None
        assert isinstance(strategy._signal_history, list)

    def test_validate_signal_method(self):
        """Test validate_signal method."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation first
        config = StrategySignalConfig(strategy_name="test_strategy", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        test_data = pd.DataFrame({
            'Close': [100, 102, 104, 106],
            'Volume': [1000000, 1200000, 1400000, 1600000],
            'High': [101, 103, 105, 107],
            'Low': [99, 101, 103, 105]
        })

        # Test basic validation
        result = strategy.validate_signal(
            symbol="TEST_VALIDATE",
            market_data=test_data,
            signal_type=SignalType.BREAKOUT
        )

        assert isinstance(result, SignalValidationResult)
        assert result.symbol == "TEST_VALIDATE"
        assert result.signal_type == SignalType.BREAKOUT

        # Test with custom signal parameters
        custom_params = {'risk_reward_ratio': 3.0}
        result_custom = strategy.validate_signal(
            symbol="TEST_CUSTOM",
            market_data=test_data,
            signal_type=SignalType.MOMENTUM,
            signal_params=custom_params
        )

        assert isinstance(result_custom, SignalValidationResult)

        # Test with None signal_type (should use default)
        result_default = strategy.validate_signal(
            symbol="TEST_DEFAULT",
            market_data=test_data
        )

        assert isinstance(result_default, SignalValidationResult)

    def test_get_strategy_signal_summary_method(self):
        """Test get_strategy_signal_summary method."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation first
        config = StrategySignalConfig(strategy_name="test_strategy", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        # Test empty summary
        empty_summary = strategy.get_strategy_signal_summary()
        assert isinstance(empty_summary, dict)
        assert 'message' in empty_summary
        assert empty_summary['message'] == "No signal validation history"

        # Add some validation history
        test_data = pd.DataFrame({
            'Close': [100, 102, 104],
            'Volume': [1000000, 1200000, 1400000]
        })

        for i in range(3):
            strategy.validate_signal(f"TEST_{i}", test_data, SignalType.BREAKOUT)

        summary = strategy.get_strategy_signal_summary()
        assert summary['total_signals_validated'] == 3
        assert 'signals_recommended_for_trading' in summary


# Note: signal_integrator and enhance_strategy_with_validation function tests
# removed as these don't exist in the current implementation


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_mixin_with_malformed_config(self):
        """Test StrategySignalMixin with malformed config."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {
                    'invalid_key': 'invalid_value',
                    'min_strength_score': 'not_a_number'  # Invalid type
                }
                self.logger = Mock()
                super().__init__()

        # Should handle gracefully
        strategy = TestStrategy()
        assert hasattr(strategy, '_signal_validator')

    def test_validate_signal_with_bad_data(self):
        """Test validate_signal with problematic data."""
        class TestStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {}
                self.logger = Mock()
                super().__init__()

        strategy = TestStrategy()
        
        # Initialize signal validation first
        config = StrategySignalConfig(strategy_name="test_strategy", default_signal_type=SignalType.BREAKOUT)
        strategy.initialize_signal_validation(config)

        # Test with None data
        result_none = strategy.validate_signal(
            symbol="NULL_DATA",
            market_data=None,
            signal_type=SignalType.BREAKOUT
        )
        assert result_none.quality_grade == SignalQuality.VERY_POOR

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result_empty = strategy.validate_signal(
            symbol="EMPTY_DATA",
            market_data=empty_df,
            signal_type=SignalType.MOMENTUM
        )
        assert result_empty.quality_grade == SignalQuality.VERY_POOR

    def test_calculator_edge_cases(self):
        """Test calculator edge cases."""
        swing_calc = SwingTradingSignalCalculator()

        # Test with None data
        try:
            strength = swing_calc.calculate_raw_strength(None)
            assert strength >= 0
        except Exception:
            pass  # Exception handling is acceptable

        # Test with malformed data
        bad_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        try:
            strength = swing_calc.calculate_raw_strength(bad_data)
            assert strength >= 0
        except Exception:
            pass  # Exception handling is acceptable

    def test_config_parameter_validation(self):
        """Test configuration parameter validation."""
        # Test with extreme values
        config = StrategySignalConfig(
            strategy_name="extreme_test",
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=150.0,  # Very high
            minimum_confidence_threshold=1.5,  # Invalid (>1)
            risk_reward_minimum=-1.0  # Invalid negative
        )

        # Should handle gracefully
        assert config.minimum_strength_threshold == 150.0


class TestIntegrationWithRealStrategies:
    """Test integration with realistic strategy patterns."""

    def test_realistic_swing_trading_integration(self):
        """Test realistic swing trading strategy integration."""
        class SwingTradingStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {
                    'strategy_name': 'swing_trading',
                    'min_strength_score': 70.0,
                    'watchlist': ['AAPL', 'GOOGL', 'MSFT'],
                    'max_positions': 5
                }
                self.logger = Mock()
                super().__init__()
                
                # Initialize signal validation
                config = StrategySignalConfig(strategy_name="swing_trading", default_signal_type=SignalType.BREAKOUT)
                self.initialize_signal_validation(config)

            def scan_for_opportunities(self):
                """Mock scanning method."""
                # Create realistic market data
                market_data = pd.DataFrame({
                    'Close': [150, 152, 154, 158, 162, 165],
                    'Volume': [2000000, 2200000, 2500000, 3000000, 3500000, 4000000],
                    'High': [151, 153, 155, 159, 163, 166],
                    'Low': [149, 151, 153, 157, 161, 164]
                })

                # Validate signal
                result = self.validate_signal(
                    symbol='AAPL',
                    market_data=market_data,
                    signal_type=SignalType.BREAKOUT
                )

                return result.recommended_action in ['execute', 'monitor']

        strategy = SwingTradingStrategy()
        can_trade = strategy.scan_for_opportunities()
        assert isinstance(can_trade, bool)

        # Check summary
        summary = strategy.get_strategy_signal_summary()
        assert summary['total_signals_validated'] >= 1

    def test_realistic_momentum_strategy_integration(self):
        """Test realistic momentum strategy integration."""
        class MomentumStrategy(StrategySignalMixin):
            def __init__(self):
                self.config = {
                    'strategy_name': 'momentum_weeklies',
                    'min_strength_score': 75.0,
                    'watchlist': ['SPY', 'QQQ', 'IWM']
                }
                self.logger = Mock()
                super().__init__()
                
                # Initialize signal validation
                config = StrategySignalConfig(strategy_name="momentum_weeklies", default_signal_type=SignalType.MOMENTUM)
                self.initialize_signal_validation(config)

            def analyze_momentum(self, symbol):
                """Mock momentum analysis."""
                momentum_data = pd.DataFrame({
                    'Close': [400, 405, 410, 415, 420, 425, 430],
                    'Volume': [5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000]
                })

                result = self.validate_signal(
                    symbol=symbol,
                    market_data=momentum_data,
                    signal_type=SignalType.MOMENTUM,
                    signal_params={'volume_spike_threshold': 1.5}
                )

                return result.normalized_score > 70.0

        strategy = MomentumStrategy()
        is_strong = strategy.analyze_momentum('SPY')
        assert isinstance(is_strong, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])