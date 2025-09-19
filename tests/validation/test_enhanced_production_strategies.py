#!/usr/bin/env python3
"""
Enhanced test suite for production strategies with signal validation.

Focuses on testing the newly enhanced signal validation functionality
to achieve >75% test coverage for the modified code.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import logging

# Import production strategies
from backend.tradingbot.strategies.production.production_swing_trading import ProductionSwingTrading
from backend.tradingbot.strategies.production.production_momentum_weeklies import ProductionMomentumWeeklies
from backend.tradingbot.strategies.production.production_leaps_tracker import ProductionLEAPSTracker

# Import validation framework
from backend.tradingbot.validation.signal_strength_validator import SignalType, SignalQuality
from backend.tradingbot.validation.strategy_signal_integration import StrategySignalConfig


class TestProductionStrategyEnhancements:
    """Test the enhanced functionality in production strategies."""

    @pytest.fixture
    def mock_integration_manager(self):
        """Mock integration manager."""
        manager = Mock()
        manager.logger = logging.getLogger('test')
        return manager

    @pytest.fixture
    def mock_data_provider(self):
        """Mock data provider."""
        provider = Mock()
        provider.logger = logging.getLogger('test')

        # Create mock async method
        async def mock_get_intraday_data(symbol, interval="15min", period="5d"):
            return pd.DataFrame({
                'close': [100, 101, 102, 105, 108, 110],
                'high': [101, 102, 103, 106, 109, 111],
                'low': [99, 100, 101, 104, 107, 109],
                'volume': [1000000, 1100000, 1200000, 2000000, 2200000, 2500000],
                'timestamp': pd.date_range(start='2023-01-01', periods=6, freq='15min')
            })

        provider.get_intraday_data = mock_get_intraday_data
        return provider

    def test_swing_trading_signal_validation_integration(self, mock_integration_manager, mock_data_provider):
        """Test signal validation is properly integrated in swing trading."""
        config = {
            'watchlist': ['AAPL', 'GOOGL'],
            'max_positions': 3,
            'max_position_size': 0.02,
            'min_strength_score': 70.0
        }

        strategy = ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)

        # Test initialization of signal validation components
        assert hasattr(strategy, '_signal_validator')
        assert hasattr(strategy, '_signal_config')
        assert hasattr(strategy, '_signal_history')

        # Test configuration
        assert strategy._signal_config.minimum_strength_threshold == 70.0

    def test_momentum_strategy_signal_validation_integration(self, mock_integration_manager, mock_data_provider):
        """Test signal validation is properly integrated in momentum strategy."""
        config = {
            'watchlist': ['SPY', 'QQQ'],
            'max_positions': 3,
            'min_volume_spike': 2.5,
            'min_momentum_strength': 75.0
        }

        strategy = ProductionMomentumWeeklies(mock_integration_manager, mock_data_provider, config)

        # Test initialization
        assert hasattr(strategy, '_signal_validator')
        assert hasattr(strategy, '_signal_config')

        # Test momentum-specific configuration
        assert strategy._signal_config.volume_threshold >= 1.5

    def test_leaps_strategy_signal_validation_integration(self, mock_integration_manager, mock_data_provider):
        """Test signal validation is properly integrated in LEAPS strategy."""
        config = {
            'themes': ['AI', 'CLOUD'],
            'max_positions_per_theme': 2,
            'min_trend_score': 60.0
        }

        strategy = ProductionLEAPSTracker(mock_integration_manager, mock_data_provider, config)

        # Test initialization
        assert hasattr(strategy, '_signal_validator')
        assert hasattr(strategy, '_signal_config')

        # Test LEAPS-specific configuration
        assert strategy._signal_config.minimum_strength_threshold >= 50.0

    def test_signal_validation_method_functionality(self, mock_integration_manager, mock_data_provider):
        """Test the validate_signal method works across strategies."""
        strategies = [
            (ProductionSwingTrading, {'watchlist': ['AAPL'], 'max_positions': 1}, SignalType.BREAKOUT),
            (ProductionMomentumWeeklies, {'watchlist': ['SPY'], 'max_positions': 1}, SignalType.MOMENTUM),
            (ProductionLEAPSTracker, {'themes': ['AI'], 'max_positions_per_theme': 1}, SignalType.TREND)
        ]

        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 105, 108],
            'Volume': [1000000, 1100000, 1200000, 2000000, 2200000],
            'High': [101, 102, 103, 106, 109],
            'Low': [99, 100, 101, 104, 107]
        })

        for strategy_class, config, signal_type in strategies:
            strategy = strategy_class(mock_integration_manager, mock_data_provider, config)

            result = strategy.validate_signal(
                symbol='TEST_SYMBOL',
                market_data=test_data,
                signal_type=signal_type
            )

            assert result is not None
            assert result.signal_type == signal_type
            assert hasattr(result, 'normalized_score')
            assert hasattr(result, 'quality_grade')
            assert hasattr(result, 'recommended_action')

    def test_signal_history_tracking(self, mock_integration_manager, mock_data_provider):
        """Test that signal validation history is tracked properly."""
        config = {'watchlist': ['AAPL'], 'max_positions': 1}
        strategy = ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)

        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000],
            'High': [101, 102, 103],
            'Low': [99, 100, 101]
        })

        # Validate multiple signals
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        for symbol in symbols:
            strategy.validate_signal(
                symbol=symbol,
                market_data=test_data,
                signal_type=SignalType.BREAKOUT
            )

        # Check history is tracked
        assert len(strategy._signal_history) == len(symbols)

        # Test summary method
        summary = strategy.get_strategy_signal_summary()
        assert summary['total_signals_validated'] == len(symbols)
        assert 'average_strength_score' in summary

    def test_signal_filtering_by_strength(self, mock_integration_manager, mock_data_provider):
        """Test that signals are filtered by strength threshold."""
        config = {
            'watchlist': ['AAPL'],
            'max_positions': 1,
            'min_strength_score': 80.0  # High threshold
        }
        strategy = ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)

        # Create weak signal data
        weak_data = pd.DataFrame({
            'Close': [100, 100.1, 100.2],  # Minimal movement
            'Volume': [500000, 500000, 500000],  # Low volume
            'High': [100.5, 100.6, 100.7],
            'Low': [99.5, 99.4, 99.3]
        })

        result = strategy.validate_signal(
            symbol='WEAK_SIGNAL',
            market_data=weak_data,
            signal_type=SignalType.BREAKOUT
        )

        # Should be filtered out due to low strength
        assert result.normalized_score < 80.0
        assert not result.passes_minimum_threshold

    def test_error_handling_in_validation(self, mock_integration_manager, mock_data_provider):
        """Test error handling in signal validation."""
        config = {'watchlist': ['AAPL'], 'max_positions': 1}
        strategy = ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)

        # Test with empty data
        empty_data = pd.DataFrame()
        result = strategy.validate_signal(
            symbol='TEST',
            market_data=empty_data,
            signal_type=SignalType.BREAKOUT
        )

        assert result is not None
        assert result.quality_grade == SignalQuality.VERY_POOR

        # Test with malformed data
        bad_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        result = strategy.validate_signal(
            symbol='TEST',
            market_data=bad_data,
            signal_type=SignalType.BREAKOUT
        )

        assert result is not None
        assert result.recommended_action == 'reject'

    def test_custom_signal_parameters(self, mock_integration_manager, mock_data_provider):
        """Test custom signal parameters are handled correctly."""
        config = {'watchlist': ['AAPL'], 'max_positions': 1}
        strategy = ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)

        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000, 2100000, 2200000, 2300000, 2400000, 2500000, 2600000, 2700000, 2800000, 2900000, 3000000],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        })

        # Test with custom parameters
        custom_params = {
            'risk_reward_ratio': 3.0,
            'max_hold_hours': 2,
            'volume_spike_threshold': 2.5
        }

        result = strategy.validate_signal(
            symbol='TEST',
            market_data=test_data,
            signal_type=SignalType.BREAKOUT,
            signal_params=custom_params
        )

        assert result is not None
        # Custom parameters should influence the validation
        assert result.raw_metrics.risk_reward_ratio == 3.0


class TestStrategySignalConfig:
    """Test strategy signal configuration customization."""

    def test_signal_config_creation(self):
        """Test creating strategy signal configurations."""
        config = StrategySignalConfig(
            strategy_name='test_strategy',
            default_signal_type=SignalType.BREAKOUT,
            minimum_strength_threshold=75.0,
            minimum_confidence_threshold=0.7,
            risk_reward_minimum=2.5
        )

        assert config.strategy_name == 'test_strategy'
        assert config.default_signal_type == SignalType.BREAKOUT
        assert config.minimum_strength_threshold == 75.0
        assert config.minimum_confidence_threshold == 0.7
        assert config.risk_reward_minimum == 2.5

    def test_default_config_values(self):
        """Test default configuration values."""
        config = StrategySignalConfig(
            strategy_name='default_test',
            default_signal_type=SignalType.BREAKOUT
        )

        # Should have sensible defaults
        assert config.minimum_strength_threshold > 0
        assert config.minimum_confidence_threshold > 0
        assert config.risk_reward_minimum > 0

    def test_config_custom_params(self):
        """Test custom validation parameters."""
        custom_params = {'volatility_threshold': 0.25, 'volume_spike': 3.0}

        config = StrategySignalConfig(
            strategy_name='custom_test',
            default_signal_type=SignalType.MOMENTUM,
            custom_validation_params=custom_params
        )

        assert config.custom_validation_params == custom_params


class TestProductionStrategySpecificMethods:
    """Test strategy-specific methods that use signal validation."""

    @pytest.fixture
    def base_strategy_setup(self):
        """Base setup for strategy testing."""
        mock_manager = Mock()
        mock_manager.logger = logging.getLogger('test')

        mock_provider = Mock()
        mock_provider.logger = logging.getLogger('test')

        return mock_manager, mock_provider

    def test_swing_trading_opportunity_validation(self, base_strategy_setup):
        """Test swing trading opportunity validation logic."""
        mock_manager, mock_provider = base_strategy_setup

        config = {
            'watchlist': ['AAPL', 'GOOGL'],
            'max_positions': 3,
            'min_strength_score': 60.0
        }

        strategy = ProductionSwingTrading(mock_manager, mock_provider, config)

        # Test internal validation methods if they exist
        if hasattr(strategy, '_validate_swing_setup'):
            # Create mock swing setup data
            setup_data = {
                'symbol': 'AAPL',
                'signal_strength': 75.0,
                'volume_confirmation': True,
                'technical_confluence': 0.8
            }

            is_valid = strategy._validate_swing_setup(setup_data)
            assert isinstance(is_valid, bool)

    def test_momentum_signal_strength_calculation(self, base_strategy_setup):
        """Test momentum-specific signal strength calculation."""
        mock_manager, mock_provider = base_strategy_setup

        config = {
            'watchlist': ['SPY'],
            'min_momentum_strength': 70.0
        }

        strategy = ProductionMomentumWeeklies(mock_manager, mock_provider, config)

        # Test momentum calculation if method exists
        base_prices = [100, 102, 105, 108, 112, 115, 118, 122, 125, 128, 130, 133, 135, 138, 140, 143, 145, 148, 150, 153, 155, 158, 160, 163, 165]
        base_volumes = [1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000, 10500000, 11000000, 11500000, 12000000, 12500000, 13000000]
        
        momentum_data = pd.DataFrame({
            'Close': base_prices,  # Strong momentum
            'Volume': base_volumes
        })

        result = strategy.validate_signal(
            symbol='SPY',
            market_data=momentum_data,
            signal_type=SignalType.MOMENTUM
        )

        # Should detect strong momentum
        assert result.normalized_score > 0

    def test_leaps_trend_validation(self, base_strategy_setup):
        """Test LEAPS trend validation logic."""
        mock_manager, mock_provider = base_strategy_setup

        config = {
            'themes': ['AI'],
            'min_trend_score': 55.0
        }

        strategy = ProductionLEAPSTracker(mock_manager, mock_provider, config)

        # Test trend validation
        base_prices = [100, 102, 104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200]
        base_volumes = [1000000, 1100000, 1300000, 1600000, 2000000, 2200000, 2400000, 2600000, 2800000, 3000000, 3200000, 3400000, 3600000, 3800000, 4000000, 4200000, 4400000, 4600000, 4800000, 5000000, 5200000, 5400000, 5600000, 5800000, 6000000, 6200000, 6400000, 6600000, 6800000, 7000000, 7200000, 7400000, 7600000, 7800000, 8000000]
        
        trend_data = pd.DataFrame({
            'Close': base_prices,  # Consistent uptrend
            'Volume': base_volumes
        })

        result = strategy.validate_signal(
            symbol='AI_STOCK',
            market_data=trend_data,
            signal_type=SignalType.TREND
        )

        # Should detect trend
        assert result.normalized_score > 0
        assert result.signal_type == SignalType.TREND


if __name__ == '__main__':
    pytest.main([__file__, '-v'])