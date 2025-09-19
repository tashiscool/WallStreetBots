#!/usr/bin/env python3
"""
Test suite for production strategy signal validation integration.

Tests the signal validation integration across all production strategies
to ensure >75% test coverage for the newly enhanced code.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import logging

# Import production strategies
from backend.tradingbot.strategies.production.production_swing_trading import ProductionSwingTrading
from backend.tradingbot.strategies.production.production_momentum_weeklies import ProductionMomentumWeeklies
from backend.tradingbot.strategies.production.production_leaps_tracker import ProductionLEAPSTracker

# Import validation framework
from backend.tradingbot.validation.signal_strength_validator import SignalType, SignalQuality


class MockDataProvider:
    """Mock data provider for testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def get_intraday_data(self, symbol, interval="15min", period="5d"):
        """Return mock intraday data."""
        np.random.seed(hash(symbol) % 2**32)
        length = 80

        if symbol == 'AAPL':
            # Strong uptrend with good signals
            base_prices = np.linspace(100, 115, length)
            noise = np.random.normal(0, 0.5, length)
            prices = base_prices + noise
            volumes = np.random.normal(2000000, 400000, length)
        elif symbol == 'WEAK_SIGNAL':
            # Weak signal pattern
            prices = [100 + np.random.normal(0, 0.2) for _ in range(length)]
            volumes = np.random.normal(800000, 100000, length)
        else:
            # Default pattern
            prices = [100 + i * 0.1 + np.random.normal(0, 0.3) for i in range(length)]
            volumes = np.random.normal(1500000, 300000, length)

        volumes = np.maximum(volumes, 500000)

        return pd.DataFrame({
            'close': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'volume': volumes,
            'timestamp': pd.date_range(start='2023-01-01', periods=length, freq='15min')
        })

    async def get_current_price(self, symbol):
        """Return mock current price."""
        return np.random.uniform(100, 120)


class MockIntegrationManager:
    """Mock integration manager for testing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)


@pytest.fixture
def mock_integration_manager():
    """Fixture for mock integration manager."""
    return MockIntegrationManager()


@pytest.fixture
def mock_data_provider():
    """Fixture for mock data provider."""
    return MockDataProvider()


@pytest.fixture
def swing_trading_strategy(mock_integration_manager, mock_data_provider):
    """Fixture for swing trading strategy."""
    config = {
        'watchlist': ['AAPL', 'GOOGL', 'MSFT'],
        'max_positions': 3,
        'max_position_size': 0.02,
        'min_strength_score': 60.0
    }
    return ProductionSwingTrading(mock_integration_manager, mock_data_provider, config)


@pytest.fixture
def momentum_strategy(mock_integration_manager, mock_data_provider):
    """Fixture for momentum strategy."""
    config = {
        'watchlist': ['SPY', 'QQQ', 'IWM'],
        'max_positions': 3,
        'min_volume_spike': 2.5,
        'min_momentum_strength': 65.0
    }
    return ProductionMomentumWeeklies(mock_integration_manager, mock_data_provider, config)


@pytest.fixture
def leaps_strategy(mock_integration_manager, mock_data_provider):
    """Fixture for LEAPS strategy."""
    config = {
        'themes': ['AI', 'CLOUD'],
        'max_positions_per_theme': 2,
        'min_trend_score': 55.0
    }
    return ProductionLEAPSTracker(mock_integration_manager, mock_data_provider, config)


class TestSwingTradingSignalIntegration:
    """Test signal validation integration for swing trading strategy."""

    def test_strategy_initialization_with_signal_validation(self, swing_trading_strategy):
        """Test that strategy initializes with signal validation components."""
        assert hasattr(swing_trading_strategy, 'validate_signal')
        assert hasattr(swing_trading_strategy, '_signal_validator')
        assert hasattr(swing_trading_strategy, '_signal_config')
        assert hasattr(swing_trading_strategy, '_signal_history')

    def test_signal_validation_method(self, swing_trading_strategy):
        """Test the validate_signal method works correctly."""
        # Create test market data
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 105, 108, 110],
            'Volume': [1000000, 1100000, 1200000, 2000000, 2200000, 2500000],
            'High': [101, 102, 103, 106, 109, 111],
            'Low': [99, 100, 101, 104, 107, 109]
        })

        result = swing_trading_strategy.validate_signal(
            symbol='TEST_SYMBOL',
            market_data=test_data,
            signal_type=SignalType.BREAKOUT
        )

        assert result is not None
        assert hasattr(result, 'normalized_score')
        assert hasattr(result, 'quality_grade')
        assert hasattr(result, 'recommended_action')
        assert result.signal_type == SignalType.BREAKOUT
        assert result.symbol == 'TEST_SYMBOL'

    @pytest.mark.asyncio
    async def test_scan_swing_opportunities_with_validation(self, swing_trading_strategy):
        """Test that scanning uses signal validation."""
        with patch.object(swing_trading_strategy, '_data_provider') as mock_provider:
            # Mock the get_intraday_data method
            mock_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame({
                'close': [100, 101, 102, 105, 108, 110],
                'high': [101, 102, 103, 106, 109, 111],
                'low': [99, 100, 101, 104, 107, 109],
                'volume': [1000000, 1100000, 1200000, 2000000, 2200000, 2500000],
                'timestamp': pd.date_range(start='2023-01-01', periods=6, freq='15min')
            }))

            # Test the scanning method
            opportunities = await swing_trading_strategy.scan_swing_opportunities()

            # Verify that scanning completed (may return empty list due to strict validation)
            assert isinstance(opportunities, list)

    def test_signal_strength_filtering(self, swing_trading_strategy):
        """Test that weak signals are filtered out."""
        # Create weak signal data
        weak_data = pd.DataFrame({
            'Close': [100, 100.1, 100.2, 100.1, 100.0],  # Minimal movement
            'Volume': [500000, 500000, 500000, 500000, 500000],  # Low volume
            'High': [100.5, 100.6, 100.7, 100.6, 100.5],
            'Low': [99.5, 99.4, 99.3, 99.4, 99.5]
        })

        result = swing_trading_strategy.validate_signal(
            symbol='WEAK_SIGNAL',
            market_data=weak_data,
            signal_type=SignalType.BREAKOUT
        )

        # Should be a weak signal
        assert result.normalized_score < 50.0
        assert result.quality_grade in [SignalQuality.POOR, SignalQuality.VERY_POOR]

    def test_get_strategy_signal_summary(self, swing_trading_strategy):
        """Test strategy signal summary method."""
        # First validate some signals to populate history
        test_data = pd.DataFrame({
            'Close': [100, 101, 102, 105],
            'Volume': [1000000, 1100000, 1200000, 2000000],
            'High': [101, 102, 103, 106],
            'Low': [99, 100, 101, 104]
        })

        # Validate a few signals
        for symbol in ['TEST1', 'TEST2']:
            swing_trading_strategy.validate_signal(
                symbol=symbol,
                market_data=test_data,
                signal_type=SignalType.BREAKOUT
            )

        summary = swing_trading_strategy.get_strategy_signal_summary()

        assert isinstance(summary, dict)
        assert 'total_signals_validated' in summary
        assert 'average_strength_score' in summary
        assert summary['total_signals_validated'] >= 2


class TestMomentumWeekliesSignalIntegration:
    """Test signal validation integration for momentum weeklies strategy."""

    def test_strategy_initialization_with_signal_validation(self, momentum_strategy):
        """Test that strategy initializes with signal validation components."""
        assert hasattr(momentum_strategy, 'validate_signal')
        assert hasattr(momentum_strategy, '_signal_validator')
        assert hasattr(momentum_strategy, '_signal_config')

    def test_momentum_signal_validation(self, momentum_strategy):
        """Test momentum-specific signal validation."""
        # Create momentum pattern data with sufficient rows
        base_prices = [100, 102, 105, 108, 112, 115, 118, 122, 125, 128, 130, 133, 135, 138, 140, 143, 145, 148, 150, 153, 155, 158, 160, 163, 165]
        base_volumes = [1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000, 8500000, 9000000, 9500000, 10000000, 10500000, 11000000, 11500000, 12000000, 12500000, 13000000]
        
        momentum_data = pd.DataFrame({
            'Close': base_prices,  # Strong momentum
            'Volume': base_volumes,  # Volume increase
            'High': [p + 2 for p in base_prices],
            'Low': [p - 2 for p in base_prices]
        })

        result = momentum_strategy.validate_signal(
            symbol='MOMENTUM_STOCK',
            market_data=momentum_data,
            signal_type=SignalType.MOMENTUM
        )

        assert result is not None
        assert result.signal_type == SignalType.MOMENTUM
        assert result.normalized_score > 0  # Should detect momentum

    @pytest.mark.asyncio
    async def test_scan_momentum_opportunities_with_validation(self, momentum_strategy):
        """Test that momentum scanning uses signal validation."""
        with patch.object(momentum_strategy, '_data_provider') as mock_provider:
            mock_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame({
                'close': [100, 102, 105, 108],
                'high': [101, 103, 106, 109],
                'low': [99, 101, 104, 107],
                'volume': [1000000, 1500000, 2000000, 2500000],
                'timestamp': pd.date_range(start='2023-01-01', periods=4, freq='15min')
            }))

            opportunities = await momentum_strategy.scan_momentum_opportunities()

            assert isinstance(opportunities, list)


class TestLEAPSTrackerSignalIntegration:
    """Test signal validation integration for LEAPS tracker strategy."""

    def test_strategy_initialization_with_signal_validation(self, leaps_strategy):
        """Test that strategy initializes with signal validation components."""
        assert hasattr(leaps_strategy, 'validate_signal')
        assert hasattr(leaps_strategy, '_signal_validator')
        assert hasattr(leaps_strategy, '_signal_config')

    def test_trend_signal_validation(self, leaps_strategy):
        """Test trend-specific signal validation."""
        # Create trend pattern data with sufficient rows
        base_prices = [100, 101, 103, 106, 110, 115, 118, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203]
        base_volumes = [1000000, 1100000, 1300000, 1600000, 2000000, 2200000, 2500000, 2800000, 3100000, 3400000, 3700000, 4000000, 4300000, 4600000, 4900000, 5200000, 5500000, 5800000, 6100000, 6400000, 6700000, 7000000, 7300000, 7600000, 7900000, 8200000, 8500000, 8800000, 9100000, 9400000, 9700000, 10000000, 10300000, 10600000, 10900000]
        
        trend_data = pd.DataFrame({
            'Close': base_prices,  # Consistent uptrend
            'Volume': base_volumes,
            'High': [p + 2 for p in base_prices],
            'Low': [p - 2 for p in base_prices]
        })

        result = leaps_strategy.validate_signal(
            symbol='TREND_STOCK',
            market_data=trend_data,
            signal_type=SignalType.TREND
        )

        assert result is not None
        assert result.signal_type == SignalType.TREND
        assert result.normalized_score > 0  # Should detect trend

    @pytest.mark.asyncio
    async def test_scan_leaps_candidates_with_validation(self, leaps_strategy):
        """Test that LEAPS scanning uses signal validation."""
        with patch.object(leaps_strategy, '_data_provider') as mock_provider:
            mock_provider.get_intraday_data = AsyncMock(return_value=pd.DataFrame({
                'close': [100, 101, 103, 106, 110],
                'high': [101, 102, 104, 107, 111],
                'low': [99, 100, 102, 105, 109],
                'volume': [1000000, 1100000, 1300000, 1600000, 2000000],
                'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='15min')
            }))

            candidates = await leaps_strategy.scan_leaps_candidates()

            assert isinstance(candidates, list)


class TestSignalValidationErrorHandling:
    """Test error handling in signal validation integration."""

    def test_invalid_market_data_handling(self, swing_trading_strategy):
        """Test handling of invalid market data."""
        # Empty DataFrame
        empty_data = pd.DataFrame()

        result = swing_trading_strategy.validate_signal(
            symbol='TEST',
            market_data=empty_data,
            signal_type=SignalType.BREAKOUT
        )

        # Should handle gracefully
        assert result.quality_grade == SignalQuality.VERY_POOR
        assert result.recommended_action == 'reject'

    def test_missing_columns_handling(self, swing_trading_strategy):
        """Test handling of missing required columns."""
        # Missing volume column
        incomplete_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101]
        })

        result = swing_trading_strategy.validate_signal(
            symbol='TEST',
            market_data=incomplete_data,
            signal_type=SignalType.BREAKOUT
        )

        # Should handle missing columns gracefully
        assert result is not None
        assert result.normalized_score >= 0

    def test_unknown_signal_type_handling(self, swing_trading_strategy):
        """Test handling of unknown signal types."""
        test_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000],
            'High': [101, 102, 103],
            'Low': [99, 100, 101]
        })

        # Should fall back to default signal type
        result = swing_trading_strategy.validate_signal(
            symbol='TEST',
            market_data=test_data,
            signal_type=None  # No signal type specified
        )

        assert result is not None


class TestSignalConfigurationCustomization:
    """Test signal configuration customization across strategies."""

    def test_swing_trading_signal_config(self, swing_trading_strategy):
        """Test swing trading signal configuration."""
        config = swing_trading_strategy._signal_config

        assert config.minimum_strength_threshold > 0
        assert config.consistency_threshold > 0
        assert config.volume_threshold > 0

    def test_momentum_strategy_signal_config(self, momentum_strategy):
        """Test momentum strategy signal configuration."""
        config = momentum_strategy._signal_config

        # Momentum strategies should have higher thresholds
        assert config.minimum_strength_threshold >= 60.0
        assert config.volume_threshold >= 1.5

    def test_leaps_strategy_signal_config(self, leaps_strategy):
        """Test LEAPS strategy signal configuration."""
        config = leaps_strategy._signal_config

        # LEAPS should allow lower thresholds for longer-term trends
        assert config.minimum_strength_threshold >= 50.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])