"""Comprehensive tests for market regime analysis to achieve >85% coverage."""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.tradingbot.market_regime import (
    MarketRegimeFilter,
    MarketRegime,
    TechnicalIndicators,
    SignalGenerator,
    MarketSignal,
    TechnicalAnalysis,
    SignalType,
    create_sample_indicators
)


class TestMarketRegimeEnums:
    """Test market regime enumeration classes."""

    def test_market_regime_values(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.SIDEWAYS.value == "sideways"
        assert MarketRegime.UNDEFINED.value == "undefined"

    def test_signal_type_values(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.NO_SIGNAL.value == "no_signal"


class TestTechnicalIndicators:
    """Test TechnicalIndicators dataclass."""

    def test_technical_indicators_creation(self):
        """Test creation of TechnicalIndicators."""
        indicators = TechnicalIndicators(
            price=100.0,
            ema_20=98.0,
            ema_50=95.0,
            ema_200=90.0,
            rsi_14=45.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=101.0,
            low_24h=99.0
        )

        assert indicators.price == 100.0
        assert indicators.ema_20 == 98.0
        assert indicators.distance_from_20ema == (100.0 - 98.0) / 100.0
        assert indicators.distance_from_50ema == (100.0 - 95.0) / 100.0

    def test_technical_indicators_derived_calculations(self):
        """Test derived indicator calculations."""
        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=190.0,
            ema_50=180.0,
            ema_200=170.0,
            rsi_14=60.0,
            atr_14=4.0,
            volume=2000000,
            high_24h=202.0,
            low_24h=198.0
        )

        expected_distance_20ema = (200.0 - 190.0) / 200.0
        expected_distance_50ema = (200.0 - 180.0) / 200.0

        assert abs(indicators.distance_from_20ema - expected_distance_20ema) < 1e-6
        assert abs(indicators.distance_from_50ema - expected_distance_50ema) < 1e-6

    def test_technical_indicators_zero_price(self):
        """Test TechnicalIndicators with zero price."""
        indicators = TechnicalIndicators(
            price=0.0,
            ema_20=98.0,
            ema_50=95.0,
            ema_200=90.0,
            rsi_14=45.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=101.0,
            low_24h=99.0
        )

        assert indicators.distance_from_20ema == 0.0
        assert indicators.distance_from_50ema == 0.0


class TestTechnicalAnalysis:
    """Test TechnicalAnalysis calculation methods."""

    def test_calculate_ema_basic(self):
        """Test basic EMA calculation."""
        prices = [100, 102, 104, 103, 105, 107, 106, 108]
        period = 5

        ema = TechnicalAnalysis.calculate_ema(prices, period)

        assert len(ema) == len(prices)
        assert ema[0] == prices[0]  # First value should be the first price
        assert ema[-1] > ema[0]  # Should trend upward with rising prices

    def test_calculate_ema_insufficient_data(self):
        """Test EMA with insufficient data."""
        prices = [100, 102]
        period = 5

        ema = TechnicalAnalysis.calculate_ema(prices, period)

        assert len(ema) == len(prices)
        assert all(np.isnan(val) for val in ema)

    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create trending up prices
        prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 112, 111, 113, 115, 114, 116]

        rsi = TechnicalAnalysis.calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)
        assert rsi[-1] > 50  # Should be above 50 for uptrend
        assert rsi[-1] <= 100  # Should not exceed 100

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = [100, 102, 104]

        rsi = TechnicalAnalysis.calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)
        assert all(np.isnan(val) for val in rsi[:14])

    def test_calculate_atr_basic(self):
        """Test basic ATR calculation."""
        highs = [102, 104, 106, 105, 107, 109, 108, 110, 112, 114, 113, 115, 117, 116, 118]
        lows = [98, 100, 102, 101, 103, 105, 104, 106, 108, 110, 109, 111, 113, 112, 114]
        closes = [100, 102, 104, 103, 105, 107, 106, 108, 110, 112, 111, 113, 115, 114, 116]

        atr = TechnicalAnalysis.calculate_atr(highs, lows, closes, period=14)

        assert len(atr) == len(closes)
        assert atr[-1] > 0  # ATR should be positive

    def test_calculate_slope(self):
        """Test slope calculation."""
        # Upward trending values
        values = [100, 102, 104, 106, 108]

        slope = TechnicalAnalysis.calculate_slope(values, period=5)

        assert slope > 0  # Should be positive for uptrend

    def test_calculate_slope_insufficient_data(self):
        """Test slope with insufficient data."""
        values = [100, 102]

        slope = TechnicalAnalysis.calculate_slope(values, period=5)

        assert slope == 0.0


class TestMarketRegimeFilter:
    """Test MarketRegimeFilter class."""

    def test_regime_filter_initialization(self):
        """Test MarketRegimeFilter initialization."""
        filter_obj = MarketRegimeFilter()

        assert hasattr(filter_obj, 'ta')
        assert hasattr(filter_obj, 'min_trend_slope')
        assert filter_obj.min_trend_slope == 0.001

    def test_determine_regime_bull(self):
        """Test bull regime detection."""
        filter_obj = MarketRegimeFilter()

        # Bull market conditions: price > 50ema, 50ema > 200ema, positive 20ema slope
        indicators = TechnicalIndicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=55.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=212.0,
            low_24h=208.0,
            ema_20_slope=0.002,  # Positive slope
            ema_50_slope=0.001
        )

        regime = filter_obj.determine_regime(indicators)
        assert regime == MarketRegime.BULL

    def test_determine_regime_bear(self):
        """Test bear regime detection."""
        filter_obj = MarketRegimeFilter()

        # Bear market conditions: price < 50ema, 50ema < 200ema, negative 20ema slope
        indicators = TechnicalIndicators(
            price=190.0,
            ema_20=192.0,
            ema_50=195.0,
            ema_200=200.0,
            rsi_14=35.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=194.0,
            low_24h=188.0,
            ema_20_slope=-0.002,  # Negative slope
            ema_50_slope=-0.001
        )

        regime = filter_obj.determine_regime(indicators)
        assert regime == MarketRegime.BEAR

    def test_determine_regime_sideways(self):
        """Test sideways regime detection."""
        filter_obj = MarketRegimeFilter()

        # Sideways conditions: flat 20ema slope
        indicators = TechnicalIndicators(
            price=200.0,
            ema_20=199.0,
            ema_50=198.0,
            ema_200=197.0,
            rsi_14=50.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=201.0,
            low_24h=199.0,
            ema_20_slope=0.0005,  # Very small slope
            ema_50_slope=0.0002
        )

        regime = filter_obj.determine_regime(indicators)
        assert regime == MarketRegime.SIDEWAYS

    def test_detect_pullback_setup(self):
        """Test pullback setup detection."""
        filter_obj = MarketRegimeFilter()

        # Previous indicators (higher price)
        prev_indicators = TechnicalIndicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=60.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=212.0,
            low_24h=208.0
        )

        # Current indicators (pullback to 20ema)
        current_indicators = TechnicalIndicators(
            price=208.5,  # Near 20ema, price decline from previous
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=42.0,  # RSI in pullback range (35-50)
            atr_14=4.0,
            volume=1000000,
            high_24h=210.0,
            low_24h=207.0  # Low touched 20ema
        )

        is_pullback = filter_obj.detect_pullback_setup(current_indicators, prev_indicators)
        assert is_pullback

    def test_detect_reversal_trigger(self):
        """Test reversal trigger detection."""
        filter_obj = MarketRegimeFilter()

        # Previous indicators
        prev_indicators = TechnicalIndicators(
            price=208.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=42.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=210.0,
            low_24h=207.0
        )

        # Current indicators (reversal trigger)
        current_indicators = TechnicalIndicators(
            price=211.0,  # Above 20ema and previous high
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=48.0,
            atr_14=4.0,
            volume=1500000,  # Volume expansion (1.5x)
            high_24h=211.0,
            low_24h=209.0
        )

        is_reversal = filter_obj.detect_reversal_trigger(current_indicators, prev_indicators)
        assert is_reversal


class TestSignalGenerator:
    """Test SignalGenerator class."""

    def test_signal_generator_initialization(self):
        """Test SignalGenerator initialization."""
        generator = SignalGenerator()

        assert hasattr(generator, 'regime_filter')
        assert isinstance(generator.regime_filter, MarketRegimeFilter)

    def test_generate_signal_bull_setup_with_trigger(self):
        """Test signal generation for bull setup with reversal trigger."""
        generator = SignalGenerator()

        # Previous indicators
        prev_indicators = TechnicalIndicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=60.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=212.0,
            low_24h=208.0,
            ema_20_slope=0.002
        )

        # Current indicators (pullback + reversal)
        current_indicators = TechnicalIndicators(
            price=211.0,  # Above 20ema and previous high
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=45.0,  # In pullback range
            atr_14=4.0,
            volume=1500000,  # Volume expansion
            high_24h=211.0,
            low_24h=207.0,
            ema_20_slope=0.002
        )

        signal = generator.generate_signal(current_indicators, prev_indicators)

        # Should generate buy signal
        assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
        assert signal.regime == MarketRegime.BULL
        assert signal.confidence > 0

    def test_generate_signal_earnings_risk(self):
        """Test signal generation with earnings risk."""
        generator = SignalGenerator()

        # Bull market setup but with earnings risk
        current_indicators = TechnicalIndicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=55.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=212.0,
            low_24h=208.0,
            ema_20_slope=0.002
        )

        prev_indicators = current_indicators

        signal = generator.generate_signal(
            current_indicators,
            prev_indicators,
            earnings_risk=True
        )

        assert signal.signal_type == SignalType.HOLD
        assert "Earnings risk detected" in signal.reasoning[0]

    def test_generate_signal_bear_regime(self):
        """Test signal generation in bear regime."""
        generator = SignalGenerator()

        # Bear market indicators
        current_indicators = TechnicalIndicators(
            price=190.0,
            ema_20=192.0,
            ema_50=195.0,
            ema_200=200.0,
            rsi_14=35.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=194.0,
            low_24h=188.0,
            ema_20_slope=-0.002
        )

        prev_indicators = current_indicators

        signal = generator.generate_signal(current_indicators, prev_indicators)

        assert signal.signal_type == SignalType.NO_SIGNAL
        assert signal.regime == MarketRegime.BEAR

    def test_calculate_signal_confidence(self):
        """Test signal confidence calculation."""
        generator = SignalGenerator()

        # Strong bull setup
        current = TechnicalIndicators(
            price=220.0,
            ema_20=218.0,
            ema_50=215.0,
            ema_200=210.0,
            rsi_14=60.0,
            atr_14=4.0,
            volume=2000000,  # High volume
            high_24h=220.0,
            low_24h=217.0,
            ema_20_slope=0.003  # Strong positive slope
        )

        previous = TechnicalIndicators(
            price=215.0,
            ema_20=215.0,
            ema_50=212.0,
            ema_200=208.0,
            rsi_14=55.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=216.0,
            low_24h=214.0,
            ema_20_slope=0.002
        )

        confidence = generator._calculate_signal_confidence(current, previous)

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high confidence


class TestCreateSampleIndicators:
    """Test the create_sample_indicators helper function."""

    def test_create_sample_indicators_basic(self):
        """Test basic creation of sample indicators."""
        indicators = create_sample_indicators(
            price=200.0,
            ema_20=198.0,
            ema_50=195.0,
            ema_200=190.0,
            rsi=60.0
        )

        assert indicators.price == 200.0
        assert indicators.ema_20 == 198.0
        assert indicators.rsi_14 == 60.0
        assert indicators.atr_14 == 200.0 * 0.02  # 2% ATR
        assert indicators.volume == 1000000  # Default volume

    def test_create_sample_indicators_with_highs_lows(self):
        """Test creation with custom highs and lows."""
        indicators = create_sample_indicators(
            price=200.0,
            ema_20=198.0,
            ema_50=195.0,
            ema_200=190.0,
            rsi=60.0,
            high=205.0,
            low=195.0,
            volume=2000000
        )

        assert indicators.high_24h == 205.0
        assert indicators.low_24h == 195.0
        assert indicators.volume == 2000000


class TestMarketSignal:
    """Test MarketSignal dataclass."""

    def test_market_signal_creation(self):
        """Test MarketSignal creation."""
        signal = MarketSignal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            regime=MarketRegime.BULL,
            price_target=220.0,
            stop_loss=200.0
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.regime == MarketRegime.BULL
        assert signal.price_target == 220.0
        assert signal.stop_loss == 200.0
        assert isinstance(signal.timestamp, datetime)
        assert isinstance(signal.reasoning, list)

    def test_market_signal_default_values(self):
        """Test MarketSignal with default values."""
        signal = MarketSignal(
            signal_type=SignalType.HOLD,
            confidence=0.5,
            regime=MarketRegime.SIDEWAYS
        )

        assert signal.reasoning == []
        assert isinstance(signal.timestamp, datetime)


class TestIntegrationScenarios:
    """Test integration scenarios combining all components."""

    def test_complete_bull_market_scenario(self):
        """Test complete bull market analysis scenario."""
        generator = SignalGenerator()

        # Bull market progression
        scenarios = [
            # Initial bull setup
            (210.0, 208.0, 205.0, 200.0, 55.0, 0.002),
            # Pullback
            (207.0, 208.0, 205.0, 200.0, 42.0, 0.002),
            # Reversal trigger
            (211.0, 208.0, 205.0, 200.0, 48.0, 0.002),
        ]

        prev_indicators = None

        for price, ema20, ema50, ema200, rsi, slope in scenarios:
            current_indicators = TechnicalIndicators(
                price=price,
                ema_20=ema20,
                ema_50=ema50,
                ema_200=ema200,
                rsi_14=rsi,
                atr_14=price * 0.02,
                volume=1500000,
                high_24h=price * 1.01,
                low_24h=price * 0.99,
                ema_20_slope=slope
            )

            if prev_indicators:
                signal = generator.generate_signal(current_indicators, prev_indicators)
                assert signal.regime == MarketRegime.BULL
                assert signal.confidence >= 0

            prev_indicators = current_indicators

    def test_regime_transition_detection(self):
        """Test detection of regime transitions."""
        filter_obj = MarketRegimeFilter()

        # Bull to bear transition
        bull_indicators = TechnicalIndicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi_14=60.0,
            atr_14=4.0,
            volume=1000000,
            high_24h=212.0,
            low_24h=208.0,
            ema_20_slope=0.002
        )

        bear_indicators = TechnicalIndicators(
            price=190.0,
            ema_20=192.0,
            ema_50=195.0,
            ema_200=200.0,
            rsi_14=35.0,
            atr_14=6.0,
            volume=1500000,
            high_24h=194.0,
            low_24h=188.0,
            ema_20_slope=-0.002
        )

        bull_regime = filter_obj.determine_regime(bull_indicators)
        bear_regime = filter_obj.determine_regime(bear_indicators)

        assert bull_regime == MarketRegime.BULL
        assert bear_regime == MarketRegime.BEAR