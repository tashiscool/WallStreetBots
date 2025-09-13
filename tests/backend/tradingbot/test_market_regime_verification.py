#!/usr / bin/env python3
"""
Behavioral Verification Tests for Market Regime Detection System
Tests mathematical accuracy, signal logic, and strategy behavior validation
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.tradingbot.market_regime import (  # noqa: E402
    MarketRegime, SignalType, TechnicalIndicators, MarketSignal,
    TechnicalAnalysis, MarketRegimeFilter, SignalGenerator,
    create_sample_indicators
)


class TestTechnicalAnalysisAccuracy(unittest.TestCase): 
    """Test mathematical accuracy of technical analysis calculations"""
    
    def setUp(self): 
        self.ta=TechnicalAnalysis()
    
    def test_ema_calculation_accuracy(self): 
        """Test EMA calculation against known mathematical values"""
        # Known test case with verified EMA values
        prices=[100, 102, 101, 103, 104, 102, 105, 107, 106, 108]
        period=5
        
        ema_values=self.ta.calculate_ema(prices, period)
        
        # EMA formula: EMA_today=(Price_today * (2 / (period + 1))) + (EMA_yesterday * (1 - (2 / (period + 1))))
        alpha=2.0 / (period + 1)
        
        # Verify alpha calculation
        self.assertAlmostEqual(alpha, 0.3333333333333333, places=10)
        
        # First EMA value should be first price
        self.assertEqual(ema_values[0], prices[0])
        
        # Verify subsequent calculations
        expected_ema_1=alpha * prices[1] + (1 - alpha) * ema_values[0]
        self.assertAlmostEqual(ema_values[1], expected_ema_1, places=10)
        
        # EMA values should be reasonable (between min and max of prices)
        self.assertGreater(ema_values[-1], min(prices))
        self.assertLess(ema_values[-1], max(prices))
    
    def test_rsi_calculation_mathematical_accuracy(self): 
        """Test RSI calculation against known benchmark values"""
        # Known RSI test case
        prices=[44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 
                 46.08, 45.89, 46.03, 46.83, 47.69, 46.49, 46.26, 47.09, 47.37]
        
        rsi_values=self.ta.calculate_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        for rsi in rsi_values[14: ]: # Skip NaN values
            if not np.isnan(rsi): 
                self.assertGreaterEqual(rsi, 0)
                self.assertLessEqual(rsi, 100)
        
        # For this specific case, RSI should be around 70 - 80 (overbought territory)
        final_rsi=rsi_values[-1]
        self.assertGreater(final_rsi, 65)
        self.assertLess(final_rsi, 85)
    
    def test_atr_calculation_accuracy(self): 
        """Test ATR calculation mathematical correctness"""
        # Sample OHLC data
        highs=[45, 46, 47, 46, 45, 46, 47, 48, 47, 46]
        lows=[44, 45, 45, 44, 43, 44, 45, 46, 45, 44]
        closes=[44.5, 45.5, 46, 45, 44, 45, 46, 47, 46, 45]
        
        atr_values=self.ta.calculate_atr(highs, lows, closes, 5)
        
        # ATR should be positive and reasonable
        for atr in atr_values: 
            if not np.isnan(atr): 
                self.assertGreater(atr, 0)
                self.assertLess(atr, 10)  # Reasonable upper bound for this data
        
        # ATR should reflect average volatility
        final_atr=atr_values[-1]
        avg_range=sum(highs[i] - lows[i] for i in range(len(highs))) / len(highs)
        self.assertAlmostEqual(final_atr, avg_range, delta=avg_range * 0.5)
    
    def test_slope_calculation_mathematical_accuracy(self): 
        """Test linear regression slope calculation accuracy"""
        # Perfect upward trend: y=2x + 10
        values=[10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        slope=self.ta.calculate_slope(values, 5)
        
        # Should be exactly 2.0 for perfect linear trend
        self.assertAlmostEqual(slope, 2.0, places=10)
        
        # Perfect downward trend: y=-1x + 20
        down_values=[20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
        down_slope=self.ta.calculate_slope(down_values, 5)
        
        self.assertAlmostEqual(down_slope, -1.0, places=10)
        
        # Flat trend should have near - zero slope
        flat_values=[15, 15.1, 14.9, 15.05, 14.95, 15]
        flat_slope=self.ta.calculate_slope(flat_values, 5)
        self.assertAlmostEqual(flat_slope, 0.0, delta=0.1)


class TestMarketRegimeAccuracy(unittest.TestCase): 
    """Test market regime detection mathematical logic and accuracy"""
    
    def setUp(self): 
        self.regime_filter=MarketRegimeFilter()
    
    def test_bull_regime_detection_accuracy(self): 
        """Test bull regime detection with precise mathematical criteria"""
        # Perfect bull setup: Price > 50EMA > 200EMA, positive 20EMA slope
        indicators=TechnicalIndicators(
            price=110.0,
            ema_20=108.0,
            ema_50=105.0,
            ema_200=100.0,
            rsi_14=55.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=111.0,
            low_24h=109.0,
            ema_20_slope=0.002  # Positive slope above minimum
        )
        
        regime=self.regime_filter.determine_regime(indicators)
        self.assertEqual(regime, MarketRegime.BULL)
    
    def test_bear_regime_detection_accuracy(self): 
        """Test bear regime detection with precise mathematical criteria"""
        # Perfect bear setup: Price < 50EMA < 200EMA, negative 20EMA slope
        indicators=TechnicalIndicators(
            price=90.0,
            ema_20=92.0,
            ema_50=95.0,
            ema_200=100.0,
            rsi_14=35.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=91.0,
            low_24h=89.0,
            ema_20_slope=-0.002  # Negative slope below minimum
        )
        
        regime=self.regime_filter.determine_regime(indicators)
        self.assertEqual(regime, MarketRegime.BEAR)
    
    def test_sideways_regime_detection_accuracy(self): 
        """Test sideways regime detection with flat slope criteria"""
        indicators=TechnicalIndicators(
            price=102.0,
            ema_20=101.0,
            ema_50=100.0,
            ema_200=99.0,
            rsi_14=50.0,
            atr_14=1.5,
            volume=1000000,
            high_24h=103.0,
            low_24h=101.0,
            ema_20_slope=0.0005  # Flat slope within tolerance
        )
        
        regime=self.regime_filter.determine_regime(indicators)
        self.assertEqual(regime, MarketRegime.SIDEWAYS)
    
    def test_regime_boundary_conditions(self): 
        """Test regime detection at boundary conditions"""
        # Test exact boundary case: price=50EMA
        boundary_indicators=TechnicalIndicators(
            price=100.0,
            ema_20=101.0,
            ema_50=100.0,  # Exactly at price
            ema_200=99.0,
            rsi_14=50.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=101.0,
            low_24h=99.0,
            ema_20_slope=0.001  # Minimum positive slope
        )
        
        regime=self.regime_filter.determine_regime(boundary_indicators)
        # Should not be bull because price is not > 50EMA
        self.assertNotEqual(regime, MarketRegime.BULL)
    
    def test_pullback_setup_detection_accuracy(self): 
        """Test pullback setup detection mathematical logic"""
        # Previous day indicators (higher price)
        prev_indicators=TechnicalIndicators(
            price=105.0,
            ema_20=102.0,
            ema_50=100.0,
            ema_200=98.0,
            rsi_14=55.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=106.0,
            low_24h=104.0
        )
        
        # Current day - pullback to 20EMA with RSI in range  
        # Must decline at least 0.2% from previous day
        current_indicators=TechnicalIndicators(
            price=104.7,  # 0.29% decline from 105.0, above 50EMA but near 20EMA
            ema_20=102.0,
            ema_50=100.0,
            ema_200=98.0,
            rsi_14=42.0,  # In pullback range (35 - 50)
            atr_14=2.0,
            volume=1200000,
            high_24h=105.0,
            low_24h=101.8  # Touched 20EMA
        )
        
        is_pullback_setup=self.regime_filter.detect_pullback_setup(
            current_indicators, prev_indicators
        )
        self.assertTrue(is_pullback_setup)
    
    def test_reversal_trigger_detection_accuracy(self): 
        """Test reversal trigger detection mathematical precision"""
        prev_indicators=TechnicalIndicators(
            price=102.0,
            ema_20=102.0,
            ema_50=100.0,
            ema_200=98.0,
            rsi_14=42.0,
            atr_14=2.0,
            volume=1000000,
            high_24h=103.0,
            low_24h=101.0
        )
        
        # Current - recovery above 20EMA and previous high with volume
        current_indicators=TechnicalIndicators(
            price=103.5,  # Above 20EMA and previous high
            ema_20=102.0,
            ema_50=100.0,
            ema_200=98.0,
            rsi_14=48.0,
            atr_14=2.0,
            volume=1300000,  # 30% volume increase
            high_24h=103.5,
            low_24h=102.0
        )
        
        is_reversal_trigger=self.regime_filter.detect_reversal_trigger(
            current_indicators, prev_indicators
        )
        self.assertTrue(is_reversal_trigger)


class TestSignalGenerationAccuracy(unittest.TestCase): 
    """Test signal generation logic and mathematical confidence calculations"""
    
    def setUp(self): 
        self.signal_gen=SignalGenerator()
    
    def test_pullback_and_reversal_setup_detection(self): 
        """Test pullback setup and reversal trigger detection separately"""
        # Test pullback setup detection
        prev_high_price=create_sample_indicators(
            price=210.0,
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi=55,
            volume=1500000
        )
        
        # Pullback day - decline and near 20EMA with RSI in range
        pullback_day=create_sample_indicators(
            price=207.6,  # 0.2%+ decline and near 20EMA
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi=42,  # In pullback range 35 - 50
            volume=1800000,
            high=209.0,
            low=207.0  # Touched 20EMA
        )
        
        # Test pullback setup
        has_pullback=self.signal_gen.regime_filter.detect_pullback_setup(
            pullback_day, prev_high_price
        )
        self.assertTrue(has_pullback, "Should detect pullback setup")
        
        # Test reversal trigger
        reversal_day=create_sample_indicators(
            price=209.5,  # Above 20EMA
            ema_20=208.0,
            ema_50=205.0,
            ema_200=200.0,
            rsi=48,
            volume=2200000,  # Volume expansion
            high=210.0,  # Above previous high
            low=208.0
        )
        
        has_reversal=self.signal_gen.regime_filter.detect_reversal_trigger(
            reversal_day, pullback_day
        )
        self.assertTrue(has_reversal, "Should detect reversal trigger")
    
    def test_signal_generation_logic_flow(self): 
        """Test signal generation follows proper logical flow"""
        # Bull regime with no setup should give HOLD
        bull_indicators=create_sample_indicators(
            price=110.0, ema_20=108.0, ema_50=105.0, ema_200=100.0, rsi=55
        )
        bull_indicators.ema_20_slope=0.002
        
        signal=self.signal_gen.generate_signal(bull_indicators, bull_indicators)
        self.assertEqual(signal.regime, MarketRegime.BULL)
        self.assertIn(signal.signal_type, [SignalType.HOLD, SignalType.NO_SIGNAL])
        
        # Test with earnings risk should block signals
        signal_earnings=self.signal_gen.generate_signal(
            bull_indicators, bull_indicators, earnings_risk=True
        )
        self.assertEqual(signal_earnings.signal_type, SignalType.HOLD)
        self.assertEqual(signal_earnings.confidence, 0.0)
    
    def test_signal_confidence_calculation_accuracy(self): 
        """Test mathematical accuracy of confidence calculation"""
        current=create_sample_indicators(
            price=110.0,
            ema_20=109.0,
            ema_50=105.0,
            ema_200=100.0,
            rsi=55,
            volume=2000000
        )
        current.ema_20_slope=0.003  # Strong positive slope
        
        previous=create_sample_indicators(
            price=108.0,
            ema_20=108.0,
            ema_50=104.0,
            ema_200=100.0,
            rsi=52,
            volume=1000000
        )
        
        confidence=self.signal_gen._calculate_signal_confidence(current, previous)
        
        # Break down expected confidence calculation
        expected_confidence=0.0
        expected_confidence += 0.3  # Base bull regime
        expected_confidence += 0.2  # EMA alignment (20 > 50 > 200)
        expected_confidence += 0.2  # Strong positive slope (> 0.002)
        expected_confidence += 0.2  # Strong volume (2x expansion)
        expected_confidence += 0.1  # Strong price momentum (> 1%)
        
        self.assertAlmostEqual(confidence, min(expected_confidence, 1.0), places=2)
    
    def test_risk_filter_accuracy(self): 
        """Test risk filter mathematical logic"""
        indicators=create_sample_indicators(
            price=110.0, ema_20=109.0, ema_50=105.0, ema_200=100.0, rsi=55
        )
        indicators.ema_20_slope=0.002
        
        # Test earnings risk filter
        signal_earnings=self.signal_gen.generate_signal(
            indicators, indicators, earnings_risk=True
        )
        self.assertEqual(signal_earnings.signal_type, SignalType.HOLD)
        self.assertEqual(signal_earnings.confidence, 0.0)
        self.assertIn("Earnings risk", " ".join(signal_earnings.reasoning))
        
        # Test macro risk filter
        signal_macro=self.signal_gen.generate_signal(
            indicators, indicators, macro_risk=True
        )
        self.assertEqual(signal_macro.signal_type, SignalType.HOLD)
        self.assertEqual(signal_macro.confidence, 0.0)
        self.assertIn("Major macro event", " ".join(signal_macro.reasoning))
    
    def test_regime_filter_signal_blocking(self): 
        """Test that non - bull regimes properly block signals"""
        # Bear regime indicators
        bear_indicators=create_sample_indicators(
            price=95.0,
            ema_20=98.0,
            ema_50=102.0,
            ema_200=105.0,
            rsi=35
        )
        bear_indicators.ema_20_slope=-0.002
        
        signal=self.signal_gen.generate_signal(bear_indicators, bear_indicators)
        self.assertEqual(signal.signal_type, SignalType.NO_SIGNAL)
        self.assertEqual(signal.regime, MarketRegime.BEAR)
        self.assertIn("not bull market", " ".join(signal.reasoning))


class TestTechnicalIndicatorsAccuracy(unittest.TestCase): 
    """Test TechnicalIndicators dataclass mathematical calculations"""
    
    def test_derived_indicators_calculation_accuracy(self): 
        """Test automatic calculation of derived indicators"""
        indicators=TechnicalIndicators(
            price=110.0,
            ema_20=105.0,
            ema_50=100.0,
            ema_200=95.0,
            rsi_14=60.0,
            atr_14=2.5,
            volume=1500000,
            high_24h=111.0,
            low_24h=108.0
        )
        
        # Test distance calculations
        expected_distance_20=(110.0 - 105.0) / 110.0
        expected_distance_50=(110.0 - 100.0) / 110.0
        
        self.assertAlmostEqual(indicators.distance_from_20ema, expected_distance_20, places=10)
        self.assertAlmostEqual(indicators.distance_from_50ema, expected_distance_50, places=10)
    
    def test_zero_price_handling(self): 
        """Test handling of zero price in distance calculations"""
        indicators=TechnicalIndicators(
            price=0.0,  # Zero price edge case
            ema_20=105.0,
            ema_50=100.0,
            ema_200=95.0,
            rsi_14=60.0,
            atr_14=2.5,
            volume=1500000,
            high_24h=111.0,
            low_24h=108.0
        )
        
        # Should handle zero price gracefully
        self.assertEqual(indicators.distance_from_20ema, 0.0)
        self.assertEqual(indicators.distance_from_50ema, 0.0)


class TestMarketRegimeIntegration(unittest.TestCase): 
    """Test integration scenarios and edge cases"""
    
    def setUp(self): 
        self.signal_gen=SignalGenerator()
        self.regime_filter=MarketRegimeFilter()
    
    def test_complete_signal_generation_pipeline(self): 
        """Test complete signal generation from indicators to final signal"""
        # Create realistic market scenario
        prev_day=create_sample_indicators(
            price=207.50,  # GOOGL - like price
            ema_20=206.80,
            ema_50=204.20,
            ema_200=201.50,
            rsi=48,
            volume=1800000
        )
        
        current_day=create_sample_indicators(
            price=206.90,  # Slight pullback
            ema_20=206.80,
            ema_50=204.20,
            ema_200=201.50,
            rsi=44,
            volume=2200000,
            high=207.60,
            low=206.40
        )
        current_day.ema_20_slope=0.0015  # Positive trend
        
        # Generate signal
        signal=self.signal_gen.generate_signal(current_day, prev_day)
        
        # Validate complete signal object
        self.assertIsInstance(signal, MarketSignal)
        self.assertIsInstance(signal.signal_type, SignalType)
        self.assertIsInstance(signal.regime, MarketRegime)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
        self.assertIsInstance(signal.reasoning, list)
        self.assertIsInstance(signal.timestamp, datetime)
    
    def test_edge_case_handling(self): 
        """Test handling of edge cases and extreme values"""
        # Extreme RSI values
        extreme_indicators=create_sample_indicators(
            price=100.0,
            ema_20=99.0,
            ema_50=98.0,
            ema_200=95.0,
            rsi=0.0,  # Extreme oversold
            volume=0  # Zero volume
        )
        
        # Should handle gracefully without errors
        signal=self.signal_gen.generate_signal(extreme_indicators, extreme_indicators)
        self.assertIsInstance(signal, MarketSignal)
        
        # Extreme overbought
        overbought_indicators=create_sample_indicators(
            price=100.0,
            ema_20=99.0,
            ema_50=98.0,
            ema_200=95.0,
            rsi=100.0,  # Extreme overbought
            volume=999999999  # Very high volume
        )
        
        signal2=self.signal_gen.generate_signal(overbought_indicators, overbought_indicators)
        self.assertIsInstance(signal2, MarketSignal)


def run_market_regime_verification_tests(): 
    """Run all market regime verification tests"""
    print("=" * 70)
    print("MARKET REGIME DETECTION - BEHAVIORAL VERIFICATION TEST SUITE")
    print("=" * 70)
    
    # Create test suite
    test_suite=unittest.TestSuite()
    
    # Add test classes
    test_classes=[
        TestTechnicalAnalysisAccuracy,
        TestMarketRegimeAccuracy,
        TestSignalGenerationAccuracy,
        TestTechnicalIndicatorsAccuracy,
        TestMarketRegimeIntegration
    ]
    
    for test_class in test_classes: 
        tests=unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner=unittest.TextTestRunner(verbosity=2)
    result=runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("MARKET REGIME VERIFICATION TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate=((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
    print(f"SUCCESS RATE: {success_rate:.1f}%")
    
    if result.failures: 
        print(f"\nFAILURES ({len(result.failures)}): ")
        for test, traceback in result.failures: 
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors: 
        print(f"\nERRORS ({len(result.errors)}): ")
        for test, traceback in result.errors: 
            print(f"  - {test}")
    
    return result


if __name__== "__main__": run_market_regime_verification_tests()