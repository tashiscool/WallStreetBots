"""
Signal Strength Validation Framework - Usage Demo
================================================

Demonstrates how to use the comprehensive signal validation framework
with existing trading strategies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
from datetime import datetime
import json

from tradingbot.validation import (
    SignalStrengthValidator,
    SignalType,
    signal_integrator,
    StrategySignalMixin,
    StrategySignalConfig
)


def create_demo_market_data(symbol: str, pattern: str = "breakout") -> pd.DataFrame:
    """Create demo market data for different patterns."""
    np.random.seed(hash(symbol) % 2**32)  # Deterministic but varied
    length = 40

    if pattern == "breakout":
        # Consolidation followed by breakout
        prices = []
        volumes = []

        # Consolidation phase (30 periods)
        base_price = 100.0
        for i in range(30):
            price = base_price + np.random.normal(0, 1)
            prices.append(max(98, min(102, price)))  # Range-bound
            volumes.append(np.random.normal(1000000, 100000))

        # Breakout phase (10 periods)
        for i in range(10):
            price = 103 + i * 0.8 + np.random.normal(0, 0.5)
            prices.append(price)
            volumes.append(np.random.normal(2500000, 300000))  # Volume spike

    elif pattern == "momentum":
        # Strong momentum pattern
        base_price = 100.0
        prices = [base_price]
        volumes = []

        for i in range(length - 1):
            momentum = 0.015 + np.random.normal(0, 0.005)  # Strong upward momentum
            new_price = prices[-1] * (1 + momentum)
            prices.append(new_price)

            # Increasing volume with momentum
            volume = 1000000 + i * 50000 + np.random.normal(0, 100000)
            volumes.append(max(500000, volume))

        volumes.append(volumes[-1])  # Match length

    elif pattern == "trend":
        # Steady long-term trend
        base_price = 100.0
        prices = [base_price]
        volumes = []

        for i in range(length - 1):
            trend = 0.008 + np.random.normal(0, 0.003)  # Steady trend
            new_price = prices[-1] * (1 + trend)
            prices.append(new_price)
            volumes.append(np.random.normal(1200000, 200000))

        volumes.append(volumes[-1])

    else:  # choppy
        # Choppy, directionless market
        prices = [100 + np.random.normal(0, 3) for _ in range(length)]
        volumes = [np.random.normal(800000, 150000) for _ in range(length)]

    # Ensure positive values
    volumes = [max(100000, v) for v in volumes]

    return pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=length, freq='D'),
        'Open': [p * 0.999 for p in prices],
        'High': [p * 1.015 for p in prices],
        'Low': [p * 0.985 for p in prices],
        'Close': prices,
        'Volume': volumes
    })


class DemoSwingTradingStrategy(StrategySignalMixin):
    """Demo swing trading strategy with signal validation."""

    def __init__(self):
        super().__init__()
        self.name = "Demo Swing Trading"
        self.signals_found = []

    def scan_for_signals(self, symbols: list) -> list:
        """Scan for swing trading signals with validation."""
        validated_signals = []

        for symbol in symbols:
            try:
                # Get market data (in real implementation, this would fetch from API)
                market_data = create_demo_market_data(symbol, "breakout")

                # Validate signal
                validation_result = self.validate_signal(
                    symbol=symbol,
                    market_data=market_data,
                    signal_type=SignalType.BREAKOUT,
                    signal_params={
                        'risk_reward_ratio': 2.5,
                        'max_hold_hours': 6
                    }
                )

                # Only include signals that pass validation
                if validation_result.recommended_action == "trade":
                    signal = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'signal_type': 'breakout',
                        'entry_price': market_data['Close'].iloc[-1],
                        'validation_score': validation_result.normalized_score,
                        'confidence': validation_result.confidence_level,
                        'quality_grade': validation_result.quality_grade.value,
                        'suggested_position_size': validation_result.suggested_position_size,
                        'validation_notes': validation_result.validation_notes
                    }
                    validated_signals.append(signal)

                print(f"✅ {symbol}: Score={validation_result.normalized_score:.1f}, "
                      f"Action={validation_result.recommended_action}, "
                      f"Quality={validation_result.quality_grade.value}")

            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")

        return validated_signals


def demo_basic_validation():
    """Demonstrate basic signal validation."""
    print("=" * 60)
    print("🎯 BASIC SIGNAL VALIDATION DEMO")
    print("=" * 60)

    validator = SignalStrengthValidator()

    # Test different market patterns
    patterns = {
        'Strong Breakout': create_demo_market_data("STRONG", "breakout"),
        'Momentum Stock': create_demo_market_data("MOMENTUM", "momentum"),
        'Trending Stock': create_demo_market_data("TREND", "trend"),
        'Choppy Stock': create_demo_market_data("CHOPPY", "choppy")
    }

    for name, data in patterns.items():
        result = validator.validate_signal(
            signal_type=SignalType.BREAKOUT,
            symbol=name.replace(' ', '_').upper(),
            market_data=data
        )

        print(f"\n📊 {name}:")
        print(f"   Score: {result.normalized_score:.1f}/100")
        print(f"   Quality: {result.quality_grade.value}")
        print(f"   Recommendation: {result.recommended_action}")
        print(f"   Confidence: {result.confidence_level:.2f}")
        print(f"   Position Size: {result.suggested_position_size:.1%}")


def demo_strategy_integration():
    """Demonstrate strategy integration."""
    print("\n" + "=" * 60)
    print("🔧 STRATEGY INTEGRATION DEMO")
    print("=" * 60)

    # Create strategy instance
    strategy = DemoSwingTradingStrategy()

    # Enhance with signal validation
    signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

    # Scan for signals
    test_symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NFLX', 'META', 'NVDA']
    print(f"\n🔍 Scanning {len(test_symbols)} symbols for validated signals...")

    validated_signals = strategy.scan_for_signals(test_symbols)

    print(f"\n📈 VALIDATED SIGNALS FOUND: {len(validated_signals)}")
    print("-" * 40)

    for signal in validated_signals:
        print(f"🎯 {signal['symbol']}:")
        print(f"   Entry Price: ${signal['entry_price']:.2f}")
        print(f"   Validation Score: {signal['validation_score']:.1f}")
        print(f"   Quality: {signal['quality_grade']}")
        print(f"   Position Size: {signal['suggested_position_size']:.1%}")
        print()


def demo_multiple_strategies():
    """Demonstrate multiple strategies with different signal types."""
    print("\n" + "=" * 60)
    print("🎛️  MULTIPLE STRATEGIES DEMO")
    print("=" * 60)

    strategies = {
        'Swing Trading': ('swing_trading', SignalType.BREAKOUT),
        'Momentum Weeklies': ('momentum_weeklies', SignalType.MOMENTUM),
        'LEAPS Tracker': ('leaps_tracker', SignalType.TREND)
    }

    test_symbol = "DEMO_STOCK"
    results = {}

    for strategy_name, (config_name, signal_type) in strategies.items():
        print(f"\n📊 Testing {strategy_name}...")

        # Create appropriate market data
        if signal_type == SignalType.BREAKOUT:
            data = create_demo_market_data(test_symbol, "breakout")
        elif signal_type == SignalType.MOMENTUM:
            data = create_demo_market_data(test_symbol, "momentum")
        else:  # TREND
            data = create_demo_market_data(test_symbol, "trend")

        # Create and enhance strategy
        class TempStrategy(StrategySignalMixin):
            def __init__(self, name): self.name = name

        strategy = TempStrategy(strategy_name)
        signal_integrator.enhance_strategy_with_validation(strategy, config_name)

        # Validate signal
        result = strategy.validate_signal(test_symbol, data, signal_type)
        results[strategy_name] = result

        print(f"   Score: {result.normalized_score:.1f}")
        print(f"   Action: {result.recommended_action}")
        print(f"   Quality: {result.quality_grade.value}")

    # Compare results
    print("\n📊 STRATEGY COMPARISON:")
    print("-" * 40)
    for name, result in results.items():
        print(f"{name:20s}: {result.normalized_score:5.1f} ({result.quality_grade.value})")


def demo_validation_summary():
    """Demonstrate validation summary and reporting."""
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY DEMO")
    print("=" * 60)

    strategy = DemoSwingTradingStrategy()
    signal_integrator.enhance_strategy_with_validation(strategy, "swing_trading")

    # Generate multiple signals for summary
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    patterns = ['breakout', 'momentum', 'trend', 'choppy', 'breakout']

    print("🔄 Generating signals for summary...")
    for symbol, pattern in zip(symbols, patterns):
        data = create_demo_market_data(symbol, pattern)
        strategy.validate_signal(symbol, data)

    # Get summary
    summary = strategy.get_strategy_signal_summary()

    print("\n📈 STRATEGY VALIDATION SUMMARY:")
    print("-" * 40)
    print(f"Strategy: {summary['strategy_name']}")
    print(f"Total Signals: {summary['total_signals_validated']}")
    print(f"Average Score: {summary['average_strength_score']:.1f}")
    print(f"Trading Recommendations: {summary['signals_recommended_for_trading']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")

    print("\n📊 Quality Distribution:")
    for quality, count in summary['quality_distribution'].items():
        if count > 0:
            print(f"   {quality.title()}: {count}")


def demo_export_functionality():
    """Demonstrate export functionality."""
    print("\n" + "=" * 60)
    print("💾 EXPORT FUNCTIONALITY DEMO")
    print("=" * 60)

    validator = SignalStrengthValidator()

    # Generate some validation results
    test_data = create_demo_market_data("EXPORT_TEST", "breakout")
    for i in range(5):
        validator.validate_signal(
            SignalType.BREAKOUT,
            f"TEST_{i}",
            test_data
        )

    # Get summary
    summary = validator.get_validation_summary()
    print("📊 Validation Summary:")
    print(json.dumps(summary, indent=2, default=str))

    # Export to file
    export_file = "validation_history_demo.json"
    validator.export_validation_history(export_file)
    print(f"\n💾 Validation history exported to: {export_file}")

    # Check file size
    if os.path.exists(export_file):
        size = os.path.getsize(export_file)
        print(f"📁 File size: {size} bytes")


def main():
    """Run all demos."""
    print("🚀 SIGNAL STRENGTH VALIDATION FRAMEWORK DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive signal validation system.")
    print("The framework provides standardized signal quality assessment")
    print("across all trading strategies with statistical rigor.")
    print("=" * 60)

    try:
        demo_basic_validation()
        demo_strategy_integration()
        demo_multiple_strategies()
        demo_validation_summary()
        demo_export_functionality()

        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("The signal validation framework is ready for production use.")
        print("Key benefits:")
        print("• Standardized signal quality scoring (0-100)")
        print("• Comprehensive validation across multiple criteria")
        print("• Strategy-specific signal calculators")
        print("• Automated filtering of poor-quality signals")
        print("• Performance tracking and reporting")
        print("• Easy integration with existing strategies")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()