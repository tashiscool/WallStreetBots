#!/usr / bin / env python3
"""WSB Strategy #2: Momentum Weeklies Scanner
Detects intraday reversals and news momentum for weekly options plays.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

# Constants for momentum strategy
DAYS_THRESHOLD_EARLY_WEEK = 2
VOLATILITY_MULTIPLE_THRESHOLD = 3.0
MIN_DATA_POINTS_SHORT = 20
MIN_DATA_POINTS_RECENT = 24
BOUNCE_PERCENTAGE_THRESHOLD = 0.015
MIN_DATA_POINTS_MEDIUM = 50
VOL_MULTIPLE_HIGH_RISK = 5
VOL_MULTIPLE_MEDIUM_RISK = 4
STRONG_MOMENTUM_THRESHOLD = 0.03
MEDIUM_MOMENTUM_THRESHOLD = 0.02

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    sys.exit(1)


@dataclass
class MomentumSignal:
    ticker: str
    signal_time: datetime
    current_price: float
    reversal_type: str  # "bullish_reversal", "news_momentum", "breakout"
    volume_spike: float  # Multiple of average volume
    price_momentum: float  # % change triggering signal
    weekly_expiry: str
    target_strike: int
    premium_estimate: float
    risk_level: str  # "low", "medium", "high"
    exit_target: float  # Price target for quick exit
    stop_loss: float


class MomentumWeekliesScanner:
    def __init__(self):
        self.mega_caps = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "CRM",
            "ADBE",
            "ORCL",
            "INTC",
            "AMD",
            "QCOM",
            "TXN",
            "AVGO",
        ]

    def get_next_weekly_expiry(self) -> str:
        """Find next weekly expiry (typically Friday)."""
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7  # Friday = 4
        if days_until_friday == 0:  # If today is Friday
            days_until_friday = 7
        elif days_until_friday <= DAYS_THRESHOLD_EARLY_WEEK:  # If Mon / Tue, use this Friday
            pass
        else:  # Wed / Thu, use next Friday
            days_until_friday += 7

        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime("%Y-%m-%d")

    def detect_volume_spike(self, ticker: str) -> tuple[bool, float]:
        """Detect unusual volume spike (3x+ average)."""
        try:
            stock = yf.Ticker(ticker)
            # Get intraday data
            data = stock.history(period="5d", interval="5m")
            if data.empty:
                return False, 0.0

            # Current volume (last 5 bars average)
            current_vol = data["Volume"].tail(5).mean()

            # Average volume over past 5 days (same time of day)
            avg_vol = data["Volume"].mean()

            if avg_vol == 0:
                return False, 0.0

            vol_multiple = current_vol / avg_vol
            return vol_multiple >= VOLATILITY_MULTIPLE_THRESHOLD, vol_multiple

        except Exception:
            return False, 0.0

    def detect_reversal_pattern(self, ticker: str) -> tuple[bool, str, float]:
        """Detect bullish reversal patterns."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="2d", interval="5m")
            if len(data) < MIN_DATA_POINTS_SHORT:
                return False, "insufficient_data", 0.0

            # Get recent price action
            prices = data["Close"].values
            data["Volume"].values

            current_price = prices[-1]

            # Look for V - shaped reversal in last 2 hours (24 5 - min bars)
            recent_prices = prices[-24:]
            if len(recent_prices) < MIN_DATA_POINTS_RECENT:
                return False, "insufficient_recent_data", 0.0

            # Find the low point in recent action
            low_idx = np.argmin(recent_prices)
            low_price = recent_prices[low_idx]

            # Check if we've bounced significantly from low
            bounce_pct = (current_price - low_price) / low_price

            # Reversal criteria:
            # 1. At least 1.5% bounce from recent low
            # 2. Low occurred in first half of window (early decline, late recovery)
            # 3. Current price  >  recent average
            if (
                bounce_pct >= BOUNCE_PERCENTAGE_THRESHOLD
                and low_idx < len(recent_prices) * 0.6
                and current_price > np.mean(recent_prices[-12:])
            ):
                return True, "bullish_reversal", bounce_pct

            return False, "no_pattern", 0.0

        except Exception:
            return False, "error", 0.0

    def detect_breakout_momentum(self, ticker: str) -> tuple[bool, float]:
        """Detect breakout above resistance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="5d", interval="15m")
            if len(data) < MIN_DATA_POINTS_MEDIUM:
                return False, 0.0

            prices = data["Close"].values
            volumes = data["Volume"].values
            current_price = prices[-1]

            # Calculate resistance level (highest high in past 3 days)
            resistance = np.max(prices[-100:-10])  # Exclude very recent to avoid false signals

            # Check if breaking above resistance with volume
            if current_price > resistance * 1.002:  # 0.2% above resistance
                current_vol = volumes[-5:].mean()  # Recent volume
                avg_vol = volumes[:-5].mean()  # Historical volume

                if current_vol > avg_vol * 1.5:  # Volume confirmation
                    breakout_strength = (current_price - resistance) / resistance
                    return True, breakout_strength

            return False, 0.0

        except Exception:
            return False, 0.0

    def get_weekly_option_premium(self, ticker: str, strike: int, expiry: str) -> float:
        """Estimate weekly option premium."""
        try:
            stock = yf.Ticker(ticker)

            # Try to get actual options chain
            try:
                chain = stock.option_chain(expiry)
                if not chain.calls.empty:
                    # Find closest strike
                    calls = chain.calls.copy()
                    calls["strike_diff"] = abs(calls["strike"] - strike)
                    closest = calls.loc[calls["strike_diff"].idxmin()]

                    # Use mid price
                    return (closest["bid"] + closest["ask"]) / 2.0
            except Exception:
                pass

            # Fallback: estimate using simplified Black - Scholes
            current_price = stock.history(period="1d")["Close"].iloc[-1]

            # Rough weekly premium estimate for OTM calls
            days_to_exp = (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days
            time_value = max(0.5, 5.0 - days_to_exp * 0.3)  # Rough time value

            if strike > current_price:  # OTM
                otm_discount = max(0.1, 1.0 - (strike - current_price) / current_price * 10)
                return time_value * otm_discount
            else:  # ITM
                intrinsic = current_price - strike
                return intrinsic + time_value * 0.3

        except Exception:
            return 2.0  # Default estimate

    def scan_momentum_signals(self) -> list[MomentumSignal]:
        """Scan for momentum weekly opportunities."""
        signals = []
        weekly_expiry = self.get_next_weekly_expiry()

        print(f"Scanning for momentum weeklies targeting {weekly_expiry}...")

        for ticker in self.mega_caps:
            try:
                # Check volume spike
                has_volume_spike, vol_multiple = self.detect_volume_spike(ticker)

                # Check for reversal pattern
                has_reversal, _pattern_type, bounce_pct = self.detect_reversal_pattern(ticker)

                # Check for breakout
                has_breakout, breakout_strength = self.detect_breakout_momentum(ticker)

                # Need at least volume spike+(reversal OR breakout)
                if has_volume_spike and (has_reversal or has_breakout):
                    # Get current price
                    stock = yf.Ticker(ticker)
                    current_data = stock.history(period="1d", interval="1m")
                    if current_data.empty:
                        continue

                    current_price = current_data["Close"].iloc[-1]

                    # Determine signal type and strength
                    if has_breakout:
                        signal_type = "breakout"
                        momentum = breakout_strength
                        risk = "medium" if vol_multiple < VOL_MULTIPLE_HIGH_RISK else "high"
                    else:
                        signal_type = "bullish_reversal"
                        momentum = bounce_pct
                        risk = "low" if vol_multiple < VOL_MULTIPLE_MEDIUM_RISK else "medium"

                    # Target strike: 2 - 5% OTM depending on momentum strength
                    if momentum > STRONG_MOMENTUM_THRESHOLD:  # Strong momentum - closer to money
                        otm_pct = 0.02
                    elif momentum > MEDIUM_MOMENTUM_THRESHOLD:
                        otm_pct = 0.03
                    else:
                        otm_pct = 0.05

                    target_strike = round(current_price * (1 + otm_pct))

                    # Get premium estimate
                    premium = self.get_weekly_option_premium(ticker, target_strike, weekly_expiry)

                    # Exit targets
                    exit_target = current_price * (1 + otm_pct + 0.02)  # 2% above strike
                    stop_loss = current_price * 0.985  # 1.5% stop

                    signal = MomentumSignal(
                        ticker=ticker,
                        signal_time=datetime.now(),
                        current_price=current_price,
                        reversal_type=signal_type,
                        volume_spike=vol_multiple,
                        price_momentum=momentum,
                        weekly_expiry=weekly_expiry,
                        target_strike=target_strike,
                        premium_estimate=premium,
                        risk_level=risk,
                        exit_target=exit_target,
                        stop_loss=stop_loss,
                    )

                    signals.append(signal)
                    print(f"📈 MOMENTUM SIGNAL: {ticker}")
                    print(f"   Type: {signal_type}")
                    print(f"   Volume: {vol_multiple:.1f}x average")
                    print(f"   Momentum: {momentum:.2%}")
                    print(f"   Strike: ${target_strike}")
                    print(f"   Premium: ${premium:.2f}")

            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue

        return signals

    def format_signals_output(self, signals: list[MomentumSignal]) -> str:
        """Format signals for display."""
        if not signals:
            return "🔍 No momentum weekly signals found at this time."

        output = f"\n🚀 MOMENTUM WEEKLIES SIGNALS ({len(signals)} found)\n"
        output += " = " * 60 + "\n"

        for i, signal in enumerate(signals, 1):
            output += f"\n{i}. {signal.ticker} - {signal.reversal_type.upper()}\n"
            output += f"   Current: ${signal.current_price:.2f}\n"
            output += f"   Volume Spike: {signal.volume_spike:.1f}x + n"
            output += f"   Momentum: {signal.price_momentum:.2%}\n"
            output += f"   Target Strike: ${signal.target_strike} ({signal.weekly_expiry})\n"
            output += f"   Premium Est: ${signal.premium_estimate:.2f}\n"
            output += f"   Exit Target: ${signal.exit_target:.2f}\n"
            output += f"   Stop Loss: ${signal.stop_loss:.2f}\n"
            output += f"   Risk Level: {signal.risk_level.upper()}\n"

        output += "\n⚠️  WEEKLY OPTIONS WARNING: \n"
        output += "• Extreme time decay - exit same / next day + n"
        output += "• High IV crush risk if momentum stalls + n"
        output += "• Use small position sizes (1 - 3% account risk)\n"

        return output


def main():
    parser = argparse.ArgumentParser(description="WSB Momentum Weeklies Scanner")
    parser.add_argument("--output", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument(
        "--min - volume-spike", type=float, default=3.0, help="Minimum volume spike multiple"
    )
    parser.add_argument(
        "--continuous", action="store_true", help="Run continuous scanning (5 - minute intervals)"
    )

    args = parser.parse_args()

    scanner = MomentumWeekliesScanner()

    if args.continuous:
        print("🔄 Starting continuous momentum scanning (Ctrl + C to stop)...")
        try:
            while True:
                signals = scanner.scan_momentum_signals()

                if signals:
                    if args.output == "json":
                        print(json.dumps([asdict(s) for s in signals], indent=2, default=str))
                    else:
                        print(scanner.format_signals_output(signals))
                else:
                    print(
                        f"⏰ {datetime.now().strftime('%H: %M:%S')} - No signals found, continuing..."
                    )

                time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n🛑 Scanning stopped by user")
    else:
        # Single scan
        signals = scanner.scan_momentum_signals()

        if args.output == "json":
            print(json.dumps([asdict(s) for s in signals], indent=2, default=str))
        else:
            print(scanner.format_signals_output(signals))


if __name__ == "__main__":
    main()
