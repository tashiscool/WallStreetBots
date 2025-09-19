#!/usr / bin / env python3
"""WSB Strategy: Enhanced Breakout Swing Trading
Fast profit - taking swing trades with same-day exit discipline
Based on WSB successful swing trading patterns with ≤30 day expiries.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Run: pip install -r wsb_requirements.txt")
    sys.exit(1)


@dataclass
class SwingSignal:
    ticker: str
    signal_time: datetime
    signal_type: str  # "breakout", "momentum", "reversal"
    entry_price: float
    breakout_level: float
    volume_confirmation: float  # Volume vs average
    strength_score: float  # 0 - 100 breakout strength
    target_strike: int
    target_expiry: str
    option_premium: float
    max_hold_hours: int  # WSB rule: same-day exits preferred
    profit_target_1: float  # 25% profit
    profit_target_2: float  # 50% profit
    profit_target_3: float  # 100% profit
    stop_loss: float
    risk_level: str


@dataclass
class ActiveSwingTrade:
    signal: SwingSignal
    entry_time: datetime
    entry_premium: float
    current_premium: float
    unrealized_pnl: float
    unrealized_pct: float
    hours_held: float
    hit_profit_target: int  # 0=none, 1=25%, 2=50%, 3 = 100%
    should_exit: bool
    exit_reason: str


class SwingTradingScanner:
    def __init__(self):
        # Phase 1 Critical Fixes - Risk Controls
        self.max_daily_loss = 0.02  # 2% of portfolio max daily loss
        self.position_stop_loss = 0.03  # 3% stop loss per position
        self.min_signal_strength = 70.0  # Minimum signal strength threshold
        self.max_position_size = 0.025  # 2.5% max position size
        self.daily_loss_tracker = 0.0  # Track daily losses
        self.consecutive_losses = 0  # Track consecutive losses
        self.cooling_off_period = 0  # Days to wait after losses

        # Enhanced logging for debugging
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Focus on liquid, high - beta names for swing trading
        self.swing_tickers = [
            # Mega caps with options liquidity
            "SPY",
            "QQQ",
            "IWM",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            # High - beta swing favorites
            "AMD",
            "NFLX",
            "CRM",
            "ADBE",
            "PYPL",
            "SQ",
            "ROKU",
            "ZM",
            "PLTR",
            "COIN",
            # Volatile sectors good for swings
            "XLF",
            "XLE",
            "XLK",
            "XBI",
            "ARKK",
            "TQQQ",
            "SOXL",
            "SPXL",
        ]

        self.active_trades: list[ActiveSwingTrade] = []

    def reset_daily_limits(self):
        """Reset daily tracking variables (call at start of each trading day)."""
        self.daily_loss_tracker = 0.0
        if self.cooling_off_period > 0:
            self.cooling_off_period -= 1
        self.logger.info(f"Daily limits reset. Cooling off days remaining: {self.cooling_off_period}")

    def record_trade_result(self, pnl_pct: float):
        """Record trade result for risk tracking."""
        if pnl_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 5:
                self.cooling_off_period = 3  # 3-day cooling off
                self.logger.warning(f"Cooling off period activated after {self.consecutive_losses} losses")
        else:
            self.consecutive_losses = 0
            self.logger.info(f"Profitable trade recorded: {pnl_pct:+.1f}%")

    def get_risk_metrics(self) -> dict:
        """Get current risk metrics for monitoring."""
        return {
            'daily_loss_tracker': self.daily_loss_tracker,
            'consecutive_losses': self.consecutive_losses,
            'cooling_off_period': self.cooling_off_period,
            'active_positions': len(self.active_trades),
            'max_daily_loss': self.max_daily_loss,
            'position_stop_loss': self.position_stop_loss,
            'min_signal_strength': self.min_signal_strength
        }

    def detect_breakout(self, ticker: str) -> tuple[bool, float, float]:
        """Detect breakout above resistance with volume confirmation."""
        try:
            stock = yf.Ticker(ticker)

            # Get intraday data for breakout detection
            data = stock.history(period="5d", interval="15m")
            if len(data) < 50:
                return False, 0.0, 0.0

            prices = data["Close"].values
            volumes = data["Volume"].values
            data["High"].values

            current_price = prices[-1]
            current_volume = volumes[-5:].mean()  # Recent 5 periods
            avg_volume = volumes[:-10].mean()  # Historical average

            # Calculate resistance levels (pivot highs)
            resistance_levels = []
            for i in range(20, len(prices) - 5):
                if prices[i] == max(prices[i - 3 : i + 4]) and prices[
                    i
                ] > np.percentile(prices[:i], 80):
                    resistance_levels.append(prices[i])

            if not resistance_levels:
                return False, 0.0, 0.0

            # Find nearest resistance level
            key_resistance = (
                max(resistance_levels[-3:])
                if len(resistance_levels) >= 3
                else max(resistance_levels)
            )

            # Breakout criteria:
            # 1. Price breaks above key resistance by  > 0.5%
            # 2. Volume is  > 2x average
            # 3. Strong momentum in last few bars

            volume_multiple = current_volume / avg_volume if avg_volume > 0 else 0
            breakout_strength = (current_price - key_resistance) / key_resistance

            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]  # Last 5 bars

            # Calculate strength score first
            strength_score = min(
                100,
                (breakout_strength * 100 + volume_multiple * 10 + recent_momentum * 50),
            )

            # CRITICAL FIX: More stringent breakout criteria
            is_breakout = (
                current_price > key_resistance * 1.008  # 0.8% above resistance (more conservative)
                and volume_multiple >= 2.5  # 2.5x volume (higher threshold)
                and breakout_strength > 0.005  # Stronger breakout required
                and recent_momentum > 0.008  # Higher momentum threshold
                and strength_score >= self.min_signal_strength  # Signal strength filter
            )

            # Log signal quality for analysis
            self.logger.info(f"Breakout analysis {ticker}: price={current_price:.2f}, "
                           f"resistance={key_resistance:.2f}, volume_mult={volume_multiple:.1f}, "
                           f"strength={strength_score:.1f}, is_breakout={is_breakout}")

            return is_breakout, key_resistance, strength_score

        except Exception:
            return False, 0.0, 0.0

    def detect_momentum_continuation(self, ticker: str) -> tuple[bool, float]:
        """Detect strong momentum continuation patterns."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="2d", interval="5m")

            if len(data) < 30:
                return False, 0.0

            prices = data["Close"].values
            volumes = data["Volume"].values

            # Look for accelerating momentum
            short_ma = np.mean(prices[-5:])  # 25min MA
            medium_ma = np.mean(prices[-10:])  # 50min MA
            long_ma = np.mean(prices[-20:])  # 100min MA

            # Momentum strength
            if short_ma > medium_ma > long_ma:
                momentum_strength = (short_ma / long_ma - 1) * 100

                # Volume confirmation
                recent_vol = volumes[-10:].mean()
                earlier_vol = volumes[-30:-10].mean()
                vol_increase = recent_vol / earlier_vol if earlier_vol > 0 else 1

                # CRITICAL FIX: Higher momentum criteria
                if (momentum_strength > 1.5 and vol_increase > 1.5 and
                    momentum_strength >= self.min_signal_strength):
                    self.logger.info(f"Momentum signal {ticker}: strength={momentum_strength:.1f}, "
                                   f"vol_increase={vol_increase:.1f}")
                    return True, momentum_strength

            return False, 0.0

        except Exception:
            return False, 0.0

    def detect_reversal_setup(self, ticker: str) -> tuple[bool, str, float]:
        """Detect oversold bounce setups."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="3d", interval="15m")

            if len(data) < 40:
                return False, "insufficient_data", 0.0

            prices = data["Close"].values
            lows = data["Low"].values
            volumes = data["Volume"].values

            current_price = prices[-1]

            # Look for bounce from oversold levels
            recent_low = min(lows[-20:])  # 20 - period low
            bounce_strength = (current_price - recent_low) / recent_low

            # Volume spike on bounce
            current_vol = volumes[-3:].mean()
            avg_vol = volumes[:-10].mean()
            vol_spike = current_vol / avg_vol if avg_vol > 0 else 1

            # RSI - like oversold condition (simplified)
            up_moves = sum(
                1
                for i in range(len(prices) - 10, len(prices) - 1)
                if prices[i + 1] > prices[i]
            )
            down_moves = 10 - up_moves

            # CRITICAL FIX: Stricter reversal criteria
            reversal_score = bounce_strength * 100
            if (
                bounce_strength > 0.020  # 2.0% bounce from low (higher threshold)
                and vol_spike > 2.5  # Higher volume spike
                and down_moves >= 8  # More oversold
                and reversal_score >= self.min_signal_strength  # Signal strength filter
            ):
                self.logger.info(f"Reversal signal {ticker}: bounce={bounce_strength:.3f}, "
                               f"vol_spike={vol_spike:.1f}, score={reversal_score:.1f}")
                return True, "oversold_bounce", reversal_score

            return False, "no_setup", 0.0

        except Exception:
            return False, "error", 0.0

    def get_optimal_expiry(self, max_days: int = 30) -> str:
        """Get optimal expiry (WSB rule: ≤30 days for swing trades)."""
        today = date.today()

        # Prefer weekly expirations for faster theta decay management
        target_days = min(max_days, 21)  # Max 3 weeks

        # Find next Friday
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:  # If today is Friday
            days_to_friday = 7

        # If too far out, use closer weekly
        if days_to_friday > target_days:
            days_to_friday -= 7

        if days_to_friday <= 0:
            days_to_friday = 7

        expiry_date = today + timedelta(days=days_to_friday)
        return expiry_date.strftime("%Y-%m-%d")

    def calculate_option_targets(
        self, current_price: float, strike: int, premium: float
    ) -> tuple[float, float, float, float]:
        """Calculate profit targets and stop loss for swing trade."""
        # WSB swing trading targets: fast profit - taking
        profit_25 = premium * 1.25  # 25% profit - take some off
        profit_50 = premium * 1.50  # 50% profit - take more off
        profit_100 = premium * 2.00  # 100% profit - close position

        stop_loss = premium * 0.70  # 30% stop loss - cut losses fast

        return profit_25, profit_50, profit_100, stop_loss

    def estimate_swing_premium(self, ticker: str, strike: int, expiry: str) -> float:
        """Estimate option premium for swing trade."""
        try:
            stock = yf.Ticker(ticker)

            # Try actual options chain first
            try:
                chain = stock.option_chain(expiry)
                if not chain.calls.empty:
                    calls = chain.calls
                    closest = calls.iloc[(calls["strike"] - strike).abs().argsort()[:1]]
                    if not closest.empty:
                        bid = closest["bid"].iloc[0]
                        ask = closest["ask"].iloc[0]
                        if bid > 0 and ask > 0:
                            return (bid + ask) / 2
            except Exception:
                pass

            # Fallback estimate
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            days_to_exp = (
                datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()
            ).days

            # Swing trade premium estimate (higher IV assumption)
            time_premium = max(0.5, current_price * 0.08 * (days_to_exp / 21))

            if strike > current_price:  # OTM
                otm_discount = max(
                    0.2, 1 - (strike - current_price) / current_price * 5
                )
                return time_premium * otm_discount
            else:  # ITM
                intrinsic = current_price - strike
                return intrinsic + time_premium * 0.5

        except Exception:
            return 2.0  # Conservative fallback

    def check_risk_limits(self) -> bool:
        """CRITICAL FIX: Check if we can take new positions based on risk limits."""
        # Check daily loss limit
        if self.daily_loss_tracker >= self.max_daily_loss:
            self.logger.warning(f"Daily loss limit reached: {self.daily_loss_tracker:.2%}")
            return False

        # Check cooling off period after consecutive losses
        if self.consecutive_losses >= 3 and self.cooling_off_period > 0:
            self.logger.warning(f"In cooling off period: {self.consecutive_losses} consecutive losses")
            return False

        return True

    def scan_swing_opportunities(self) -> list[SwingSignal]:
        """Scan for swing trading opportunities with enhanced risk controls."""
        # CRITICAL FIX: Check risk limits before scanning
        if not self.check_risk_limits():
            self.logger.warning("Risk limits exceeded - no new signals generated")
            return []

        signals = []
        expiry = self.get_optimal_expiry()

        print(f"🎯 Scanning swing opportunities targeting {expiry}...")
        self.logger.info(f"Signal scan started with min_strength={self.min_signal_strength}")

        for ticker in self.swing_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="1m")

                if hist.empty:
                    continue

                current_price = hist["Close"].iloc[-1]

                # Check for different signal types
                signals_found = []

                # 1. Breakout detection
                is_breakout, resistance_level, breakout_strength = self.detect_breakout(
                    ticker
                )
                if is_breakout:
                    signals_found.append(
                        ("breakout", breakout_strength, resistance_level)
                    )

                # 2. Momentum continuation
                is_momentum, momentum_strength = self.detect_momentum_continuation(
                    ticker
                )
                if is_momentum:
                    signals_found.append(("momentum", momentum_strength, current_price))

                # 3. Reversal setup
                is_reversal, _reversal_type, reversal_strength = (
                    self.detect_reversal_setup(ticker)
                )
                if is_reversal:
                    signals_found.append(("reversal", reversal_strength, current_price))

                # Process signals with enhanced filtering
                for signal_type, strength, ref_level in signals_found:
                    # CRITICAL FIX: Apply signal strength filter
                    if strength < self.min_signal_strength:
                        self.logger.info(f"Signal rejected - low strength: {ticker} {signal_type} {strength:.1f}")
                        continue

                    # Target strike selection based on signal type (MORE CONSERVATIVE)
                    if signal_type == "breakout":
                        strike_multiplier = 1.015  # 1.5% OTM (more conservative)
                        max_hold_hours = 4  # Shorter hold times
                    elif signal_type == "momentum":
                        strike_multiplier = 1.01  # 1% OTM (more conservative)
                        max_hold_hours = 3  # Shorter hold times
                    else:  # reversal
                        strike_multiplier = 1.02  # 2% OTM (more conservative)
                        max_hold_hours = 6  # Shorter hold times

                    target_strike = round(current_price * strike_multiplier)
                    premium = self.estimate_swing_premium(ticker, target_strike, expiry)

                    # CRITICAL FIX: Higher minimum premium and size limits
                    if premium < 0.50:  # Higher minimum premium threshold
                        self.logger.info(f"Signal rejected - low premium: {ticker} premium={premium:.2f}")
                        continue

                    # Position size check
                    position_value = premium * 100  # Assume 1 contract = $100 multiplier
                    if position_value > self.max_position_size * 10000:  # Assume $10k portfolio
                        self.logger.warning(f"Position too large: {ticker} value=${position_value:.0f}")
                        continue

                    # Calculate targets
                    profit_25, profit_50, profit_100, stop_loss = (
                        self.calculate_option_targets(
                            current_price, target_strike, premium
                        )
                    )

                    # CRITICAL FIX: More stringent risk assessment
                    if strength > 85:
                        risk_level = "low"
                    elif strength > 75:
                        risk_level = "medium"
                    else:
                        risk_level = "high"
                        # Skip high risk signals in current validation phase
                        self.logger.info(f"High risk signal skipped: {ticker} {signal_type} {strength:.1f}")
                        continue

                    signal = SwingSignal(
                        ticker=ticker,
                        signal_time=datetime.now(),
                        signal_type=signal_type,
                        entry_price=current_price,
                        breakout_level=ref_level,
                        volume_confirmation=2.0,  # Simplified
                        strength_score=strength,
                        target_strike=target_strike,
                        target_expiry=expiry,
                        option_premium=premium,
                        max_hold_hours=max_hold_hours,
                        profit_target_1=profit_25,
                        profit_target_2=profit_50,
                        profit_target_3=profit_100,
                        stop_loss=stop_loss,
                        risk_level=risk_level,
                    )

                    signals.append(signal)
                    print(
                        f"  🎯 {ticker} {signal_type.upper()} - Strength: {strength:.0f}"
                    )

            except Exception as e:
                print(f"  ❌ {ticker}: Error - {e}")
                continue

        # Sort by strength score
        signals.sort(key=lambda x: x.strength_score, reverse=True)
        return signals

    def monitor_active_trades(self) -> list[str]:
        """Monitor active swing trades and generate exit signals."""
        exit_recommendations = []

        if not self.active_trades:
            return exit_recommendations

        print("📊 Monitoring active swing trades...")

        for trade in self.active_trades:
            try:
                # Update trade status
                hours_held = (datetime.now() - trade.entry_time).total_seconds() / 3600
                trade.hours_held = hours_held

                # Get current option price (simplified)
                stock = yf.Ticker(trade.signal.ticker)
                current_stock_price = stock.history(period="1d", interval="1m")[
                    "Close"
                ].iloc[-1]

                # CRITICAL FIX: Enhanced premium estimation
                if current_stock_price >= trade.signal.target_strike:
                    # ITM - estimate intrinsic + remaining time value
                    intrinsic = current_stock_price - trade.signal.target_strike
                    time_value = trade.entry_premium * max(0.1, 1 - hours_held / 24)
                    current_premium = intrinsic + time_value
                else:
                    # OTM - faster time decay (more realistic)
                    decay_factor = max(
                        0.05, 1 - (hours_held / trade.signal.max_hold_hours) ** 1.5
                    )
                    current_premium = trade.entry_premium * decay_factor

                trade.current_premium = current_premium
                trade.unrealized_pnl = current_premium - trade.entry_premium
                trade.unrealized_pct = (
                    trade.unrealized_pnl / trade.entry_premium
                ) * 100

                # Log trade monitoring
                self.logger.info(f"Monitoring {trade.signal.ticker}: Stock=${current_stock_price:.2f}, "
                               f"Premium=${current_premium:.2f}, P&L={trade.unrealized_pct:+.1f}%")

                # CRITICAL FIX: Enhanced exit conditions with strict risk management
                exit_reason = None

                # 1. STOP LOSS - Priority #1 (more aggressive)
                if trade.unrealized_pct <= -self.position_stop_loss * 100:  # 3% stop loss
                    exit_reason = f"STOP LOSS: {trade.unrealized_pct:+.1f}% - EXIT IMMEDIATELY"
                    self.daily_loss_tracker += abs(trade.unrealized_pnl) / 10000  # Track losses
                    self.consecutive_losses += 1
                    self.logger.warning(f"Stop loss triggered: {trade.signal.ticker}")

                # 2. Profit targets (take profits earlier)
                elif trade.unrealized_pct >= 50:  # 50% profit - close immediately
                    trade.hit_profit_target = 3
                    exit_reason = "50%+ profit target - CLOSE POSITION"
                    self.consecutive_losses = 0  # Reset on profit
                    self.logger.info(f"Major profit target hit: {trade.signal.ticker}")
                elif trade.unrealized_pct >= 25:  # 25% profit - close position
                    trade.hit_profit_target = 2
                    exit_reason = "25% profit target - CLOSE POSITION"
                    self.consecutive_losses = 0  # Reset on profit
                elif trade.unrealized_pct >= 15:  # 15% profit - take some profits
                    if trade.hit_profit_target < 1:
                        trade.hit_profit_target = 1
                        exit_reason = "15% profit - consider partial exit"

                # 3. Time-based exits (MUCH more aggressive)
                elif hours_held >= trade.signal.max_hold_hours * 0.8:  # 80% of max time
                    exit_reason = f"Time limit approaching ({hours_held:.1f}h) - EXIT"

                # 4. Theta decay protection - exit OTM positions losing value fast
                elif (current_stock_price < trade.signal.target_strike * 0.98 and
                      hours_held >= 2 and trade.unrealized_pct <= -10):
                    exit_reason = "Theta decay protection - OTM position deteriorating"

                # 5. End of day exit rule (earlier exit)
                elif datetime.now().hour >= 14:  # 2 PM ET
                    exit_reason = "End of day approach - close all positions"

                if exit_reason:
                    trade.should_exit = True
                    trade.exit_reason = exit_reason
                    exit_recommendations.append(
                        f"{trade.signal.ticker} {trade.signal.target_strike}C: {exit_reason} "
                        f"(P & L: {trade.unrealized_pct:+.1f}%)"
                    )

            except Exception as e:
                self.logger.error(f"Error monitoring {trade.signal.ticker}: {e}")
                print(f"Error monitoring {trade.signal.ticker}: {e}")
                # If we can't monitor, close position for safety
                trade.should_exit = True
                trade.exit_reason = "Monitoring error - close for safety"
                exit_recommendations.append(f"EXIT {trade.signal.ticker} - Monitoring Error")

        # Remove completed trades
        self.active_trades = [t for t in self.active_trades if not t.should_exit]

        self.logger.info(f"Trade monitoring complete. {len(exit_recommendations)} exits recommended")
        return exit_recommendations

    def format_signals(self, signals: list[SwingSignal]) -> str:
        """Format swing signals for display."""
        if not signals:
            return "🎯 No swing trading signals found at this time."

        output = f"\n🎯 SWING TRADING SIGNALS ({len(signals)} found)\n"
        output += " = " * 70 + "\n"

        for i, signal in enumerate(signals[:10], 1):  # Top 10
            hours_display = f"{signal.max_hold_hours}h max"

            output += f"\n{i}. {signal.ticker} - {signal.signal_type.upper()} 🎯\n"
            output += f"   Entry: ${signal.entry_price:.2f} | Strike: ${signal.target_strike}\n"
            output += f"   Premium: ${signal.option_premium:.2f} | Expiry: {signal.target_expiry}\n"
            output += f"   Strength: {signal.strength_score:.0f}/100 | Risk: {signal.risk_level.upper()}\n"
            output += f"   Max Hold: {hours_display} | Profit Targets: 25%/50%/100%\n"
            output += f"   Stop Loss: ${signal.stop_loss:.2f} (-30%)\n"

        output += "\n" + " = " * 70
        output += "\n🎯 WSB SWING TRADING RULES: \n"
        output += "• FAST profit - taking: 25% → 50% → 100%\n"
        output += "• MAX 30 days expiry (prefer weeklies)\n"
        output += "• Same-day exits preferred for momentum + n"
        output += "• 30% stop loss - cut losses FAST + n"
        output += "• Don't hold overnight unless strong setup + n"
        output += "• Volume confirmation is MANDATORY + n"

        output += "\n⚠️  SWING TRADING WARNINGS: \n"
        output += "• Options decay fast - time is your enemy + n"
        output += "• Don't chase breakouts that already moved  > 5%\n"
        output += "• Avoid earnings weeks (IV crush risk)\n"
        output += "• Use 1 - 2% position sizing max + n"

        return output


def main():
    parser = argparse.ArgumentParser(description="WSB Enhanced Swing Trading Scanner")
    parser.add_argument(
        "command", choices=["scan", "monitor", "continuous"], help="Command to execute"
    )
    parser.add_argument(
        "--max - expiry - days",
        type=int,
        default=21,
        help="Maximum days to expiry (WSB rule: ≤30)",
    )
    parser.add_argument(
        "--output", choices=["json", "text"], default="text", help="Output format"
    )
    parser.add_argument(
        "--min - strength",
        type=float,
        default=60.0,
        help="Minimum signal strength score",
    )

    args = parser.parse_args()

    scanner = SwingTradingScanner()

    if args.command == "scan":
        signals = scanner.scan_swing_opportunities()

        # Filter by minimum strength
        signals = [s for s in signals if s.strength_score >= args.min_strength]

        if args.output == "json":
            print(json.dumps([asdict(s) for s in signals], indent=2, default=str))
        else:
            print(scanner.format_signals(signals))

    elif args.command == "monitor":
        exit_signals = scanner.monitor_active_trades()

        if exit_signals:
            print("\n🚨 SWING TRADE EXIT SIGNALS: ")
            for signal in exit_signals:
                print(f"  • {signal}")
        else:
            print("📊 No active trades to monitor")

    elif args.command == "continuous":
        print("🔄 Starting continuous swing scanning (Ctrl + C to stop)...")
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%H: %M:%S')} - Scanning...")

                signals = scanner.scan_swing_opportunities()
                signals = [s for s in signals if s.strength_score >= args.min_strength]

                if signals:
                    print(scanner.format_signals(signals))

                # Also monitor any active trades
                exit_signals = scanner.monitor_active_trades()
                if exit_signals:
                    print("\n🚨 EXIT SIGNALS: ")
                    for signal in exit_signals:
                        print(f"  • {signal}")

                print("\nNext scan in 5 minutes...")
                time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n🛑 Scanning stopped by user")


if __name__ == "__main__":
    main()
