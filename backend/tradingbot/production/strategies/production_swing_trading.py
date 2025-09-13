#!/usr / bin / env python3
"""Production Swing Trading Strategy
Fast profit - taking swing trades with same-day exit discipline.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from ...options.pricing_engine import BlackScholesEngine
from ...options.smart_selection import SmartOptionsSelector
from ...risk.real_time_risk_manager import RealTimeRiskManager
from ..core.production_integration import ProductionTradeSignal
from ..data.production_data_integration import ReliableDataProvider


@dataclass
class SwingSignal:
    """Production swing signal with enhanced metadata."""

    ticker: str
    signal_time: datetime
    signal_type: str  # "breakout", "momentum", "reversal"
    entry_price: float
    breakout_level: float
    volume_confirmation: float
    strength_score: float  # 0 - 100
    target_strike: float
    target_expiry: str
    option_premium: float
    max_hold_hours: int
    profit_target_1: float  # 25%
    profit_target_2: float  # 50%
    profit_target_3: float  # 100%
    stop_loss: float
    risk_level: str


class ProductionSwingTrading:
    """Production Swing Trading Strategy.

    Strategy Logic:
    1. Scans for breakouts, momentum continuation, and reversal setups
    2. Uses short - term options (≤30 DTE) with fast profit - taking
    3. Implements WSB - style same-day exit discipline
    4. Focuses on liquid, high - beta names for maximum movement
    5. Uses volume confirmation for all entries

    Risk Management:
    - Maximum 2% account risk per swing trade
    - Maximum 5 concurrent swing positions
    - 30% stop loss with immediate exits
    - Profit targets at 25%, 50%, 100% levels
    - Time-based exits: same-day preferred, 8 hours max
    - End - of - day liquidation for momentum trades
    """

    def __init__(
        self, integration_manager, data_provider: ReliableDataProvider, config: dict
    ):
        self.strategy_name = "swing_trading"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.options_selector = SmartOptionsSelector(data_provider)
        self.risk_manager = RealTimeRiskManager()
        self.bs_engine = BlackScholesEngine()

        # Strategy configuration
        self.swing_tickers = config.get(
            "watchlist",
            [
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
                # Volatile ETFs good for swings
                "XLF",
                "XLE",
                "XLK",
                "XBI",
                "ARKK",
                "TQQQ",
                "SOXL",
                "SPXL",
            ],
        )

        # Risk parameters
        self.max_positions = config.get("max_positions", 5)
        self.max_position_size = config.get("max_position_size", 0.02)  # 2% per swing
        self.max_expiry_days = config.get("max_expiry_days", 21)  # 3 weeks max
        self.min_strength_score = config.get("min_strength_score", 60.0)

        # Swing parameters
        self.min_volume_multiple = config.get("min_volume_multiple", 2.0)
        self.min_breakout_strength = config.get("min_breakout_strength", 0.002)  # 0.2%
        self.min_premium = config.get("min_premium", 0.25)

        # Exit criteria
        self.profit_targets = config.get("profit_targets", [25, 50, 100])  # % gains
        self.stop_loss_pct = config.get("stop_loss_pct", 30)  # 30% loss
        self.max_hold_hours = config.get("max_hold_hours", 8)
        self.end_of_day_exit_hour = config.get("end_of_day_exit_hour", 15)  # 3pm ET

        # Active positions tracking
        self.active_positions: list[dict[str, Any]] = []

        self.logger.info("ProductionSwingTrading strategy initialized")

    async def detect_breakout(self, ticker: str) -> tuple[bool, float, float]:
        """Detect breakout above resistance with volume confirmation."""
        try:
            # Get 5 days of 15 - minute data for breakout analysis
            data = await self.data_provider.get_intraday_data(
                ticker, interval="15min", period="5d"
            )

            if data.empty or len(data) < 50:
                return False, 0.0, 0.0

            prices = data["close"].values
            volumes = data["volume"].values
            data["high"].values

            current_price = prices[-1]
            current_volume = volumes[-5:].mean()  # Recent 5 periods
            avg_volume = volumes[:-10].mean()  # Historical average

            # Calculate resistance levels (pivot highs)
            resistance_levels = []
            for i in range(20, len(prices) - 5):
                if (
                    prices[i] == max(prices[i - 3 : i + 4])
                    and prices[i] > prices[:i].max() * 0.98
                ):  # Near high water mark
                    resistance_levels.append(prices[i])

            if not resistance_levels:
                return False, 0.0, 0.0

            # Find key resistance level
            key_resistance = (
                max(resistance_levels[-3:])
                if len(resistance_levels) >= 3
                else max(resistance_levels)
            )

            # Volume confirmation
            volume_multiple = current_volume / avg_volume if avg_volume > 0 else 0

            # Breakout strength
            breakout_strength = (current_price - key_resistance) / key_resistance

            # Recent momentum
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5]

            # Breakout criteria
            is_breakout = (
                current_price > key_resistance * (1 + self.min_breakout_strength)
                and volume_multiple >= self.min_volume_multiple
                and breakout_strength > self.min_breakout_strength
                and recent_momentum > 0.005  # 0.5% momentum
            )

            strength_score = min(
                100,
                (breakout_strength * 100 + volume_multiple * 10 + recent_momentum * 50),
            )

            return is_breakout, key_resistance, strength_score

        except Exception as e:
            self.logger.error(f"Error detecting breakout for {ticker}: {e}")
            return False, 0.0, 0.0

    async def detect_momentum_continuation(self, ticker: str) -> tuple[bool, float]:
        """Detect strong momentum continuation patterns."""
        try:
            # Get 2 days of 5 - minute data for momentum analysis
            data = await self.data_provider.get_intraday_data(
                ticker, interval="5min", period="2d"
            )

            if data.empty or len(data) < 30:
                return False, 0.0

            prices = data["close"].values
            volumes = data["volume"].values

            # Moving averages for trend confirmation
            short_ma = prices[-5:].mean()  # 25min MA
            medium_ma = prices[-10:].mean()  # 50min MA
            long_ma = prices[-20:].mean()  # 100min MA

            # Check for accelerating momentum
            if short_ma > medium_ma > long_ma:
                momentum_strength = (short_ma / long_ma - 1) * 100

                # Volume confirmation
                recent_vol = volumes[-10:].mean()
                earlier_vol = volumes[-30:-10].mean()
                vol_increase = recent_vol / earlier_vol if earlier_vol > 0 else 1

                # Strong momentum criteria
                if momentum_strength > 1.0 and vol_increase > 1.3:
                    return True, momentum_strength

            return False, 0.0

        except Exception as e:
            self.logger.error(f"Error detecting momentum for {ticker}: {e}")
            return False, 0.0

    async def detect_reversal_setup(self, ticker: str) -> tuple[bool, str, float]:
        """Detect oversold bounce setups."""
        try:
            # Get 3 days of 15 - minute data for reversal analysis
            data = await self.data_provider.get_intraday_data(
                ticker, interval="15min", period="3d"
            )

            if data.empty or len(data) < 40:
                return False, "insufficient_data", 0.0

            prices = data["close"].values
            lows = data["low"].values
            volumes = data["volume"].values

            current_price = prices[-1]

            # Look for bounce from oversold levels
            recent_low = lows[-20:].min()  # 20 - period low
            bounce_strength = (current_price - recent_low) / recent_low

            # Volume spike confirmation
            current_vol = volumes[-3:].mean()
            avg_vol = volumes[:-10].mean()
            vol_spike = current_vol / avg_vol if avg_vol > 0 else 1

            # Simple oversold condition
            up_moves = sum(
                1
                for i in range(len(prices) - 10, len(prices) - 1)
                if prices[i + 1] > prices[i]
            )
            down_moves = 10 - up_moves

            # Reversal criteria
            if (
                bounce_strength > 0.015  # 1.5% bounce from low
                and vol_spike > 2.0  # Volume spike
                and down_moves >= 7
            ):  # Was oversold
                return True, "oversold_bounce", bounce_strength * 100

            return False, "no_setup", 0.0

        except Exception as e:
            self.logger.error(f"Error detecting reversal for {ticker}: {e}")
            return False, "error", 0.0

    def get_optimal_expiry(self) -> str:
        """Get optimal expiry for swing trades (≤30 days)."""
        today = date.today()

        # Prefer weekly expirations for faster management
        target_days = min(self.max_expiry_days, 21)  # Max 3 weeks

        # Find next Friday
        days_to_friday = (4 - today.weekday()) % 7
        if days_to_friday == 0:  # If today is Friday
            days_to_friday = 7

        # Adjust if too far out
        if days_to_friday > target_days:
            days_to_friday -= 7

        if days_to_friday <= 0:
            days_to_friday = 7

        expiry_date = today + timedelta(days=days_to_friday)
        return expiry_date.strftime("%Y-%m-%d")

    def calculate_option_targets(
        self, premium: float
    ) -> tuple[float, float, float, float]:
        """Calculate profit targets and stop loss."""
        profit_25 = premium * 1.25  # 25% profit
        profit_50 = premium * 1.50  # 50% profit
        profit_100 = premium * 2.00  # 100% profit
        stop_loss = premium * 0.70  # 30% stop loss

        return profit_25, profit_50, profit_100, stop_loss

    async def estimate_swing_premium(
        self, ticker: str, strike: float, expiry: str
    ) -> float:
        """Estimate option premium for swing trade."""
        try:
            # Try to get actual options data
            options_data = await self.data_provider.get_options_chain(ticker, expiry)
            if options_data and "calls" in options_data:
                calls = options_data["calls"]

                # Find closest strike
                closest_call = None
                min_diff = float("inf")

                for call in calls:
                    diff = abs(call["strike"] - strike)
                    if diff < min_diff:
                        min_diff = diff
                        closest_call = call

                if (
                    closest_call
                    and closest_call.get("bid", 0) > 0
                    and closest_call.get("ask", 0) > 0
                ):
                    return (closest_call["bid"] + closest_call["ask"]) / 2

            # Fallback estimate
            current_price = await self.data_provider.get_current_price(ticker)
            if not current_price:
                return 2.0

            days_to_exp = (
                datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()
            ).days

            # Higher IV assumption for swing trades
            time_premium = max(0.5, current_price * 0.08 * (days_to_exp / 21))

            if strike > current_price:  # OTM
                otm_discount = max(
                    0.2, 1 - (strike - current_price) / current_price * 5
                )
                return time_premium * otm_discount
            else:  # ITM
                intrinsic = current_price - strike
                return intrinsic + time_premium * 0.5

        except Exception as e:
            self.logger.error(f"Error estimating premium for {ticker}: {e}")
            return 2.0

    async def scan_swing_opportunities(self) -> list[SwingSignal]:
        """Scan for swing trading opportunities."""
        signals = []
        expiry = self.get_optimal_expiry()

        self.logger.info(f"Scanning swing opportunities targeting {expiry}")

        for ticker in self.swing_tickers:
            try:
                # Skip if we already have a position
                if any(pos["ticker"] == ticker for pos in self.active_positions):
                    continue

                # Get current price
                current_price = await self.data_provider.get_current_price(ticker)
                if not current_price:
                    continue

                # Check for different signal types
                signals_found = []

                # 1. Breakout detection
                (
                    is_breakout,
                    resistance_level,
                    breakout_strength,
                ) = await self.detect_breakout(ticker)
                if is_breakout:
                    signals_found.append(
                        ("breakout", breakout_strength, resistance_level)
                    )

                # 2. Momentum continuation
                (
                    is_momentum,
                    momentum_strength,
                ) = await self.detect_momentum_continuation(ticker)
                if is_momentum:
                    signals_found.append(("momentum", momentum_strength, current_price))

                # 3. Reversal setup
                (
                    is_reversal,
                    reversal_type,
                    reversal_strength,
                ) = await self.detect_reversal_setup(ticker)
                if is_reversal and reversal_type == "oversold_bounce":
                    signals_found.append(("reversal", reversal_strength, current_price))

                # Process signals
                for signal_type, strength, ref_level in signals_found:
                    if strength < self.min_strength_score:
                        continue

                    # Target strike selection based on signal type
                    if signal_type == "breakout":
                        strike_multiplier = 1.02  # 2% OTM for breakouts
                        max_hold_hours = 6  # Breakouts can be held longer
                    elif signal_type == "momentum":
                        strike_multiplier = 1.015  # 1.5% OTM for momentum
                        max_hold_hours = 4  # Momentum fades fast
                    else:  # reversal
                        strike_multiplier = 1.025  # 2.5% OTM for reversals
                        max_hold_hours = 8  # Reversals take time

                    target_strike = round(current_price * strike_multiplier, 1)
                    premium = await self.estimate_swing_premium(
                        ticker, target_strike, expiry
                    )

                    if premium < self.min_premium:
                        continue

                    # Calculate targets
                    profit_25, profit_50, profit_100, stop_loss = (
                        self.calculate_option_targets(premium)
                    )

                    # Risk assessment
                    if strength > 80:
                        risk_level = "low"
                    elif strength > 60:
                        risk_level = "medium"
                    else:
                        risk_level = "high"

                    signal = SwingSignal(
                        ticker=ticker,
                        signal_time=datetime.now(),
                        signal_type=signal_type,
                        entry_price=current_price,
                        breakout_level=ref_level,
                        volume_confirmation=self.min_volume_multiple,
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
                    self.logger.info(
                        f"Swing signal: {ticker} {signal_type} "
                        f"strength={strength: .0f} strike = ${target_strike} premium = ${premium: .2f}"
                    )

            except Exception as e:
                self.logger.error(f"Error scanning {ticker}: {e}")
                continue

        # Sort by strength score
        signals.sort(key=lambda x: x.strength_score, reverse=True)
        return signals

    async def execute_swing_trade(self, signal: SwingSignal) -> bool:
        """Execute swing trade."""
        try:
            # Check if we can add more positions
            if len(self.active_positions) >= self.max_positions:
                self.logger.info("Max swing positions reached, skipping trade")
                return False

            # Calculate position size
            portfolio_value = await self.integration_manager.get_portfolio_value()
            max_risk = portfolio_value * self.max_position_size

            # Size based on premium cost
            contracts = max(1, int(max_risk / (signal.option_premium * 100)))
            contracts = min(contracts, 3)  # Max 3 contracts for swing trades

            # Create trade signal
            trade_signal = ProductionTradeSignal(
                symbol=signal.ticker,
                action="BUY",
                quantity=contracts,
                option_type="CALL",
                strike_price=Decimal(str(signal.target_strike)),
                expiration_date=datetime.strptime(
                    signal.target_expiry, "%Y-%m-%d"
                ).date(),
                premium=Decimal(str(signal.option_premium)),
                confidence=min(1.0, signal.strength_score / 100.0),
                strategy_name=self.strategy_name,
                signal_strength=min(1.0, signal.strength_score / 100.0),
                metadata={
                    "swing_type": signal.signal_type,
                    "strength_score": signal.strength_score,
                    "breakout_level": signal.breakout_level,
                    "max_hold_hours": signal.max_hold_hours,
                    "profit_targets": [
                        signal.profit_target_1,
                        signal.profit_target_2,
                        signal.profit_target_3,
                    ],
                    "stop_loss": signal.stop_loss,
                    "risk_level": signal.risk_level,
                    "volume_confirmation": signal.volume_confirmation,
                },
            )

            # Execute the trade
            success = await self.integration_manager.execute_trade_signal(trade_signal)

            if success:
                # Track position
                position = {
                    "ticker": signal.ticker,
                    "signal_type": signal.signal_type,
                    "trade_signal": trade_signal,
                    "entry_time": signal.signal_time,
                    "entry_premium": signal.option_premium,
                    "contracts": contracts,
                    "cost_basis": signal.option_premium * contracts * 100,
                    "max_hold_hours": signal.max_hold_hours,
                    "profit_targets": [
                        signal.profit_target_1,
                        signal.profit_target_2,
                        signal.profit_target_3,
                    ],
                    "stop_loss": signal.stop_loss,
                    "hit_profit_target": 0,  # Track which targets hit
                    "expiry_date": datetime.strptime(
                        signal.target_expiry, "%Y-%m-%d"
                    ).date(),
                }

                self.active_positions.append(position)

                await self.integration_manager.alert_system.send_alert(
                    "SWING_ENTRY",
                    "HIGH",  # Swing trades are urgent due to time sensitivity
                    f"Swing Entry: {signal.ticker} {signal.signal_type.upper()} "
                    f"${signal.target_strike} call {signal.target_expiry} "
                    f"{contracts} contracts @ ${signal.option_premium: .2f}",
                )

                self.logger.info(f"Swing trade executed: {signal.ticker}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error executing swing trade for {signal.ticker}: {e}")
            return False

    async def manage_positions(self):
        """Manage existing swing positions."""
        positions_to_remove = []
        current_time = datetime.now()

        for i, position in enumerate(self.active_positions):
            try:
                ticker = position["ticker"]
                contracts = position["contracts"]
                entry_premium = position["entry_premium"]

                # Calculate hours held
                hours_held = (
                    current_time - position["entry_time"]
                ).total_seconds() / 3600

                # Get current option price
                current_premium = await self.data_provider.get_option_price(
                    ticker,
                    position["trade_signal"].strike_price,
                    position["trade_signal"].expiration_date,
                    "call",
                )

                if not current_premium:
                    continue

                # Calculate P & L
                pnl_per_contract = current_premium - entry_premium
                total_pnl = pnl_per_contract * contracts * 100
                pnl_pct = (
                    (pnl_per_contract / entry_premium) * 100 if entry_premium > 0 else 0
                )

                # Check exit conditions
                should_exit = False
                should_scale_out = False
                exit_reason = ""
                scale_out_percentage = 0

                # 1. Profit targets
                if current_premium >= position["profit_targets"][2]:  # 100% target
                    should_exit = True
                    exit_reason = "PROFIT_TARGET_100"
                elif (
                    current_premium >= position["profit_targets"][1]
                    and position["hit_profit_target"] < 2
                ):
                    should_scale_out = True
                    scale_out_percentage = 50
                    position["hit_profit_target"] = 2
                    exit_reason = "PROFIT_TARGET_50"
                elif (
                    current_premium >= position["profit_targets"][0]
                    and position["hit_profit_target"] < 1
                ):
                    should_scale_out = True
                    scale_out_percentage = 25
                    position["hit_profit_target"] = 1
                    exit_reason = "PROFIT_TARGET_25"

                # 2. Stop loss
                elif current_premium <= position["stop_loss"]:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                # 3. Time-based exits
                elif hours_held >= position["max_hold_hours"]:
                    should_exit = True
                    exit_reason = "MAX_HOLD_TIME"

                # 4. End of day exit for momentum trades
                elif (
                    position["signal_type"] == "momentum"
                    and current_time.hour >= self.end_of_day_exit_hour
                ):
                    should_exit = True
                    exit_reason = "END_OF_DAY"

                # 5. Same day exit for swing trades (WSB rule)
                elif current_time.date() > position["entry_time"].date():
                    if (
                        position["signal_type"] in ["momentum", "breakout"]
                        and pnl_pct > -15
                    ):
                        should_exit = True
                        exit_reason = "NEXT_DAY_EXIT"

                # Execute exit or scale-out
                if should_exit or should_scale_out:
                    exit_contracts = (
                        contracts
                        if should_exit
                        else max(1, int(contracts * scale_out_percentage / 100))
                    )

                    exit_signal = ProductionTradeSignal(
                        symbol=ticker,
                        action="SELL",
                        quantity=exit_contracts,
                        option_type="CALL",
                        strike_price=position["trade_signal"].strike_price,
                        expiration_date=position["trade_signal"].expiration_date,
                        premium=Decimal(str(current_premium)),
                        strategy_name=self.strategy_name,
                        metadata={
                            "swing_action": "full_exit" if should_exit else "scale_out",
                            "exit_reason": exit_reason,
                            "hours_held": hours_held,
                            "pnl_per_contract": pnl_per_contract,
                            "total_pnl": total_pnl,
                            "pnl_pct": pnl_pct,
                            "profit_target_level": position["hit_profit_target"],
                        },
                    )

                    success = await self.integration_manager.execute_trade_signal(
                        exit_signal
                    )

                    if success:
                        if should_exit:
                            await self.integration_manager.alert_system.send_alert(
                                "SWING_EXIT",
                                "HIGH",
                                f"Swing Exit: {ticker} {exit_reason} "
                                f"P & L: ${total_pnl:.0f} ({pnl_pct: .1%}) "
                                f"Held: {hours_held:.1f}h",
                            )
                            positions_to_remove.append(i)
                            self.logger.info(
                                f"Swing position closed: {ticker} {exit_reason}"
                            )
                        else:
                            # Update position after scale-out
                            position["contracts"] -= exit_contracts
                            position["cost_basis"] -= (
                                entry_premium * exit_contracts * 100
                            )

                            await self.integration_manager.alert_system.send_alert(
                                "SWING_SCALE_OUT",
                                "MEDIUM",
                                f"Swing Scale-Out: {ticker} {exit_reason} "
                                f"Sold {exit_contracts} contracts at {pnl_pct: .1%} gain",
                            )
                            self.logger.info(
                                f"Swing scaled out: {ticker} {exit_reason}"
                            )

            except Exception as e:
                self.logger.error(f"Error managing swing position {i}: {e}")

        # Remove fully closed positions
        for i in reversed(positions_to_remove):
            self.active_positions.pop(i)

    async def scan_opportunities(self) -> list[ProductionTradeSignal]:
        """Main strategy execution: scan and generate trade signals."""
        try:
            # First manage existing positions
            await self.manage_positions()

            # Then scan for new opportunities if we have capacity
            if len(self.active_positions) >= self.max_positions:
                return []

            # Only scan during market hours for swing trades
            if not await self.data_provider.is_market_open():
                return []

            # Don't initiate new swings near market close
            current_hour = datetime.now().hour
            if current_hour >= 15:  # After 3pm ET
                return []

            # Scan for swing opportunities
            swing_signals = await self.scan_swing_opportunities()

            # Execute top signals
            trade_signals = []
            max_new_positions = self.max_positions - len(self.active_positions)

            for signal in swing_signals[:max_new_positions]:
                success = await self.execute_swing_trade(signal)
                if success:
                    # Return trade signal for tracking
                    trade_signal = ProductionTradeSignal(
                        symbol=signal.ticker,
                        action="BUY",
                        quantity=1,  # Will be recalculated in execute_trade
                        option_type="CALL",
                        strike_price=Decimal(str(signal.target_strike)),
                        expiration_date=datetime.strptime(
                            signal.target_expiry, "%Y-%m-%d"
                        ).date(),
                        premium=Decimal(str(signal.option_premium)),
                        confidence=min(1.0, signal.strength_score / 100.0),
                        strategy_name=self.strategy_name,
                        signal_strength=min(1.0, signal.strength_score / 100.0),
                    )
                    trade_signals.append(trade_signal)

            return trade_signals

        except Exception as e:
            self.logger.error(f"Error in swing trading scan: {e}")
            return []

    def get_strategy_status(self) -> dict[str, Any]:
        """Get current strategy status."""
        try:
            total_cost_basis = sum(pos["cost_basis"] for pos in self.active_positions)
            position_details = []

            for position in self.active_positions:
                hours_held = (
                    datetime.now() - position["entry_time"]
                ).total_seconds() / 3600

                position_details.append(
                    {
                        "ticker": position["ticker"],
                        "signal_type": position["signal_type"],
                        "strike": float(position["trade_signal"].strike_price),
                        "expiry": position["expiry_date"].isoformat(),
                        "contracts": position["contracts"],
                        "entry_premium": position["entry_premium"],
                        "cost_basis": position["cost_basis"],
                        "max_hold_hours": position["max_hold_hours"],
                        "hours_held": round(hours_held, 1),
                        "hit_profit_target": position["hit_profit_target"],
                        "entry_time": position["entry_time"].isoformat(),
                    }
                )

            return {
                "strategy_name": self.strategy_name,
                "is_active": True,
                "active_positions": len(self.active_positions),
                "max_positions": self.max_positions,
                "total_cost_basis": total_cost_basis,
                "position_details": position_details,
                "last_scan": datetime.now().isoformat(),
                "config": {
                    "max_positions": self.max_positions,
                    "max_position_size": self.max_position_size,
                    "max_expiry_days": self.max_expiry_days,
                    "min_strength_score": self.min_strength_score,
                    "profit_targets": self.profit_targets,
                    "stop_loss_pct": self.stop_loss_pct,
                    "max_hold_hours": self.max_hold_hours,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting swing trading status: {e}")
            return {"strategy_name": self.strategy_name, "error": str(e)}

    async def run_strategy(self):
        """Main strategy execution loop."""
        self.logger.info("Starting Production Swing Trading Strategy")

        try:
            while True:
                # Scan for swing trading opportunities
                signals = await self.scan_opportunities()

                # Execute trades for signals
                if signals:
                    await self.execute_trades(signals)

                # Wait before next scan (swing trading is active)
                await asyncio.sleep(120)  # 2 minutes between scans

        except Exception as e:
            self.logger.error(f"Error in swing trading strategy main loop: {e}")


def create_production_swing_trading(
    integration_manager, data_provider: ReliableDataProvider, config: dict
) -> ProductionSwingTrading:
    """Factory function to create ProductionSwingTrading strategy."""
    return ProductionSwingTrading(integration_manager, data_provider, config)
