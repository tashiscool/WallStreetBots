#!/usr / bin / env python3
"""Production Momentum Weeklies Strategy
Detects intraday reversals and news momentum for weekly options plays
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
class MomentumSignal:
    """Production momentum signal with enhanced metadata"""

    ticker: str
    signal_time: datetime
    current_price: float
    reversal_type: str  # "bullish_reversal", "news_momentum", "breakout"
    volume_spike: float  # Multiple of average volume
    price_momentum: float  # % change triggering signal
    weekly_expiry: str
    target_strike: float
    premium_estimate: float
    risk_level: str  # "low", "medium", "high"
    exit_target: float
    stop_loss: float
    confidence: float  # 0 - 1 confidence score
    expected_move: float  # Expected stock move needed for profit


class ProductionMomentumWeeklies:
    """Production Momentum Weeklies Strategy

    Strategy Logic:
    1. Scans mega - cap stocks for intraday momentum patterns
    2. Detects volume spikes (3x+ average) combined with:
       - Bullish reversal patterns (V - shaped bounces)
       - Breakout momentum above resistance
    3. Targets weekly options 2 - 5% OTM based on momentum strength
    4. Quick exits: 25 - 50% profit target or 50% loss stop
    5. Time-based exits: Close before final trading day

    Risk Management:
    - Maximum 5% account risk per position
    - Maximum 3 concurrent momentum plays
    - Strict time decay management for weeklies
    - Volume and liquidity requirements
    """

    def __init__(self, integration_manager, data_provider: ReliableDataProvider, config: dict):
        self.strategy_name = "momentum_weeklies"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.options_selector = SmartOptionsSelector(data_provider)
        self.risk_manager = RealTimeRiskManager()
        self.bs_engine = BlackScholesEngine()

        # Strategy configuration
        self.mega_caps = config.get(
            "watchlist",
            [
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
                "PYPL",
                "DIS",
                "V",
                "MA",
                "JPM",
                "BAC",
                "WMT",
                "HD",
            ],
        )

        # Risk parameters
        self.max_positions = config.get("max_positions", 3)
        self.max_position_size = config.get("max_position_size", 0.05)  # 5% per position
        self.min_volume_spike = config.get("min_volume_spike", 3.0)
        self.min_momentum_threshold = config.get("min_momentum_threshold", 0.015)  # 1.5%

        # Option targeting
        self.target_dte_range = config.get("target_dte_range", (0, 4))  # 0 - 4 days for weeklies
        self.otm_range = config.get("otm_range", (0.02, 0.05))  # 2 - 5% OTM
        self.min_premium = config.get("min_premium", 0.50)  # Minimum $0.50 premium

        # Exit criteria
        self.profit_target = config.get("profit_target", 0.25)  # 25% profit target
        self.stop_loss = config.get("stop_loss", 0.50)  # 50% loss stop
        self.time_exit_hours = config.get("time_exit_hours", 4)  # Exit if no profit in 4 hours

        # Active positions tracking
        self.active_positions: list[dict[str, Any]] = []

        self.logger.info("ProductionMomentumWeeklies strategy initialized")

    def get_next_weekly_expiry(self) -> str:
        """Get next weekly expiry date"""
        today = date.today()

        # For weeklies, target this Friday or next Friday
        days_until_friday = (4 - today.weekday()) % 7  # Friday = 4

        # If it's Friday after market hours, or very close to expiry, use next Friday
        if days_until_friday == 0 or days_until_friday <= 1:
            days_until_friday = 7 if days_until_friday == 0 else 7 + days_until_friday

        next_friday = today + timedelta(days=days_until_friday)
        return next_friday.strftime("%Y-%m-%d")

    async def detect_volume_spike(self, ticker: str) -> tuple[bool, float]:
        """Detect unusual volume spike"""
        try:
            # Get intraday volume data
            volume_data = await self.data_provider.get_intraday_data(
                ticker, interval="5min", period="5d"
            )

            if volume_data.empty:
                return False, 0.0

            # Current volume (last 5 bars average)
            current_vol = volume_data["volume"].tail(5).mean()

            # Average volume (excluding current spike)
            avg_vol = volume_data["volume"].iloc[:-10].mean()

            if avg_vol <= 0:
                return False, 0.0

            vol_multiple = float(current_vol / avg_vol)
            return vol_multiple >= self.min_volume_spike, vol_multiple

        except Exception as e:
            self.logger.error(f"Error detecting volume spike for {ticker}: {e}")
            return False, 0.0

    async def detect_reversal_pattern(self, ticker: str) -> tuple[bool, str, float]:
        """Detect bullish reversal patterns"""
        try:
            # Get recent price action (2 hours of 5 - minute bars)
            price_data = await self.data_provider.get_intraday_data(
                ticker, interval="5min", period="2d"
            )

            if len(price_data) < 24:  # Need at least 2 hours of data
                return False, "insufficient_data", 0.0

            # Focus on last 2 hours (24 bars)
            recent_data = price_data.tail(24)
            prices = recent_data["close"].values
            current_price = prices[-1]

            # Find the lowest point in recent action
            low_idx = prices.argmin()
            low_price = prices[low_idx]

            # Calculate bounce from low
            bounce_pct = (current_price - low_price) / low_price

            # Reversal criteria:
            # 1. Significant bounce from recent low ( > 1.5%)
            # 2. Low occurred in first 60% of window (early decline, late recovery)
            # 3. Current price above recent average
            recent_avg = prices[-12:].mean()  # Last hour average

            is_reversal = (
                bounce_pct >= self.min_momentum_threshold
                and low_idx < len(prices) * 0.6
                and current_price > recent_avg
            )

            return is_reversal, "bullish_reversal", bounce_pct

        except Exception as e:
            self.logger.error(f"Error detecting reversal for {ticker}: {e}")
            return False, "error", 0.0

    async def detect_breakout_momentum(self, ticker: str) -> tuple[bool, float]:
        """Detect breakout above resistance"""
        try:
            # Get extended price data for resistance calculation
            price_data = await self.data_provider.get_intraday_data(
                ticker, interval="15min", period="5d"
            )

            if len(price_data) < 50:
                return False, 0.0

            prices = price_data["close"].values
            volumes = price_data["volume"].values
            current_price = prices[-1]

            # Calculate resistance (highest high in past 3 days, excluding very recent)
            resistance_window = prices[-100:-5]  # Exclude last 5 bars
            if len(resistance_window) < 20:
                return False, 0.0

            resistance = resistance_window.max()

            # Check breakout with volume confirmation
            if current_price > resistance * 1.005:  # 0.5% above resistance
                # Volume confirmation
                recent_vol = volumes[-5:].mean()
                avg_vol = volumes[-50:-5].mean()

                if recent_vol > avg_vol * 1.5:  # 50% volume increase
                    breakout_strength = (current_price - resistance) / resistance
                    return True, breakout_strength

            return False, 0.0

        except Exception as e:
            self.logger.error(f"Error detecting breakout for {ticker}: {e}")
            return False, 0.0

    async def calculate_option_premium(
        self, ticker: str, strike: float, expiry: str, option_type: str = "call"
    ) -> float:
        """Calculate theoretical option premium"""
        try:
            # Get current stock data
            current_price = await self.data_provider.get_current_price(ticker)
            if not current_price:
                return 0.0

            # Get volatility data
            iv_data = await self.data_provider.get_implied_volatility(ticker)
            volatility = iv_data.get("iv_rank", 30) / 100.0  # Convert IV rank to volatility

            # Calculate time to expiry
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            days_to_expiry = (expiry_date - date.today()).days
            time_to_expiry = max(0.01, days_to_expiry / 365.0)  # Minimum 0.01 years

            # Risk - free rate (approximate)
            risk_free_rate = 0.05

            # Calculate theoretical premium
            premium = self.bs_engine.calculate_option_price(
                current_price, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )

            return max(0.10, premium)  # Minimum $0.10

        except Exception as e:
            self.logger.error(f"Error calculating premium for {ticker}: {e}")
            return 1.0  # Default estimate

    async def scan_momentum_opportunities(self) -> list[MomentumSignal]:
        """Scan for momentum weekly opportunities"""
        signals = []
        weekly_expiry = self.get_next_weekly_expiry()

        self.logger.info(f"Scanning {len(self.mega_caps)} tickers for momentum signals")

        for ticker in self.mega_caps:
            try:
                # Skip if we already have a position in this ticker
                if any(pos["ticker"] == ticker for pos in self.active_positions):
                    continue

                # Check volume spike
                has_volume_spike, vol_multiple = await self.detect_volume_spike(ticker)
                if not has_volume_spike:
                    continue

                # Check for reversal pattern
                has_reversal, pattern_type, bounce_pct = await self.detect_reversal_pattern(ticker)

                # Check for breakout
                has_breakout, breakout_strength = await self.detect_breakout_momentum(ticker)

                # Need volume spike+(reversal OR breakout)
                if not (has_reversal or has_breakout):
                    continue

                # Get current price
                current_price = await self.data_provider.get_current_price(ticker)
                if not current_price:
                    continue

                # Determine signal characteristics
                if has_breakout:
                    signal_type = "breakout"
                    momentum = breakout_strength
                    confidence = 0.8 if vol_multiple > 5 else 0.6
                    risk = "high" if vol_multiple > 6 else "medium"
                else:
                    signal_type = "bullish_reversal"
                    momentum = bounce_pct
                    confidence = 0.7 if vol_multiple > 4 else 0.5
                    risk = "medium" if vol_multiple > 4 else "low"

                # Skip if momentum is too weak
                if momentum < self.min_momentum_threshold:
                    continue

                # Calculate target strike based on momentum strength
                if momentum > 0.04:  # Very strong momentum
                    otm_pct = self.otm_range[0]  # 2% OTM
                elif momentum > 0.025:  # Strong momentum
                    otm_pct = 0.03  # 3% OTM
                else:  # Moderate momentum
                    otm_pct = self.otm_range[1]  # 5% OTM

                target_strike = round(current_price * (1 + otm_pct), 1)

                # Calculate premium estimate
                premium = await self.calculate_option_premium(ticker, target_strike, weekly_expiry)

                if premium < self.min_premium:
                    continue  # Skip if premium too low

                # Calculate targets
                expected_move = otm_pct + 0.02  # Need to move beyond strike+2%
                exit_target = current_price * (1 + expected_move)
                stop_loss = current_price * 0.985  # 1.5% stock stop

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
                    confidence=confidence,
                    expected_move=expected_move,
                )

                signals.append(signal)

                self.logger.info(
                    f"Momentum signal: {ticker} {signal_type} "
                    f"vol={vol_multiple: .1f}x momentum={momentum: .2%} "
                    f"strike=${target_strike} premium = ${premium: .2f}"
                )

            except Exception as e:
                self.logger.error(f"Error scanning {ticker}: {e}")
                continue

        # Sort by confidence and momentum strength
        signals.sort(key=lambda s: (s.confidence, s.price_momentum), reverse=True)

        return signals

    async def execute_momentum_trade(self, signal: MomentumSignal) -> bool:
        """Execute momentum trade"""
        try:
            # Check if we can add more positions
            if len(self.active_positions) >= self.max_positions:
                self.logger.info("Max positions reached, skipping trade")
                return False

            # Calculate position size
            portfolio_value = await self.integration_manager.get_portfolio_value()
            max_risk = portfolio_value * self.max_position_size

            # Size position based on premium and risk
            contracts = max(
                1, int(max_risk / (signal.premium_estimate * 100))
            )  # Options are 100 shares
            contracts = min(contracts, 10)  # Max 10 contracts for weekly momentum plays

            # Create trade signal
            trade_signal = ProductionTradeSignal(
                symbol=signal.ticker,
                action="BUY",
                quantity=contracts,
                option_type="CALL",
                strike_price=Decimal(str(signal.target_strike)),
                expiration_date=datetime.strptime(signal.weekly_expiry, "%Y-%m-%d").date(),
                premium=Decimal(str(signal.premium_estimate)),
                confidence=signal.confidence,
                strategy_name=self.strategy_name,
                signal_strength=signal.price_momentum,
                metadata={
                    "reversal_type": signal.reversal_type,
                    "volume_spike": signal.volume_spike,
                    "exit_target": signal.exit_target,
                    "stop_loss": signal.stop_loss,
                    "expected_move": signal.expected_move,
                    "risk_level": signal.risk_level,
                    "entry_time": signal.signal_time.isoformat(),
                },
            )

            # Execute the trade
            success = await self.integration_manager.execute_trade_signal(trade_signal)

            if success:
                # Track position
                position = {
                    "ticker": signal.ticker,
                    "trade_signal": trade_signal,
                    "entry_time": signal.signal_time,
                    "entry_premium": signal.premium_estimate,
                    "contracts": contracts,
                    "exit_target": signal.exit_target,
                    "stop_loss": signal.stop_loss,
                    "risk_level": signal.risk_level,
                    "time_limit": signal.signal_time + timedelta(hours=self.time_exit_hours),
                }

                self.active_positions.append(position)

                await self.integration_manager.alert_system.send_alert(
                    "MOMENTUM_ENTRY",
                    "HIGH",
                    f"Momentum Weekly Entry: {signal.ticker} "
                    f"{signal.reversal_type} ${signal.target_strike} call "
                    f"{contracts} contracts @ ${signal.premium_estimate: .2f}",
                )

                self.logger.info(f"Momentum trade executed: {signal.ticker}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error executing momentum trade for {signal.ticker}: {e}")
            return False

    async def manage_positions(self):
        """Manage existing momentum positions"""
        positions_to_remove = []

        for i, position in enumerate(self.active_positions):
            try:
                ticker = position["ticker"]
                entry_premium = position["entry_premium"]
                contracts = position["contracts"]

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
                pnl_pct = pnl_per_contract / entry_premium if entry_premium > 0 else 0

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                # Profit target
                if pnl_pct >= self.profit_target:
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"

                # Stop loss
                elif pnl_pct <= -self.stop_loss:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                # Time-based exit
                elif datetime.now() >= position["time_limit"]:
                    should_exit = True
                    exit_reason = "TIME_LIMIT"

                # Day before expiry exit
                elif position["trade_signal"].expiration_date <= date.today() + timedelta(days=1):
                    should_exit = True
                    exit_reason = "EXPIRY_RISK"

                if should_exit:
                    # Create exit trade signal
                    exit_signal = ProductionTradeSignal(
                        symbol=ticker,
                        action="SELL",
                        quantity=contracts,
                        option_type="CALL",
                        strike_price=position["trade_signal"].strike_price,
                        expiration_date=position["trade_signal"].expiration_date,
                        premium=Decimal(str(current_premium)),
                        strategy_name=self.strategy_name,
                        metadata={
                            "exit_reason": exit_reason,
                            "entry_premium": entry_premium,
                            "pnl_per_contract": pnl_per_contract,
                            "total_pnl": total_pnl,
                            "pnl_pct": pnl_pct,
                        },
                    )

                    # Execute exit
                    success = await self.integration_manager.execute_trade_signal(exit_signal)

                    if success:
                        await self.integration_manager.alert_system.send_alert(
                            "MOMENTUM_EXIT",
                            "MEDIUM",
                            f"Momentum Weekly Exit: {ticker} {exit_reason} "
                            f"P & L: ${total_pnl:.2f} ({pnl_pct: .1%})",
                        )

                        positions_to_remove.append(i)
                        self.logger.info(f"Momentum position closed: {ticker} {exit_reason}")

            except Exception as e:
                self.logger.error(f"Error managing position {i}: {e}")

        # Remove closed positions
        for i in reversed(positions_to_remove):
            self.active_positions.pop(i)

    async def scan_opportunities(self) -> list[ProductionTradeSignal]:
        """Main strategy execution: scan and generate trade signals"""
        try:
            # First manage existing positions
            await self.manage_positions()

            # Then scan for new opportunities if we have capacity
            if len(self.active_positions) >= self.max_positions:
                return []

            # Skip during after - hours
            if not await self.data_provider.is_market_open():
                return []

            # Scan for momentum signals
            momentum_signals = await self.scan_momentum_opportunities()

            # Execute top signals
            trade_signals = []
            for signal in momentum_signals[: self.max_positions - len(self.active_positions)]:
                success = await self.execute_momentum_trade(signal)
                if success:
                    # Convert to ProductionTradeSignal for consistency
                    trade_signal = ProductionTradeSignal(
                        symbol=signal.ticker,
                        action="BUY",
                        quantity=1,  # Will be recalculated in execute_trade
                        option_type="CALL",
                        strike_price=Decimal(str(signal.target_strike)),
                        expiration_date=datetime.strptime(signal.weekly_expiry, "%Y-%m-%d").date(),
                        premium=Decimal(str(signal.premium_estimate)),
                        confidence=signal.confidence,
                        strategy_name=self.strategy_name,
                        signal_strength=signal.price_momentum,
                    )
                    trade_signals.append(trade_signal)

            return trade_signals

        except Exception as e:
            self.logger.error(f"Error in momentum weeklies scan: {e}")
            return []

    def get_strategy_status(self) -> dict[str, Any]:
        """Get current strategy status"""
        try:
            total_pnl = 0.0
            position_details = []

            for position in self.active_positions:
                entry_value = position["entry_premium"] * position["contracts"] * 100
                total_pnl += (
                    entry_value  # This would be updated with current values in real implementation
                )

                position_details.append(
                    {
                        "ticker": position["ticker"],
                        "strike": float(position["trade_signal"].strike_price),
                        "expiry": position["trade_signal"].expiration_date.isoformat(),
                        "contracts": position["contracts"],
                        "entry_premium": position["entry_premium"],
                        "risk_level": position["risk_level"],
                        "time_remaining": (position["time_limit"] - datetime.now()).total_seconds()
                        / 3600,
                    }
                )

            return {
                "strategy_name": self.strategy_name,
                "is_active": True,
                "active_positions": len(self.active_positions),
                "max_positions": self.max_positions,
                "total_pnl": total_pnl,
                "position_details": position_details,
                "last_scan": datetime.now().isoformat(),
                "config": {
                    "max_positions": self.max_positions,
                    "max_position_size": self.max_position_size,
                    "min_volume_spike": self.min_volume_spike,
                    "profit_target": self.profit_target,
                    "stop_loss": self.stop_loss,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"strategy_name": self.strategy_name, "error": str(e)}

    async def run_strategy(self):
        """Main strategy execution loop"""
        self.logger.info("Starting Production Momentum Weeklies Strategy")

        try:
            while True:
                # Scan for momentum opportunities
                signals = await self.scan_opportunities()

                # Execute trades for signals
                if signals:
                    await self.execute_trades(signals)

                # Wait before next scan (momentum runs frequently)
                await asyncio.sleep(60)  # 1 minute between scans

        except Exception as e:
            self.logger.error(f"Error in momentum weeklies strategy main loop: {e}")


def create_production_momentum_weeklies(
    integration_manager, data_provider: ReliableDataProvider, config: dict
) -> ProductionMomentumWeeklies:
    """Factory function to create ProductionMomentumWeeklies strategy"""
    return ProductionMomentumWeeklies(integration_manager, data_provider, config)
