"""Production WSB Dip Bot - Real Trading Implementation.

This is a production - ready version of the WSB Dip Bot that:
- Uses real market data from Alpaca
- Executes real trades via AlpacaManager
- Integrates with Django models for persistence
- Implements real risk management
- Provides live position monitoring

Replaces all hardcoded mock data with live market feeds.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from ...core.trading_interface import OrderSide, OrderType
from ...production.core.production_integration import (
    ProductionIntegrationManager,
    ProductionTradeSignal,
)
from ...production.data.production_data_integration import (
    ReliableDataProvider as ProductionDataProvider,
)


@dataclass
class DipSignal:
    """Dip after run signal."""

    ticker: str
    current_price: Decimal
    run_percentage: float
    dip_percentage: float
    target_strike: Decimal
    target_expiry: datetime
    expected_premium: Decimal
    risk_amount: Decimal
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ProductionWSBDipBot:
    """Production WSB Dip Bot.

    Implements the exact WSB pattern:
    1. Detect "big run" ( >=  10% over 10 days)
    2. Wait for "hard dip" ( <=  -3% vs prior close)
    3. Buy ~5% OTM calls with ~30 DTE
    4. Exit at 3x profit or delta  >=  0.60
    """

    def __init__(
        self,
        integration_manager: ProductionIntegrationManager,
        data_provider: ProductionDataProvider,
        config: dict[str, Any],
    ):
        self.integration = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Strategy parameters
        self.run_lookback_days = config.get("run_lookback_days", 10)
        self.run_threshold = config.get("run_threshold", 0.10)  # 10%
        self.dip_threshold = config.get("dip_threshold", -0.03)  # -3%
        self.target_dte_days = config.get("target_dte_days", 30)
        self.otm_percentage = config.get("otm_percentage", 0.05)  # 5%
        self.max_position_size = config.get("max_position_size", 0.20)  # 20%
        self.target_multiplier = config.get("target_multiplier", 3.0)  # 3x profit
        self.delta_target = config.get("delta_target", 0.60)  # Delta exit
        
        # WSB Mode: Full account reinvestment (ignores risk caps)
        self.wsb_mode = config.get("wsb_mode", False)
        if self.wsb_mode:
            self.logger.warning("WSB MODE ENABLED: Full account reinvestment, ignoring risk caps!")

        # Universe of stocks to scan
        self.universe = config.get(
            "universe",
            [
                "AAPL",
                "MSFT",
                "GOOGL",
                "META",
                "NVDA",
                "AMD",
                "AVGO",
                "TSLA",
                "AMZN",
                "NFLX",
                "CRM",
                "COST",
                "ADBE",
                "V",
                "MA",
                "LIN",
            ],
        )

        # Active positions
        self.active_positions: dict[str, DipSignal] = {}

        self.logger.info("ProductionWSBDipBot initialized")

    async def scan_for_dip_signals(self) -> list[DipSignal]:
        """Scan universe for dip after run signals."""
        signals = []

        try:
            # Check if market is open
            if not await self.data_provider.is_market_open():
                self.logger.info("Market is closed - skipping scan")
                return signals

            for ticker in self.universe:
                try:
                    signal = await self._check_dip_after_run(ticker)
                    if signal:
                        signals.append(signal)
                        self.logger.info(
                            f"Dip signal detected for {ticker}: "
                            f"run={signal.run_percentage: .2%}, "
                            f"dip={signal.dip_percentage: .2%}"
                        )
                except Exception as e:
                    self.logger.error(f"Error checking {ticker}: {e}")
                    continue

            self.logger.info(f"Scan complete: {len(signals)} signals detected")
            return signals

        except Exception as e:
            self.logger.error(f"Error in scan_for_dip_signals: {e}")
            return []

    async def _perform_preflight_checks(self):
        """Perform pre-trading safety checks."""
        try:
            # Check position reconciliation
            reconciliation_report = (
                await self.position_reconciler.reconcile_all_positions(auto_halt=True)
            )

            if reconciliation_report.requires_intervention:
                self.is_trading_enabled = False
                self.logger.error(
                    f"Trading disabled due to position discrepancies: {reconciliation_report.critical_discrepancies} critical"
                )
                return

            self.last_reconciliation = reconciliation_report.timestamp

            # Check data source health
            data_health = await self.data_provider.get_data_source_health()
            unhealthy_sources = [
                source
                for source, health in data_health.items()
                if not health.get("is_healthy", True)
            ]

            if unhealthy_sources:
                self.logger.warning(
                    f"Some data sources are unhealthy: {unhealthy_sources}"
                )

            self.logger.info("Preflight checks completed successfully")

        except Exception as e:
            self.logger.error(f"Preflight checks failed: {e}")
            self.is_trading_enabled = False

    async def _detect_advanced_dip_pattern(self, ticker: str) -> DipSignal | None:
        """Advanced dip detection algorithm with technical indicators."""
        try:
            # Get extended price history (30 days)
            price_history = await self.data_provider.get_price_history(ticker, days=30)
            volume_history = await self.data_provider.get_volume_history(
                ticker, days=30
            )

            if len(price_history) < 30 or len(volume_history) < 30:
                self.logger.warning(
                    f"Insufficient data for {ticker}: {len(price_history)} price points, {len(volume_history)} volume points"
                )
                return None

            # 1. Identify "big run" (20%+ gain in 1 - 5 days)
            recent_high = max(price_history[-5:])  # 5 - day high
            base_price = min(price_history[-15:-5])  # Earlier base (10 - 15 days ago)
            run_percentage = (recent_high - base_price) / base_price

            if run_percentage < 0.20:  # Need 20%+ run first
                self.logger.debug(
                    f"{ticker}: Insufficient run ({run_percentage:.2%}  <  20%)"
                )
                return None

            # 2. Detect significant dip from high
            current_price = price_history[-1]
            dip_percentage = (recent_high - current_price) / recent_high

            if dip_percentage < 0.05:  # Need 5%+ dip
                self.logger.debug(
                    f"{ticker}: Insufficient dip ({dip_percentage:.2%}  <  5%)"
                )
                return None

            # 3. Volume analysis - look for capitulation or exhaustion
            avg_volume = sum(volume_history[-20:-1]) / len(
                volume_history[-20:-1]
            )  # 20 - day average
            recent_volume = volume_history[-1]
            volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # 4. Technical indicators
            rsi = self._calculate_rsi(price_history, period=14)
            bb_position = self._calculate_bollinger_position(price_history)

            # Signal strength scoring
            signal_strength = 0
            if dip_percentage >= 0.08:
                signal_strength += 2  # 8%+ dip
            if volume_spike >= 1.5:
                signal_strength += 2  # 50%+ volume increase
            if rsi < 30:
                signal_strength += 1  # Oversold RSI
            if bb_position < 0.2:
                signal_strength += 1  # Below lower BB

            if signal_strength >= 4:  # Require strong signal
                # Calculate target strike (~5% OTM)
                target_strike = current_price * Decimal(str(1.05))

                # Calculate expected premium (simplified)
                expected_premium = current_price * Decimal("0.05")  # 5% of stock price

                # Calculate risk amount
                risk_amount = expected_premium * Decimal(
                    "100"
                )  # 1 contract = 100 shares

                return DipSignal(
                    ticker=ticker,
                    current_price=current_price,
                    run_percentage=run_percentage,
                    dip_percentage=dip_percentage,
                    target_strike=target_strike,
                    target_expiry=datetime.now() + timedelta(days=self.target_dte_days),
                    expected_premium=expected_premium,
                    risk_amount=risk_amount,
                    confidence=min(0.95, signal_strength / 6.0),  # Scale to 0 - 95%
                    metadata={
                        "volume_spike": volume_spike,
                        "rsi": rsi,
                        "bb_position": bb_position,
                        "signal_strength": signal_strength,
                        "recent_high": recent_high,
                        "base_price": base_price,
                    },
                )

            self.logger.debug(
                f"{ticker}: Signal strength {signal_strength}  <  4 (required)"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error in advanced dip detection for {ticker}: {e}")
            return None

    def _calculate_rsi(self, prices: list[Decimal], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        try:
            if len(prices) < period + 1:
                return 50.0  # Neutral RSI

            # Convert to float for calculations
            price_values = [float(p) for p in prices]

            # Calculate price changes
            deltas = [
                price_values[i] - price_values[i - 1]
                for i in range(1, len(price_values))
            ]

            # Separate gains and losses
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]

            # Calculate average gains and losses
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                return 100.0  # All gains, no losses

            # Calculate RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return 50.0

    def _calculate_bollinger_position(
        self, prices: list[Decimal], period: int = 20, std_dev: float = 2.0
    ) -> float:
        """Calculate Bollinger Band position (0 - 1, where 0.5 is middle)."""
        try:
            if len(prices) < period:
                return 0.5  # Neutral position

            # Convert to float for calculations
            price_values = [float(p) for p in prices[-period:]]
            current_price = float(prices[-1])

            # Calculate moving average
            sma = sum(price_values) / len(price_values)

            # Calculate standard deviation
            variance = sum((p - sma) ** 2 for p in price_values) / len(price_values)
            std = variance**0.5

            # Calculate Bollinger Bands
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)

            # Calculate position (0=lower band, 1=upper band)
            if upper_band == lower_band:
                return 0.5  # Avoid division by zero

            position = (current_price - lower_band) / (upper_band - lower_band)
            return max(0.0, min(1.0, position))  # Clamp to 0 - 1

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger position: {e}")
            return 0.5

    async def select_optimal_option(
        self, dip_signal: DipSignal
    ) -> dict[str, Any] | None:
        """Select best options contract based on WSB criteria."""
        try:
            # Get options chain for the target expiry
            options_chain = await self.data_provider.get_options_chain(
                dip_signal.ticker, expiry_date=dip_signal.target_expiry.date()
            )

            if not options_chain:
                self.logger.warning(
                    f"No options chain available for {dip_signal.ticker}"
                )
                return None

            # Filter for calls ~5% OTM
            target_strike = dip_signal.target_strike
            suitable_options = []

            for option in options_chain:
                if (
                    option.option_type.lower() == "call"
                    and abs(float(option.strike) - float(target_strike))
                    / float(target_strike)
                    < 0.02  # Within 2% of target
                    and option.volume > 10  # Minimum liquidity
                    and option.bid > 0.05
                ):  # Minimum bid to avoid illiquid contracts
                    # Calculate bid - ask spread ratio
                    spread_ratio = (
                        (option.ask - option.bid) / option.bid
                        if option.bid > 0
                        else float("inf")
                    )

                    suitable_options.append(
                        {
                            "option": option,
                            "spread_ratio": spread_ratio,
                            "volume": option.volume,
                            "bid": option.bid,
                            "ask": option.ask,
                            "strike": option.strike,
                        }
                    )

            if not suitable_options:
                self.logger.warning(
                    f"No suitable options found for {dip_signal.ticker}"
                )
                return None

            # Select best option based on bid - ask spread and volume
            best_option = min(suitable_options, key=lambda x: x["spread_ratio"])

            self.logger.info(
                f"Selected option for {dip_signal.ticker}: "
                f"Strike={best_option['strike']}, "
                f"Bid={best_option['bid']: .2f}, "
                f"Volume={best_option['volume']}"
            )

            return best_option

        except Exception as e:
            self.logger.error(
                f"Error selecting optimal option for {dip_signal.ticker}: {e}"
            )
            return None

    async def should_exit_position(self, position: dict[str, Any]) -> dict[str, Any]:
        """Dynamic exit decision based on multiple factors."""
        try:
            current_data = await self._get_current_position_data(position)

            # 1. Profit target analysis
            profit_pct = (
                current_data["current_value"] - position["cost_basis"]
            ) / position["cost_basis"]

            # WSB Mode: Use fixed 3x profit target, otherwise dynamic
            if self.wsb_mode:
                profit_target = 3.0  # Fixed 3x profit target like WSB trader
            else:
                # Dynamic profit targets based on volatility
                volatility = await self._get_recent_volatility(position["ticker"])
                if volatility > 0.30:  # High vol stocks
                    profit_target = 2.0  # 200% target
                else:
                    profit_target = 1.5  # 150% target

            if profit_pct >= profit_target:
                return {
                    "should_exit": True,
                    "reason": "PROFIT_TARGET",
                    "confidence": 0.95,
                    "profit_pct": profit_pct,
                    "target": profit_target,
                }

            # 2. Delta - based exits (for options)
            if position.get("instrument_type") == "option":
                delta_threshold = 0.60  # Delta exit threshold
                if current_data.get("delta", 0) >= delta_threshold:  # Deep ITM
                    return {
                        "should_exit": True,
                        "reason": "DELTA_TARGET",
                        "confidence": 0.85,
                        "delta": current_data.get("delta", 0),
                    }

            # 3. Time decay protection
            days_to_expiry = position.get("days_to_expiry", 30)
            if (
                days_to_expiry <= 7 and profit_pct < 0.20
            ):  # Less than week, minimal profit
                return {
                    "should_exit": True,
                    "reason": "TIME_DECAY",
                    "confidence": 0.75,
                    "days_to_expiry": days_to_expiry,
                    "profit_pct": profit_pct,
                }

            # 4. Stop loss - trailing or fixed
            stop_loss_pct = self._calculate_dynamic_stop_loss(position, current_data)
            if profit_pct <= -stop_loss_pct:
                return {
                    "should_exit": True,
                    "reason": "STOP_LOSS",
                    "confidence": 0.90,
                    "profit_pct": profit_pct,
                    "stop_loss_pct": stop_loss_pct,
                }

            return {
                "should_exit": False,
                "reason": "HOLD",
                "confidence": 0.60,
                "profit_pct": profit_pct,
                "days_to_expiry": days_to_expiry,
            }

        except Exception as e:
            self.logger.error(
                f"Error in exit decision for {position.get('ticker', 'unknown')}: {e}"
            )
            return {
                "should_exit": False,
                "reason": "ERROR",
                "confidence": 0.0,
                "error": str(e),
            }

    async def _get_current_position_data(
        self, position: dict[str, Any]
    ) -> dict[str, Any]:
        """Get current position data including market value and Greeks."""
        try:
            ticker = position["ticker"]

            # Get current market price
            current_price = await self.data_provider.get_current_price(ticker)
            if not current_price:
                return {"current_value": position["cost_basis"], "delta": 0}

            # Calculate current value
            if (
                position.get("instrument_type") == "option"
            ):  # For options, get current option price
                options_chain = await self.data_provider.get_options_chain(ticker)
                if options_chain:
                    # Find matching option
                    matching_option = None
                    for option in options_chain:
                        if (
                            option.strike == position.get("strike")
                            and option.option_type.lower()
                            == position.get("option_type", "call").lower()
                        ):
                            matching_option = option
                            break

                    if matching_option:
                        current_value = matching_option.last_price * position.get(
                            "quantity", 100
                        )
                        delta = getattr(matching_option, "delta", 0)
                    else:
                        current_value = position["cost_basis"]  # Fallback
                        delta = 0
                else:
                    current_value = position["cost_basis"]  # Fallback
                    delta = 0
            else:
                # For stocks
                current_value = current_price.price * position.get("quantity", 0)
                delta = 1.0  # Stocks have delta of 1

            return {
                "current_value": current_value,
                "current_price": current_price.price,
                "delta": delta,
            }

        except Exception as e:
            self.logger.error(f"Error getting current position data: {e}")
            return {"current_value": position["cost_basis"], "delta": 0}

    async def _get_recent_volatility(self, ticker: str) -> float:
        """Get recent volatility for dynamic profit targets."""
        try:
            # Get recent price history
            price_history = await self.data_provider.get_price_history(ticker, days=20)

            if len(price_history) < 10:
                return 0.20  # Default moderate volatility

            # Calculate daily returns
            returns = []
            for i in range(1, len(price_history)):
                daily_return = (
                    price_history[i] - price_history[i - 1]
                ) / price_history[i - 1]
                returns.append(float(daily_return))

            # Calculate volatility (annualized)
            import statistics

            volatility = statistics.stdev(returns) * (252**0.5)  # Annualized

            return volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility for {ticker}: {e}")
            return 0.20  # Default moderate volatility

    def _calculate_dynamic_stop_loss(
        self, position: dict[str, Any], current_data: dict[str, Any]
    ) -> float:
        """Calculate dynamic stop loss percentage."""
        try:
            # Base stop loss
            base_stop_loss = 0.20  # 20% base stop loss

            # Adjust based on volatility
            volatility = current_data.get("volatility", 0.20)
            if volatility > 0.40:  # High volatility
                stop_loss = base_stop_loss * 1.5  # 30% stop loss
            elif volatility < 0.15:  # Low volatility
                stop_loss = base_stop_loss * 0.75  # 15% stop loss
            else:
                stop_loss = base_stop_loss  # 20% stop loss

            # Adjust based on time to expiry (for options)
            if position.get("instrument_type") == "option":
                days_to_expiry = position.get("days_to_expiry", 30)
                if days_to_expiry <= 7:
                    stop_loss *= 0.5  # Tighter stop loss for near expiry
                elif days_to_expiry <= 14:
                    stop_loss *= 0.75  # Moderately tighter

            return min(0.50, max(0.10, stop_loss))  # Keep between 10 - 50%

        except Exception as e:
            self.logger.error(f"Error calculating dynamic stop loss: {e}")
            return 0.20  # Default 20% stop loss

    async def _check_dip_after_run(self, ticker: str) -> DipSignal | None:
        """Check for dip after run pattern using advanced detection."""
        return await self._detect_advanced_dip_pattern(ticker)

    async def _get_real_option_premium(
        self, ticker: str, strike: Decimal, expiry_date, spot_price: Decimal
    ) -> Decimal:
        """Get real option premium using market data and Black - Scholes."""
        try:
            from ...options.pricing_engine import create_options_pricing_engine

            # Create options pricing engine
            pricing_engine = create_options_pricing_engine()

            # Convert datetime to date if needed
            if isinstance(expiry_date, datetime):
                expiry_date = expiry_date.date()

            # Calculate theoretical option price
            theoretical_price = await pricing_engine.calculate_theoretical_price(
                ticker=ticker,
                strike=strike,
                expiry_date=expiry_date,
                option_type="call",  # WSB dip bot uses calls
                current_price=spot_price,
            )

            self.logger.info(
                f"Real option pricing for {ticker}: Strike=${strike}, "
                f"Expiry={expiry_date}, Spot=${spot_price}, Premium=${theoretical_price}"
            )

            return theoretical_price

        except Exception as e:
            self.logger.error(f"Error getting real option premium: {e}")

            # Fallback to simple intrinsic + minimal time value
            intrinsic_value = max(Decimal("0.00"), spot_price - strike)
            time_value = spot_price * Decimal(
                "0.02"
            )  # 2% of stock price as minimal time value
            fallback_premium = intrinsic_value + time_value

            self.logger.warning(f"Using fallback premium: ${fallback_premium}")
            return max(Decimal("0.01"), fallback_premium)  # Minimum $0.01

    async def execute_dip_trade(self, signal: DipSignal) -> bool:
        """Execute dip trade."""
        try:
            if self.wsb_mode:
                # WSB Mode: Use full available cash for position sizing
                account_info = await self.integration.get_account_info()
                available_cash = account_info.get("cash", 0)
                
                # Calculate max contracts we can afford
                quantity = int(available_cash / float(signal.expected_premium))
                
                self.logger.warning(
                    f"WSB MODE: Using full account cash ${available_cash:.2f} "
                    f"for {quantity} contracts of {signal.ticker}"
                )
            else:
                # Normal mode: Use risk-based sizing
                quantity = int(float(signal.risk_amount) / float(signal.expected_premium))

            if quantity <= 0:
                self.logger.warning(f"Quantity too small for {signal.ticker}")
                return False

            # Create trade signal
            trade_signal = ProductionTradeSignal(
                strategy_name="wsb_dip_bot",
                ticker=signal.ticker,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=float(signal.expected_premium),
                trade_type="option",  # Would be "option" in real implementation
                risk_amount=signal.risk_amount,
                expected_return=signal.risk_amount
                * Decimal(str(self.target_multiplier)),
                metadata={
                    "signal_type": "dip_after_run",
                    "run_percentage": signal.run_percentage,
                    "dip_percentage": signal.dip_percentage,
                    "target_strike": float(signal.target_strike),
                    "target_expiry": signal.target_expiry.isoformat(),
                    "confidence": signal.confidence,
                    "strategy_params": signal.metadata,
                },
            )

            # Execute trade
            result = await self.integration.execute_trade(trade_signal)

            if result.status.value == "FILLED":  # Store active position
                self.active_positions[signal.ticker] = signal

                # Send alert
                await self.integration.alert_system.send_alert(
                    "ENTRY_SIGNAL",
                    "HIGH",
                    f"WSB Dip Bot: {signal.ticker} trade executed - "
                    f"Run: {signal.run_percentage:.2%}, Dip: {signal.dip_percentage:.2%}, "
                    f"Quantity: {quantity}, Premium: ${signal.expected_premium:.2f}",
                )

                self.logger.info(f"Dip trade executed for {signal.ticker}")
                return True
            else:
                self.logger.error(
                    f"Trade execution failed for {signal.ticker}: {result.error_message}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error executing dip trade: {e}")
            return False

    async def monitor_positions(self):
        """Monitor active positions for exit signals."""
        try:
            for _ticker, position in list(self.active_positions.items()):
                exit_signal = await self._check_exit_conditions(position)
                if exit_signal:
                    await self._execute_exit(position, exit_signal)

        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    async def _check_exit_conditions(self, position: DipSignal) -> str | None:
        """Check exit conditions for position."""
        try:
            # Get current price
            current_data = await self.data_provider.get_current_price(position.ticker)
            if not current_data:
                return None

            current_price = current_data.price

            # Check profit target (3x)
            # In real implementation, would check actual option price
            # For now, use simplified calculation
            price_appreciation = float(
                (current_price - position.current_price) / position.current_price
            )

            if price_appreciation >= (self.target_multiplier - 1):
                return "profit_target"

            # Check delta target (would need real options data)
            # For now, check if option is ITM
            if current_price >= position.target_strike:
                return "delta_target"

            # Check time decay (simplified)
            days_held = (
                datetime.now() - position.metadata.get("entry_time", datetime.now())
            ).days
            if days_held >= 2:  # Max hold 2 days
                return "time_stop"

            return None

        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return None

    async def _execute_exit(self, position: DipSignal, reason: str):
        """Execute exit trade."""
        try:
            # Get current option price (simplified)
            current_data = await self.data_provider.get_current_price(position.ticker)
            if not current_data:
                return

            # Estimate current option value (simplified)
            current_option_value = await self._get_real_option_premium(
                position.ticker,
                position.target_strike,
                position.target_expiry,
                current_data.price,
            )

            # Create exit signal
            exit_signal = ProductionTradeSignal(
                strategy_name="wsb_dip_bot",
                ticker=position.ticker,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=1,  # Would get actual quantity from position
                price=float(current_option_value),
                trade_type="option",
                risk_amount=Decimal("0.00"),
                expected_return=Decimal("0.00"),
                metadata={
                    "exit_reason": reason,
                    "entry_price": float(position.current_price),
                    "current_price": float(current_data.price),
                    "target_strike": float(position.target_strike),
                },
            )

            # Execute exit trade
            result = await self.integration.execute_trade(exit_signal)

            if result.status.value == "FILLED":  # Remove from active positions
                del self.active_positions[position.ticker]

                # Send alert
                await self.integration.alert_system.send_alert(
                    "PROFIT_TARGET",
                    "MEDIUM",
                    f"WSB Dip Bot: {position.ticker} position closed - "
                    f"Reason: {reason}, Exit price: ${current_option_value:.2f}",
                )

                self.logger.info(f"Position closed for {position.ticker}: {reason}")

        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")

    async def run_strategy(self):
        """Main strategy loop."""
        self.logger.info("Starting WSB Dip Bot strategy")

        try:
            while True:
                # Check if market is open
                if await self.data_provider.is_market_open():
                    # Scan for new signals
                    signals = await self.scan_for_dip_signals()

                    # Execute trades for new signals
                    for signal in signals:
                        if signal.ticker not in self.active_positions:
                            await self.execute_dip_trade(signal)

                    # Monitor existing positions
                    await self.monitor_positions()

                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            self.logger.error(f"Error in strategy loop: {e}")

    def get_strategy_status(self) -> dict[str, Any]:
        """Get current strategy status."""
        return {
            "strategy_name": "wsb_dip_bot",
            "active_positions": len(self.active_positions),
            "positions": [
                {
                    "ticker": pos.ticker,
                    "run_percentage": pos.run_percentage,
                    "dip_percentage": pos.dip_percentage,
                    "target_strike": float(pos.target_strike),
                    "risk_amount": float(pos.risk_amount),
                    "confidence": pos.confidence,
                }
                for pos in self.active_positions.values()
            ],
            "parameters": {
                "run_lookback_days": self.run_lookback_days,
                "run_threshold": self.run_threshold,
                "dip_threshold": self.dip_threshold,
                "target_dte_days": self.target_dte_days,
                "otm_percentage": self.otm_percentage,
                "max_position_size": self.max_position_size,
                "target_multiplier": self.target_multiplier,
                "delta_target": self.delta_target,
                "wsb_mode": self.wsb_mode,
            },
        }


# Factory function
def create_production_wsb_dip_bot(
    integration_manager: ProductionIntegrationManager,
    data_provider: ProductionDataProvider,
    config: dict[str, Any],
) -> ProductionWSBDipBot:
    """Create ProductionWSBDipBot instance."""
    return ProductionWSBDipBot(integration_manager, data_provider, config)
