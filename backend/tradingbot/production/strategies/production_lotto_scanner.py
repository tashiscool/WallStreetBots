"""Production Lotto Scanner Strategy
0DTE and earnings lotto plays with extreme risk management.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from ...options.pricing_engine import RealOptionsPricingEngine
from ...options.smart_selection import SmartOptionsSelector
from ...risk.real_time_risk_manager import RealTimeRiskManager
from ..core.production_integration import (
    ProductionIntegrationManager,
    ProductionTradeSignal,
)
from ..data.production_data_integration import ReliableDataProvider


@dataclass
class LottoOpportunity:
    """Individual lotto opportunity."""

    ticker: str
    play_type: str  # "0dte", "earnings", "catalyst"
    expiry_date: str
    days_to_expiry: int
    strike: float
    option_type: str  # "call" or "put"
    current_premium: float
    breakeven: float
    current_spot: float
    catalyst_event: str
    expected_move: float
    max_position_size: float
    max_contracts: int
    risk_level: str
    win_probability: float
    potential_return: float
    stop_loss_price: float
    profit_target_price: float
    risk_score: float


class ProductionLottoScanner:
    """Production Lotto Scanner Strategy.

    Extreme high - risk, high - reward plays with strict position sizing:
    - 0DTE options on momentum / catalysts
    - Earnings lotto plays
    - Disciplined 1% max risk per play
    - Quick profit - taking at 3 - 5x returns
    - Hard stop losses at 50%
    """

    def __init__(
        self,
        integration_manager: ProductionIntegrationManager,
        data_provider: ReliableDataProvider,
        config: dict,
    ):
        self.strategy_name = "lotto_scanner"
        self.integration_manager = integration_manager
        self.data_provider = data_provider

        # Configuration
        self.max_risk_pct = config.get("max_risk_pct", 0.01)  # 1% max risk per play
        self.max_concurrent_positions = config.get("max_concurrent_positions", 3)
        self.profit_targets = config.get(
            "profit_targets", [300, 500, 800]
        )  # 3x, 5x, 8x
        self.stop_loss_pct = config.get("stop_loss_pct", 0.50)  # 50% stop loss
        self.min_win_probability = config.get(
            "min_win_probability", 0.15
        )  # 15% minimum
        self.max_dte = config.get("max_dte", 5)  # Max 5 days to expiry

        # Watchlists
        self.high_volume_tickers = config.get(
            "high_volume_tickers",
            ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
        )

        self.meme_tickers = config.get(
            "meme_tickers", ["GME", "AMC", "PLTR", "COIN", "HOOD", "RIVN", "SOFI"]
        )

        self.earnings_tickers = config.get(
            "earnings_tickers",
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
                "AMD",
                "QCOM",
                "UBER",
                "SNOW",
            ],
        )

        # Components
        self.options_selector = SmartOptionsSelector(data_provider)
        self.pricing_engine = RealOptionsPricingEngine()
        self.risk_manager = RealTimeRiskManager()

        # State tracking
        self.active_positions: dict[str, Any] = {}
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_scan_time: datetime | None = None

        # Logging
        self.logger = logging.getLogger(f"production.{self.strategy_name}")
        self.logger.info(
            f"Initialized ProductionLottoScanner with max risk {self.max_risk_pct: .1%}"
        )

    async def run_strategy(self):
        """Main strategy execution loop."""
        self.logger.info("Starting Lotto Scanner production strategy")

        while True:
            try:
                # Check market conditions
                if not await self._should_trade():
                    await asyncio.sleep(300)  # 5 minute wait
                    continue

                # Monitor existing positions first
                await self._monitor_positions()

                # Look for new opportunities if we have capacity
                if len(self.active_positions) < self.max_concurrent_positions:
                    await self._scan_opportunities()

                # Brief pause between cycles
                await asyncio.sleep(60)  # 1 minute between scans

            except Exception as e:
                self.logger.error(f"Strategy error: {e}")
                await asyncio.sleep(300)

    async def _should_trade(self) -> bool:
        """Check if conditions are right for lotto trading."""
        try:
            # Check market hours (lotto plays need active market)
            if not await self.data_provider.is_market_open():
                return False

            # Check portfolio risk
            portfolio_value = await self.integration_manager.get_portfolio_value()
            if portfolio_value < 10000:  # Minimum account size for lotto plays
                self.logger.warning("Account too small for lotto plays")
                return False

            # Check daily loss limits
            if self.daily_pnl < -portfolio_value * 0.05:  # -5% daily limit
                self.logger.warning("Daily loss limit reached")
                return False

            # Check position limits
            if len(self.active_positions) >= self.max_concurrent_positions:
                return False

            # Check time of day (avoid last 30 minutes for new positions)
            now = datetime.now()
            if now.hour == 15 and now.minute >= 30:  # After 3: 30 PM EST
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking trade conditions: {e}")
            return False

    async def _monitor_positions(self):
        """Monitor and manage existing lotto positions."""
        positions_to_close = []

        for position_id, position_info in self.active_positions.items():
            try:
                current_price = await self._get_option_price(
                    position_info["ticker"],
                    position_info["expiry"],
                    position_info["strike"],
                    position_info["option_type"],
                )

                if current_price is None:
                    continue

                entry_price = position_info["entry_price"]
                pnl_pct = (current_price - entry_price) / entry_price

                # Check profit targets (scale out)
                if pnl_pct >= 3.0:  # 300% gain
                    await self._close_position(
                        position_id, "profit_target_300", 0.5
                    )  # Close 50%
                elif pnl_pct >= 5.0:  # 500% gain
                    await self._close_position(
                        position_id, "profit_target_500", 1.0
                    )  # Close all

                # Check stop loss
                elif pnl_pct <= -self.stop_loss_pct:
                    positions_to_close.append((position_id, "stop_loss"))

                # Check time decay (0DTE positions)
                elif position_info["days_to_expiry"] <= 0:
                    # Close 0DTE positions in last hour
                    now = datetime.now()
                    if now.hour >= 15:  # After 3 PM on expiry day
                        positions_to_close.append((position_id, "time_decay"))

                # Update position tracking
                position_info["current_price"] = current_price
                position_info["unrealized_pnl"] = pnl_pct

            except Exception as e:
                self.logger.error(f"Error monitoring position {position_id}: {e}")

        # Close flagged positions
        for position_id, reason in positions_to_close:
            await self._close_position(position_id, reason)

    async def _scan_opportunities(self):
        """Scan for lotto opportunities."""
        try:
            opportunities = []

            # Scan 0DTE opportunities
            zero_dte_opps = await self._scan_0dte_opportunities()
            opportunities.extend(zero_dte_opps)

            # Scan earnings opportunities
            earnings_opps = await self._scan_earnings_opportunities()
            opportunities.extend(earnings_opps)

            # Scan catalyst opportunities (news / events)
            catalyst_opps = await self._scan_catalyst_opportunities()
            opportunities.extend(catalyst_opps)

            if not opportunities:
                return

            # Rank opportunities by risk - adjusted expected value
            opportunities.sort(
                key=lambda x: x.win_probability
                * x.potential_return
                / (1 + x.risk_score),
                reverse=True,
            )

            # Take best opportunity if it meets criteria
            best_opp = opportunities[0]
            if (
                best_opp.win_probability >= self.min_win_probability
                and best_opp.risk_score <= 8.0
            ):  # Max risk score of 8
                await self._execute_lotto_trade(best_opp)

            self.last_scan_time = datetime.now()

        except Exception as e:
            self.logger.error(f"Error scanning opportunities: {e}")

    async def _scan_0dte_opportunities(self) -> list[LottoOpportunity]:
        """Scan for 0DTE opportunities."""
        opportunities = []

        try:
            expiry = await self._get_0dte_expiry()
            if not expiry:
                return opportunities

            for ticker in self.high_volume_tickers:
                try:
                    # Get current price and momentum
                    current_price = await self.data_provider.get_current_price(ticker)
                    if not current_price:
                        continue

                    # Check for momentum / volume spike
                    momentum_score = await self._calculate_momentum_score(ticker)
                    if momentum_score < 60:  # Need strong momentum for 0DTE
                        continue

                    # Get options chain
                    chain = await self.options_selector.get_options_chain(
                        ticker, expiry
                    )
                    if not chain:
                        continue

                    # Calculate expected move
                    expected_move = await self._estimate_intraday_move(ticker)

                    # Look for liquid strikes around momentum levels
                    call_strike = current_price * (1 + expected_move * 0.5)
                    put_strike = current_price * (1 - expected_move * 0.5)

                    for option_type, target_strike in [
                        ("call", call_strike),
                        ("put", put_strike),
                    ]:
                        opp = await self._evaluate_option_opportunity(
                            ticker,
                            expiry,
                            target_strike,
                            option_type,
                            "0dte",
                            f"Intraday momentum ({momentum_score})",
                            expected_move,
                            current_price,
                        )

                        if opp:
                            opportunities.append(opp)

                except Exception as e:
                    self.logger.error(f"Error scanning 0DTE for {ticker}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in 0DTE scan: {e}")

        return opportunities

    async def _scan_earnings_opportunities(self) -> list[LottoOpportunity]:
        """Scan for earnings lotto opportunities."""
        opportunities = []

        try:
            # Get earnings calendar for next 7 days
            earnings_events = await self._get_upcoming_earnings()

            for event in earnings_events:
                try:
                    ticker = event["ticker"]
                    earnings_date = event["date"]
                    expected_move = event.get("expected_move", 0.05)

                    current_price = await self.data_provider.get_current_price(ticker)
                    if not current_price:
                        continue

                    # Find best expiry for earnings play
                    expiry = await self._find_best_earnings_expiry(
                        ticker, earnings_date
                    )
                    if not expiry:
                        continue

                    # Create straddle / strangle opportunities
                    call_strike = current_price * (1 + expected_move)
                    put_strike = current_price * (1 - expected_move)

                    catalyst = f"Earnings {earnings_date.strftime('%m/%d')} {event.get('time', 'AMC')}"

                    for option_type, target_strike in [
                        ("call", call_strike),
                        ("put", put_strike),
                    ]:
                        opp = await self._evaluate_option_opportunity(
                            ticker,
                            expiry,
                            target_strike,
                            option_type,
                            "earnings",
                            catalyst,
                            expected_move,
                            current_price,
                        )

                        if opp:
                            opportunities.append(opp)

                except Exception as e:
                    self.logger.error(
                        f"Error scanning earnings for {event.get('ticker', 'unknown')}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Error in earnings scan: {e}")

        return opportunities

    async def _scan_catalyst_opportunities(self) -> list[LottoOpportunity]:
        """Scan for catalyst - driven opportunities (news, events, etc.)."""
        opportunities = []

        try:
            # Check meme stocks for unusual activity
            for ticker in self.meme_tickers:
                try:
                    # Get volume and price momentum
                    volume_ratio = await self._get_volume_ratio(ticker)
                    price_momentum = await self._get_price_momentum(ticker)

                    if (
                        volume_ratio > 3.0 and abs(price_momentum) > 0.03
                    ):  # 3x volume+3% move
                        current_price = await self.data_provider.get_current_price(
                            ticker
                        )
                        if not current_price:
                            continue

                        # Use weekly expiry for catalyst plays
                        expiry = await self._get_weekly_expiry()
                        expected_move = min(
                            0.20, abs(price_momentum) * 2
                        )  # 2x current move

                        catalyst = f"Unusual activity (V: {volume_ratio:.1f}x, P: {price_momentum:+.1%})"

                        # Directional play based on momentum
                        if price_momentum > 0:
                            target_strike = current_price * (1 + expected_move * 0.7)
                            option_type = "call"
                        else:
                            target_strike = current_price * (1 - expected_move * 0.7)
                            option_type = "put"

                        opp = await self._evaluate_option_opportunity(
                            ticker,
                            expiry,
                            target_strike,
                            option_type,
                            "catalyst",
                            catalyst,
                            expected_move,
                            current_price,
                        )

                        if opp:
                            opportunities.append(opp)

                except Exception as e:
                    self.logger.error(f"Error scanning catalyst for {ticker}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in catalyst scan: {e}")

        return opportunities

    async def _evaluate_option_opportunity(
        self,
        ticker: str,
        expiry: str,
        target_strike: float,
        option_type: str,
        play_type: str,
        catalyst: str,
        expected_move: float,
        current_spot: float,
    ) -> LottoOpportunity | None:
        """Evaluate a specific option opportunity."""
        try:
            # Find best strike near target
            best_option = await self.options_selector.find_best_strike(
                ticker, expiry, target_strike, option_type
            )

            if not best_option:
                return None

            strike = best_option["strike"]
            bid = best_option.get("bid", 0)
            ask = best_option.get("ask", 0)
            volume = best_option.get("volume", 0)
            open_interest = best_option.get("openInterest", 0)

            # Liquidity filters
            if bid <= 0.05 or ask <= 0.05 or ask - bid > bid * 0.5:  # Wide spreads
                return None

            if volume < 50 or open_interest < 100:  # Minimum liquidity
                return None

            mid_price = (bid + ask) / 2

            # Position sizing
            portfolio_value = await self.integration_manager.get_portfolio_value()
            max_dollar_risk = portfolio_value * self.max_risk_pct
            max_contracts = int(max_dollar_risk / (mid_price * 100))
            max_contracts = min(max_contracts, 10)  # Hard cap

            if max_contracts <= 0:
                return None

            # Calculate metrics
            days_to_expiry = (
                datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()
            ).days

            if option_type == "call":
                breakeven = strike + mid_price
                distance_to_breakeven = (breakeven - current_spot) / current_spot
            else:
                breakeven = strike - mid_price
                distance_to_breakeven = (current_spot - breakeven) / current_spot

            # Win probability estimation (very rough)
            base_prob = 0.35 if play_type == "0dte" else 0.25
            win_probability = max(0.10, base_prob - distance_to_breakeven * 8)

            # Risk scoring (higher=riskier)
            risk_score = (
                (distance_to_breakeven * 10)  # Distance to breakeven
                + (max(0, 5 - days_to_expiry))  # Time decay risk
                + (10 - min(10, volume / 20))  # Liquidity risk
                + (2 if play_type == "0dte" else 0)  # 0DTE extra risk
            )

            # Potential return (target 5x)
            potential_return = 5.0
            profit_target_price = mid_price * (1 + potential_return)
            stop_loss_price = mid_price * (1 - self.stop_loss_pct)

            return LottoOpportunity(
                ticker=ticker,
                play_type=play_type,
                expiry_date=expiry,
                days_to_expiry=days_to_expiry,
                strike=strike,
                option_type=option_type,
                current_premium=mid_price,
                breakeven=breakeven,
                current_spot=current_spot,
                catalyst_event=catalyst,
                expected_move=expected_move,
                max_position_size=max_dollar_risk,
                max_contracts=max_contracts,
                risk_level="extreme" if risk_score > 7 else "very_high",
                win_probability=win_probability,
                potential_return=potential_return,
                stop_loss_price=stop_loss_price,
                profit_target_price=profit_target_price,
                risk_score=risk_score,
            )

        except Exception as e:
            self.logger.error(f"Error evaluating option opportunity: {e}")
            return None

    async def _execute_lotto_trade(self, opportunity: LottoOpportunity):
        """Execute a lotto trade."""
        try:
            # Create trade signal
            signal = ProductionTradeSignal(
                strategy_name=self.strategy_name,
                ticker=opportunity.ticker,
                signal_type="LOTTO_BUY",
                confidence_score=opportunity.win_probability,
                entry_price=opportunity.current_premium,
                stop_loss=opportunity.stop_loss_price,
                take_profit=opportunity.profit_target_price,
                position_size=opportunity.max_contracts,
                expiry_date=opportunity.expiry_date,
                strike_price=opportunity.strike,
                option_type=opportunity.option_type.upper(),
                reasoning=f"Lotto {opportunity.play_type}: {opportunity.catalyst_event}",
            )

            # Execute trade
            success = await self.integration_manager.execute_trade_signal(signal)

            if success:
                # Track position
                position_id = f"{opportunity.ticker}_{opportunity.expiry_date}_{opportunity.strike}_{opportunity.option_type}"
                self.active_positions[position_id] = {
                    "ticker": opportunity.ticker,
                    "expiry": opportunity.expiry_date,
                    "strike": opportunity.strike,
                    "option_type": opportunity.option_type,
                    "entry_price": opportunity.current_premium,
                    "entry_time": datetime.now(),
                    "contracts": opportunity.max_contracts,
                    "play_type": opportunity.play_type,
                    "catalyst": opportunity.catalyst_event,
                    "days_to_expiry": opportunity.days_to_expiry,
                }

                self.trade_count += 1

                self.logger.info(
                    f"Executed lotto trade: {opportunity.ticker} "
                    f"{opportunity.strike} {opportunity.option_type} "
                    f"@ ${opportunity.current_premium: .2f}"
                )

                # Send alert
                await self.integration_manager.send_alert(
                    f"🎰 LOTTO TRADE: {opportunity.ticker} "
                    f"{opportunity.strike} {opportunity.option_type.upper()} "
                    f"@ ${opportunity.current_premium: .2f} "
                    f"({opportunity.play_type}: {opportunity.catalyst_event})"
                )

        except Exception as e:
            self.logger.error(f"Error executing lotto trade: {e}")

    async def _close_position(
        self, position_id: str, reason: str, close_pct: float = 1.0
    ):
        """Close a lotto position."""
        try:
            if position_id not in self.active_positions:
                return

            position = self.active_positions[position_id]

            # Create close signal
            signal = ProductionTradeSignal(
                strategy_name=self.strategy_name,
                ticker=position["ticker"],
                signal_type="LOTTO_CLOSE",
                confidence_score=0.8,
                position_size=int(position["contracts"] * close_pct),
                expiry_date=position["expiry"],
                strike_price=position["strike"],
                option_type=position["option_type"].upper(),
                reasoning=f"Close lotto position: {reason}",
            )

            success = await self.integration_manager.execute_trade_signal(signal)

            if success:
                if close_pct >= 1.0:
                    # Remove position completely
                    del self.active_positions[position_id]
                else:
                    # Reduce position size
                    position["contracts"] = int(position["contracts"] * (1 - close_pct))

                self.logger.info(
                    f"Closed {close_pct: .0%} of lotto position {position_id}: {reason}"
                )

                # Send alert
                await self.integration_manager.send_alert(
                    f"🎰 CLOSED {close_pct: .0%}: {position['ticker']} "
                    f"{position['strike']} {position['option_type'].upper()} "
                    f"({reason})"
                )

        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")

    # Helper methods
    async def _get_0dte_expiry(self) -> str | None:
        """Get 0DTE expiry if available."""
        today = date.today()
        weekday = today.weekday()  # 0=Monday, 4 = Friday

        # Check if today has 0DTE options (Mon / Wed / Fri for major indices)
        if weekday in [0, 2, 4]:
            return today.strftime("%Y-%m-%d")

        return None

    async def _get_weekly_expiry(self) -> str:
        """Get next weekly expiry (Friday)."""
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7

        friday = today + timedelta(days=days_until_friday)
        return friday.strftime("%Y-%m-%d")

    async def _calculate_momentum_score(self, ticker: str) -> float:
        """Calculate momentum score (0 - 100)."""
        try:
            # Get recent price data
            prices = await self.data_provider.get_recent_prices(ticker, periods=20)
            if not prices or len(prices) < 10:
                return 0

            # Calculate various momentum indicators
            current = prices[-1]
            sma_5 = sum(prices[-5:]) / 5
            sma_20 = sum(prices[-20:]) / 20

            # Price vs moving averages
            ma_score = 0
            if current > sma_5 > sma_20:
                ma_score = 40
            elif current > sma_5:
                ma_score = 25

            # Recent price change
            change_1d = (current - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
            change_score = min(30, abs(change_1d) * 1000)  # Cap at 30 points

            # Volume momentum (simplified)
            volume_score = 30  # Placeholder - would use actual volume data

            return min(100, ma_score + change_score + volume_score)

        except Exception:
            return 0

    async def _estimate_intraday_move(self, ticker: str) -> float:
        """Estimate expected intraday move."""
        try:
            # Get recent volatility
            prices = await self.data_provider.get_recent_prices(ticker, periods=20)
            if not prices or len(prices) < 10:
                return 0.02  # 2% default

            # Calculate daily volatility
            returns = [
                (prices[i] - prices[i - 1]) / prices[i - 1]
                for i in range(1, len(prices))
            ]
            daily_vol = (sum(r * r for r in returns) / len(returns)) ** 0.5

            # Scale for intraday (roughly 0.6x daily)
            intraday_move = daily_vol * 0.6

            return min(0.15, max(0.01, intraday_move))  # Cap between 1 - 15%

        except Exception:
            return 0.03  # 3% fallback

    async def _get_upcoming_earnings(self) -> list[dict]:
        """Get upcoming earnings events (mock implementation)."""
        # In production, this would use a real earnings calendar API
        today = date.today()

        mock_earnings = [
            {
                "ticker": "AAPL",
                "date": today + timedelta(days=3),
                "expected_move": 0.04,
                "time": "AMC",
            },
            {
                "ticker": "GOOGL",
                "date": today + timedelta(days=5),
                "expected_move": 0.05,
                "time": "AMC",
            },
            {
                "ticker": "TSLA",
                "date": today + timedelta(days=2),
                "expected_move": 0.08,
                "time": "AMC",
            },
        ]

        return [e for e in mock_earnings if e["date"] >= today]

    async def _find_best_earnings_expiry(
        self, ticker: str, earnings_date: date
    ) -> str | None:
        """Find best expiry for earnings play."""
        # Look for expiry within 3 days after earnings
        earnings_date + timedelta(days=1)

        # In production, would query actual available expiries
        # For now, return weekly expiry
        return await self._get_weekly_expiry()

    async def _get_volume_ratio(self, ticker: str) -> float:
        """Get current volume vs average ratio."""
        # Mock implementation - would use real volume data
        return 2.5  # Placeholder

    async def _get_price_momentum(self, ticker: str) -> float:
        """Get current price momentum."""
        try:
            prices = await self.data_provider.get_recent_prices(ticker, periods=2)
            if not prices or len(prices) < 2:
                return 0

            return (prices[-1] - prices[-2]) / prices[-2]
        except Exception:
            return 0

    async def _get_option_price(
        self, ticker: str, expiry: str, strike: float, option_type: str
    ) -> float | None:
        """Get current option price."""
        try:
            chain = await self.options_selector.get_options_chain(ticker, expiry)
            if not chain:
                return None

            options_list = (
                chain.get("calls", [])
                if option_type == "call"
                else chain.get("puts", [])
            )

            for option in options_list:
                if abs(option.get("strike", 0) - strike) < 0.01:
                    bid = option.get("bid", 0)
                    ask = option.get("ask", 0)
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2

            return None

        except Exception:
            return None


def create_production_lotto_scanner(
    integration_manager: ProductionIntegrationManager,
    data_provider: ReliableDataProvider,
    config: dict,
) -> ProductionLottoScanner:
    """Factory function to create ProductionLottoScanner."""
    return ProductionLottoScanner(integration_manager, data_provider, config)
