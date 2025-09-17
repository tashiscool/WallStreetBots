#!/usr / bin / env python3
"""Production LEAPS Tracker Strategy
Long - term positions on secular growth trends with systematic profit - taking.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from ...options.pricing_engine import BlackScholesEngine
from ...options.smart_selection import SmartOptionsSelector
from ...risk.managers.real_time_risk_manager import RealTimeRiskManager
from ...production.core.production_integration import ProductionTradeSignal
from ...production.data.production_data_integration import ReliableDataProvider


@dataclass
class SecularTrend:
    """Secular growth trend definition."""

    theme: str
    description: str
    tickers: list[str]
    growth_drivers: list[str]
    time_horizon: str


@dataclass
class MovingAverageCross:
    """Moving average cross analysis."""

    cross_type: str  # "golden_cross", "death_cross", "neutral"
    cross_date: date | None
    days_since_cross: int | None
    sma_50: float
    sma_200: float
    price_above_50sma: bool
    price_above_200sma: bool
    cross_strength: float  # 0 - 100
    trend_direction: str  # "bullish", "bearish", "sideways"


@dataclass
class LEAPSCandidate:
    """LEAPS candidate with comprehensive scoring."""

    ticker: str
    company_name: str
    theme: str
    current_price: float
    trend_score: float
    financial_score: float
    momentum_score: float
    valuation_score: float
    composite_score: float
    expiry_date: str
    recommended_strike: float
    premium_estimate: float
    break_even: float
    target_return_1y: float
    target_return_3y: float
    risk_factors: list[str]
    ma_cross_signal: MovingAverageCross
    entry_timing_score: float
    exit_timing_score: float


class ProductionLEAPSTracker:
    """Production LEAPS Tracker Strategy.

    Strategy Logic:
    1. Identifies secular growth trends and themes
    2. Analyzes technical timing using golden / death cross signals
    3. Targets LEAPS options (12+ months) on strongest candidates
    4. Implements systematic profit - taking at 2x, 3x, 4x levels
    5. Uses moving average signals for entry / exit timing
    6. Focuses on diversification across multiple themes

    Risk Management:
    - Maximum 30% account allocation to LEAPS
    - Maximum 10% per individual LEAPS position
    - Maximum 5 concurrent LEAPS positions
    - Stop loss at -50% to preserve capital
    - Time-based exits before expiration year
    - Theme diversification requirements
    """

    def __init__(
        self, integration_manager, data_provider: ReliableDataProvider, config: dict
    ):
        self.strategy_name = "leaps_tracker"
        self.integration_manager = integration_manager
        self.data_provider = data_provider
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.options_selector = SmartOptionsSelector(data_provider)
        self.risk_manager = RealTimeRiskManager()
        self.bs_engine = BlackScholesEngine()

        # Strategy configuration
        self.secular_themes = config.get(
            "secular_themes",
            {
                "ai_revolution": SecularTrend(
                    theme="AI Revolution",
                    description="Artificial intelligence transforming industries",
                    tickers=[
                        "NVDA",
                        "AMD",
                        "GOOGL",
                        "MSFT",
                        "META",
                        "ORCL",
                        "CRM",
                        "SNOW",
                    ],
                    growth_drivers=[
                        "GPU compute",
                        "Cloud AI",
                        "Enterprise adoption",
                        "Consumer AI",
                    ],
                    time_horizon="5 - 10 years",
                ),
                "cloud_transformation": SecularTrend(
                    theme="Cloud Transformation",
                    description="Enterprise digital transformation",
                    tickers=[
                        "MSFT",
                        "AMZN",
                        "GOOGL",
                        "CRM",
                        "SNOW",
                        "DDOG",
                        "NET",
                        "OKTA",
                    ],
                    growth_drivers=[
                        "Remote work",
                        "Digital transformation",
                        "Data analytics",
                        "Security",
                    ],
                    time_horizon="3 - 7 years",
                ),
                "electric_mobility": SecularTrend(
                    theme="Electric Mobility",
                    description="Transportation electrification",
                    tickers=["TSLA", "RIVN", "LCID", "NIO", "XPEV", "GM", "F"],
                    growth_drivers=[
                        "Battery tech",
                        "Charging infrastructure",
                        "Regulation",
                        "Cost parity",
                    ],
                    time_horizon="5 - 15 years",
                ),
                "cybersecurity": SecularTrend(
                    theme="Cybersecurity",
                    description="Digital security imperative",
                    tickers=["CRWD", "ZS", "PANW", "OKTA", "NET", "S", "FTNT"],
                    growth_drivers=[
                        "Remote work security",
                        "Cloud security",
                        "Compliance",
                        "Threat landscape",
                    ],
                    time_horizon="5 - 10 years",
                ),
            },
        )

        # Risk parameters
        self.max_positions = config.get("max_positions", 5)
        self.max_position_size = config.get("max_position_size", 0.10)  # 10% per LEAPS
        self.max_total_allocation = config.get(
            "max_total_allocation", 0.30
        )  # 30% total
        self.min_dte = config.get("min_dte", 365)  # Minimum 1 year
        self.max_dte = config.get("max_dte", 730)  # Maximum 2 years

        # Scoring thresholds
        self.min_composite_score = config.get("min_composite_score", 60)
        self.min_entry_timing_score = config.get("min_entry_timing_score", 50)
        self.max_exit_timing_score = config.get("max_exit_timing_score", 70)

        # Profit taking levels
        self.profit_levels = config.get(
            "profit_levels", [100, 200, 300, 400]
        )  # 2x, 3x, 4x, 5x
        self.scale_out_percentage = config.get(
            "scale_out_percentage", 25
        )  # 25% each level

        # Exit criteria
        self.stop_loss = config.get("stop_loss", 0.50)  # 50% loss
        self.time_exit_dte = config.get(
            "time_exit_dte", 90
        )  # Exit 3 months before expiry

        # Active positions tracking
        self.active_positions: list[dict[str, Any]] = []

        self.logger.info("ProductionLEAPSTracker strategy initialized")

    async def analyze_moving_average_cross(self, ticker: str) -> MovingAverageCross:
        """Analyze golden cross / death cross signals."""
        try:
            # Get 1 year of historical data
            hist_data = await self.data_provider.get_historical_data(ticker, "1y")

            if hist_data.empty or len(hist_data) < 250:
                return MovingAverageCross(
                    cross_type="neutral",
                    cross_date=None,
                    days_since_cross=None,
                    sma_50=0.0,
                    sma_200=0.0,
                    price_above_50sma=False,
                    price_above_200sma=False,
                    cross_strength=0.0,
                    trend_direction="sideways",
                )

            prices = hist_data["close"].values
            current_price = prices[-1]

            # Calculate moving averages
            sma_50 = prices[-50:].mean()
            sma_200 = prices[-200:].mean()

            # Check current position relative to MAs
            price_above_50 = current_price > sma_50
            price_above_200 = current_price > sma_200

            # Calculate historical SMAs to find crosses
            sma_50_series = [
                prices[max(0, i - 49) : i + 1].mean() for i in range(49, len(prices))
            ]
            sma_200_series = [
                prices[max(0, i - 199) : i + 1].mean() for i in range(199, len(prices))
            ]

            if len(sma_50_series) < 50 or len(sma_200_series) < 50:
                return MovingAverageCross(
                    cross_type="neutral",
                    cross_date=None,
                    days_since_cross=None,
                    sma_50=sma_50,
                    sma_200=sma_200,
                    price_above_50sma=price_above_50,
                    price_above_200sma=price_above_200,
                    cross_strength=0.0,
                    trend_direction="sideways",
                )

            # Find most recent cross in last 120 days
            cross_type = "neutral"
            cross_date = None
            days_since_cross = None
            cross_strength = 0.0

            lookback_days = min(120, len(sma_50_series) - 1)
            recent_50 = sma_50_series[-lookback_days:]
            recent_200 = sma_200_series[-lookback_days:]

            # Find crossovers
            for i in range(1, len(recent_50)):
                prev_50 = recent_50[i - 1]
                prev_200 = recent_200[i - 1]
                curr_50 = recent_50[i]
                curr_200 = recent_200[i]

                # Golden cross: 50 SMA crosses above 200 SMA
                if prev_50 <= prev_200 and curr_50 > curr_200:
                    cross_type = "golden_cross"
                    days_ago = len(recent_50) - i - 1
                    cross_date = date.today() - timedelta(days=days_ago)
                    days_since_cross = days_ago
                    separation = abs(curr_50 - curr_200) / curr_200
                    cross_strength = min(100, separation * 1000)

                # Death cross: 50 SMA crosses below 200 SMA
                elif prev_50 >= prev_200 and curr_50 < curr_200:
                    cross_type = "death_cross"
                    days_ago = len(recent_50) - i - 1
                    cross_date = date.today() - timedelta(days=days_ago)
                    days_since_cross = days_ago
                    separation = abs(curr_50 - curr_200) / curr_200
                    cross_strength = min(100, separation * 1000)

            # Determine trend direction
            if sma_50 > sma_200 and price_above_50 and price_above_200:
                trend_direction = "bullish"
            elif sma_50 < sma_200 and not price_above_50 and not price_above_200:
                trend_direction = "bearish"
            else:
                trend_direction = "sideways"

            return MovingAverageCross(
                cross_type=cross_type,
                cross_date=cross_date,
                days_since_cross=days_since_cross,
                sma_50=sma_50,
                sma_200=sma_200,
                price_above_50sma=price_above_50,
                price_above_200sma=price_above_200,
                cross_strength=cross_strength,
                trend_direction=trend_direction,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing MA cross for {ticker}: {e}")
            return MovingAverageCross(
                cross_type="neutral",
                cross_date=None,
                days_since_cross=None,
                sma_50=0.0,
                sma_200=0.0,
                price_above_50sma=False,
                price_above_200sma=False,
                cross_strength=0.0,
                trend_direction="sideways",
            )

    def calculate_entry_exit_timing_scores(
        self, ma_cross: MovingAverageCross, current_price: float
    ) -> tuple[float, float]:
        """Calculate entry and exit timing scores based on MA cross analysis."""
        entry_score = 50.0  # Default neutral
        exit_score = 50.0  # Default neutral

        if (
            ma_cross.cross_type == "golden_cross"
        ):  # Golden cross scenarios - good for LEAPS entries
            if ma_cross.days_since_cross is not None:
                if ma_cross.days_since_cross <= 30:
                    # Recent golden cross - excellent entry timing
                    entry_score = 85.0 + ma_cross.cross_strength * 0.15
                    exit_score = 20.0  # Don't exit on golden cross
                elif ma_cross.days_since_cross <= 60:
                    # Still good entry window
                    entry_score = 75.0 + ma_cross.cross_strength * 0.10
                    exit_score = 25.0
                else:
                    # Golden cross getting old but still positive
                    entry_score = 65.0
                    exit_score = 35.0

        elif (
            ma_cross.cross_type == "death_cross"
        ):  # Death cross scenarios - poor for LEAPS entries
            if ma_cross.days_since_cross is not None:
                if ma_cross.days_since_cross <= 30:
                    # Recent death cross - avoid new entries
                    entry_score = 15.0
                    exit_score = 90.0  # Strong exit signal
                elif ma_cross.days_since_cross <= 90:
                    # Death cross period - still cautious
                    entry_score = 25.0
                    exit_score = 75.0
                else:
                    # Death cross effects may be waning
                    entry_score = 40.0
                    exit_score = 60.0

        # Neutral / sideways - use MA position for guidance
        elif ma_cross.price_above_50sma and ma_cross.price_above_200sma:
            if ma_cross.sma_50 > ma_cross.sma_200:
                # Bullish setup without recent cross
                entry_score = 65.0
                exit_score = 35.0
            else:
                # Mixed signals
                entry_score = 55.0
                exit_score = 45.0
        else:
            # Price below key MAs - cautious
            entry_score = 35.0
            exit_score = 65.0

        # Adjust for trend strength
        if ma_cross.trend_direction == "bullish":
            entry_score = min(95.0, entry_score + 5.0)
            exit_score = max(5.0, exit_score - 5.0)
        elif ma_cross.trend_direction == "bearish":
            entry_score = max(5.0, entry_score - 10.0)
            exit_score = min(95.0, exit_score + 10.0)

        return entry_score, exit_score

    async def calculate_comprehensive_score(
        self, ticker: str
    ) -> tuple[float, float, float, float]:
        """Calculate comprehensive scoring for LEAPS candidate."""
        try:
            # Get 2 years of data for comprehensive analysis
            hist_data = await self.data_provider.get_historical_data(ticker, "2y")
            if hist_data.empty or len(hist_data) < 250:
                return 50.0, 50.0, 50.0, 50.0

            prices = hist_data["close"].values
            volumes = hist_data["volume"].values
            current = prices[-1]

            # 1. Momentum Score
            returns = {}
            if len(prices) > 21:
                returns["1m"] = (current / prices[-21] - 1) * 100
            if len(prices) > 63:
                returns["3m"] = (current / prices[-63] - 1) * 100
            if len(prices) > 126:
                returns["6m"] = (current / prices[-126] - 1) * 100
            if len(prices) > 252:
                returns["1y"] = (current / prices[-252] - 1) * 100

            returns["2y"] = (current / prices[0] - 1) * 100 if len(prices) > 0 else 0

            # Weight longer - term returns for LEAPS
            momentum_raw = 0
            total_weight = 0
            if "1m" in returns:
                momentum_raw += returns["1m"] * 0.1
                total_weight += 0.1
            if "3m" in returns:
                momentum_raw += returns["3m"] * 0.2
                total_weight += 0.2
            if "6m" in returns:
                momentum_raw += returns["6m"] * 0.3
                total_weight += 0.3
            if "1y" in returns:
                momentum_raw += returns["1y"] * 0.4
                total_weight += 0.4

            if total_weight > 0:
                momentum_raw /= total_weight

            momentum_score = max(0, min(100, 50 + momentum_raw))

            # 2. Trend Score
            sma_20 = prices[-20:].mean()
            sma_50 = prices[-50:].mean()
            sma_200 = prices[-200:].mean()

            trend_score = 0
            if current > sma_20 > sma_50 > sma_200:
                trend_score = 100
            elif current > sma_20 > sma_50:
                trend_score = 75
            elif current > sma_20:
                trend_score = 50
            else:
                trend_score = 25

            # 3. Financial Score (simplified)
            financial_score = 50.0  # Default for production
            try:
                # Get some basic financial metrics if available
                current_price_data = await self.data_provider.get_current_price(ticker)
                if current_price_data:
                    # Use price momentum and volume as proxy for financial strength
                    vol_trend = (
                        volumes[-30:].mean() / volumes[-60:-30].mean()
                        if len(volumes) > 60
                        else 1.0
                    )
                    financial_score = min(100, max(0, 50 + vol_trend * 25))
            except Exception:
                pass

            # 4. Valuation Score (simplified - higher prices get lower scores)
            try:
                # Use price relative to moving averages as valuation proxy
                price_to_sma200 = current / sma_200
                if price_to_sma200 < 1.2:
                    valuation_score = 80
                elif price_to_sma200 < 1.5:
                    valuation_score = 60
                elif price_to_sma200 < 2.0:
                    valuation_score = 40
                else:
                    valuation_score = 20
            except Exception:
                valuation_score = 50

            return momentum_score, trend_score, financial_score, valuation_score

        except Exception as e:
            self.logger.error(f"Error calculating scores for {ticker}: {e}")
            return 50.0, 50.0, 50.0, 50.0

    async def get_leaps_expiries(self, ticker: str) -> list[str]:
        """Get LEAPS expiries (12+ months out)."""
        try:
            expiries = await self.data_provider.get_option_expiries(ticker)
            if not expiries:
                return []

            leaps_expiries = []
            today = date.today()

            for exp_str in expiries:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                    days_out = (exp_date - today).days

                    # LEAPS: within our DTE range and prefer January expiries
                    if self.min_dte <= days_out <= self.max_dte:
                        leaps_expiries.append(exp_str)

                except Exception:
                    continue

            return sorted(leaps_expiries)[:3]  # Top 3 LEAPS dates

        except Exception:
            return []

    async def estimate_leaps_premium(
        self, ticker: str, strike: float, expiry: str
    ) -> float:
        """Estimate LEAPS premium."""
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

            # Fallback: rough estimate using Black - Scholes
            current_price = await self.data_provider.get_current_price(ticker)
            if not current_price:
                return 10.0

            days_to_exp = (
                datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()
            ).days
            time_to_expiry = days_to_exp / 365.0

            # Estimate volatility and use BS pricing
            volatility = 0.30  # Default volatility assumption for LEAPS
            risk_free_rate = 0.04

            premium = self.bs_engine.calculate_option_price(
                current_price,
                strike,
                time_to_expiry,
                risk_free_rate,
                volatility,
                "call",
            )

            return max(5.0, premium)

        except Exception as e:
            self.logger.error(f"Error estimating premium for {ticker}: {e}")
            return 10.0

    async def scan_leaps_candidates(self) -> list[LEAPSCandidate]:
        """Scan all themes for LEAPS candidates."""
        candidates = []

        self.logger.info("Scanning secular growth themes for LEAPS opportunities")

        for theme in self.secular_themes.values():
            self.logger.info(f"Analyzing {theme.theme}")

            for ticker in theme.tickers:
                try:
                    # Skip if we already have a position in this ticker
                    if any(pos["ticker"] == ticker for pos in self.active_positions):
                        continue

                    # Get current price
                    current_price = await self.data_provider.get_current_price(ticker)
                    if not current_price:
                        continue

                    # Calculate comprehensive scores
                    (
                        momentum_score,
                        trend_score,
                        financial_score,
                        valuation_score,
                    ) = await self.calculate_comprehensive_score(ticker)

                    # Analyze golden / death cross signals
                    ma_cross_signal = await self.analyze_moving_average_cross(ticker)
                    entry_timing_score, exit_timing_score = (
                        self.calculate_entry_exit_timing_scores(
                            ma_cross_signal, current_price
                        )
                    )

                    # Enhanced composite score including timing
                    composite_score = (
                        trend_score * 0.25
                        + momentum_score * 0.20
                        + financial_score * 0.20
                        + valuation_score * 0.15
                        + entry_timing_score
                        * 0.20  # Weight timing significantly for LEAPS
                    )

                    # Skip if composite score too low
                    if composite_score < self.min_composite_score:
                        continue

                    # Skip if poor entry timing
                    if entry_timing_score < self.min_entry_timing_score:
                        continue

                    # Skip if strong exit signals
                    if exit_timing_score > self.max_exit_timing_score:
                        continue

                    # Get LEAPS expiries
                    leaps_expiries = await self.get_leaps_expiries(ticker)
                    if not leaps_expiries:
                        continue

                    # Target strike: 10 - 20% OTM for growth names
                    target_strike = round(current_price * 1.15, 1)

                    # Use nearest LEAPS expiry
                    expiry = leaps_expiries[0]
                    premium = await self.estimate_leaps_premium(
                        ticker, target_strike, expiry
                    )
                    breakeven = target_strike + premium

                    # Return targets
                    target_1y = ((target_strike * 1.3) / current_price - 1) * 100
                    target_3y = ((target_strike * 2.0) / current_price - 1) * 100

                    # Risk factors
                    risk_factors = []
                    if valuation_score < 30:
                        risk_factors.append("High valuation")
                    if momentum_score < 40:
                        risk_factors.append("Weak momentum")
                    if financial_score < 40:
                        risk_factors.append("Financial concerns")
                    if premium > current_price * 0.25:
                        risk_factors.append("High premium cost")
                    if (
                        ma_cross_signal.cross_type == "death_cross"
                        and ma_cross_signal.days_since_cross
                        and ma_cross_signal.days_since_cross < 30
                    ):
                        risk_factors.append("Recent death cross")

                    candidate = LEAPSCandidate(
                        ticker=ticker,
                        company_name=ticker,  # Simplified for production
                        theme=theme.theme,
                        current_price=current_price,
                        trend_score=trend_score,
                        financial_score=financial_score,
                        momentum_score=momentum_score,
                        valuation_score=valuation_score,
                        composite_score=composite_score,
                        expiry_date=expiry,
                        recommended_strike=target_strike,
                        premium_estimate=premium,
                        break_even=breakeven,
                        target_return_1y=target_1y,
                        target_return_3y=target_3y,
                        risk_factors=risk_factors,
                        ma_cross_signal=ma_cross_signal,
                        entry_timing_score=entry_timing_score,
                        exit_timing_score=exit_timing_score,
                    )

                    candidates.append(candidate)
                    self.logger.info(
                        f"LEAPS candidate: {ticker} Score: {composite_score:.0f}"
                    )

                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {e}")
                    continue

        # Sort by composite score
        candidates.sort(key=lambda x: x.composite_score, reverse=True)
        return candidates

    async def execute_leaps_trade(self, candidate: LEAPSCandidate) -> bool:
        """Execute LEAPS trade."""
        try:
            # Check if we can add more positions
            if len(self.active_positions) >= self.max_positions:
                self.logger.info("Max LEAPS positions reached, skipping trade")
                return False

            # Check total allocation limit
            portfolio_value = await self.integration_manager.get_portfolio_value()
            current_leaps_allocation = (
                sum(pos["cost_basis"] for pos in self.active_positions)
                / portfolio_value
                if portfolio_value > 0
                else 0
            )

            if current_leaps_allocation >= self.max_total_allocation:
                self.logger.info("Max LEAPS allocation reached, skipping trade")
                return False

            # Calculate position size
            max_position_value = portfolio_value * self.max_position_size
            contracts = max(
                1, int(max_position_value / (candidate.premium_estimate * 100))
            )
            contracts = min(contracts, 3)  # Max 3 contracts per LEAPS

            # Create trade signal
            trade_signal = ProductionTradeSignal(
                symbol=candidate.ticker,
                action="BUY",
                quantity=contracts,
                option_type="CALL",
                strike_price=Decimal(str(candidate.recommended_strike)),
                expiration_date=datetime.strptime(
                    candidate.expiry_date, "%Y-%m-%d"
                ).date(),
                premium=Decimal(str(candidate.premium_estimate)),
                confidence=candidate.composite_score / 100.0,
                strategy_name=self.strategy_name,
                signal_strength=candidate.entry_timing_score / 100.0,
                metadata={
                    "leaps_type": "secular_growth",
                    "theme": candidate.theme,
                    "composite_score": candidate.composite_score,
                    "entry_timing_score": candidate.entry_timing_score,
                    "ma_cross_type": candidate.ma_cross_signal.cross_type,
                    "target_return_1y": candidate.target_return_1y,
                    "target_return_3y": candidate.target_return_3y,
                    "breakeven": candidate.break_even,
                    "risk_factors": candidate.risk_factors,
                },
            )

            # Execute the trade
            success = await self.integration_manager.execute_trade_signal(trade_signal)

            if success:
                # Track position
                position = {
                    "ticker": candidate.ticker,
                    "theme": candidate.theme,
                    "trade_signal": trade_signal,
                    "entry_time": datetime.now(),
                    "entry_premium": candidate.premium_estimate,
                    "entry_price": candidate.current_price,
                    "contracts": contracts,
                    "cost_basis": candidate.premium_estimate * contracts * 100,
                    "breakeven": candidate.break_even,
                    "scale_out_level": 0,  # Track profit - taking levels
                    "profit_levels": self.profit_levels.copy(),
                    "expiry_date": datetime.strptime(
                        candidate.expiry_date, "%Y-%m-%d"
                    ).date(),
                    "ma_cross_type": candidate.ma_cross_signal.cross_type,
                }

                self.active_positions.append(position)

                await self.integration_manager.alert_system.send_alert(
                    "LEAPS_ENTRY",
                    "LOW",  # LEAPS are long - term, lower urgency
                    f"LEAPS Entry: {candidate.ticker} {candidate.theme} "
                    f"${candidate.recommended_strike} call {candidate.expiry_date} "
                    f"{contracts} contracts @ ${candidate.premium_estimate: .2f}",
                )

                self.logger.info(f"LEAPS trade executed: {candidate.ticker}")
                return True

            return False

        except Exception as e:
            self.logger.error(
                f"Error executing LEAPS trade for {candidate.ticker}: {e}"
            )
            return False

    async def manage_positions(self):
        """Manage existing LEAPS positions."""
        positions_to_remove = []

        for i, position in enumerate(self.active_positions):
            try:
                ticker = position["ticker"]
                contracts = position["contracts"]
                entry_premium = position["entry_premium"]

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
                current_value = current_premium * contracts * 100
                pnl = current_value - position["cost_basis"]
                pnl_pct = (
                    (pnl / position["cost_basis"]) * 100
                    if position["cost_basis"] > 0
                    else 0
                )

                # Check exit conditions
                should_exit = False
                should_scale_out = False
                exit_reason = ""
                scale_out_percentage = 0

                # Check profit - taking levels
                for level_idx, profit_level in enumerate(position["profit_levels"]):
                    if (
                        pnl_pct >= profit_level
                        and position["scale_out_level"] <= level_idx
                    ):
                        should_scale_out = True
                        scale_out_percentage = self.scale_out_percentage
                        position["scale_out_level"] = level_idx + 1
                        exit_reason = f"PROFIT_TARGET_{profit_level}%"
                        break

                # Stop loss
                if pnl_pct <= -self.stop_loss * 100:
                    should_exit = True
                    exit_reason = "STOP_LOSS"

                # Time-based exit
                elif (
                    position["expiry_date"] - date.today()
                ).days <= self.time_exit_dte:
                    should_exit = True
                    exit_reason = "TIME_EXIT"

                # Death cross exit signal
                elif (
                    position.get("ma_cross_type") != "death_cross"
                ):  # Wasn't death cross at entry
                    ma_cross = await self.analyze_moving_average_cross(ticker)
                    if (
                        ma_cross.cross_type == "death_cross"
                        and ma_cross.days_since_cross
                        and ma_cross.days_since_cross <= 30
                    ):
                        should_exit = True
                        exit_reason = "DEATH_CROSS_EXIT"

                # Execute scale-out or full exit
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
                            "leaps_action": "scale_out"
                            if should_scale_out
                            else "full_exit",
                            "exit_reason": exit_reason,
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                            "scale_out_level": position["scale_out_level"],
                        },
                    )

                    success = await self.integration_manager.execute_trade_signal(
                        exit_signal
                    )

                    if success:
                        if should_exit:
                            await self.integration_manager.alert_system.send_alert(
                                "LEAPS_EXIT",
                                "LOW",
                                f"LEAPS Exit: {ticker} {exit_reason} "
                                f"P & L: ${pnl:.0f} ({pnl_pct: .1%})",
                            )
                            positions_to_remove.append(i)
                            self.logger.info(
                                f"LEAPS position closed: {ticker} {exit_reason}"
                            )
                        else:
                            # Update position after scale-out
                            position["contracts"] -= exit_contracts
                            position["cost_basis"] -= (
                                entry_premium * exit_contracts * 100
                            )

                            await self.integration_manager.alert_system.send_alert(
                                "LEAPS_SCALE_OUT",
                                "LOW",
                                f"LEAPS Scale-Out: {ticker} {exit_reason} "
                                f"Sold {exit_contracts} contracts at {pnl_pct: .1%} gain",
                            )
                            self.logger.info(
                                f"LEAPS scaled out: {ticker} {exit_reason}"
                            )

            except Exception as e:
                self.logger.error(f"Error managing LEAPS position {i}: {e}")

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

            # LEAPS can be scanned less frequently - check market hours for efficiency
            if not await self.data_provider.is_market_open():
                return []

            # Scan for LEAPS candidates
            candidates = await self.scan_leaps_candidates()

            # Execute top candidates
            trade_signals = []
            max_new_positions = self.max_positions - len(self.active_positions)

            for candidate in candidates[:max_new_positions]:
                success = await self.execute_leaps_trade(candidate)
                if success:
                    # Return trade signal for tracking
                    trade_signal = ProductionTradeSignal(
                        symbol=candidate.ticker,
                        action="BUY",
                        quantity=1,  # Will be recalculated in execute_trade
                        option_type="CALL",
                        strike_price=Decimal(str(candidate.recommended_strike)),
                        expiration_date=datetime.strptime(
                            candidate.expiry_date, "%Y-%m-%d"
                        ).date(),
                        premium=Decimal(str(candidate.premium_estimate)),
                        confidence=candidate.composite_score / 100.0,
                        strategy_name=self.strategy_name,
                        signal_strength=candidate.entry_timing_score / 100.0,
                    )
                    trade_signals.append(trade_signal)

            return trade_signals

        except Exception as e:
            self.logger.error(f"Error in LEAPS tracker scan: {e}")
            return []

    def get_strategy_status(self) -> dict[str, Any]:
        """Get current strategy status."""
        try:
            total_cost_basis = sum(pos["cost_basis"] for pos in self.active_positions)
            position_details = []

            for position in self.active_positions:
                position_details.append(
                    {
                        "ticker": position["ticker"],
                        "theme": position["theme"],
                        "strike": float(position["trade_signal"].strike_price),
                        "expiry": position["expiry_date"].isoformat(),
                        "contracts": position["contracts"],
                        "entry_premium": position["entry_premium"],
                        "cost_basis": position["cost_basis"],
                        "breakeven": position["breakeven"],
                        "scale_out_level": position["scale_out_level"],
                        "days_to_expiry": (position["expiry_date"] - date.today()).days,
                        "entry_date": position["entry_time"].date().isoformat(),
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
                    "max_total_allocation": self.max_total_allocation,
                    "min_composite_score": self.min_composite_score,
                    "profit_levels": self.profit_levels,
                    "stop_loss": self.stop_loss,
                },
                "themes": list(self.secular_themes.keys()),
            }

        except Exception as e:
            self.logger.error(f"Error getting LEAPS strategy status: {e}")
            return {"strategy_name": self.strategy_name, "error": str(e)}

    async def run_strategy(self):
        """Main strategy execution loop."""
        self.logger.info("Starting Production LEAPS Tracker Strategy")

        try:
            while True:
                # Scan for LEAPS opportunities
                signals = await self.scan_opportunities()

                # Execute trades for signals
                if signals:
                    await self.execute_trades(signals)

                # Wait before next scan (LEAPS run less frequently)
                await asyncio.sleep(600)  # 10 minutes between scans

        except Exception as e:
            self.logger.error(f"Error in LEAPS tracker strategy main loop: {e}")


def create_production_leaps_tracker(
    integration_manager, data_provider: ReliableDataProvider, config: dict
) -> ProductionLEAPSTracker:
    """Factory function to create ProductionLEAPSTracker strategy."""
    return ProductionLEAPSTracker(integration_manager, data_provider, config)
