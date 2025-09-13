"""Smart Options Selection with Liquidity Analysis
Implements intelligent options contract selection based on WSB criteria and market liquidity.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LiquidityRating(Enum):
    """Options liquidity ratings."""

    EXCELLENT = "excellent"  # Very liquid, tight spreads
    GOOD = "good"  # Good liquidity, reasonable spreads
    FAIR = "fair"  # Moderate liquidity, wider spreads
    POOR = "poor"  # Low liquidity, very wide spreads
    ILLIQUID = "illiquid"  # Essentially no liquidity


@dataclass
class OptionsAnalysis:
    """Comprehensive options contract analysis."""

    ticker: str
    strike: Decimal
    expiry: date
    option_type: str  # 'call' or 'put'

    # Pricing data
    bid: Decimal
    ask: Decimal
    mid_price: Decimal
    last_price: Decimal

    # Liquidity metrics
    volume: int
    open_interest: int
    bid_ask_spread: Decimal
    spread_percentage: float
    liquidity_rating: LiquidityRating

    # Greeks and risk metrics
    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal
    implied_volatility: Decimal

    # Selection scoring
    wsb_suitability_score: float  # 0 - 10 scale
    liquidity_score: float  # 0 - 10 scale
    value_score: float  # 0 - 10 scale
    overall_score: float  # 0 - 10 scale

    # Metadata
    days_to_expiry: int
    moneyness: float  # Strike / Spot price
    premium_to_stock_ratio: float
    break_even_price: Decimal
    max_profit_potential: Decimal | None

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SelectionCriteria:
    """Options selection criteria."""

    target_dte_min: int = 21  # Minimum days to expiry
    target_dte_max: int = 45  # Maximum days to expiry
    target_otm_min: float = 0.02  # Minimum 2% OTM
    target_otm_max: float = 0.10  # Maximum 10% OTM
    min_volume: int = 50  # Minimum daily volume
    min_open_interest: int = 100  # Minimum open interest
    max_spread_percentage: float = 0.15  # Maximum 15% bid - ask spread
    min_liquidity_rating: LiquidityRating = LiquidityRating.FAIR
    target_delta_min: float = 0.15  # Minimum delta
    target_delta_max: float = 0.45  # Maximum delta
    max_premium_to_stock_ratio: float = 0.05  # Max 5% of stock price


class SmartOptionsSelector:
    """Smart Options Selection Engine.

    Implements sophisticated options contract selection optimized for WSB - style trading:
    - Liquidity analysis and filtering
    - Spread analysis and cost optimization
    - Greeks - based risk assessment
    - Volume and open interest validation
    - WSB suitability scoring
    """

    def __init__(self, data_provider=None, pricing_engine=None):
        self.data_provider = data_provider
        self.pricing_engine = pricing_engine
        self.logger = logging.getLogger(__name__)

        # Default selection criteria (can be overridden)
        self.default_criteria = SelectionCriteria()

        # Liquidity thresholds
        self.liquidity_thresholds = {
            "excellent": {"min_volume": 1000, "min_oi": 5000, "max_spread": 0.05},
            "good": {"min_volume": 500, "min_oi": 2000, "max_spread": 0.08},
            "fair": {"min_volume": 100, "min_oi": 500, "max_spread": 0.15},
            "poor": {"min_volume": 25, "min_oi": 100, "max_spread": 0.25},
        }

        self.logger.info("SmartOptionsSelector initialized")

    async def select_optimal_call_option(
        self,
        ticker: str,
        spot_price: Decimal,
        signal_strength: int = 5,
        custom_criteria: SelectionCriteria | None = None,
    ) -> OptionsAnalysis | None:
        """Select optimal call option for WSB dip bot strategy.

        Args:
            ticker: Stock symbol
            spot_price: Current stock price
            signal_strength: Pattern signal strength (1 - 10)
            custom_criteria: Custom selection criteria

        Returns:
            Best options contract analysis or None if no suitable options found
        """
        try:
            # Use custom criteria or defaults
            criteria = custom_criteria or self._get_dynamic_criteria(signal_strength)

            self.logger.info(
                f"Selecting optimal call option for {ticker} at ${spot_price} (signal strength: {signal_strength})"
            )

            # Get available expiry dates
            expiry_dates = await self._get_available_expiries(ticker, criteria)

            if not expiry_dates:
                self.logger.warning(f"No suitable expiry dates found for {ticker}")
                return None

            # Get options chains for each expiry
            all_options = []
            for expiry_date in expiry_dates:
                options_chain = await self.data_provider.get_options_chain(ticker, expiry_date)
                calls = [opt for opt in options_chain if opt.option_type.lower() == "call"]
                all_options.extend(calls)

            if not all_options:
                self.logger.warning(f"No call options found for {ticker}")
                return None

            # Analyze all options
            analyzed_options = []
            for option in all_options:
                analysis = await self._analyze_option(option, spot_price, criteria)
                if analysis and self._meets_criteria(analysis, criteria):
                    analyzed_options.append(analysis)

            if not analyzed_options:
                self.logger.warning(f"No options meet selection criteria for {ticker}")
                return None

            # Sort by overall score and return best option
            analyzed_options.sort(key=lambda x: x.overall_score, reverse=True)
            best_option = analyzed_options[0]

            self.logger.info(
                f"Selected optimal call for {ticker}: ${best_option.strike} strike, "
                f"{best_option.days_to_expiry}DTE, Score: {best_option.overall_score:.2f}, "
                f"Liquidity: {best_option.liquidity_rating.value}"
            )

            return best_option

        except Exception as e:
            self.logger.error(f"Error selecting optimal call option for {ticker}: {e}")
            return None

    def _get_dynamic_criteria(self, signal_strength: int) -> SelectionCriteria:
        """Get dynamic selection criteria based on signal strength."""
        criteria = SelectionCriteria()

        # Adjust criteria based on signal strength
        if signal_strength >= 8:  # Very strong signal
            criteria.target_otm_min = 0.01  # 1% OTM
            criteria.target_otm_max = 0.05  # 5% OTM
            criteria.target_dte_min = 25  # Slightly more time
            criteria.target_dte_max = 40
            criteria.min_volume = 25  # Accept lower volume for better strikes
            criteria.min_open_interest = 50
        elif signal_strength >= 6:  # Strong signal
            criteria.target_otm_min = 0.02  # 2% OTM
            criteria.target_otm_max = 0.07  # 7% OTM
            criteria.target_dte_min = 21
            criteria.target_dte_max = 42
        elif signal_strength >= 4:  # Medium signal
            criteria.target_otm_min = 0.03  # 3% OTM
            criteria.target_otm_max = 0.08  # 8% OTM
            criteria.target_dte_min = 21
            criteria.target_dte_max = 45
        else:  # Weaker signal
            criteria.target_otm_min = 0.05  # 5% OTM
            criteria.target_otm_max = 0.10  # 10% OTM
            criteria.target_dte_min = 28  # More time for weaker signals
            criteria.target_dte_max = 50
            criteria.min_volume = 100  # Higher liquidity requirements
            criteria.min_open_interest = 250

        return criteria

    async def _get_available_expiries(
        self, ticker: str, criteria: SelectionCriteria
    ) -> list[datetime]:
        """Get available expiry dates within criteria."""
        try:
            # Get options chain with first available expiry to see all expiry dates
            options_chain = await self.data_provider.get_options_chain(ticker)

            if not options_chain:
                return []

            # Extract unique expiry dates
            expiry_dates = list({opt.expiry for opt in options_chain})
            expiry_dates.sort()

            # Filter by DTE criteria
            now = datetime.now()
            suitable_expiries = []

            for expiry in expiry_dates:
                if isinstance(expiry, date):
                    expiry = datetime.combine(expiry, datetime.min.time())

                days_to_expiry = (expiry - now).days

                if criteria.target_dte_min <= days_to_expiry <= criteria.target_dte_max:
                    suitable_expiries.append(expiry)

            # Prefer Friday expiries (standard for WSB)
            friday_expiries = [exp for exp in suitable_expiries if exp.weekday() == 4]

            return (
                friday_expiries if friday_expiries else suitable_expiries[:3]
            )  # Top 3 if no Fridays

        except Exception as e:
            self.logger.error(f"Error getting available expiries: {e}")
            return []

    async def _analyze_option(
        self, option, spot_price: Decimal, criteria: SelectionCriteria
    ) -> OptionsAnalysis | None:
        """Perform comprehensive analysis of options contract."""
        try:
            # Calculate basic metrics
            mid_price = (
                (option.bid + option.ask) / 2
                if option.bid > 0 and option.ask > 0
                else option.last_price
            )
            bid_ask_spread = (
                option.ask - option.bid if option.bid > 0 and option.ask > 0 else Decimal("0")
            )
            spread_percentage = float(bid_ask_spread / mid_price) if mid_price > 0 else 1.0

            # Calculate moneyness and days to expiry
            moneyness = float(option.strike / spot_price)

            expiry_date = option.expiry
            if isinstance(expiry_date, datetime):
                expiry_date = expiry_date.date()

            days_to_expiry = (expiry_date - date.today()).days

            # Calculate premium to stock ratio
            premium_to_stock_ratio = float(mid_price / spot_price) if spot_price > 0 else 0

            # Calculate break - even
            break_even_price = option.strike + mid_price

            # Assess liquidity
            liquidity_rating = self._assess_liquidity(
                option.volume, option.open_interest, spread_percentage
            )

            # Calculate scores
            wsb_suitability_score = self._calculate_wsb_suitability_score(
                moneyness, days_to_expiry, premium_to_stock_ratio, option.delta, spread_percentage
            )

            liquidity_score = self._calculate_liquidity_score(
                option.volume, option.open_interest, spread_percentage, liquidity_rating
            )

            value_score = self._calculate_value_score(
                option.implied_volatility, spread_percentage, premium_to_stock_ratio, option.delta
            )

            # Overall score (weighted combination)
            overall_score = (
                wsb_suitability_score * 0.4  # 40% WSB fit
                + liquidity_score * 0.35  # 35% liquidity
                + value_score * 0.25  # 25% value
            )

            return OptionsAnalysis(
                ticker=option.ticker,
                strike=option.strike,
                expiry=expiry_date,
                option_type=option.option_type,
                bid=option.bid,
                ask=option.ask,
                mid_price=mid_price,
                last_price=option.last_price,
                volume=option.volume,
                open_interest=option.open_interest,
                bid_ask_spread=bid_ask_spread,
                spread_percentage=spread_percentage,
                liquidity_rating=liquidity_rating,
                delta=option.delta,
                gamma=option.gamma,
                theta=option.theta,
                vega=option.vega,
                implied_volatility=option.implied_volatility,
                wsb_suitability_score=wsb_suitability_score,
                liquidity_score=liquidity_score,
                value_score=value_score,
                overall_score=overall_score,
                days_to_expiry=days_to_expiry,
                moneyness=moneyness,
                premium_to_stock_ratio=premium_to_stock_ratio,
                break_even_price=break_even_price,
                max_profit_potential=None,  # Unlimited for calls
            )

        except Exception as e:
            self.logger.error(f"Error analyzing option: {e}")
            return None

    def _assess_liquidity(
        self, volume: int, open_interest: int, spread_percentage: float
    ) -> LiquidityRating:
        """Assess options liquidity based on volume, OI, and spread."""
        try:
            # Check against thresholds in order of quality
            for rating, thresholds in self.liquidity_thresholds.items():
                if (
                    volume >= thresholds["min_volume"]
                    and open_interest >= thresholds["min_oi"]
                    and spread_percentage <= thresholds["max_spread"]
                ):
                    return LiquidityRating(rating)

            return LiquidityRating.ILLIQUID

        except Exception:
            return LiquidityRating.POOR

    def _calculate_wsb_suitability_score(
        self, moneyness: float, dte: int, premium_ratio: float, delta: Decimal, spread_pct: float
    ) -> float:
        """Calculate how suitable this option is for WSB - style trading."""
        try:
            score = 0.0

            # OTM preference (WSB likes ~5% OTM)
            if 1.02 <= moneyness <= 1.08:  # 2 - 8% OTM sweet spot
                score += 3.0
            elif 1.01 <= moneyness <= 1.10:  # 1 - 10% OTM acceptable
                score += 2.0
            elif 1.00 <= moneyness <= 1.12:  # ATM to 12% OTM
                score += 1.0

            # DTE preference (WSB likes 3 - 6 weeks)
            if 21 <= dte <= 42:  # 3 - 6 weeks ideal
                score += 2.5
            elif 14 <= dte <= 56:  # 2 - 8 weeks acceptable
                score += 2.0
            elif 7 <= dte <= 70:  # 1 - 10 weeks
                score += 1.0

            # Premium affordability (not too expensive)
            if premium_ratio <= 0.02:  #  <=  2% of stock price
                score += 2.0
            elif premium_ratio <= 0.03:  #  <=  3% of stock price
                score += 1.5
            elif premium_ratio <= 0.05:  #  <=  5% of stock price
                score += 1.0

            # Delta preference (moderate leverage)
            delta_val = float(delta) if delta else 0
            if 0.20 <= delta_val <= 0.40:  # Sweet spot for leverage
                score += 2.0
            elif 0.15 <= delta_val <= 0.50:  # Acceptable range
                score += 1.5
            elif 0.10 <= delta_val <= 0.60:  # Usable range
                score += 1.0

            # Spread penalty (WSB hates wide spreads)
            if spread_pct <= 0.05:  # Tight spread
                score += 0.5
            elif spread_pct <= 0.10:  # Reasonable spread
                score += 0.0
            elif spread_pct <= 0.20:  # Wide spread
                score -= 0.5
            else:  # Very wide spread
                score -= 1.0

            return max(0.0, min(10.0, score))

        except Exception:
            return 0.0

    def _calculate_liquidity_score(
        self, volume: int, open_interest: int, spread_pct: float, liquidity_rating: LiquidityRating
    ) -> float:
        """Calculate liquidity score."""
        try:
            score = 0.0

            # Volume scoring
            if volume >= 1000:
                score += 3.0
            elif volume >= 500:
                score += 2.5
            elif volume >= 200:
                score += 2.0
            elif volume >= 100:
                score += 1.5
            elif volume >= 50:
                score += 1.0
            elif volume >= 25:
                score += 0.5

            # Open interest scoring
            if open_interest >= 5000:
                score += 3.0
            elif open_interest >= 2000:
                score += 2.5
            elif open_interest >= 1000:
                score += 2.0
            elif open_interest >= 500:
                score += 1.5
            elif open_interest >= 200:
                score += 1.0
            elif open_interest >= 100:
                score += 0.5

            # Spread scoring
            if spread_pct <= 0.03:
                score += 2.0
            elif spread_pct <= 0.05:
                score += 1.5
            elif spread_pct <= 0.08:
                score += 1.0
            elif spread_pct <= 0.12:
                score += 0.5
            elif spread_pct <= 0.20:
                score += 0.0
            else:
                score -= 1.0

            # Liquidity rating bonus / penalty
            rating_scores = {
                LiquidityRating.EXCELLENT: 2.0,
                LiquidityRating.GOOD: 1.5,
                LiquidityRating.FAIR: 1.0,
                LiquidityRating.POOR: 0.5,
                LiquidityRating.ILLIQUID: -1.0,
            }
            score += rating_scores.get(liquidity_rating, 0)

            return max(0.0, min(10.0, score))

        except Exception:
            return 0.0

    def _calculate_value_score(
        self, iv: Decimal, spread_pct: float, premium_ratio: float, delta: Decimal
    ) -> float:
        """Calculate value score based on various metrics."""
        try:
            score = 5.0  # Start with neutral score

            # IV scoring (prefer reasonable IV, not too high or low)
            iv_val = float(iv) if iv else 0.25
            if 0.20 <= iv_val <= 0.40:  # Reasonable IV range
                score += 1.0
            elif 0.15 <= iv_val <= 0.50:  # Acceptable range
                score += 0.5
            elif iv_val > 0.60:  # Very high IV (expensive)
                score -= 1.0
            elif iv_val < 0.10:  # Very low IV (suspicious)
                score -= 0.5

            # Premium efficiency (delta per dollar spent)
            if delta and premium_ratio > 0:
                efficiency = float(delta) / premium_ratio
                if efficiency >= 15:  # High efficiency
                    score += 1.5
                elif efficiency >= 10:  # Good efficiency
                    score += 1.0
                elif efficiency >= 7:  # Fair efficiency
                    score += 0.5
                elif efficiency < 3:  # Poor efficiency
                    score -= 1.0

            # Spread impact on value
            if spread_pct <= 0.05:  # Low spread = better value
                score += 0.5
            elif spread_pct >= 0.20:  # High spread = worse value
                score -= 1.0

            return max(0.0, min(10.0, score))

        except Exception:
            return 5.0

    def _meets_criteria(self, analysis: OptionsAnalysis, criteria: SelectionCriteria) -> bool:
        """Check if option meets minimum selection criteria."""
        try:
            # Basic liquidity requirements
            if analysis.volume < criteria.min_volume:
                return False

            if analysis.open_interest < criteria.min_open_interest:
                return False

            if analysis.spread_percentage > criteria.max_spread_percentage:
                return False

            # Liquidity rating requirement
            rating_values = {
                LiquidityRating.EXCELLENT: 5,
                LiquidityRating.GOOD: 4,
                LiquidityRating.FAIR: 3,
                LiquidityRating.POOR: 2,
                LiquidityRating.ILLIQUID: 1,
            }

            if rating_values.get(analysis.liquidity_rating, 0) < rating_values.get(
                criteria.min_liquidity_rating, 3
            ):
                return False

            # DTE requirements
            if not (criteria.target_dte_min <= analysis.days_to_expiry <= criteria.target_dte_max):
                return False

            # OTM requirements
            otm_percentage = analysis.moneyness - 1.0
            if not (criteria.target_otm_min <= otm_percentage <= criteria.target_otm_max):
                return False

            # Delta requirements
            delta_val = float(analysis.delta) if analysis.delta else 0
            if not (criteria.target_delta_min <= delta_val <= criteria.target_delta_max):
                return False

            # Premium to stock ratio
            return not analysis.premium_to_stock_ratio > criteria.max_premium_to_stock_ratio

        except Exception as e:
            self.logger.error(f"Error checking criteria: {e}")
            return False

    async def get_selection_summary(
        self, ticker: str, spot_price: Decimal, signal_strength: int = 5
    ) -> dict[str, Any]:
        """Get comprehensive summary of available options."""
        try:
            criteria = self._get_dynamic_criteria(signal_strength)

            # Get all available options
            expiry_dates = await self._get_available_expiries(ticker, criteria)
            all_options = []

            for expiry_date in expiry_dates:
                options_chain = await self.data_provider.get_options_chain(ticker, expiry_date)
                calls = [opt for opt in options_chain if opt.option_type.lower() == "call"]
                all_options.extend(calls)

            # Analyze all options
            analyzed_options = []
            meeting_criteria = []

            for option in all_options:
                analysis = await self._analyze_option(option, spot_price, criteria)
                if analysis:
                    analyzed_options.append(analysis)
                    if self._meets_criteria(analysis, criteria):
                        meeting_criteria.append(analysis)

            # Sort by score
            analyzed_options.sort(key=lambda x: x.overall_score, reverse=True)
            meeting_criteria.sort(key=lambda x: x.overall_score, reverse=True)

            return {
                "ticker": ticker,
                "spot_price": float(spot_price),
                "signal_strength": signal_strength,
                "criteria": {
                    "dte_range": f"{criteria.target_dte_min}-{criteria.target_dte_max}",
                    "otm_range": f"{criteria.target_otm_min:.1%}-{criteria.target_otm_max: .1%}",
                    "min_volume": criteria.min_volume,
                    "min_open_interest": criteria.min_open_interest,
                },
                "total_options_analyzed": len(analyzed_options),
                "options_meeting_criteria": len(meeting_criteria),
                "best_option": {
                    "strike": float(meeting_criteria[0].strike),
                    "expiry": meeting_criteria[0].expiry.isoformat(),
                    "score": meeting_criteria[0].overall_score,
                    "liquidity_rating": meeting_criteria[0].liquidity_rating.value,
                }
                if meeting_criteria
                else None,
                "top_5_options": [
                    {
                        "strike": float(opt.strike),
                        "expiry": opt.expiry.isoformat(),
                        "score": opt.overall_score,
                        "liquidity_rating": opt.liquidity_rating.value,
                        "volume": opt.volume,
                        "spread_pct": opt.spread_percentage,
                    }
                    for opt in meeting_criteria[:5]
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting selection summary: {e}")
            return {"error": str(e)}


def create_smart_options_selector(data_provider=None, pricing_engine=None) -> SmartOptionsSelector:
    """Factory function to create smart options selector."""
    return SmartOptionsSelector(data_provider, pricing_engine)
