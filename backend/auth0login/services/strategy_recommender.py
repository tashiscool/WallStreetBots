"""
Strategy Recommendation Service

Provides personalized strategy recommendations based on user's risk profile,
capital, and investment timeline.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Trading strategy definition."""
    id: str
    name: str
    short_description: str
    risk_level: str  # 'low', 'medium', 'high', 'very_high'
    min_capital: int  # Minimum capital required in dollars
    typical_hold_period: str
    suitable_profiles: list[str]  # Which risk profiles this fits
    fit_explanation: dict[str, str]  # Profile -> explanation why it fits


# Define all available strategies with their characteristics
STRATEGIES = [
    Strategy(
        id="wheel",
        name="Wheel Strategy",
        short_description="Sell puts and covered calls to collect premium income",
        risk_level="low",
        min_capital=5000,  # Need enough for 100 shares of something
        typical_hold_period="Weeks to months",
        suitable_profiles=["conservative", "moderate"],
        fit_explanation={
            "conservative": "Perfect for you! The Wheel generates steady income with defined risk. You'll sell puts on stocks you'd want to own, and if assigned, sell covered calls.",
            "moderate": "Great fit for part of your portfolio. Wheel provides consistent income while you use more aggressive strategies elsewhere.",
        },
    ),
    Strategy(
        id="leaps-tracker",
        name="LEAPS Tracker",
        short_description="Long-term options (1+ year) for leveraged buy-and-hold",
        risk_level="low",
        min_capital=3000,
        typical_hold_period="6-18 months",
        suitable_profiles=["conservative", "moderate"],
        fit_explanation={
            "conservative": "LEAPS give you time on your side. Long expiration means less stress and time decay works slowly. Great for patient investors.",
            "moderate": "LEAPS offer leveraged exposure with more forgiving timeframes than short-term options.",
        },
    ),
    Strategy(
        id="swing-trading",
        name="Swing Trading",
        short_description="Multi-day trades on technical patterns",
        risk_level="medium",
        min_capital=2000,
        typical_hold_period="2-10 days",
        suitable_profiles=["conservative", "moderate"],
        fit_explanation={
            "conservative": "Swing trading with proper position sizing provides good risk/reward. Focus on high-probability setups with clear stop losses.",
            "moderate": "Classic approach that balances opportunity with risk management. Great for building trading skills.",
        },
    ),
    Strategy(
        id="index-baseline",
        name="Index Baseline",
        short_description="Track major indexes (SPY, QQQ) for steady growth",
        risk_level="low",
        min_capital=1000,
        typical_hold_period="Ongoing",
        suitable_profiles=["conservative", "moderate", "aggressive"],
        fit_explanation={
            "conservative": "The foundation of any portfolio. Market-matching returns with minimal active management needed.",
            "moderate": "Solid core holding while you explore other strategies.",
            "aggressive": "Even aggressive traders need a stable base. This provides benchmark comparison.",
        },
    ),
    Strategy(
        id="momentum-weeklies",
        name="Momentum Weeklies",
        short_description="Quick trades on stocks with strong momentum",
        risk_level="high",
        min_capital=2000,
        typical_hold_period="Hours to 1 day",
        suitable_profiles=["moderate", "aggressive"],
        fit_explanation={
            "moderate": "Can add excitement to your portfolio in small allocations. Use strict position sizing.",
            "aggressive": "Perfect for your style! Capture quick moves with weekly options for maximum leverage.",
        },
    ),
    Strategy(
        id="earnings-protection",
        name="Earnings Protection",
        short_description="Options strategies around earnings announcements",
        risk_level="medium",
        min_capital=2500,
        typical_hold_period="1-3 days",
        suitable_profiles=["moderate", "aggressive"],
        fit_explanation={
            "moderate": "Earnings plays offer defined-risk opportunities with known catalysts.",
            "aggressive": "Earnings events provide the volatility you're looking for with clear timing.",
        },
    ),
    Strategy(
        id="spx-credit-spreads",
        name="SPX Credit Spreads",
        short_description="Iron condors and credit spreads on S&P 500 index",
        risk_level="medium",
        min_capital=5000,
        typical_hold_period="1-4 weeks",
        suitable_profiles=["moderate", "aggressive"],
        fit_explanation={
            "moderate": "High probability trades with defined risk. SPX is highly liquid with favorable tax treatment.",
            "aggressive": "Can size up positions for more premium with manageable risk profiles.",
        },
    ),
    Strategy(
        id="wsb-dip-bot",
        name="WSB Dip Bot",
        short_description="Buy call options on oversold momentum stocks",
        risk_level="high",
        min_capital=1000,
        typical_hold_period="1-5 days",
        suitable_profiles=["moderate", "aggressive"],
        fit_explanation={
            "moderate": "Small allocation can provide upside exposure. Only use 5-10% of portfolio.",
            "aggressive": "Core strategy for your profile! Captures rebounds with leveraged options.",
        },
    ),
    Strategy(
        id="lotto-scanner",
        name="Lotto Scanner",
        short_description="High-risk, high-reward plays on cheap options",
        risk_level="very_high",
        min_capital=500,
        typical_hold_period="Same day to 1 week",
        suitable_profiles=["aggressive"],
        fit_explanation={
            "aggressive": "Asymmetric bets with limited downside (premium paid) and unlimited upside. Keep to 1-2% of portfolio per trade.",
        },
    ),
    Strategy(
        id="debit-spreads",
        name="Debit Spreads",
        short_description="Directional plays with defined risk using spreads",
        risk_level="medium",
        min_capital=1500,
        typical_hold_period="Days to weeks",
        suitable_profiles=["moderate", "aggressive"],
        fit_explanation={
            "moderate": "Get directional exposure with capped risk. Great for learning options.",
            "aggressive": "Efficient use of capital for directional bets with known max loss.",
        },
    ),
    Strategy(
        id="exotic-spreads",
        name="Exotic Spreads (Straddles/Strangles)",
        short_description="Volatility plays using straddles, strangles, and butterflies",
        risk_level="high",
        min_capital=3000,
        typical_hold_period="Days to weeks",
        suitable_profiles=["aggressive"],
        fit_explanation={
            "aggressive": "Profit from big moves in either direction. Ideal when you expect volatility but unsure of direction.",
        },
    ),
]


@dataclass
class StrategyRecommendation:
    """A single strategy recommendation."""
    strategy: Strategy
    fit_score: float  # 0-100 indicating how well it fits
    fit_explanation: str
    allocation_suggestion: str
    warnings: list[str]


class StrategyRecommenderService:
    """Service for recommending strategies based on user profile."""

    def __init__(self):
        self.strategies = STRATEGIES

    def get_recommendations(
        self,
        risk_profile: str,
        capital_amount: float,
        investment_timeline: Optional[str] = None,
    ) -> dict:
        """
        Get personalized strategy recommendations.

        Args:
            risk_profile: 'conservative', 'moderate', or 'aggressive'
            capital_amount: Available capital in dollars
            investment_timeline: Optional timeline preference

        Returns:
            Dict with recommendations organized by fit level
        """
        recommendations = []
        excluded = []

        for strategy in self.strategies:
            # Check if profile matches
            if risk_profile not in strategy.suitable_profiles:
                # May still recommend with warnings for borderline cases
                if self._is_adjacent_profile(risk_profile, strategy.suitable_profiles):
                    warnings = [self._get_profile_mismatch_warning(risk_profile, strategy)]
                    fit_score = 40  # Lower score for mismatch
                else:
                    excluded.append({
                        'strategy_id': strategy.id,
                        'strategy_name': strategy.name,
                        'reason': f"Not recommended for {risk_profile} profile",
                    })
                    continue
            else:
                warnings = []
                fit_score = 80

            # Check capital requirements
            if capital_amount < strategy.min_capital:
                excluded.append({
                    'strategy_id': strategy.id,
                    'strategy_name': strategy.name,
                    'reason': f"Requires minimum ${strategy.min_capital:,} (you have ${capital_amount:,.0f})",
                })
                continue

            # Adjust fit score based on capital headroom
            capital_headroom = capital_amount / strategy.min_capital
            if capital_headroom >= 3:
                fit_score += 10
            elif capital_headroom >= 2:
                fit_score += 5

            # Add capital-related warnings
            if capital_headroom < 1.5:
                warnings.append(
                    f"You meet the minimum capital requirement, but consider allocating "
                    f"at least ${int(strategy.min_capital * 1.5):,} for proper position sizing."
                )

            # Get fit explanation
            fit_explanation = strategy.fit_explanation.get(
                risk_profile,
                f"This strategy can work for your {risk_profile} profile."
            )

            # Determine allocation suggestion
            allocation = self._get_allocation_suggestion(
                risk_profile, strategy, capital_amount
            )

            recommendations.append(StrategyRecommendation(
                strategy=strategy,
                fit_score=min(100, fit_score),
                fit_explanation=fit_explanation,
                allocation_suggestion=allocation,
                warnings=warnings,
            ))

        # Sort by fit score
        recommendations.sort(key=lambda r: r.fit_score, reverse=True)

        # Categorize into tiers
        highly_recommended = [r for r in recommendations if r.fit_score >= 70]
        consider = [r for r in recommendations if 40 <= r.fit_score < 70]

        return {
            'status': 'success',
            'profile': risk_profile,
            'capital': capital_amount,
            'highly_recommended': [
                self._recommendation_to_dict(r) for r in highly_recommended
            ],
            'consider': [
                self._recommendation_to_dict(r) for r in consider
            ],
            'excluded': excluded,
            'portfolio_suggestion': self._get_portfolio_suggestion(
                risk_profile, capital_amount, highly_recommended
            ),
        }

    def _is_adjacent_profile(
        self,
        user_profile: str,
        strategy_profiles: list[str],
    ) -> bool:
        """Check if user's profile is adjacent to strategy's supported profiles."""
        profile_order = ['conservative', 'moderate', 'aggressive']
        user_idx = profile_order.index(user_profile)

        for profile in strategy_profiles:
            strategy_idx = profile_order.index(profile)
            if abs(user_idx - strategy_idx) == 1:
                return True
        return False

    def _get_profile_mismatch_warning(
        self,
        user_profile: str,
        strategy: Strategy,
    ) -> str:
        """Generate a warning for profile mismatch."""
        if user_profile == 'conservative':
            return (
                f"{strategy.name} is typically recommended for more aggressive traders. "
                "If you proceed, use smaller position sizes and strict stop losses."
            )
        elif user_profile == 'aggressive':
            return (
                f"{strategy.name} may not generate the returns you're looking for. "
                "Consider it for portfolio stabilization."
            )
        return f"This strategy is designed for {', '.join(strategy.suitable_profiles)} profiles."

    def _get_allocation_suggestion(
        self,
        profile: str,
        strategy: Strategy,
        capital: float,
    ) -> str:
        """Get suggested capital allocation for a strategy."""
        base_allocations = {
            'conservative': {
                'low': '20-30%',
                'medium': '10-15%',
                'high': '5-10%',
                'very_high': '0-2%',
            },
            'moderate': {
                'low': '15-25%',
                'medium': '15-20%',
                'high': '10-15%',
                'very_high': '2-5%',
            },
            'aggressive': {
                'low': '10-20%',
                'medium': '15-25%',
                'high': '15-20%',
                'very_high': '5-10%',
            },
        }

        allocation_pct = base_allocations.get(profile, {}).get(strategy.risk_level, '10-15%')
        return f"{allocation_pct} of portfolio (${int(capital * 0.15):,} - ${int(capital * 0.25):,})"

    def _recommendation_to_dict(self, rec: StrategyRecommendation) -> dict:
        """Convert recommendation to serializable dict."""
        return {
            'strategy_id': rec.strategy.id,
            'strategy_name': rec.strategy.name,
            'short_description': rec.strategy.short_description,
            'risk_level': rec.strategy.risk_level,
            'min_capital': rec.strategy.min_capital,
            'typical_hold_period': rec.strategy.typical_hold_period,
            'fit_score': rec.fit_score,
            'fit_explanation': rec.fit_explanation,
            'allocation_suggestion': rec.allocation_suggestion,
            'warnings': rec.warnings,
        }

    def _get_portfolio_suggestion(
        self,
        profile: str,
        capital: float,
        recommendations: list[StrategyRecommendation],
    ) -> dict:
        """Generate an overall portfolio suggestion."""
        if not recommendations:
            return {
                'message': "Start with Index Baseline to build experience.",
                'strategies': ['index-baseline'],
            }

        if profile == 'conservative':
            return {
                'message': (
                    "We recommend starting with 2-3 lower-risk strategies focused on "
                    "income generation and capital preservation."
                ),
                'suggested_mix': [
                    {'strategy': 'wheel', 'allocation': '30%'},
                    {'strategy': 'index-baseline', 'allocation': '40%'},
                    {'strategy': 'leaps-tracker', 'allocation': '20%'},
                    {'strategy': 'cash_reserve', 'allocation': '10%'},
                ],
            }
        elif profile == 'moderate':
            return {
                'message': (
                    "A balanced portfolio with core income strategies plus "
                    "some growth opportunities."
                ),
                'suggested_mix': [
                    {'strategy': 'index-baseline', 'allocation': '25%'},
                    {'strategy': 'wheel', 'allocation': '20%'},
                    {'strategy': 'momentum-weeklies', 'allocation': '15%'},
                    {'strategy': 'wsb-dip-bot', 'allocation': '10%'},
                    {'strategy': 'spx-credit-spreads', 'allocation': '20%'},
                    {'strategy': 'cash_reserve', 'allocation': '10%'},
                ],
            }
        else:  # aggressive
            return {
                'message': (
                    "High-growth portfolio with focus on momentum and volatility plays. "
                    "Remember to keep some stable positions for drawdown management."
                ),
                'suggested_mix': [
                    {'strategy': 'wsb-dip-bot', 'allocation': '25%'},
                    {'strategy': 'momentum-weeklies', 'allocation': '20%'},
                    {'strategy': 'lotto-scanner', 'allocation': '10%'},
                    {'strategy': 'exotic-spreads', 'allocation': '15%'},
                    {'strategy': 'index-baseline', 'allocation': '20%'},
                    {'strategy': 'cash_reserve', 'allocation': '10%'},
                ],
            }

    def get_strategy_details(self, strategy_id: str) -> Optional[dict]:
        """Get detailed information about a specific strategy."""
        for strategy in self.strategies:
            if strategy.id == strategy_id:
                return {
                    'id': strategy.id,
                    'name': strategy.name,
                    'short_description': strategy.short_description,
                    'risk_level': strategy.risk_level,
                    'min_capital': strategy.min_capital,
                    'typical_hold_period': strategy.typical_hold_period,
                    'suitable_profiles': strategy.suitable_profiles,
                    'fit_explanations': strategy.fit_explanation,
                }
        return None


# Singleton instance
strategy_recommender_service = StrategyRecommenderService()
