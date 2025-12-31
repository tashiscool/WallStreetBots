"""Portfolio Builder Service

Provides portfolio templates, custom portfolio building, and strategy combination analysis.
Integrates with PortfolioCapitalAllocator for correlation-aware allocation optimization.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from django.utils import timezone
from django.db.models import Q

from backend.tradingbot.models.models import StrategyPortfolio

logger = logging.getLogger(__name__)


# Strategy metadata for portfolio building
AVAILABLE_STRATEGIES = {
    'wsb-dip-bot': {
        'name': 'WSB Dip Bot',
        'description': 'Buys dips on popular momentum stocks',
        'risk_level': 'aggressive',
        'expected_return': 0.25,
        'volatility': 0.35,
        'correlation_group': 'momentum',
        'min_allocation': 5,
        'max_allocation': 30,
    },
    'wheel': {
        'name': 'Wheel Strategy',
        'description': 'Cash-secured puts and covered calls',
        'risk_level': 'moderate',
        'expected_return': 0.12,
        'volatility': 0.15,
        'correlation_group': 'income',
        'min_allocation': 10,
        'max_allocation': 40,
    },
    'momentum-weeklies': {
        'name': 'Momentum Weeklies',
        'description': 'Weekly options on momentum breakouts',
        'risk_level': 'aggressive',
        'expected_return': 0.30,
        'volatility': 0.45,
        'correlation_group': 'momentum',
        'min_allocation': 5,
        'max_allocation': 20,
    },
    'earnings-protection': {
        'name': 'Earnings Protection',
        'description': 'Hedges positions around earnings',
        'risk_level': 'conservative',
        'expected_return': 0.05,
        'volatility': 0.10,
        'correlation_group': 'hedging',
        'min_allocation': 5,
        'max_allocation': 15,
    },
    'debit-spreads': {
        'name': 'Debit Spreads',
        'description': 'Directional plays with limited risk',
        'risk_level': 'moderate',
        'expected_return': 0.18,
        'volatility': 0.25,
        'correlation_group': 'directional',
        'min_allocation': 5,
        'max_allocation': 25,
    },
    'leaps-tracker': {
        'name': 'LEAPS Tracker',
        'description': 'Long-term equity anticipation securities',
        'risk_level': 'moderate',
        'expected_return': 0.20,
        'volatility': 0.20,
        'correlation_group': 'long_term',
        'min_allocation': 10,
        'max_allocation': 35,
    },
    'lotto-scanner': {
        'name': 'Lotto Scanner',
        'description': 'High-risk/high-reward speculative plays',
        'risk_level': 'aggressive',
        'expected_return': 0.50,
        'volatility': 0.80,
        'correlation_group': 'speculative',
        'min_allocation': 2,
        'max_allocation': 10,
    },
    'swing-trading': {
        'name': 'Swing Trading',
        'description': 'Multi-day trend following',
        'risk_level': 'moderate',
        'expected_return': 0.15,
        'volatility': 0.20,
        'correlation_group': 'directional',
        'min_allocation': 10,
        'max_allocation': 30,
    },
    'spx-credit-spreads': {
        'name': 'SPX Credit Spreads',
        'description': 'Index credit spreads for consistent income',
        'risk_level': 'moderate',
        'expected_return': 0.10,
        'volatility': 0.12,
        'correlation_group': 'income',
        'min_allocation': 10,
        'max_allocation': 40,
    },
    'index-baseline': {
        'name': 'Index Baseline',
        'description': 'Index tracking baseline comparison',
        'risk_level': 'conservative',
        'expected_return': 0.10,
        'volatility': 0.15,
        'correlation_group': 'passive',
        'min_allocation': 0,
        'max_allocation': 50,
    },
}

# Pre-built portfolio templates
PORTFOLIO_TEMPLATES = {
    'conservative-income': {
        'name': 'Conservative Income',
        'description': 'Focus on steady income with low volatility. '
                       'Ideal for capital preservation with modest growth.',
        'risk_profile': 'conservative',
        'strategies': {
            'wheel': {'allocation_pct': 35, 'enabled': True},
            'spx-credit-spreads': {'allocation_pct': 30, 'enabled': True},
            'earnings-protection': {'allocation_pct': 15, 'enabled': True},
            'leaps-tracker': {'allocation_pct': 20, 'enabled': True},
        },
        'expected_sharpe': 1.2,
        'expected_return': 0.12,
        'expected_volatility': 0.12,
    },
    'balanced-growth': {
        'name': 'Balanced Growth',
        'description': 'Balance between income and growth. '
                       'Moderate risk with diversified strategies.',
        'risk_profile': 'moderate',
        'strategies': {
            'wheel': {'allocation_pct': 25, 'enabled': True},
            'swing-trading': {'allocation_pct': 20, 'enabled': True},
            'debit-spreads': {'allocation_pct': 20, 'enabled': True},
            'leaps-tracker': {'allocation_pct': 20, 'enabled': True},
            'spx-credit-spreads': {'allocation_pct': 15, 'enabled': True},
        },
        'expected_sharpe': 1.4,
        'expected_return': 0.16,
        'expected_volatility': 0.18,
    },
    'aggressive-momentum': {
        'name': 'Aggressive Momentum',
        'description': 'Maximize returns through momentum strategies. '
                       'Higher risk, higher potential reward.',
        'risk_profile': 'aggressive',
        'strategies': {
            'wsb-dip-bot': {'allocation_pct': 25, 'enabled': True},
            'momentum-weeklies': {'allocation_pct': 20, 'enabled': True},
            'debit-spreads': {'allocation_pct': 20, 'enabled': True},
            'swing-trading': {'allocation_pct': 20, 'enabled': True},
            'lotto-scanner': {'allocation_pct': 8, 'enabled': True},
            'wheel': {'allocation_pct': 7, 'enabled': True},
        },
        'expected_sharpe': 1.1,
        'expected_return': 0.25,
        'expected_volatility': 0.30,
    },
    'income-focused': {
        'name': 'Income Focused',
        'description': 'Premium collection strategies for consistent monthly income.',
        'risk_profile': 'moderate',
        'strategies': {
            'wheel': {'allocation_pct': 40, 'enabled': True},
            'spx-credit-spreads': {'allocation_pct': 35, 'enabled': True},
            'earnings-protection': {'allocation_pct': 10, 'enabled': True},
            'swing-trading': {'allocation_pct': 15, 'enabled': True},
        },
        'expected_sharpe': 1.3,
        'expected_return': 0.11,
        'expected_volatility': 0.13,
    },
    'growth-seeker': {
        'name': 'Growth Seeker',
        'description': 'Long-term capital appreciation focus with moderate speculation.',
        'risk_profile': 'moderate',
        'strategies': {
            'leaps-tracker': {'allocation_pct': 30, 'enabled': True},
            'swing-trading': {'allocation_pct': 25, 'enabled': True},
            'debit-spreads': {'allocation_pct': 20, 'enabled': True},
            'wsb-dip-bot': {'allocation_pct': 15, 'enabled': True},
            'lotto-scanner': {'allocation_pct': 5, 'enabled': True},
            'wheel': {'allocation_pct': 5, 'enabled': True},
        },
        'expected_sharpe': 1.25,
        'expected_return': 0.20,
        'expected_volatility': 0.22,
    },
}

# Correlation matrix between strategy correlation groups
CORRELATION_GROUPS = {
    ('momentum', 'momentum'): 0.8,
    ('momentum', 'income'): 0.2,
    ('momentum', 'hedging'): -0.3,
    ('momentum', 'directional'): 0.5,
    ('momentum', 'long_term'): 0.4,
    ('momentum', 'speculative'): 0.6,
    ('momentum', 'passive'): 0.4,
    ('income', 'income'): 0.7,
    ('income', 'hedging'): 0.1,
    ('income', 'directional'): 0.3,
    ('income', 'long_term'): 0.4,
    ('income', 'speculative'): 0.1,
    ('income', 'passive'): 0.3,
    ('hedging', 'hedging'): 0.5,
    ('hedging', 'directional'): -0.2,
    ('hedging', 'long_term'): 0.1,
    ('hedging', 'speculative'): -0.4,
    ('hedging', 'passive'): 0.0,
    ('directional', 'directional'): 0.6,
    ('directional', 'long_term'): 0.5,
    ('directional', 'speculative'): 0.4,
    ('directional', 'passive'): 0.5,
    ('long_term', 'long_term'): 0.5,
    ('long_term', 'speculative'): 0.3,
    ('long_term', 'passive'): 0.6,
    ('speculative', 'speculative'): 0.7,
    ('speculative', 'passive'): 0.2,
    ('passive', 'passive'): 1.0,
}


@dataclass
class PortfolioAnalysis:
    """Analysis results for a portfolio configuration."""
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    diversification_score: float
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_contribution: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]


class PortfolioBuilderService:
    """Service for building and managing strategy portfolios."""

    def __init__(self, user=None):
        """Initialize portfolio builder.

        Args:
            user: Django User instance
        """
        self.user = user
        self.logger = logging.getLogger(__name__)

    def get_available_strategies(self) -> Dict[str, dict]:
        """Get all available strategies with metadata.

        Returns:
            Dictionary of strategy_id -> metadata
        """
        return AVAILABLE_STRATEGIES.copy()

    def get_portfolio_templates(self) -> Dict[str, dict]:
        """Get all pre-built portfolio templates.

        Returns:
            Dictionary of template_id -> template configuration
        """
        return PORTFOLIO_TEMPLATES.copy()

    def get_template_details(self, template_id: str) -> Optional[dict]:
        """Get details for a specific template.

        Args:
            template_id: Template identifier

        Returns:
            Template details or None
        """
        template = PORTFOLIO_TEMPLATES.get(template_id)
        if not template:
            return None

        # Analyze the template
        analysis = self.analyze_portfolio(template['strategies'])

        return {
            **template,
            'template_id': template_id,
            'analysis': {
                'expected_return': analysis.expected_return,
                'expected_volatility': analysis.expected_volatility,
                'expected_sharpe': analysis.expected_sharpe,
                'diversification_score': analysis.diversification_score,
                'correlation_matrix': analysis.correlation_matrix,
                'risk_contribution': analysis.risk_contribution,
                'warnings': analysis.warnings,
                'recommendations': analysis.recommendations,
            },
            'strategy_details': {
                sid: {
                    **AVAILABLE_STRATEGIES.get(sid, {}),
                    **config,
                }
                for sid, config in template['strategies'].items()
            },
        }

    def create_from_template(self, template_id: str, name: str = None) -> StrategyPortfolio:
        """Create a portfolio from a template.

        Args:
            template_id: Template to use
            name: Optional custom name

        Returns:
            Created StrategyPortfolio instance

        Raises:
            ValueError: If template not found
        """
        template = PORTFOLIO_TEMPLATES.get(template_id)
        if not template:
            raise ValueError(f"Template '{template_id}' not found")

        # Analyze to get metrics
        analysis = self.analyze_portfolio(template['strategies'])

        portfolio = StrategyPortfolio.objects.create(
            user=self.user,
            name=name or template['name'],
            description=template['description'],
            risk_profile=template['risk_profile'],
            is_template=False,
            is_active=False,
            strategies=template['strategies'],
            correlation_matrix=analysis.correlation_matrix,
            diversification_score=analysis.diversification_score,
            expected_sharpe=analysis.expected_sharpe,
        )

        self.logger.info(
            f"Created portfolio '{portfolio.name}' from template '{template_id}' "
            f"for user {self.user}"
        )

        return portfolio

    def create_custom_portfolio(
        self,
        name: str,
        strategies: Dict[str, dict],
        description: str = '',
        risk_profile: str = 'custom'
    ) -> StrategyPortfolio:
        """Create a custom portfolio with specified strategies.

        Args:
            name: Portfolio name
            strategies: Strategy allocations
            description: Optional description
            risk_profile: Risk profile classification

        Returns:
            Created StrategyPortfolio instance
        """
        # Validate strategies
        for sid in strategies.keys():
            if sid not in AVAILABLE_STRATEGIES:
                raise ValueError(f"Unknown strategy: {sid}")

        # Analyze portfolio
        analysis = self.analyze_portfolio(strategies)

        portfolio = StrategyPortfolio.objects.create(
            user=self.user,
            name=name,
            description=description,
            risk_profile=risk_profile,
            is_template=False,
            is_active=False,
            strategies=strategies,
            correlation_matrix=analysis.correlation_matrix,
            diversification_score=analysis.diversification_score,
            expected_sharpe=analysis.expected_sharpe,
        )

        self.logger.info(
            f"Created custom portfolio '{name}' for user {self.user}"
        )

        return portfolio

    def get_user_portfolios(self) -> List[StrategyPortfolio]:
        """Get all portfolios for the current user.

        Returns:
            List of StrategyPortfolio instances
        """
        if not self.user:
            return []

        return list(
            StrategyPortfolio.objects.filter(
                Q(user=self.user) | Q(is_template=True)
            ).order_by('-is_active', '-updated_at')
        )

    def get_active_portfolio(self) -> Optional[StrategyPortfolio]:
        """Get the user's active portfolio.

        Returns:
            Active StrategyPortfolio or None
        """
        if not self.user:
            return None

        return StrategyPortfolio.objects.filter(
            user=self.user,
            is_active=True
        ).first()

    def analyze_portfolio(self, strategies: Dict[str, dict]) -> PortfolioAnalysis:
        """Analyze a portfolio configuration.

        Args:
            strategies: Strategy allocations

        Returns:
            PortfolioAnalysis with metrics
        """
        warnings = []
        recommendations = []

        # Calculate total allocation
        total_allocation = sum(
            s.get('allocation_pct', 0)
            for s in strategies.values()
            if s.get('enabled', True)
        )

        if total_allocation < 95:
            warnings.append(
                f"Total allocation ({total_allocation:.1f}%) is below 95%. "
                "Consider adding more strategies."
            )
        elif total_allocation > 100:
            warnings.append(
                f"Total allocation ({total_allocation:.1f}%) exceeds 100%. "
                "Please reduce allocations."
            )

        # Build correlation matrix
        correlation_matrix = self._build_correlation_matrix(strategies)

        # Calculate portfolio metrics
        expected_return = 0.0
        expected_variance = 0.0
        risk_contribution = {}

        for sid, config in strategies.items():
            if not config.get('enabled', True):
                continue

            strategy_meta = AVAILABLE_STRATEGIES.get(sid, {})
            weight = config.get('allocation_pct', 0) / 100.0

            # Expected return
            strat_return = strategy_meta.get('expected_return', 0.10)
            expected_return += weight * strat_return

            # Variance contribution
            strat_vol = strategy_meta.get('volatility', 0.20)
            expected_variance += (weight * strat_vol) ** 2

            risk_contribution[sid] = (weight * strat_vol) ** 2

        # Portfolio volatility (simplified)
        expected_volatility = np.sqrt(expected_variance) if expected_variance > 0 else 0.20

        # Sharpe ratio
        risk_free_rate = 0.05  # 5% risk-free rate
        expected_sharpe = (
            (expected_return - risk_free_rate) / expected_volatility
            if expected_volatility > 0 else 0.0
        )

        # Normalize risk contribution
        total_risk = sum(risk_contribution.values())
        if total_risk > 0:
            risk_contribution = {
                k: v / total_risk * 100
                for k, v in risk_contribution.items()
            }

        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(
            strategies, correlation_matrix
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            strategies, diversification_score, expected_sharpe
        )

        return PortfolioAnalysis(
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            expected_sharpe=expected_sharpe,
            diversification_score=diversification_score,
            correlation_matrix=correlation_matrix,
            risk_contribution=risk_contribution,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _build_correlation_matrix(
        self,
        strategies: Dict[str, dict]
    ) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix for strategies.

        Args:
            strategies: Strategy configurations

        Returns:
            Correlation matrix as nested dict
        """
        matrix = {}
        strategy_ids = list(strategies.keys())

        for sid1 in strategy_ids:
            matrix[sid1] = {}
            group1 = AVAILABLE_STRATEGIES.get(sid1, {}).get('correlation_group', 'unknown')

            for sid2 in strategy_ids:
                if sid1 == sid2:
                    matrix[sid1][sid2] = 1.0
                else:
                    group2 = AVAILABLE_STRATEGIES.get(sid2, {}).get('correlation_group', 'unknown')

                    # Look up correlation (try both orderings)
                    corr = CORRELATION_GROUPS.get(
                        (group1, group2),
                        CORRELATION_GROUPS.get((group2, group1), 0.3)
                    )
                    matrix[sid1][sid2] = corr

        return matrix

    def _calculate_diversification_score(
        self,
        strategies: Dict[str, dict],
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate diversification score (0-100).

        Higher score = better diversified portfolio.

        Args:
            strategies: Strategy configurations
            correlation_matrix: Correlation matrix

        Returns:
            Diversification score (0-100)
        """
        if len(strategies) < 2:
            return 50.0  # Single strategy = moderate diversification

        # Calculate average pairwise correlation
        correlations = []
        strategy_ids = [
            sid for sid, config in strategies.items()
            if config.get('enabled', True)
        ]

        for i, sid1 in enumerate(strategy_ids):
            for sid2 in strategy_ids[i + 1:]:
                corr = correlation_matrix.get(sid1, {}).get(sid2, 0.3)
                correlations.append(corr)

        if not correlations:
            return 50.0

        avg_correlation = np.mean(correlations)

        # Score: lower correlation = higher score
        # -1 correlation = 100, +1 correlation = 0
        diversification_score = (1 - avg_correlation) * 50 + 25

        # Bonus for number of strategies (up to 25 points)
        strategy_count_bonus = min(len(strategy_ids) * 5, 25)
        diversification_score += strategy_count_bonus

        # Cap at 100
        return min(max(diversification_score, 0), 100)

    def _generate_recommendations(
        self,
        strategies: Dict[str, dict],
        diversification_score: float,
        expected_sharpe: float
    ) -> List[str]:
        """Generate portfolio recommendations.

        Args:
            strategies: Strategy configurations
            diversification_score: Diversification score
            expected_sharpe: Expected Sharpe ratio

        Returns:
            List of recommendations
        """
        recommendations = []

        # Diversification recommendations
        if diversification_score < 40:
            recommendations.append(
                "Low diversification score. Consider adding strategies "
                "from different correlation groups (income vs momentum)."
            )

        # Sharpe recommendations
        if expected_sharpe < 1.0:
            recommendations.append(
                "Expected Sharpe ratio is below 1.0. Consider reducing "
                "allocation to high-volatility strategies."
            )

        # Strategy-specific recommendations
        enabled_strategies = [
            sid for sid, config in strategies.items()
            if config.get('enabled', True)
        ]

        # Check for hedging
        has_hedging = any(
            AVAILABLE_STRATEGIES.get(sid, {}).get('correlation_group') == 'hedging'
            for sid in enabled_strategies
        )
        if not has_hedging:
            recommendations.append(
                "Consider adding earnings-protection for portfolio hedging."
            )

        # Check for income
        has_income = any(
            AVAILABLE_STRATEGIES.get(sid, {}).get('correlation_group') == 'income'
            for sid in enabled_strategies
        )
        if not has_income:
            recommendations.append(
                "Consider adding wheel or SPX credit spreads for income generation."
            )

        # Check for concentration
        max_allocation = max(
            config.get('allocation_pct', 0)
            for config in strategies.values()
            if config.get('enabled', True)
        ) if strategies else 0

        if max_allocation > 40:
            recommendations.append(
                f"High concentration detected ({max_allocation}% in single strategy). "
                "Consider redistributing for better diversification."
            )

        return recommendations

    def optimize_portfolio(
        self,
        strategies: List[str],
        risk_profile: str = 'moderate',
        total_capital: Decimal = Decimal('100000')
    ) -> Dict[str, dict]:
        """Optimize allocation for given strategies.

        Uses risk parity and Sharpe weighting to determine optimal allocations.

        Args:
            strategies: List of strategy IDs to include
            risk_profile: Risk profile (conservative, moderate, aggressive)
            total_capital: Total capital to allocate

        Returns:
            Optimized strategy allocations
        """
        if not strategies:
            return {}

        # Risk profile multipliers
        risk_multipliers = {
            'conservative': {'income': 1.5, 'hedging': 1.3, 'momentum': 0.5, 'speculative': 0.2},
            'moderate': {'income': 1.0, 'hedging': 1.0, 'momentum': 1.0, 'speculative': 0.5},
            'aggressive': {'income': 0.6, 'hedging': 0.5, 'momentum': 1.5, 'speculative': 1.2},
        }
        multipliers = risk_multipliers.get(risk_profile, risk_multipliers['moderate'])

        # Calculate base weights using inverse volatility
        weights = {}
        for sid in strategies:
            meta = AVAILABLE_STRATEGIES.get(sid, {})
            volatility = meta.get('volatility', 0.20)
            sharpe = (
                meta.get('expected_return', 0.10) / volatility
                if volatility > 0 else 0.5
            )

            # Apply risk profile multiplier
            corr_group = meta.get('correlation_group', 'directional')
            multiplier = multipliers.get(corr_group, 1.0)

            # Weight = Sharpe / volatility * multiplier
            weights[sid] = (sharpe / volatility) * multiplier if volatility > 0 else 0.5

        # Normalize weights to sum to 100%
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight * 100 for k, v in weights.items()}
        else:
            equal_weight = 100 / len(strategies)
            weights = {sid: equal_weight for sid in strategies}

        # Apply min/max constraints
        for sid, weight in weights.items():
            meta = AVAILABLE_STRATEGIES.get(sid, {})
            min_alloc = meta.get('min_allocation', 5)
            max_alloc = meta.get('max_allocation', 30)
            weights[sid] = max(min_alloc, min(weight, max_alloc))

        # Renormalize after constraints
        total_weight = sum(weights.values())
        if total_weight != 100:
            scale = 100 / total_weight if total_weight > 0 else 1
            weights = {k: v * scale for k, v in weights.items()}

        # Build allocation structure
        allocations = {
            sid: {
                'allocation_pct': round(weight, 1),
                'enabled': True,
                'params': {},
            }
            for sid, weight in weights.items()
        }

        return allocations

    def suggest_additions(
        self,
        current_strategies: List[str],
        risk_profile: str = 'moderate'
    ) -> List[dict]:
        """Suggest strategies to add for better diversification.

        Args:
            current_strategies: Currently selected strategies
            risk_profile: User's risk profile

        Returns:
            List of suggested strategies with reasons
        """
        suggestions = []

        # Get current correlation groups
        current_groups = set(
            AVAILABLE_STRATEGIES.get(sid, {}).get('correlation_group')
            for sid in current_strategies
        )

        # Find missing groups
        all_groups = set(
            meta.get('correlation_group')
            for meta in AVAILABLE_STRATEGIES.values()
        )
        missing_groups = all_groups - current_groups

        # Suggest strategies from missing groups
        for sid, meta in AVAILABLE_STRATEGIES.items():
            if sid in current_strategies:
                continue

            corr_group = meta.get('correlation_group')
            risk_level = meta.get('risk_level')

            # Check if this fills a gap
            if corr_group in missing_groups:
                # Filter by risk profile
                if risk_profile == 'conservative' and risk_level == 'aggressive':
                    continue
                if risk_profile == 'aggressive' and risk_level == 'conservative':
                    continue

                suggestions.append({
                    'strategy_id': sid,
                    'name': meta.get('name'),
                    'reason': f"Adds {corr_group} exposure for diversification",
                    'expected_correlation': -0.2 if corr_group == 'hedging' else 0.3,
                    'diversification_impact': 'high' if corr_group == 'hedging' else 'medium',
                })

        return suggestions[:3]  # Return top 3 suggestions


def get_portfolio_builder(user=None) -> PortfolioBuilderService:
    """Factory function to get portfolio builder service.

    Args:
        user: Django User instance

    Returns:
        PortfolioBuilderService instance
    """
    return PortfolioBuilderService(user=user)
