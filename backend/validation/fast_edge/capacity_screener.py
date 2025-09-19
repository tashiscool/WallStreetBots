"""
Trading Capacity Screening for Fast-Edge Implementation
Screens and validates trading strategies for capacity constraints and scalability.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json


class CapacityConstraint(Enum):
    LIQUIDITY_LIMITED = "liquidity_limited"
    IMPACT_COST_LIMITED = "impact_cost_limited"
    FREQUENCY_LIMITED = "frequency_limited"
    INFORMATION_DECAY = "information_decay"
    COMPETITION_CROWDING = "competition_crowding"
    REGULATORY_LIMITED = "regulatory_limited"


class StrategyCategory(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    VOLATILITY = "volatility"
    EARNINGS = "earnings"
    MACRO = "macro"


@dataclass
class CapacityMetrics:
    """Metrics for strategy capacity analysis."""
    strategy_name: str
    strategy_category: StrategyCategory
    target_aum: float  # Target Assets Under Management
    current_aum: float  # Current AUM
    max_capacity: float  # Maximum theoretical capacity
    utilization_rate: float  # Current capacity utilization
    daily_volume_requirement: float  # Daily volume needed
    market_impact_bps: float  # Expected market impact in bps
    liquidity_coverage_ratio: float  # Available liquidity vs needed
    information_decay_half_life_hours: float  # How long edge persists
    competition_pressure_score: float  # 0-1, higher = more competition
    regulatory_constraints: List[str] = field(default_factory=list)


@dataclass
class LiquidityProfile:
    """Liquidity profile for securities/strategies."""
    symbol: str
    avg_daily_volume: float
    avg_daily_dollar_volume: float
    typical_spread_bps: float
    depth_at_best: float  # Shares at best bid/ask
    depth_5_levels: float  # Shares within 5 price levels
    market_impact_model: Dict[str, float]  # Impact parameters
    intraday_volume_pattern: List[float]  # Hourly volume pattern
    volatility_20d: float


class MarketImpactModel:
    """Models market impact for different trade sizes."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.impact_cache: Dict[str, Dict] = {}

    def calculate_impact(self, symbol: str, trade_size: float,
                        liquidity_profile: LiquidityProfile) -> Dict[str, float]:
        """Calculate expected market impact for trade."""
        try:
            # Cache key
            cache_key = f"{symbol}_{trade_size}_{int(time.time() // 3600)}"
            if cache_key in self.impact_cache:
                return self.impact_cache[cache_key]

            # Participation rate
            participation_rate = trade_size / liquidity_profile.avg_daily_volume

            # Linear impact component (spread and small size impact)
            linear_impact_bps = liquidity_profile.typical_spread_bps / 2

            # Square-root impact component (market impact)
            sqrt_component = np.sqrt(participation_rate) * liquidity_profile.volatility_20d * 100

            # Temporary impact (recovers over time)
            temporary_impact_bps = linear_impact_bps + 0.1 * sqrt_component

            # Permanent impact (doesn't recover)
            permanent_impact_bps = 0.3 * sqrt_component

            # Total impact
            total_impact_bps = temporary_impact_bps + permanent_impact_bps

            # Adjust for liquidity depth
            depth_adjustment = 1.0
            if trade_size > liquidity_profile.depth_at_best:
                depth_adjustment = 1.5  # 50% penalty for going beyond best level

            if trade_size > liquidity_profile.depth_5_levels:
                depth_adjustment = 2.0  # 100% penalty for going beyond 5 levels

            result = {
                'total_impact_bps': total_impact_bps * depth_adjustment,
                'temporary_impact_bps': temporary_impact_bps * depth_adjustment,
                'permanent_impact_bps': permanent_impact_bps * depth_adjustment,
                'participation_rate': participation_rate,
                'depth_adjustment': depth_adjustment
            }

            # Cache result
            self.impact_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.error(f"Impact calculation error for {symbol}: {e}")
            return {
                'total_impact_bps': 50.0,  # Conservative fallback
                'temporary_impact_bps': 30.0,
                'permanent_impact_bps': 20.0,
                'participation_rate': 0.1,
                'depth_adjustment': 1.0
            }

    def estimate_optimal_trade_size(self, symbol: str, target_profit_bps: float,
                                  liquidity_profile: LiquidityProfile,
                                  max_participation_rate: float = 0.05) -> float:
        """Estimate optimal trade size given profit target and constraints."""
        try:
            # Binary search for optimal size
            min_size = 1000  # Minimum trade size
            max_size = liquidity_profile.avg_daily_volume * max_participation_rate

            optimal_size = min_size
            max_profit_after_impact = 0

            # Test different sizes
            test_sizes = np.logspace(np.log10(min_size), np.log10(max_size), 20)

            for size in test_sizes:
                impact = self.calculate_impact(symbol, size, liquidity_profile)
                profit_after_impact = target_profit_bps - impact['total_impact_bps']

                if profit_after_impact > max_profit_after_impact:
                    max_profit_after_impact = profit_after_impact
                    optimal_size = size

            return optimal_size

        except Exception as e:
            self.logger.error(f"Optimal size calculation error: {e}")
            return 10000  # Conservative fallback


class LiquidityDataProvider:
    """Provides liquidity data for capacity screening."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.liquidity_cache: Dict[str, Tuple[LiquidityProfile, datetime]] = {}

    def get_liquidity_profile(self, symbol: str) -> Optional[LiquidityProfile]:
        """Get liquidity profile for symbol."""
        try:
            # Check cache
            if symbol in self.liquidity_cache:
                profile, timestamp = self.liquidity_cache[symbol]
                if datetime.now() - timestamp < timedelta(hours=1):
                    return profile

            # Generate realistic liquidity profile
            # In production, this would fetch from market data provider
            profile = self._generate_liquidity_profile(symbol)

            # Cache profile
            self.liquidity_cache[symbol] = (profile, datetime.now())
            return profile

        except Exception as e:
            self.logger.error(f"Error getting liquidity profile for {symbol}: {e}")
            return None

    def _generate_liquidity_profile(self, symbol: str) -> LiquidityProfile:
        """Generate realistic liquidity profile for demo."""
        # Base profiles by symbol type
        if symbol in ['SPY', 'QQQ', 'AAPL', 'MSFT']:
            # Ultra liquid
            base_volume = np.random.uniform(50_000_000, 100_000_000)
            base_price = 150
            spread_bps = np.random.uniform(1, 3)
        elif symbol in ['TSLA', 'NVDA', 'AMD', 'META']:
            # High liquid
            base_volume = np.random.uniform(20_000_000, 50_000_000)
            base_price = 200
            spread_bps = np.random.uniform(2, 5)
        else:
            # Medium liquid
            base_volume = np.random.uniform(1_000_000, 10_000_000)
            base_price = 50
            spread_bps = np.random.uniform(3, 8)

        return LiquidityProfile(
            symbol=symbol,
            avg_daily_volume=base_volume,
            avg_daily_dollar_volume=base_volume * base_price,
            typical_spread_bps=spread_bps,
            depth_at_best=base_volume * 0.001,  # 0.1% of daily volume
            depth_5_levels=base_volume * 0.005,  # 0.5% of daily volume
            market_impact_model={'alpha': 0.6, 'beta': 0.1},
            intraday_volume_pattern=self._generate_volume_pattern(),
            volatility_20d=np.random.uniform(0.15, 0.40)
        )

    def _generate_volume_pattern(self) -> List[float]:
        """Generate realistic intraday volume pattern."""
        # U-shaped pattern with high volume at open/close
        base_pattern = [
            0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04,  # 9:30-1:30
            0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15         # 1:30-4:00
        ]
        # Add some noise
        return [max(0.01, p + np.random.normal(0, 0.01)) for p in base_pattern]


class CapacityScreener:
    """Screens strategies for capacity constraints and scalability."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.impact_model = MarketImpactModel()
        self.liquidity_provider = LiquidityDataProvider()
        self.screening_history: List[CapacityMetrics] = []

    def screen_strategy_capacity(self, strategy_name: str, strategy_category: StrategyCategory,
                               target_aum: float, universe: List[str],
                               expected_turnover: float, target_profit_bps: float,
                               holding_period_hours: float) -> CapacityMetrics:
        """Screen strategy for capacity constraints."""
        try:
            # Calculate aggregate liquidity requirements
            daily_volume_req = self._calculate_volume_requirement(
                target_aum, expected_turnover, holding_period_hours
            )

            # Get liquidity for universe
            universe_liquidity = {}
            total_available_liquidity = 0

            for symbol in universe:
                liquidity = self.liquidity_provider.get_liquidity_profile(symbol)
                if liquidity:
                    universe_liquidity[symbol] = liquidity
                    total_available_liquidity += liquidity.avg_daily_dollar_volume

            # Calculate capacity constraints
            liquidity_coverage = total_available_liquidity / daily_volume_req if daily_volume_req > 0 else float('inf')

            # Calculate market impact
            avg_trade_size = daily_volume_req / len(universe) if universe else 0
            total_impact_bps = 0

            for symbol, liquidity in universe_liquidity.items():
                if liquidity:
                    impact = self.impact_model.calculate_impact(symbol, avg_trade_size / liquidity.avg_daily_dollar_volume * liquidity.avg_daily_volume, liquidity)
                    total_impact_bps += impact['total_impact_bps'] / len(universe)

            # Estimate maximum capacity
            max_capacity = self._estimate_max_capacity(
                universe_liquidity, target_profit_bps, strategy_category
            )

            # Calculate information decay
            info_decay_half_life = self._estimate_information_decay(
                strategy_category, holding_period_hours
            )

            # Calculate competition pressure
            competition_score = self._estimate_competition_pressure(
                strategy_category, universe
            )

            # Calculate utilization rate
            utilization_rate = target_aum / max_capacity if max_capacity > 0 else 1.0

            metrics = CapacityMetrics(
                strategy_name=strategy_name,
                strategy_category=strategy_category,
                target_aum=target_aum,
                current_aum=target_aum * 0.1,  # Assume starting at 10%
                max_capacity=max_capacity,
                utilization_rate=min(1.0, utilization_rate),
                daily_volume_requirement=daily_volume_req,
                market_impact_bps=total_impact_bps,
                liquidity_coverage_ratio=liquidity_coverage,
                information_decay_half_life_hours=info_decay_half_life,
                competition_pressure_score=competition_score,
                regulatory_constraints=self._get_regulatory_constraints(strategy_category)
            )

            self.screening_history.append(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Capacity screening error for {strategy_name}: {e}")
            raise

    def _calculate_volume_requirement(self, target_aum: float, turnover: float,
                                    holding_period_hours: float) -> float:
        """Calculate daily volume requirement."""
        # Annual turnover to daily
        trading_days_per_year = 252
        daily_turnover = turnover / trading_days_per_year

        # Adjust for holding period
        intraday_factor = min(1.0, 24 / holding_period_hours)

        return target_aum * daily_turnover * intraday_factor

    def _estimate_max_capacity(self, universe_liquidity: Dict[str, LiquidityProfile],
                             target_profit_bps: float, strategy_category: StrategyCategory) -> float:
        """Estimate maximum strategy capacity."""
        if not universe_liquidity:
            return 0

        # Base capacity from liquidity
        total_liquidity = sum(liq.avg_daily_dollar_volume for liq in universe_liquidity.values())

        # Maximum participation rates by strategy type
        max_participation_rates = {
            StrategyCategory.MOMENTUM: 0.03,      # 3% max participation
            StrategyCategory.MEAN_REVERSION: 0.05, # 5% max participation
            StrategyCategory.ARBITRAGE: 0.10,      # 10% max participation
            StrategyCategory.VOLATILITY: 0.02,     # 2% max participation
            StrategyCategory.EARNINGS: 0.08,       # 8% max participation
            StrategyCategory.MACRO: 0.15           # 15% max participation
        }

        max_participation = max_participation_rates.get(strategy_category, 0.05)

        # Base capacity from liquidity constraint
        liquidity_capacity = total_liquidity * max_participation

        # Adjust for impact tolerance
        impact_tolerance_bps = target_profit_bps * 0.3  # Use max 30% of profit for impact
        impact_adjustment = min(1.0, impact_tolerance_bps / 10)  # Scale down if low tolerance

        return liquidity_capacity * impact_adjustment

    def _estimate_information_decay(self, strategy_category: StrategyCategory,
                                  holding_period_hours: float) -> float:
        """Estimate information decay half-life."""
        # Base decay rates by strategy type
        base_decay_hours = {
            StrategyCategory.MOMENTUM: 4.0,        # Fast decay
            StrategyCategory.MEAN_REVERSION: 12.0, # Medium decay
            StrategyCategory.ARBITRAGE: 0.5,       # Very fast decay
            StrategyCategory.VOLATILITY: 8.0,      # Medium-fast decay
            StrategyCategory.EARNINGS: 48.0,       # Slow decay
            StrategyCategory.MACRO: 168.0          # Very slow decay (1 week)
        }

        base_decay = base_decay_hours.get(strategy_category, 8.0)

        # Adjust based on holding period
        decay_adjustment = min(2.0, holding_period_hours / base_decay)

        return base_decay * decay_adjustment

    def _estimate_competition_pressure(self, strategy_category: StrategyCategory,
                                     universe: List[str]) -> float:
        """Estimate competition pressure score."""
        # Base competition by strategy type
        base_competition = {
            StrategyCategory.MOMENTUM: 0.8,        # High competition
            StrategyCategory.MEAN_REVERSION: 0.7,  # High competition
            StrategyCategory.ARBITRAGE: 0.9,       # Very high competition
            StrategyCategory.VOLATILITY: 0.6,      # Medium competition
            StrategyCategory.EARNINGS: 0.5,        # Medium competition
            StrategyCategory.MACRO: 0.3            # Low competition
        }

        base_score = base_competition.get(strategy_category, 0.6)

        # Adjust for universe size (smaller universe = more competition)
        universe_adjustment = max(0.5, 1.0 - len(universe) / 500)

        return min(1.0, base_score * universe_adjustment)

    def _get_regulatory_constraints(self, strategy_category: StrategyCategory) -> List[str]:
        """Get regulatory constraints by strategy type."""
        constraints = []

        if strategy_category == StrategyCategory.ARBITRAGE:
            constraints.extend(['position_limits', 'short_selling_rules'])

        if strategy_category in [StrategyCategory.MOMENTUM, StrategyCategory.VOLATILITY]:
            constraints.append('pattern_day_trader_rules')

        if strategy_category == StrategyCategory.EARNINGS:
            constraints.extend(['insider_trading_rules', 'blackout_periods'])

        return constraints

    def generate_capacity_report(self, metrics: CapacityMetrics) -> Dict[str, Any]:
        """Generate comprehensive capacity report."""
        # Determine capacity status
        if metrics.utilization_rate > 0.9:
            capacity_status = "CRITICAL"
        elif metrics.utilization_rate > 0.7:
            capacity_status = "WARNING"
        elif metrics.utilization_rate > 0.5:
            capacity_status = "GOOD"
        else:
            capacity_status = "EXCELLENT"

        # Calculate runway
        growth_rate = 0.20  # Assume 20% annual growth
        years_to_capacity = 0
        if metrics.utilization_rate < 1.0 and growth_rate > 0:
            years_to_capacity = np.log(1 / metrics.utilization_rate) / np.log(1 + growth_rate)

        # Generate recommendations
        recommendations = self._generate_capacity_recommendations(metrics)

        return {
            'strategy_name': metrics.strategy_name,
            'capacity_status': capacity_status,
            'current_utilization': f"{metrics.utilization_rate:.1%}",
            'max_capacity': f"${metrics.max_capacity:,.0f}",
            'current_aum': f"${metrics.current_aum:,.0f}",
            'years_to_capacity': f"{years_to_capacity:.1f}" if years_to_capacity < 100 else "10+",
            'daily_volume_req': f"${metrics.daily_volume_requirement:,.0f}",
            'market_impact': f"{metrics.market_impact_bps:.1f} bps",
            'liquidity_coverage': f"{metrics.liquidity_coverage_ratio:.1f}x",
            'info_decay_half_life': f"{metrics.information_decay_half_life_hours:.1f} hours",
            'competition_pressure': f"{metrics.competition_pressure_score:.1%}",
            'regulatory_constraints': metrics.regulatory_constraints,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_capacity_recommendations(self, metrics: CapacityMetrics) -> List[str]:
        """Generate actionable capacity recommendations."""
        recommendations = []

        if metrics.utilization_rate > 0.8:
            recommendations.append("Consider capacity expansion strategies")

        if metrics.market_impact_bps > 20:
            recommendations.append("Reduce position sizes or improve execution")

        if metrics.liquidity_coverage_ratio < 2.0:
            recommendations.append("Expand trading universe for better liquidity")

        if metrics.information_decay_half_life_hours < 2.0:
            recommendations.append("Focus on execution speed optimization")

        if metrics.competition_pressure_score > 0.7:
            recommendations.append("Enhance signal sophistication to reduce competition")

        if len(metrics.regulatory_constraints) > 2:
            recommendations.append("Review regulatory compliance requirements")

        if not recommendations:
            recommendations.append("Strategy has good capacity characteristics")

        return recommendations

    def compare_strategies(self, metrics_list: List[CapacityMetrics]) -> Dict[str, Any]:
        """Compare multiple strategies by capacity metrics."""
        if not metrics_list:
            return {}

        comparison = {
            'strategies': [],
            'rankings': {
                'by_capacity': [],
                'by_efficiency': [],
                'by_scalability': []
            },
            'summary_stats': {}
        }

        for metrics in metrics_list:
            strategy_data = {
                'name': metrics.strategy_name,
                'category': metrics.strategy_category.value,
                'max_capacity': metrics.max_capacity,
                'utilization_rate': metrics.utilization_rate,
                'market_impact_bps': metrics.market_impact_bps,
                'efficiency_score': self._calculate_efficiency_score(metrics),
                'scalability_score': self._calculate_scalability_score(metrics)
            }
            comparison['strategies'].append(strategy_data)

        # Rankings
        strategies = comparison['strategies']
        comparison['rankings']['by_capacity'] = sorted(strategies, key=lambda x: x['max_capacity'], reverse=True)
        comparison['rankings']['by_efficiency'] = sorted(strategies, key=lambda x: x['efficiency_score'], reverse=True)
        comparison['rankings']['by_scalability'] = sorted(strategies, key=lambda x: x['scalability_score'], reverse=True)

        # Summary statistics
        capacities = [s['max_capacity'] for s in strategies]
        impacts = [s['market_impact_bps'] for s in strategies]

        comparison['summary_stats'] = {
            'total_capacity': sum(capacities),
            'avg_capacity': np.mean(capacities),
            'median_capacity': np.median(capacities),
            'avg_impact_bps': np.mean(impacts),
            'capacity_concentration': max(capacities) / sum(capacities) if sum(capacities) > 0 else 0
        }

        return comparison

    def _calculate_efficiency_score(self, metrics: CapacityMetrics) -> float:
        """Calculate efficiency score (capacity per unit of impact)."""
        if metrics.market_impact_bps > 0:
            return metrics.max_capacity / metrics.market_impact_bps / 1_000_000
        return 0

    def _calculate_scalability_score(self, metrics: CapacityMetrics) -> float:
        """Calculate scalability score (combines multiple factors)."""
        # Components of scalability
        utilization_score = 1.0 - metrics.utilization_rate
        liquidity_score = min(1.0, metrics.liquidity_coverage_ratio / 5.0)
        decay_score = min(1.0, metrics.information_decay_half_life_hours / 24.0)
        competition_score = 1.0 - metrics.competition_pressure_score

        # Weighted average
        scalability = (
            utilization_score * 0.3 +
            liquidity_score * 0.3 +
            decay_score * 0.2 +
            competition_score * 0.2
        )

        return scalability

    def get_screening_summary(self) -> Dict[str, Any]:
        """Get summary of all capacity screenings."""
        if not self.screening_history:
            return {'message': 'No strategies screened yet'}

        total_capacity = sum(m.max_capacity for m in self.screening_history)
        total_current_aum = sum(m.current_aum for m in self.screening_history)

        by_category = defaultdict(list)
        for metrics in self.screening_history:
            by_category[metrics.strategy_category.value].append(metrics)

        category_summary = {}
        for category, metrics_list in by_category.items():
            category_summary[category] = {
                'count': len(metrics_list),
                'total_capacity': sum(m.max_capacity for m in metrics_list),
                'avg_impact_bps': np.mean([m.market_impact_bps for m in metrics_list]),
                'avg_utilization': np.mean([m.utilization_rate for m in metrics_list])
            }

        return {
            'total_strategies_screened': len(self.screening_history),
            'total_max_capacity': total_capacity,
            'total_current_aum': total_current_aum,
            'overall_utilization': total_current_aum / total_capacity if total_capacity > 0 else 0,
            'by_category': category_summary,
            'last_screening': max([m.strategy_name for m in self.screening_history]) if self.screening_history else None
        }


# Example usage and testing
if __name__ == "__main__":
    def demo_capacity_screening():
        print("=== Trading Capacity Screening Demo ===")

        screener = CapacityScreener()

        # Screen different strategy types
        strategies_to_screen = [
            {
                'name': 'Momentum_SPY_Components',
                'category': StrategyCategory.MOMENTUM,
                'target_aum': 50_000_000,
                'universe': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 20,  # 100 stocks
                'turnover': 12.0,  # 12x annual turnover
                'target_profit_bps': 150,
                'holding_period_hours': 6
            },
            {
                'name': 'Mean_Reversion_ETFs',
                'category': StrategyCategory.MEAN_REVERSION,
                'target_aum': 25_000_000,
                'universe': ['SPY', 'QQQ', 'IWM', 'VTI', 'EFA'] * 10,  # 50 ETFs
                'turnover': 6.0,   # 6x annual turnover
                'target_profit_bps': 100,
                'holding_period_hours': 24
            },
            {
                'name': 'Earnings_Drift_Large_Cap',
                'category': StrategyCategory.EARNINGS,
                'target_aum': 100_000_000,
                'universe': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA'] * 5,  # 35 stocks
                'turnover': 4.0,   # 4x annual turnover
                'target_profit_bps': 200,
                'holding_period_hours': 72
            }
        ]

        screened_metrics = []

        for strategy in strategies_to_screen:
            print(f"\nScreening: {strategy['name']}")

            metrics = screener.screen_strategy_capacity(
                strategy['name'],
                strategy['category'],
                strategy['target_aum'],
                strategy['universe'],
                strategy['turnover'],
                strategy['target_profit_bps'],
                strategy['holding_period_hours']
            )

            screened_metrics.append(metrics)

            # Generate report
            report = screener.generate_capacity_report(metrics)
            print(f"  Status: {report['capacity_status']}")
            print(f"  Max Capacity: {report['max_capacity']}")
            print(f"  Utilization: {report['current_utilization']}")
            print(f"  Market Impact: {report['market_impact']}")
            print(f"  Liquidity Coverage: {report['liquidity_coverage']}")

            if report['recommendations']:
                print("  Recommendations:")
                for rec in report['recommendations'][:2]:
                    print(f"    • {rec}")

        # Compare strategies
        print("\n=== Strategy Comparison ===")
        comparison = screener.compare_strategies(screened_metrics)

        print(f"Total Combined Capacity: ${comparison['summary_stats']['total_capacity']:,.0f}")
        print(f"Average Impact: {comparison['summary_stats']['avg_impact_bps']:.1f} bps")

        print("\nTop Strategies by Capacity:")
        for i, strategy in enumerate(comparison['rankings']['by_capacity'][:3]):
            print(f"  {i+1}. {strategy['name']}: ${strategy['max_capacity']:,.0f}")

        print("\nTop Strategies by Efficiency:")
        for i, strategy in enumerate(comparison['rankings']['by_efficiency'][:3]):
            print(f"  {i+1}. {strategy['name']}: {strategy['efficiency_score']:.1f}")

        # Get overall summary
        summary = screener.get_screening_summary()
        print("\n=== Screening Summary ===")
        print(f"Strategies Screened: {summary['total_strategies_screened']}")
        print(f"Total Capacity: ${summary['total_max_capacity']:,.0f}")
        print(f"Overall Utilization: {summary['overall_utilization']:.1%}")

    demo_capacity_screening()