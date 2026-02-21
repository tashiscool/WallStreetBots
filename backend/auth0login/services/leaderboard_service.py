"""
Leaderboard Service for Strategy Rankings and Comparisons.

Provides strategy performance rankings, side-by-side comparisons,
rank history tracking, and hypothetical portfolio calculations.
"""
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import ClassVar, Dict, List, Optional, Any, Tuple

from django.db.models import Avg, Max, Min, F, Q, Count
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)


class LeaderboardService:
    """
    Service for strategy leaderboard and comparison operations.

    Provides rankings, comparisons, historical tracking, and
    hypothetical portfolio analysis across all trading strategies.
    """

    # Available strategies in the system
    STRATEGY_REGISTRY: ClassVar[dict] = {
        'wsb_dip_bot': {
            'name': 'WSB Dip Bot',
            'description': 'Momentum-based dip buying on volatile stocks',
            'risk_level': 'high',
            'category': 'momentum'
        },
        'wheel_strategy': {
            'name': 'Wheel Strategy',
            'description': 'Cash-secured puts and covered calls',
            'risk_level': 'medium',
            'category': 'income'
        },
        'momentum_weeklies': {
            'name': 'Momentum Weeklies',
            'description': 'Weekly options on trending stocks',
            'risk_level': 'high',
            'category': 'momentum'
        },
        'earnings_protection': {
            'name': 'Earnings Protection',
            'description': 'Hedging around earnings announcements',
            'risk_level': 'medium',
            'category': 'hedging'
        },
        'debit_spreads': {
            'name': 'Debit Spreads',
            'description': 'Directional option spreads with limited risk',
            'risk_level': 'medium',
            'category': 'directional'
        },
        'leaps_tracker': {
            'name': 'LEAPS Tracker',
            'description': 'Long-term equity anticipation securities',
            'risk_level': 'medium',
            'category': 'long_term'
        },
        'lotto_scanner': {
            'name': 'Lotto Scanner',
            'description': 'High-risk short-term option plays',
            'risk_level': 'very_high',
            'category': 'speculative'
        },
        'swing_trading': {
            'name': 'Swing Trading',
            'description': 'Multi-day stock position trades',
            'risk_level': 'medium',
            'category': 'swing'
        },
        'spx_credit_spreads': {
            'name': 'SPX Credit Spreads',
            'description': 'Index credit spreads for income',
            'risk_level': 'medium',
            'category': 'income'
        },
        'index_baseline': {
            'name': 'Index Baseline',
            'description': 'SPY/QQQ baseline comparison',
            'risk_level': 'low',
            'category': 'benchmark'
        },
        'crypto_dip_bot': {
            'name': 'Crypto Dip Bot',
            'description': '24/7 crypto dip buying',
            'risk_level': 'very_high',
            'category': 'crypto'
        },
        'exotic_spreads': {
            'name': 'Exotic Spreads',
            'description': 'Iron condors, butterflies, calendars',
            'risk_level': 'medium',
            'category': 'income'
        }
    }

    # Metrics available for ranking
    RANKING_METRICS: ClassVar[dict] = {
        'sharpe_ratio': {'name': 'Sharpe Ratio', 'higher_better': True},
        'sortino_ratio': {'name': 'Sortino Ratio', 'higher_better': True},
        'total_return_pct': {'name': 'Total Return %', 'higher_better': True},
        'win_rate': {'name': 'Win Rate %', 'higher_better': True},
        'profit_factor': {'name': 'Profit Factor', 'higher_better': True},
        'max_drawdown_pct': {'name': 'Max Drawdown %', 'higher_better': False},
        'calmar_ratio': {'name': 'Calmar Ratio', 'higher_better': True},
        'risk_adjusted_score': {'name': 'Risk-Adjusted Score', 'higher_better': True},
    }

    # Period mappings
    PERIOD_DAYS: ClassVar[dict] = {
        '1W': 7,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365,
        'YTD': None,  # Calculated dynamically
        'ALL': None,  # All available data
    }

    def __init__(self, user: Optional[User] = None):
        """
        Initialize the leaderboard service.

        Args:
            user: Optional user for personalized rankings
        """
        self.user = user

    def get_leaderboard(
        self,
        period: str = '1M',
        metric: str = 'sharpe_ratio',
        limit: int = 10,
        category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ranked strategies for a period by specified metric.

        Args:
            period: Time period ('1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL')
            metric: Ranking metric (sharpe_ratio, total_return_pct, etc.)
            limit: Maximum strategies to return
            category: Optional category filter

        Returns:
            Leaderboard data with rankings, strategies, and metadata
        """
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        # Calculate date range
        start_date, end_date = self._get_date_range(period)

        # Get the most recent snapshots for each strategy
        queryset = StrategyPerformanceSnapshot.objects.filter(
            snapshot_date__gte=start_date,
            snapshot_date__lte=end_date
        )

        # Get latest snapshot per strategy
        latest_snapshots = queryset.values('strategy_name').annotate(
            latest_date=Max('snapshot_date')
        )

        # Fetch full snapshot data for latest dates
        snapshot_ids = []
        for item in latest_snapshots:
            snapshot = StrategyPerformanceSnapshot.objects.filter(
                strategy_name=item['strategy_name'],
                snapshot_date=item['latest_date']
            ).first()
            if snapshot:
                snapshot_ids.append(snapshot.id)

        snapshots = StrategyPerformanceSnapshot.objects.filter(
            id__in=snapshot_ids
        ).order_by(self._get_ordering(metric))

        # Filter by category if specified
        if category:
            strategy_names = [
                name for name, info in self.STRATEGY_REGISTRY.items()
                if info.get('category') == category
            ]
            snapshots = snapshots.filter(strategy_name__in=strategy_names)

        # Build leaderboard entries
        leaderboard = []
        for rank, snapshot in enumerate(snapshots[:limit], 1):
            entry = snapshot.to_leaderboard_entry(rank)

            # Add strategy metadata
            strategy_info = self.STRATEGY_REGISTRY.get(snapshot.strategy_name, {})
            entry['strategy_display_name'] = strategy_info.get('name', snapshot.strategy_name)
            entry['description'] = strategy_info.get('description', '')
            entry['risk_level'] = strategy_info.get('risk_level', 'unknown')
            entry['category'] = strategy_info.get('category', 'other')

            # Calculate trend
            entry['trend'] = snapshot.get_trend_vs_previous(metric)

            leaderboard.append(entry)

        return {
            'period': period,
            'metric': metric,
            'metric_name': self.RANKING_METRICS.get(metric, {}).get('name', metric),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_strategies': len(leaderboard),
            'leaderboard': leaderboard,
            'available_metrics': list(self.RANKING_METRICS.keys()),
            'available_periods': list(self.PERIOD_DAYS.keys()),
            'generated_at': datetime.now().isoformat()
        }

    def compare_strategies(
        self,
        strategy_names: List[str],
        period: str = '1M'
    ) -> Dict[str, Any]:
        """
        Side-by-side comparison of specific strategies.

        Args:
            strategy_names: List of strategy names to compare
            period: Time period for comparison

        Returns:
            Comparison data with metrics for each strategy
        """
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        if not strategy_names:
            return {'error': 'No strategies specified', 'strategies': []}

        start_date, end_date = self._get_date_range(period)

        comparison = []
        for strategy_name in strategy_names:
            # Get latest snapshot for this strategy in period
            snapshot = StrategyPerformanceSnapshot.objects.filter(
                strategy_name=strategy_name,
                snapshot_date__gte=start_date,
                snapshot_date__lte=end_date
            ).order_by('-snapshot_date').first()

            if snapshot:
                strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})

                comparison.append({
                    'strategy_name': strategy_name,
                    'display_name': strategy_info.get('name', strategy_name),
                    'description': strategy_info.get('description', ''),
                    'risk_level': strategy_info.get('risk_level', 'unknown'),
                    'category': strategy_info.get('category', 'other'),
                    'snapshot_date': snapshot.snapshot_date.isoformat(),
                    'metrics': {
                        'total_return_pct': float(snapshot.total_return_pct),
                        'sharpe_ratio': float(snapshot.sharpe_ratio),
                        'sortino_ratio': float(snapshot.sortino_ratio),
                        'max_drawdown_pct': float(snapshot.max_drawdown_pct),
                        'win_rate': float(snapshot.win_rate),
                        'profit_factor': float(snapshot.profit_factor),
                        'trades_count': snapshot.trades_count,
                        'avg_trade_pnl': float(snapshot.avg_trade_pnl),
                        'volatility': float(snapshot.volatility),
                        'calmar_ratio': float(snapshot.calmar_ratio),
                        'risk_adjusted_score': snapshot.get_risk_adjusted_score(),
                    },
                    'benchmark_comparison': {
                        'vs_spy_return': float(snapshot.vs_spy_return),
                        'beta': float(snapshot.beta),
                        'alpha': float(snapshot.alpha),
                        'correlation_spy': float(snapshot.correlation_spy),
                    },
                    'rankings': {
                        'rank_by_sharpe': snapshot.rank_by_sharpe,
                        'rank_by_return': snapshot.rank_by_return,
                        'rank_by_risk_adjusted': snapshot.rank_by_risk_adjusted,
                    }
                })
            else:
                # Strategy has no data for this period
                strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})
                comparison.append({
                    'strategy_name': strategy_name,
                    'display_name': strategy_info.get('name', strategy_name),
                    'no_data': True,
                    'message': f'No performance data for {period}'
                })

        # Calculate best/worst for each metric
        metrics_summary = self._calculate_comparison_summary(comparison)

        return {
            'period': period,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'strategies_compared': len(strategy_names),
            'comparison': comparison,
            'metrics_summary': metrics_summary,
            'generated_at': datetime.now().isoformat()
        }

    def get_strategy_rank_history(
        self,
        strategy_name: str,
        metric: str = 'sharpe_ratio',
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Get how a strategy's ranking has changed over time.

        Args:
            strategy_name: Strategy to track
            metric: Metric to track ranking by
            days: Number of days of history

        Returns:
            Rank history with trend analysis
        """
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        # Get all snapshots for this strategy
        snapshots = StrategyPerformanceSnapshot.objects.filter(
            strategy_name=strategy_name,
            snapshot_date__gte=start_date,
            snapshot_date__lte=end_date
        ).order_by('snapshot_date')

        history = []
        for snapshot in snapshots:
            rank_field = f'rank_by_{metric.replace("_ratio", "").replace("_pct", "")}'
            rank = getattr(snapshot, rank_field, None) or snapshot.rank_by_sharpe

            metric_value = getattr(snapshot, metric, 0)

            history.append({
                'date': snapshot.snapshot_date.isoformat(),
                'rank': rank,
                'metric_value': float(metric_value),
                'total_strategies': snapshot.total_strategies_ranked or 10,
            })

        # Calculate trend
        if len(history) >= 2:
            first_rank = history[0]['rank'] or 0
            last_rank = history[-1]['rank'] or 0
            rank_change = first_rank - last_rank  # Positive = improved

            if rank_change > 0:
                trend = 'improving'
            elif rank_change < 0:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
            rank_change = 0

        strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})

        return {
            'strategy_name': strategy_name,
            'display_name': strategy_info.get('name', strategy_name),
            'metric': metric,
            'metric_name': self.RANKING_METRICS.get(metric, {}).get('name', metric),
            'days': days,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'history': history,
            'trend': trend,
            'rank_change': rank_change,
            'current_rank': history[-1]['rank'] if history else None,
            'best_rank': min((h['rank'] for h in history if h['rank']), default=None),
            'worst_rank': max((h['rank'] for h in history if h['rank']), default=None),
            'generated_at': datetime.now().isoformat()
        }

    def calculate_hypothetical_portfolio(
        self,
        allocations: Dict[str, float],
        period: str = '1M',
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate hypothetical portfolio performance with given allocations.

        Args:
            allocations: Dict of {strategy_name: allocation_pct}
            period: Time period to analyze
            initial_capital: Starting capital amount

        Returns:
            Portfolio performance metrics and breakdown
        """
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        # Validate allocations sum to 100%
        total_allocation = sum(allocations.values())
        if abs(total_allocation - 100.0) > 0.01:
            return {
                'error': f'Allocations must sum to 100%, got {total_allocation}%',
                'allocations': allocations
            }

        start_date, end_date = self._get_date_range(period)

        # Calculate weighted portfolio metrics
        portfolio_return = 0.0
        portfolio_sharpe = 0.0
        portfolio_max_dd = 0.0
        portfolio_win_rate = 0.0
        portfolio_volatility = 0.0

        strategy_breakdown = []
        total_trades = 0

        for strategy_name, allocation_pct in allocations.items():
            allocation_weight = allocation_pct / 100.0
            strategy_capital = initial_capital * allocation_weight

            # Get latest snapshot
            snapshot = StrategyPerformanceSnapshot.objects.filter(
                strategy_name=strategy_name,
                snapshot_date__gte=start_date,
                snapshot_date__lte=end_date
            ).order_by('-snapshot_date').first()

            if snapshot:
                strategy_return = float(snapshot.total_return_pct)
                strategy_pnl = strategy_capital * (strategy_return / 100.0)

                # Weighted contribution
                portfolio_return += strategy_return * allocation_weight
                portfolio_sharpe += float(snapshot.sharpe_ratio) * allocation_weight
                portfolio_max_dd = max(portfolio_max_dd, float(snapshot.max_drawdown_pct))
                portfolio_win_rate += float(snapshot.win_rate) * allocation_weight
                portfolio_volatility += float(snapshot.volatility) * allocation_weight
                total_trades += snapshot.trades_count

                strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})

                strategy_breakdown.append({
                    'strategy_name': strategy_name,
                    'display_name': strategy_info.get('name', strategy_name),
                    'allocation_pct': allocation_pct,
                    'capital_allocated': strategy_capital,
                    'return_pct': strategy_return,
                    'pnl': strategy_pnl,
                    'contribution_to_return': strategy_return * allocation_weight,
                    'sharpe_ratio': float(snapshot.sharpe_ratio),
                    'max_drawdown_pct': float(snapshot.max_drawdown_pct),
                    'trades': snapshot.trades_count,
                })
            else:
                strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})
                strategy_breakdown.append({
                    'strategy_name': strategy_name,
                    'display_name': strategy_info.get('name', strategy_name),
                    'allocation_pct': allocation_pct,
                    'capital_allocated': strategy_capital,
                    'no_data': True,
                    'message': f'No data for {period}'
                })

        # Calculate portfolio-level P&L
        portfolio_pnl = initial_capital * (portfolio_return / 100.0)
        final_value = initial_capital + portfolio_pnl

        # Get SPY benchmark for comparison
        spy_snapshot = StrategyPerformanceSnapshot.objects.filter(
            strategy_name='index_baseline',
            snapshot_date__gte=start_date,
            snapshot_date__lte=end_date
        ).order_by('-snapshot_date').first()

        benchmark_return = float(spy_snapshot.benchmark_return_pct) if spy_snapshot else 0.0

        return {
            'period': period,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'initial_capital': initial_capital,
            'final_value': final_value,
            'portfolio_metrics': {
                'total_return_pct': round(portfolio_return, 2),
                'total_pnl': round(portfolio_pnl, 2),
                'sharpe_ratio': round(portfolio_sharpe, 2),
                'max_drawdown_pct': round(portfolio_max_dd, 2),
                'win_rate': round(portfolio_win_rate, 2),
                'volatility': round(portfolio_volatility, 2),
                'total_trades': total_trades,
            },
            'benchmark_comparison': {
                'spy_return_pct': benchmark_return,
                'vs_benchmark': round(portfolio_return - benchmark_return, 2),
                'beat_benchmark': portfolio_return > benchmark_return,
            },
            'strategy_breakdown': strategy_breakdown,
            'allocations': allocations,
            'diversification_score': self._calculate_diversification_score(strategy_breakdown),
            'generated_at': datetime.now().isoformat()
        }

    def get_top_performers(
        self,
        period: str = '1M',
        count: int = 3
    ) -> Dict[str, Any]:
        """
        Get top performing strategies across different metrics.

        Returns:
            Top performers by different criteria
        """
        result = {
            'period': period,
            'categories': {}
        }

        for metric in ['sharpe_ratio', 'total_return_pct', 'win_rate']:
            leaderboard = self.get_leaderboard(
                period=period,
                metric=metric,
                limit=count
            )
            result['categories'][metric] = {
                'metric_name': self.RANKING_METRICS[metric]['name'],
                'top_strategies': leaderboard['leaderboard']
            }

        return result

    def get_strategy_details(
        self,
        strategy_name: str,
        period: str = '1M'
    ) -> Dict[str, Any]:
        """
        Get comprehensive details for a single strategy.

        Args:
            strategy_name: Strategy to get details for
            period: Time period for metrics

        Returns:
            Full strategy details with metrics and history
        """
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        start_date, end_date = self._get_date_range(period)

        # Get latest snapshot
        snapshot = StrategyPerformanceSnapshot.objects.filter(
            strategy_name=strategy_name,
            snapshot_date__gte=start_date,
            snapshot_date__lte=end_date
        ).order_by('-snapshot_date').first()

        strategy_info = self.STRATEGY_REGISTRY.get(strategy_name, {})

        if not snapshot:
            return {
                'strategy_name': strategy_name,
                'display_name': strategy_info.get('name', strategy_name),
                'description': strategy_info.get('description', ''),
                'risk_level': strategy_info.get('risk_level', 'unknown'),
                'category': strategy_info.get('category', 'other'),
                'no_data': True,
                'message': f'No performance data for {period}'
            }

        # Get rank history
        rank_history = self.get_strategy_rank_history(
            strategy_name, 'sharpe_ratio', days=90
        )

        return {
            'strategy_name': strategy_name,
            'display_name': strategy_info.get('name', strategy_name),
            'description': strategy_info.get('description', ''),
            'risk_level': strategy_info.get('risk_level', 'unknown'),
            'category': strategy_info.get('category', 'other'),
            'period': period,
            'snapshot_date': snapshot.snapshot_date.isoformat(),
            'performance': {
                'total_return_pct': float(snapshot.total_return_pct),
                'sharpe_ratio': float(snapshot.sharpe_ratio),
                'sortino_ratio': float(snapshot.sortino_ratio),
                'max_drawdown_pct': float(snapshot.max_drawdown_pct),
                'win_rate': float(snapshot.win_rate),
                'profit_factor': float(snapshot.profit_factor),
                'calmar_ratio': float(snapshot.calmar_ratio),
                'risk_adjusted_score': snapshot.get_risk_adjusted_score(),
            },
            'trading_stats': {
                'trades_count': snapshot.trades_count,
                'winning_trades': snapshot.winning_trades,
                'losing_trades': snapshot.losing_trades,
                'avg_trade_pnl': float(snapshot.avg_trade_pnl),
                'best_trade_pnl': float(snapshot.best_trade_pnl),
                'worst_trade_pnl': float(snapshot.worst_trade_pnl),
                'avg_hold_duration_hours': float(snapshot.avg_hold_duration_hours),
            },
            'risk_metrics': {
                'volatility': float(snapshot.volatility),
                'var_95': float(snapshot.var_95),
                'beta': float(snapshot.beta),
            },
            'benchmark_comparison': {
                'benchmark_return_pct': float(snapshot.benchmark_return_pct),
                'vs_spy_return': float(snapshot.vs_spy_return),
                'alpha': float(snapshot.alpha),
                'correlation_spy': float(snapshot.correlation_spy),
            },
            'rankings': {
                'rank_by_sharpe': snapshot.rank_by_sharpe,
                'rank_by_return': snapshot.rank_by_return,
                'rank_by_risk_adjusted': snapshot.rank_by_risk_adjusted,
                'total_strategies': snapshot.total_strategies_ranked,
            },
            'rank_history': rank_history,
            'generated_at': datetime.now().isoformat()
        }

    def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get list of all available strategies with metadata."""
        strategies = []
        for name, info in self.STRATEGY_REGISTRY.items():
            strategies.append({
                'strategy_name': name,
                'display_name': info['name'],
                'description': info['description'],
                'risk_level': info['risk_level'],
                'category': info['category'],
            })
        return strategies

    def _get_date_range(self, period: str) -> Tuple[date, date]:
        """Calculate start and end dates for a period."""
        end_date = date.today()

        if period == 'YTD':
            start_date = date(end_date.year, 1, 1)
        elif period == 'ALL':
            start_date = date(2020, 1, 1)  # Beginning of data
        else:
            days = self.PERIOD_DAYS.get(period, 30)
            start_date = end_date - timedelta(days=days)

        return start_date, end_date

    def _get_ordering(self, metric: str) -> str:
        """Get Django ORM ordering for metric (higher is better by default)."""
        metric_info = self.RANKING_METRICS.get(metric, {'higher_better': True})

        if metric_info['higher_better']:
            return f'-{metric}'
        else:
            return metric

    def _calculate_comparison_summary(
        self,
        comparison: List[Dict]
    ) -> Dict[str, Dict]:
        """Calculate best/worst for each metric in comparison."""
        summary = {}

        metrics_to_compare = [
            'total_return_pct', 'sharpe_ratio', 'win_rate', 'max_drawdown_pct'
        ]

        for metric in metrics_to_compare:
            values = []
            for item in comparison:
                if not item.get('no_data') and 'metrics' in item:
                    values.append({
                        'strategy': item['strategy_name'],
                        'value': item['metrics'].get(metric, 0)
                    })

            if values:
                metric_info = self.RANKING_METRICS.get(metric, {'higher_better': True})

                if metric_info['higher_better']:
                    best = max(values, key=lambda x: x['value'])
                    worst = min(values, key=lambda x: x['value'])
                else:
                    best = min(values, key=lambda x: x['value'])
                    worst = max(values, key=lambda x: x['value'])

                summary[metric] = {
                    'best': best,
                    'worst': worst,
                }

        return summary

    def _calculate_diversification_score(
        self,
        breakdown: List[Dict]
    ) -> float:
        """
        Calculate portfolio diversification score (0-100).

        Higher score = more diversified across categories and risk levels.
        """
        if not breakdown:
            return 0.0

        categories = set()
        risk_levels = set()

        for item in breakdown:
            if not item.get('no_data'):
                strategy_name = item['strategy_name']
                info = self.STRATEGY_REGISTRY.get(strategy_name, {})
                if info.get('category'):
                    categories.add(info['category'])
                if info.get('risk_level'):
                    risk_levels.add(info['risk_level'])

        # More categories and risk levels = more diversified
        category_score = min(len(categories) / 5.0, 1.0) * 50
        risk_score = min(len(risk_levels) / 4.0, 1.0) * 50

        return round(category_score + risk_score, 1)
