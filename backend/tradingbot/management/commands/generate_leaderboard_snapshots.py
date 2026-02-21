"""
Management command to generate daily strategy performance snapshots.

Usage:
    # Generate daily snapshots for all strategies
    python manage.py generate_leaderboard_snapshots

    # Generate for specific period
    python manage.py generate_leaderboard_snapshots --period weekly

    # Generate for specific date
    python manage.py generate_leaderboard_snapshots --date 2025-01-15

    # Dry run (preview without saving)
    python manage.py generate_leaderboard_snapshots --dry-run

    # Force regenerate (overwrite existing)
    python manage.py generate_leaderboard_snapshots --force

Recommended scheduling:
    # Daily at 9 PM ET (after market close processing)
    0 21 * * 1-5 cd /path/to/project && python manage.py generate_leaderboard_snapshots --period daily

    # Weekly on Sunday
    0 10 * * 0 cd /path/to/project && python manage.py generate_leaderboard_snapshots --period weekly

    # Monthly on 1st of month
    0 10 1 * * cd /path/to/project && python manage.py generate_leaderboard_snapshots --period monthly
"""
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import ClassVar, Dict, List, Optional, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Avg, Sum, Count, Max, Min

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate strategy performance snapshots for leaderboard rankings'

    # Strategy registry - mirrors LeaderboardService
    STRATEGIES: ClassVar[list] = [
        'wsb_dip_bot',
        'wheel_strategy',
        'momentum_weeklies',
        'earnings_protection',
        'debit_spreads',
        'leaps_tracker',
        'lotto_scanner',
        'swing_trading',
        'spx_credit_spreads',
        'index_baseline',
        'crypto_dip_bot',
        'exotic_spreads',
    ]

    def add_arguments(self, parser):
        parser.add_argument(
            '--period',
            type=str,
            default='daily',
            choices=['daily', 'weekly', 'monthly'],
            help='Snapshot period type (default: daily)'
        )
        parser.add_argument(
            '--date',
            type=str,
            help='Specific date for snapshot (YYYY-MM-DD format)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview snapshots without saving to database'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Overwrite existing snapshots for the date'
        )
        parser.add_argument(
            '--strategy',
            type=str,
            help='Generate for specific strategy only'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed output'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        period = options['period']
        dry_run = options['dry_run']
        force = options['force']
        verbose = options['verbose']
        specific_strategy = options.get('strategy')

        # Determine snapshot date
        if options['date']:
            try:
                snapshot_date = datetime.strptime(options['date'], '%Y-%m-%d').date()
            except ValueError as e:
                raise CommandError('Invalid date format. Use YYYY-MM-DD') from e
        else:
            snapshot_date = date.today()

        self.stdout.write(f"\nGenerating {period} snapshots for {snapshot_date}")

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - No changes will be saved'))

        # Determine strategies to process
        strategies = [specific_strategy] if specific_strategy else self.STRATEGIES

        snapshots_created = 0
        snapshots_updated = 0
        snapshots_skipped = 0

        for strategy_name in strategies:
            try:
                # Check if snapshot exists
                existing = StrategyPerformanceSnapshot.objects.filter(
                    strategy_name=strategy_name,
                    snapshot_date=snapshot_date,
                    period=period
                ).first()

                if existing and not force:
                    if verbose:
                        self.stdout.write(f"  Skipping {strategy_name} - snapshot exists")
                    snapshots_skipped += 1
                    continue

                # Generate metrics for this strategy
                metrics = self._calculate_strategy_metrics(
                    strategy_name, snapshot_date, period
                )

                if verbose:
                    self.stdout.write(f"\n  {strategy_name}:")
                    self.stdout.write(f"    Return: {metrics['total_return_pct']:.2f}%")
                    self.stdout.write(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
                    self.stdout.write(f"    Trades: {metrics['trades_count']}")

                if dry_run:
                    snapshots_created += 1
                    continue

                # Create or update snapshot
                with transaction.atomic():
                    if existing:
                        for key, value in metrics.items():
                            setattr(existing, key, value)
                        existing.save()
                        snapshots_updated += 1
                    else:
                        StrategyPerformanceSnapshot.objects.create(
                            strategy_name=strategy_name,
                            snapshot_date=snapshot_date,
                            period=period,
                            **metrics
                        )
                        snapshots_created += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  Error processing {strategy_name}: {e}")
                )
                logger.exception(f"Error generating snapshot for {strategy_name}")

        # Update rankings after all snapshots created
        if not dry_run:
            self._update_rankings(snapshot_date, period)

        # Summary
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write(
            self.style.SUCCESS(f"Snapshots created: {snapshots_created}")
        )
        self.stdout.write(
            self.style.SUCCESS(f"Snapshots updated: {snapshots_updated}")
        )
        self.stdout.write(f"Snapshots skipped: {snapshots_skipped}")
        self.stdout.write("=" * 50 + "\n")

    def _calculate_strategy_metrics(
        self,
        strategy_name: str,
        snapshot_date: date,
        period: str
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for a strategy.

        In production, this would pull from actual trade history.
        Here we simulate based on strategy characteristics.
        """
        from backend.tradingbot.models.models import TradeReasoning

        # Calculate period range
        if period == 'daily':
            start_date = snapshot_date - timedelta(days=1)
        elif period == 'weekly':
            start_date = snapshot_date - timedelta(days=7)
        else:  # monthly
            start_date = snapshot_date - timedelta(days=30)

        # Try to get real trade data
        trades = TradeReasoning.objects.filter(
            strategy_name=strategy_name,
            entry_time__date__gte=start_date,
            entry_time__date__lte=snapshot_date
        )

        trades_count = trades.count()
        winning_trades = trades.filter(actual_pnl__gt=0).count()
        losing_trades = trades.filter(actual_pnl__lt=0).count()

        # Calculate P&L metrics
        if trades_count > 0:
            pnl_stats = trades.aggregate(
                total_pnl=Sum('actual_pnl'),
                avg_pnl=Avg('actual_pnl'),
                max_pnl=Max('actual_pnl'),
                min_pnl=Min('actual_pnl'),
            )
            total_pnl = float(pnl_stats['total_pnl'] or 0)
            avg_pnl = float(pnl_stats['avg_pnl'] or 0)
            best_pnl = float(pnl_stats['max_pnl'] or 0)
            worst_pnl = float(pnl_stats['min_pnl'] or 0)
            win_rate = (winning_trades / trades_count) * 100 if trades_count > 0 else 0

            # Calculate returns
            # Assume average position size of $10000 for percentage calc
            avg_position = 10000
            total_return_pct = (total_pnl / (avg_position * max(trades_count, 1))) * 100
        else:
            # No observed trades in the period: return conservative zero metrics.
            return self._get_zero_metrics()

        # Calculate Sharpe ratio approximation
        # In production, use actual daily returns std dev
        volatility = abs(total_return_pct) * 0.5 if total_return_pct != 0 else 10.0
        risk_free_rate = 5.0  # 5% annual
        period_rf = risk_free_rate / 252 * (7 if period == 'weekly' else 30 if period == 'monthly' else 1)

        excess_return = total_return_pct - period_rf
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino - only downside deviation
        negative_trades = trades.filter(actual_pnl__lt=0)
        if negative_trades.exists():
            downside_vol = abs(float(negative_trades.aggregate(
                avg_loss=Avg('actual_pnl')
            )['avg_loss'] or 0))
            sortino_ratio = excess_return / (downside_vol / avg_position * 100) if downside_vol > 0 else sharpe_ratio
        else:
            sortino_ratio = sharpe_ratio * 1.2

        # Calculate max drawdown (simplified)
        max_drawdown = abs(worst_pnl / avg_position * 100) if worst_pnl < 0 else 0

        # Profit factor
        gross_profit = float(trades.filter(
            actual_pnl__gt=0
        ).aggregate(total=Sum('actual_pnl'))['total'] or 0)
        gross_loss = abs(float(trades.filter(
            actual_pnl__lt=0
        ).aggregate(total=Sum('actual_pnl'))['total'] or 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0

        # Calmar ratio
        calmar_ratio = total_return_pct / max_drawdown if max_drawdown > 0 else total_return_pct

        # Get SPY benchmark return (simplified)
        benchmark_return = self._get_benchmark_return(start_date, snapshot_date)

        return {
            'total_return_pct': Decimal(str(round(total_return_pct, 4))),
            'sharpe_ratio': Decimal(str(round(sharpe_ratio, 4))),
            'sortino_ratio': Decimal(str(round(sortino_ratio, 4))),
            'max_drawdown_pct': Decimal(str(round(max_drawdown, 4))),
            'win_rate': Decimal(str(round(win_rate, 2))),
            'profit_factor': Decimal(str(round(min(profit_factor, 10), 4))),
            'trades_count': trades_count,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_trade_pnl': Decimal(str(round(avg_pnl, 2))),
            'best_trade_pnl': Decimal(str(round(best_pnl, 2))),
            'worst_trade_pnl': Decimal(str(round(worst_pnl, 2))),
            'avg_hold_duration_hours': Decimal('24.0'),  # Default
            'volatility': Decimal(str(round(volatility, 4))),
            'var_95': Decimal(str(round(volatility * 1.65, 4))),
            'calmar_ratio': Decimal(str(round(calmar_ratio, 4))),
            'benchmark_return_pct': Decimal(str(round(benchmark_return, 4))),
            'vs_spy_return': Decimal(str(round(total_return_pct - benchmark_return, 4))),
            'beta': Decimal('1.0'),
            'alpha': Decimal(str(round(total_return_pct - benchmark_return, 4))),
            'correlation_spy': Decimal('0.7'),
            'total_strategies_ranked': len(self.STRATEGIES),
        }

    def _get_zero_metrics(self) -> Dict[str, Any]:
        """Return conservative metrics when no real trades exist for the period."""
        return {
            'total_return_pct': Decimal('0'),
            'sharpe_ratio': Decimal('0'),
            'sortino_ratio': Decimal('0'),
            'max_drawdown_pct': Decimal('0'),
            'win_rate': Decimal('0'),
            'profit_factor': Decimal('0'),
            'trades_count': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_trade_pnl': Decimal('0'),
            'best_trade_pnl': Decimal('0'),
            'worst_trade_pnl': Decimal('0'),
            'avg_hold_duration_hours': Decimal('0'),
            'volatility': Decimal('0'),
            'var_95': Decimal('0'),
            'calmar_ratio': Decimal('0'),
            'benchmark_return_pct': Decimal('0'),
            'vs_spy_return': Decimal('0'),
            'beta': Decimal('0'),
            'alpha': Decimal('0'),
            'correlation_spy': Decimal('0'),
            'total_strategies_ranked': len(self.STRATEGIES),
        }

    def _get_benchmark_return(self, start_date: date, end_date: date) -> float:
        """Get SPY benchmark return for period."""
        # Benchmark integration is not yet implemented: avoid synthetic returns.
        self.stdout.write(
            self.style.WARNING(
                "Benchmark return unavailable; using 0.0 until live benchmark feed is integrated"
            )
        )
        return 0.0

    def _update_rankings(self, snapshot_date: date, period: str):
        """Update rank fields for all snapshots on this date."""
        from backend.tradingbot.models.models import StrategyPerformanceSnapshot

        snapshots = StrategyPerformanceSnapshot.objects.filter(
            snapshot_date=snapshot_date,
            period=period
        )

        total = snapshots.count()

        # Rank by Sharpe
        for rank, snapshot in enumerate(
            snapshots.order_by('-sharpe_ratio'), 1
        ):
            snapshot.rank_by_sharpe = rank
            snapshot.total_strategies_ranked = total
            snapshot.save(update_fields=['rank_by_sharpe', 'total_strategies_ranked'])

        # Rank by Return
        for rank, snapshot in enumerate(
            snapshots.order_by('-total_return_pct'), 1
        ):
            snapshot.rank_by_return = rank
            snapshot.save(update_fields=['rank_by_return'])

        # Rank by Risk-Adjusted Score
        snapshots_with_scores = [
            (s, s.get_risk_adjusted_score())
            for s in snapshots
        ]
        snapshots_with_scores.sort(key=lambda x: x[1], reverse=True)

        for rank, (snapshot, _) in enumerate(snapshots_with_scores, 1):
            snapshot.rank_by_risk_adjusted = rank
            snapshot.save(update_fields=['rank_by_risk_adjusted'])

        self.stdout.write(f"  Updated rankings for {total} strategies")
