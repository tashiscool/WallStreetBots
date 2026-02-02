"""
Comprehensive tests for LeaderboardService.

Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage.
"""
import unittest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from django.contrib.auth.models import User
from django.test import TestCase

from backend.auth0login.services.leaderboard_service import LeaderboardService


class TestLeaderboardService(TestCase):
    """Test suite for LeaderboardService."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = Mock(spec=User)
        self.user.id = 1
        self.user.username = "testuser"
        self.service = LeaderboardService(user=self.user)

    def test_initialization(self):
        """Test service initialization."""
        service = LeaderboardService()
        self.assertIsNone(service.user)

        service_with_user = LeaderboardService(user=self.user)
        self.assertEqual(service_with_user.user, self.user)

    def test_strategy_registry_completeness(self):
        """Test that STRATEGY_REGISTRY contains expected strategies."""
        self.assertIn('wsb_dip_bot', LeaderboardService.STRATEGY_REGISTRY)
        self.assertIn('wheel_strategy', LeaderboardService.STRATEGY_REGISTRY)
        self.assertIn('index_baseline', LeaderboardService.STRATEGY_REGISTRY)

        # Validate structure
        for strategy_name, info in LeaderboardService.STRATEGY_REGISTRY.items():
            self.assertIn('name', info)
            self.assertIn('description', info)
            self.assertIn('risk_level', info)
            self.assertIn('category', info)

    def test_ranking_metrics_structure(self):
        """Test RANKING_METRICS structure."""
        for metric_name, info in LeaderboardService.RANKING_METRICS.items():
            self.assertIn('name', info)
            self.assertIn('higher_better', info)
            self.assertIsInstance(info['higher_better'], bool)

    def test_period_days_mapping(self):
        """Test PERIOD_DAYS contains expected periods."""
        expected_periods = ['1W', '1M', '3M', '6M', '1Y', 'YTD', 'ALL']
        for period in expected_periods:
            self.assertIn(period, LeaderboardService.PERIOD_DAYS)

    def test_get_date_range_1w(self):
        """Test _get_date_range for 1 week period."""
        start, end = self.service._get_date_range('1W')
        self.assertIsInstance(start, date)
        self.assertIsInstance(end, date)
        self.assertEqual((end - start).days, 7)

    def test_get_date_range_1m(self):
        """Test _get_date_range for 1 month period."""
        start, end = self.service._get_date_range('1M')
        self.assertEqual((end - start).days, 30)

    def test_get_date_range_ytd(self):
        """Test _get_date_range for YTD period."""
        start, end = self.service._get_date_range('YTD')
        self.assertEqual(start.year, end.year)
        self.assertEqual(start.month, 1)
        self.assertEqual(start.day, 1)

    def test_get_date_range_all(self):
        """Test _get_date_range for ALL period."""
        start, end = self.service._get_date_range('ALL')
        self.assertEqual(start.year, 2020)
        self.assertEqual(end, date.today())

    def test_get_date_range_invalid_defaults_to_30_days(self):
        """Test _get_date_range with invalid period defaults to 30 days."""
        start, end = self.service._get_date_range('INVALID')
        self.assertEqual((end - start).days, 30)

    def test_get_ordering_higher_better(self):
        """Test _get_ordering for metrics where higher is better."""
        ordering = self.service._get_ordering('sharpe_ratio')
        self.assertEqual(ordering, '-sharpe_ratio')

        ordering = self.service._get_ordering('total_return_pct')
        self.assertEqual(ordering, '-total_return_pct')

    def test_get_ordering_lower_better(self):
        """Test _get_ordering for metrics where lower is better."""
        ordering = self.service._get_ordering('max_drawdown_pct')
        self.assertEqual(ordering, 'max_drawdown_pct')

    def test_get_ordering_unknown_metric_defaults_to_higher_better(self):
        """Test _get_ordering for unknown metric."""
        ordering = self.service._get_ordering('unknown_metric')
        self.assertEqual(ordering, '-unknown_metric')

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_leaderboard_basic(self, mock_snapshot):
        """Test get_leaderboard with basic parameters."""
        # Mock queryset
        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.values.return_value.annotate.return_value = []

        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        result = self.service.get_leaderboard()

        self.assertIn('period', result)
        self.assertIn('metric', result)
        self.assertIn('leaderboard', result)
        self.assertIn('available_metrics', result)
        self.assertIn('available_periods', result)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_leaderboard_with_snapshots(self, mock_snapshot_model):
        """Test get_leaderboard with mock snapshots."""
        # Create mock snapshots
        mock_snapshot1 = Mock()
        mock_snapshot1.id = 1
        mock_snapshot1.strategy_name = 'wsb_dip_bot'
        mock_snapshot1.snapshot_date = date.today()
        mock_snapshot1.sharpe_ratio = Decimal('2.5')
        mock_snapshot1.total_return_pct = Decimal('15.5')
        mock_snapshot1.to_leaderboard_entry = Mock(return_value={
            'rank': 1,
            'strategy_name': 'wsb_dip_bot',
            'sharpe_ratio': 2.5,
        })
        mock_snapshot1.get_trend_vs_previous = Mock(return_value='up')

        # Set up mock queryset chain
        mock_queryset = MagicMock()
        mock_snapshot_model.objects.filter.return_value = mock_queryset

        # Mock the values().annotate() chain
        mock_values = MagicMock()
        mock_queryset.values.return_value = mock_values
        mock_values.annotate.return_value = [
            {'strategy_name': 'wsb_dip_bot', 'latest_date': date.today()}
        ]

        # Mock the filter for full snapshot
        mock_snapshot_model.objects.filter.return_value.first.return_value = mock_snapshot1

        # Mock final queryset with ordering
        mock_final_queryset = MagicMock()
        mock_snapshot_model.objects.filter.return_value = mock_final_queryset
        mock_final_queryset.order_by.return_value = [mock_snapshot1]

        result = self.service.get_leaderboard(period='1M', metric='sharpe_ratio', limit=10)

        self.assertEqual(result['period'], '1M')
        self.assertEqual(result['metric'], 'sharpe_ratio')
        self.assertIsInstance(result['leaderboard'], list)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_leaderboard_with_category_filter(self, mock_snapshot):
        """Test get_leaderboard with category filtering."""
        mock_queryset = MagicMock()
        # Set up proper query chain - each method returns the same mock queryset
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.values.return_value.annotate.return_value = mock_queryset
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = mock_queryset
        # Make it iterable but empty
        mock_queryset.__iter__ = lambda self: iter([])
        mock_queryset.__getitem__ = lambda self, key: []

        result = self.service.get_leaderboard(category='momentum')

        # Verify filter was called
        self.assertIn('leaderboard', result)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_compare_strategies_empty_list(self, mock_snapshot):
        """Test compare_strategies with empty strategy list."""
        result = self.service.compare_strategies([])

        self.assertIn('error', result)
        self.assertEqual(result['strategies'], [])

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_compare_strategies_with_data(self, mock_snapshot):
        """Test compare_strategies with valid data."""
        # Mock snapshot
        mock_snap = Mock()
        mock_snap.strategy_name = 'wsb_dip_bot'
        mock_snap.snapshot_date = date.today()
        mock_snap.total_return_pct = Decimal('15.5')
        mock_snap.sharpe_ratio = Decimal('2.5')
        mock_snap.sortino_ratio = Decimal('3.0')
        mock_snap.max_drawdown_pct = Decimal('-10.5')
        mock_snap.win_rate = Decimal('65.0')
        mock_snap.profit_factor = Decimal('2.0')
        mock_snap.trades_count = 100
        mock_snap.avg_trade_pnl = Decimal('50.0')
        mock_snap.volatility = Decimal('20.0')
        mock_snap.calmar_ratio = Decimal('1.5')
        mock_snap.vs_spy_return = Decimal('5.0')
        mock_snap.beta = Decimal('1.2')
        mock_snap.alpha = Decimal('2.0')
        mock_snap.correlation_spy = Decimal('0.85')
        mock_snap.rank_by_sharpe = 1
        mock_snap.rank_by_return = 2
        mock_snap.rank_by_risk_adjusted = 1
        mock_snap.get_risk_adjusted_score = Mock(return_value=8.5)

        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = mock_snap

        result = self.service.compare_strategies(['wsb_dip_bot'], period='1M')

        self.assertEqual(result['period'], '1M')
        self.assertIn('comparison', result)
        self.assertGreater(len(result['comparison']), 0)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_compare_strategies_no_data(self, mock_snapshot):
        """Test compare_strategies when strategy has no data."""
        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = None

        result = self.service.compare_strategies(['wsb_dip_bot'], period='1M')

        self.assertIn('comparison', result)
        comparison = result['comparison'][0]
        self.assertTrue(comparison.get('no_data', False))

    def test_calculate_comparison_summary_empty(self):
        """Test _calculate_comparison_summary with empty list."""
        summary = self.service._calculate_comparison_summary([])
        self.assertEqual(summary, {})

    def test_calculate_comparison_summary_with_data(self):
        """Test _calculate_comparison_summary with valid data."""
        comparison = [
            {
                'strategy_name': 'strategy1',
                'metrics': {
                    'total_return_pct': 15.0,
                    'sharpe_ratio': 2.5,
                    'win_rate': 65.0,
                    'max_drawdown_pct': -10.0,
                }
            },
            {
                'strategy_name': 'strategy2',
                'metrics': {
                    'total_return_pct': 20.0,
                    'sharpe_ratio': 3.0,
                    'win_rate': 70.0,
                    'max_drawdown_pct': -15.0,
                }
            }
        ]

        summary = self.service._calculate_comparison_summary(comparison)

        self.assertIn('total_return_pct', summary)
        self.assertEqual(summary['total_return_pct']['best']['strategy'], 'strategy2')
        self.assertEqual(summary['total_return_pct']['worst']['strategy'], 'strategy1')

    def test_calculate_comparison_summary_with_no_data_entries(self):
        """Test _calculate_comparison_summary skips no_data entries."""
        comparison = [
            {'strategy_name': 'strategy1', 'no_data': True},
            {
                'strategy_name': 'strategy2',
                'metrics': {'total_return_pct': 15.0}
            }
        ]

        summary = self.service._calculate_comparison_summary(comparison)
        # Should only have one valid entry
        self.assertIn('total_return_pct', summary)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_rank_history(self, mock_snapshot):
        """Test get_strategy_rank_history."""
        # Mock snapshots
        mock_snap1 = Mock()
        mock_snap1.strategy_name = 'wsb_dip_bot'
        mock_snap1.snapshot_date = date.today() - timedelta(days=30)
        mock_snap1.rank_by_sharpe = 3
        mock_snap1.sharpe_ratio = Decimal('2.0')
        mock_snap1.total_strategies_ranked = 10

        mock_snap2 = Mock()
        mock_snap2.strategy_name = 'wsb_dip_bot'
        mock_snap2.snapshot_date = date.today()
        mock_snap2.rank_by_sharpe = 1
        mock_snap2.sharpe_ratio = Decimal('2.5')
        mock_snap2.total_strategies_ranked = 10

        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_snap1, mock_snap2]

        result = self.service.get_strategy_rank_history('wsb_dip_bot', days=90)

        self.assertEqual(result['strategy_name'], 'wsb_dip_bot')
        self.assertIn('history', result)
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(result['trend'], 'improving')
        self.assertEqual(result['rank_change'], 2)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_rank_history_declining(self, mock_snapshot):
        """Test get_strategy_rank_history with declining trend."""
        mock_snap1 = Mock()
        mock_snap1.snapshot_date = date.today() - timedelta(days=30)
        mock_snap1.rank_by_sharpe = 1
        mock_snap1.sharpe_ratio = Decimal('2.5')
        mock_snap1.total_strategies_ranked = 10

        mock_snap2 = Mock()
        mock_snap2.snapshot_date = date.today()
        mock_snap2.rank_by_sharpe = 5
        mock_snap2.sharpe_ratio = Decimal('1.5')
        mock_snap2.total_strategies_ranked = 10

        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_snap1, mock_snap2]

        result = self.service.get_strategy_rank_history('wsb_dip_bot', days=90)

        self.assertEqual(result['trend'], 'declining')
        self.assertEqual(result['rank_change'], -4)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_rank_history_stable(self, mock_snapshot):
        """Test get_strategy_rank_history with stable trend."""
        mock_snap1 = Mock()
        mock_snap1.snapshot_date = date.today() - timedelta(days=30)
        mock_snap1.rank_by_sharpe = 2
        mock_snap1.sharpe_ratio = Decimal('2.0')
        mock_snap1.total_strategies_ranked = 10

        mock_snap2 = Mock()
        mock_snap2.snapshot_date = date.today()
        mock_snap2.rank_by_sharpe = 2
        mock_snap2.sharpe_ratio = Decimal('2.0')
        mock_snap2.total_strategies_ranked = 10

        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_snap1, mock_snap2]

        result = self.service.get_strategy_rank_history('wsb_dip_bot', days=90)

        self.assertEqual(result['trend'], 'stable')
        self.assertEqual(result['rank_change'], 0)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_rank_history_insufficient_data(self, mock_snapshot):
        """Test get_strategy_rank_history with insufficient data."""
        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        result = self.service.get_strategy_rank_history('wsb_dip_bot', days=90)

        self.assertEqual(result['trend'], 'insufficient_data')
        self.assertEqual(result['rank_change'], 0)
        self.assertIsNone(result['current_rank'])

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_calculate_hypothetical_portfolio_invalid_allocation(self, mock_snapshot):
        """Test calculate_hypothetical_portfolio with invalid allocation sum."""
        allocations = {'wsb_dip_bot': 50.0, 'wheel_strategy': 30.0}  # Only 80%

        result = self.service.calculate_hypothetical_portfolio(allocations)

        self.assertIn('error', result)
        self.assertIn('100%', result['error'])

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_calculate_hypothetical_portfolio_valid(self, mock_snapshot):
        """Test calculate_hypothetical_portfolio with valid allocations."""
        allocations = {'wsb_dip_bot': 60.0, 'wheel_strategy': 40.0}

        # Mock snapshots
        mock_snap1 = Mock()
        mock_snap1.strategy_name = 'wsb_dip_bot'
        mock_snap1.total_return_pct = Decimal('20.0')
        mock_snap1.sharpe_ratio = Decimal('2.5')
        mock_snap1.max_drawdown_pct = Decimal('-10.0')
        mock_snap1.win_rate = Decimal('65.0')
        mock_snap1.volatility = Decimal('20.0')
        mock_snap1.trades_count = 50

        mock_snap2 = Mock()
        mock_snap2.strategy_name = 'wheel_strategy'
        mock_snap2.total_return_pct = Decimal('15.0')
        mock_snap2.sharpe_ratio = Decimal('2.0')
        mock_snap2.max_drawdown_pct = Decimal('-5.0')
        mock_snap2.win_rate = Decimal('70.0')
        mock_snap2.volatility = Decimal('15.0')
        mock_snap2.trades_count = 40

        # Mock benchmark
        mock_benchmark = Mock()
        mock_benchmark.benchmark_return_pct = Decimal('10.0')

        def mock_filter_side_effect(**kwargs):
            strategy_name = kwargs.get('strategy_name')
            mock_queryset = MagicMock()
            if strategy_name == 'wsb_dip_bot':
                mock_queryset.order_by.return_value.first.return_value = mock_snap1
            elif strategy_name == 'wheel_strategy':
                mock_queryset.order_by.return_value.first.return_value = mock_snap2
            elif strategy_name == 'index_baseline':
                mock_queryset.order_by.return_value.first.return_value = mock_benchmark
            else:
                mock_queryset.order_by.return_value.first.return_value = None
            return mock_queryset

        mock_snapshot.objects.filter.side_effect = mock_filter_side_effect

        result = self.service.calculate_hypothetical_portfolio(
            allocations,
            period='1M',
            initial_capital=100000.0
        )

        self.assertIn('portfolio_metrics', result)
        self.assertIn('strategy_breakdown', result)
        self.assertIn('benchmark_comparison', result)
        self.assertEqual(len(result['strategy_breakdown']), 2)

    def test_calculate_diversification_score_empty(self):
        """Test _calculate_diversification_score with empty breakdown."""
        score = self.service._calculate_diversification_score([])
        self.assertEqual(score, 0.0)

    def test_calculate_diversification_score_single_category(self):
        """Test _calculate_diversification_score with single category."""
        breakdown = [
            {'strategy_name': 'wsb_dip_bot'},
            {'strategy_name': 'momentum_weeklies'},
        ]
        score = self.service._calculate_diversification_score(breakdown)
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 100)

    def test_calculate_diversification_score_diverse(self):
        """Test _calculate_diversification_score with diverse strategies."""
        breakdown = [
            {'strategy_name': 'wsb_dip_bot'},      # momentum, high
            {'strategy_name': 'wheel_strategy'},    # income, medium
            {'strategy_name': 'index_baseline'},    # benchmark, low
            {'strategy_name': 'lotto_scanner'},     # speculative, very_high
        ]
        score = self.service._calculate_diversification_score(breakdown)
        self.assertGreater(score, 50)  # Should be well diversified

    def test_calculate_diversification_score_no_data_entries(self):
        """Test _calculate_diversification_score skips no_data entries."""
        breakdown = [
            {'strategy_name': 'wsb_dip_bot', 'no_data': True},
            {'strategy_name': 'wheel_strategy'},
        ]
        score = self.service._calculate_diversification_score(breakdown)
        self.assertGreater(score, 0)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_top_performers(self, mock_snapshot):
        """Test get_top_performers."""
        mock_leaderboard = {
            'period': '1M',
            'leaderboard': [
                {'strategy_name': 'strategy1', 'sharpe_ratio': 2.5}
            ]
        }

        with patch.object(self.service, 'get_leaderboard', return_value=mock_leaderboard):
            result = self.service.get_top_performers(period='1M', count=3)

            self.assertIn('period', result)
            self.assertIn('categories', result)
            self.assertIn('sharpe_ratio', result['categories'])
            self.assertIn('total_return_pct', result['categories'])
            self.assertIn('win_rate', result['categories'])

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_details_with_data(self, mock_snapshot):
        """Test get_strategy_details with existing data."""
        mock_snap = Mock()
        mock_snap.strategy_name = 'wsb_dip_bot'
        mock_snap.snapshot_date = date.today()
        mock_snap.total_return_pct = Decimal('15.5')
        mock_snap.sharpe_ratio = Decimal('2.5')
        mock_snap.sortino_ratio = Decimal('3.0')
        mock_snap.max_drawdown_pct = Decimal('-10.0')
        mock_snap.win_rate = Decimal('65.0')
        mock_snap.profit_factor = Decimal('2.0')
        mock_snap.calmar_ratio = Decimal('1.5')
        mock_snap.trades_count = 100
        mock_snap.winning_trades = 65
        mock_snap.losing_trades = 35
        mock_snap.avg_trade_pnl = Decimal('50.0')
        mock_snap.best_trade_pnl = Decimal('500.0')
        mock_snap.worst_trade_pnl = Decimal('-200.0')
        mock_snap.avg_hold_duration_hours = Decimal('24.5')
        mock_snap.volatility = Decimal('20.0')
        mock_snap.var_95 = Decimal('-5.0')
        mock_snap.beta = Decimal('1.2')
        mock_snap.benchmark_return_pct = Decimal('10.0')
        mock_snap.vs_spy_return = Decimal('5.5')
        mock_snap.alpha = Decimal('2.0')
        mock_snap.correlation_spy = Decimal('0.85')
        mock_snap.rank_by_sharpe = 1
        mock_snap.rank_by_return = 2
        mock_snap.rank_by_risk_adjusted = 1
        mock_snap.total_strategies_ranked = 10
        mock_snap.get_risk_adjusted_score = Mock(return_value=8.5)

        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = mock_snap

        with patch.object(self.service, 'get_strategy_rank_history', return_value={'history': []}):
            result = self.service.get_strategy_details('wsb_dip_bot', period='1M')

            self.assertEqual(result['strategy_name'], 'wsb_dip_bot')
            self.assertIn('performance', result)
            self.assertIn('trading_stats', result)
            self.assertIn('risk_metrics', result)
            self.assertIn('benchmark_comparison', result)
            self.assertIn('rankings', result)
            self.assertIn('rank_history', result)

    @patch('backend.tradingbot.models.models.StrategyPerformanceSnapshot')
    def test_get_strategy_details_no_data(self, mock_snapshot):
        """Test get_strategy_details with no data."""
        mock_queryset = MagicMock()
        mock_snapshot.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = None

        result = self.service.get_strategy_details('wsb_dip_bot', period='1M')

        self.assertTrue(result.get('no_data'))
        self.assertIn('message', result)

    def test_get_all_strategies(self):
        """Test get_all_strategies returns all registered strategies."""
        strategies = self.service.get_all_strategies()

        self.assertIsInstance(strategies, list)
        self.assertGreater(len(strategies), 0)

        for strategy in strategies:
            self.assertIn('strategy_name', strategy)
            self.assertIn('display_name', strategy)
            self.assertIn('description', strategy)
            self.assertIn('risk_level', strategy)
            self.assertIn('category', strategy)


if __name__ == '__main__':
    unittest.main()
