"""
Comprehensive tests for TaxOptimizer.

Tests tax lot management, loss harvesting, wash sale detection,
tax impact preview, and all edge cases.
Target: 80%+ coverage.
"""
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from backend.auth0login.services.tax_optimizer import (
    TaxOptimizer,
    get_tax_optimizer_service,
    SHORT_TERM_TAX_RATE,
    LONG_TERM_TAX_RATE,
    WASH_SALE_WINDOW_DAYS,
)


class TestTaxOptimizer(TestCase):
    """Test suite for TaxOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = Mock(spec=User)
        self.user.id = 1
        self.user.username = "testuser"
        self.optimizer = TaxOptimizer(user=self.user)

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = TaxOptimizer(self.user)
        self.assertEqual(optimizer.user, self.user)

    def test_tax_rate_constants(self):
        """Test tax rate constants are defined."""
        self.assertEqual(SHORT_TERM_TAX_RATE, Decimal('0.35'))
        self.assertEqual(LONG_TERM_TAX_RATE, Decimal('0.15'))
        self.assertEqual(WASH_SALE_WINDOW_DAYS, 30)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_get_all_lots(self, mock_lot_model):
        """Test get_all_lots returns lot data."""
        mock_lot = Mock()
        mock_lot.to_dict.return_value = {
            'symbol': 'AAPL',
            'quantity': 100,
            'cost_basis': 10000.0
        }

        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_lot]

        lots = self.optimizer.get_all_lots()

        self.assertIsInstance(lots, list)
        self.assertEqual(len(lots), 1)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_get_all_lots_with_symbol_filter(self, mock_lot_model):
        """Test get_all_lots with symbol filter."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        lots = self.optimizer.get_all_lots(symbol='AAPL')

        # Should filter by symbol
        mock_queryset.filter.assert_called_once()

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_get_all_lots_include_closed(self, mock_lot_model):
        """Test get_all_lots can include closed lots."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        lots = self.optimizer.get_all_lots(include_closed=True)

        # Should not filter by is_closed when include_closed=True
        self.assertEqual(mock_queryset.filter.call_count, 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_get_lots_by_symbol(self, mock_lot_model):
        """Test get_lots_by_symbol returns summary."""
        mock_lot = Mock()
        mock_lot.to_dict.return_value = {
            'symbol': 'AAPL',
            'remaining_quantity': 100.0,
            'cost_basis_per_share': 100.0,
            'market_value': 11000.0,
            'unrealized_gain': 1000.0,
            'is_long_term': True
        }

        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_lot]

        result = self.optimizer.get_lots_by_symbol('AAPL')

        self.assertEqual(result['symbol'], 'AAPL')
        self.assertIn('lots', result)
        self.assertIn('summary', result)
        self.assertGreater(result['summary']['total_quantity'], 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_get_harvesting_opportunities(self, mock_lot_model):
        """Test get_harvesting_opportunities finds loss positions."""
        mock_lot = Mock()
        mock_lot.symbol = 'AAPL'
        mock_lot.unrealized_gain = Decimal('-500.0')
        mock_lot.is_long_term = False
        mock_lot.days_held = 10
        mock_lot.to_dict.return_value = {
            'symbol': 'AAPL',
            'unrealized_gain': -500.0
        }

        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.__getitem__ = lambda self, key: [mock_lot]

        with patch.object(self.optimizer, 'check_wash_sale_risk', return_value={'is_risky': False}):
            with patch.object(self.optimizer, '_estimate_tax_savings', return_value={'savings': 175.0}):
                with patch.object(self.optimizer, '_days_until_wash_safe', return_value=0):
                    with patch.object(self.optimizer, '_find_similar_alternatives', return_value=[]):
                        with patch.object(self.optimizer, '_get_harvest_recommendation', return_value='Safe to harvest'):
                            opportunities = self.optimizer.get_harvesting_opportunities(min_loss=100.0)

                            self.assertIsInstance(opportunities, list)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_check_wash_sale_risk_no_recent_purchases(self, mock_sale_model, mock_lot_model):
        """Test check_wash_sale_risk with no recent purchases."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        with patch.object(self.optimizer, '_check_strategy_intentions', return_value={'has_pending': False}):
            result = self.optimizer.check_wash_sale_risk('AAPL')

            self.assertEqual(result['risk_level'], 'none')
            self.assertFalse(result['is_risky'])

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_check_wash_sale_risk_recent_purchase(self, mock_lot_model):
        """Test check_wash_sale_risk with recent purchase."""
        mock_lot = Mock()
        mock_lot.acquired_at = timezone.now() - timedelta(days=10)
        mock_lot.original_quantity = Decimal('100')
        mock_lot.cost_basis_per_share = Decimal('100.0')

        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_lot]

        with patch.object(self.optimizer, '_check_strategy_intentions', return_value={'has_pending': False}):
            result = self.optimizer.check_wash_sale_risk('AAPL')

            self.assertEqual(result['risk_level'], 'high')
            self.assertTrue(result['is_risky'])

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_check_wash_sale_risk_pending_buys(self, mock_lot_model):
        """Test check_wash_sale_risk with pending strategy buys."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        with patch.object(self.optimizer, '_check_strategy_intentions', return_value={'has_pending': True}):
            result = self.optimizer.check_wash_sale_risk('AAPL')

            self.assertEqual(result['risk_level'], 'medium')
            self.assertTrue(result['is_risky'])

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_preview_sale_tax_impact_no_lots(self, mock_lot_model):
        """Test preview_sale_tax_impact with no available lots."""
        with patch.object(self.optimizer, '_select_lots', return_value=[]):
            result = self.optimizer.preview_sale_tax_impact('AAPL', 100, 110.0)

            self.assertIn('error', result)
            self.assertFalse(result['success'])

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_preview_sale_tax_impact_profit(self, mock_lot_model):
        """Test preview_sale_tax_impact with profitable sale."""
        mock_lot = Mock()
        mock_lot.id = 1
        mock_lot.acquired_at = timezone.now() - timedelta(days=400)
        mock_lot.cost_basis_per_share = Decimal('100.0')
        mock_lot.is_long_term = True
        mock_lot.days_held = 400

        lot_info = {'lot': mock_lot, 'quantity': Decimal('50')}

        with patch.object(self.optimizer, '_select_lots', return_value=[lot_info]):
            with patch.object(self.optimizer, 'check_wash_sale_risk', return_value={'is_risky': False}):
                with patch.object(self.optimizer, '_compare_lot_selection_methods', return_value={}):
                    result = self.optimizer.preview_sale_tax_impact(
                        'AAPL', 50, 120.0, lot_selection='fifo'
                    )

                    self.assertTrue(result['success'])
                    self.assertGreater(result['summary']['total_gain_loss'], 0)
                    self.assertGreater(result['summary']['long_term_gain_loss'], 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_preview_sale_tax_impact_loss(self, mock_lot_model):
        """Test preview_sale_tax_impact with loss."""
        mock_lot = Mock()
        mock_lot.id = 1
        mock_lot.acquired_at = timezone.now() - timedelta(days=10)
        mock_lot.cost_basis_per_share = Decimal('120.0')
        mock_lot.is_long_term = False
        mock_lot.days_held = 10

        lot_info = {'lot': mock_lot, 'quantity': Decimal('50')}

        with patch.object(self.optimizer, '_select_lots', return_value=[lot_info]):
            with patch.object(self.optimizer, 'check_wash_sale_risk', return_value={'is_risky': True}):
                with patch.object(self.optimizer, '_compare_lot_selection_methods', return_value={}):
                    result = self.optimizer.preview_sale_tax_impact(
                        'AAPL', 50, 100.0, lot_selection='fifo'
                    )

                    self.assertTrue(result['success'])
                    self.assertLess(result['summary']['total_gain_loss'], 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_select_lots_fifo(self, mock_lot_model):
        """Test _select_lots with FIFO method."""
        mock_lot1 = Mock()
        mock_lot1.remaining_quantity = Decimal('100')
        mock_lot2 = Mock()
        mock_lot2.remaining_quantity = Decimal('100')

        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = [mock_lot1, mock_lot2]

        result = self.optimizer._select_lots('AAPL', Decimal('150'), 'fifo')

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['quantity'], Decimal('100'))
        self.assertEqual(result[1]['quantity'], Decimal('50'))

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_select_lots_lifo(self, mock_lot_model):
        """Test _select_lots with LIFO method."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        result = self.optimizer._select_lots('AAPL', Decimal('100'), 'lifo')

        # Verify LIFO ordering was used
        mock_queryset.order_by.assert_called_with('-acquired_at')

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_select_lots_hifo(self, mock_lot_model):
        """Test _select_lots with HIFO method."""
        mock_queryset = MagicMock()
        mock_lot_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value = []

        result = self.optimizer._select_lots('AAPL', Decimal('100'), 'hifo')

        # Verify HIFO ordering was used
        mock_queryset.order_by.assert_called_with('-cost_basis_per_share')

    def test_suggest_lot_selection_minimize_tax(self):
        """Test suggest_lot_selection for minimizing tax."""
        with patch.object(self.optimizer, '_compare_lot_selection_methods', return_value={}):
            result = self.optimizer.suggest_lot_selection(
                'AAPL', 100, goal='minimize_tax'
            )

            self.assertEqual(result['recommended_method'], 'hifo')
            self.assertIn('reason', result)

    def test_suggest_lot_selection_maximize_loss(self):
        """Test suggest_lot_selection for maximizing loss."""
        comparisons = {
            'fifo': {'total_gain': -500.0},
            'lifo': {'total_gain': -200.0},
        }
        with patch.object(self.optimizer, '_compare_lot_selection_methods', return_value=comparisons):
            result = self.optimizer.suggest_lot_selection(
                'AAPL', 100, goal='maximize_loss'
            )

            self.assertIn(result['recommended_method'], ['fifo', 'lifo'])

    def test_suggest_lot_selection_long_term_priority(self):
        """Test suggest_lot_selection with long-term priority."""
        with patch.object(self.optimizer, '_compare_lot_selection_methods', return_value={}):
            result = self.optimizer.suggest_lot_selection(
                'AAPL', 100, goal='long_term_priority'
            )

            self.assertEqual(result['recommended_method'], 'specific')

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_get_year_summary(self, mock_sale_model):
        """Test get_year_summary."""
        mock_sale = Mock()
        mock_sale.proceeds = Decimal('12000.0')
        mock_sale.cost_basis_sold = Decimal('10000.0')
        mock_sale.wash_sale_disallowed = Decimal('0')
        mock_sale.realized_gain = Decimal('2000.0')
        mock_sale.is_long_term = True
        mock_sale.symbol = 'AAPL'

        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = [mock_sale]
        mock_queryset.count.return_value = 1

        result = self.optimizer.get_year_summary(2024)

        self.assertEqual(result['year'], 2024)
        self.assertIn('total_sales', result)
        self.assertIn('short_term', result)
        self.assertIn('long_term', result)

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_get_year_summary_current_year(self, mock_sale_model):
        """Test get_year_summary defaults to current year."""
        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = []

        result = self.optimizer.get_year_summary()

        self.assertEqual(result['year'], timezone.now().year)

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_get_year_summary_with_losses(self, mock_sale_model):
        """Test get_year_summary with losses."""
        mock_sale = Mock()
        mock_sale.proceeds = Decimal('8000.0')
        mock_sale.cost_basis_sold = Decimal('10000.0')
        mock_sale.wash_sale_disallowed = Decimal('0')
        mock_sale.realized_gain = Decimal('-2000.0')
        mock_sale.is_long_term = False
        mock_sale.symbol = 'AAPL'

        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = [mock_sale]

        result = self.optimizer.get_year_summary(2024)

        self.assertLess(result['net_gain_loss'], 0)
        self.assertGreater(result['loss_carryforward'], 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLot')
    def test_compare_lot_selection_methods(self, mock_lot_model):
        """Test _compare_lot_selection_methods."""
        mock_lot = Mock()
        mock_lot.cost_basis_per_share = Decimal('100.0')
        mock_lot.current_price = Decimal('110.0')

        with patch.object(self.optimizer, '_select_lots') as mock_select:
            mock_select.return_value = [
                {'lot': mock_lot, 'quantity': Decimal('50')}
            ]

            # Mock first call to get current price
            mock_first_lot = Mock()
            mock_first_lot.current_price = Decimal('110.0')
            mock_queryset = MagicMock()
            mock_lot_model.objects.filter.return_value = mock_queryset
            mock_queryset.first.return_value = mock_first_lot

            result = self.optimizer._compare_lot_selection_methods(
                'AAPL', 50, None
            )

            self.assertIn('fifo', result)
            self.assertIn('lifo', result)
            self.assertIn('hifo', result)

    def test_estimate_tax_savings_no_loss(self):
        """Test _estimate_tax_savings with no loss."""
        mock_lot = Mock()
        mock_lot.unrealized_gain = Decimal('500.0')

        result = self.optimizer._estimate_tax_savings(mock_lot)

        self.assertEqual(result['savings'], 0)

    def test_estimate_tax_savings_short_term_loss(self):
        """Test _estimate_tax_savings with short-term loss."""
        mock_lot = Mock()
        mock_lot.unrealized_gain = Decimal('-1000.0')
        mock_lot.is_long_term = False

        result = self.optimizer._estimate_tax_savings(mock_lot)

        self.assertGreater(result['estimated_savings'], 0)
        self.assertEqual(result['holding_type'], 'short_term')

    def test_estimate_tax_savings_long_term_loss(self):
        """Test _estimate_tax_savings with long-term loss."""
        mock_lot = Mock()
        mock_lot.unrealized_gain = Decimal('-1000.0')
        mock_lot.is_long_term = True

        result = self.optimizer._estimate_tax_savings(mock_lot)

        self.assertGreater(result['estimated_savings'], 0)
        self.assertEqual(result['holding_type'], 'long_term')

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_days_until_wash_safe_no_sales(self, mock_sale_model):
        """Test _days_until_wash_safe with no recent sales."""
        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = None

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 0)

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_days_until_wash_safe_recent_sale(self, mock_sale_model):
        """Test _days_until_wash_safe with recent sale."""
        mock_sale = Mock()
        mock_sale.sold_at = timezone.now() - timedelta(days=15)

        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = mock_sale

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 15)  # 30 - 15 = 15 days

    @patch('backend.auth0login.services.tax_optimizer.TaxLotSale')
    def test_days_until_wash_safe_past_window(self, mock_sale_model):
        """Test _days_until_wash_safe past wash sale window."""
        mock_sale = Mock()
        mock_sale.sold_at = timezone.now() - timedelta(days=40)

        mock_queryset = MagicMock()
        mock_sale_model.objects.filter.return_value = mock_queryset
        mock_queryset.order_by.return_value.first.return_value = mock_sale

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 0)

    def test_check_strategy_intentions(self):
        """Test _check_strategy_intentions placeholder."""
        result = self.optimizer._check_strategy_intentions('AAPL')

        self.assertFalse(result['has_pending'])
        self.assertIn('strategies', result)

    def test_find_similar_alternatives_spy(self):
        """Test _find_similar_alternatives for SPY."""
        alternatives = self.optimizer._find_similar_alternatives('SPY')

        self.assertIsInstance(alternatives, list)
        self.assertGreater(len(alternatives), 0)
        self.assertTrue(any(alt['symbol'] == 'VOO' for alt in alternatives))

    def test_find_similar_alternatives_unknown(self):
        """Test _find_similar_alternatives for unknown symbol."""
        alternatives = self.optimizer._find_similar_alternatives('UNKNOWN')

        self.assertEqual(alternatives, [])

    def test_get_harvest_recommendation_high_risk(self):
        """Test _get_harvest_recommendation with high wash risk."""
        mock_lot = Mock()
        wash_risk = {'risk_level': 'high'}

        result = self.optimizer._get_harvest_recommendation(
            mock_lot, wash_risk, days_until_safe=15
        )

        self.assertIn('WARNING', result)
        self.assertIn('wash sale', result.lower())

    def test_get_harvest_recommendation_medium_risk(self):
        """Test _get_harvest_recommendation with medium wash risk."""
        mock_lot = Mock()
        wash_risk = {'risk_level': 'medium'}

        result = self.optimizer._get_harvest_recommendation(
            mock_lot, wash_risk, days_until_safe=0
        )

        self.assertIn('CAUTION', result)

    def test_get_harvest_recommendation_no_risk(self):
        """Test _get_harvest_recommendation with no wash risk."""
        mock_lot = Mock()
        mock_lot.unrealized_gain = Decimal('-500.0')
        mock_lot.is_long_term = True

        wash_risk = {'risk_level': 'none'}

        result = self.optimizer._get_harvest_recommendation(
            mock_lot, wash_risk, days_until_safe=0
        )

        self.assertIn('OPPORTUNITY', result)

    def test_get_wash_sale_recommendation_none(self):
        """Test _get_wash_sale_recommendation with no risk."""
        result = self.optimizer._get_wash_sale_recommendation('none', [])

        self.assertIn('Safe', result)

    def test_get_wash_sale_recommendation_high(self):
        """Test _get_wash_sale_recommendation with high risk."""
        recent_purchases = [{'days_ago': 10}]

        result = self.optimizer._get_wash_sale_recommendation('high', recent_purchases)

        self.assertIn('WASH SALE RISK', result)

    def test_get_wash_sale_recommendation_medium(self):
        """Test _get_wash_sale_recommendation with medium risk."""
        result = self.optimizer._get_wash_sale_recommendation('medium', [])

        self.assertIn('POTENTIAL WASH SALE', result)

    def test_get_tax_optimizer_service(self):
        """Test get_tax_optimizer_service factory function."""
        optimizer = get_tax_optimizer_service(self.user)

        self.assertIsInstance(optimizer, TaxOptimizer)
        self.assertEqual(optimizer.user, self.user)


if __name__ == '__main__':
    unittest.main()
