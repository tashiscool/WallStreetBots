"""
Integration tests for TaxOptimizer.

Tests tax lot management, loss harvesting, wash sale detection,
tax impact preview, and lot selection methods with real database operations.
Target: 80%+ coverage.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal

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
from backend.tradingbot.models.models import TaxLot, TaxLotSale


@pytest.mark.django_db
class TestTaxOptimizerIntegration(TestCase):
    """Integration test suite for TaxOptimizer with real database operations."""

    @classmethod
    def setUpTestData(cls):
        """Set up test data once for all tests in this class."""
        cls.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass123'
        )
        cls.other_user = User.objects.create_user(
            username='otheruser',
            email='other@example.com',
            password='testpass123'
        )

    def setUp(self):
        """Set up test fixtures for each test."""
        self.optimizer = TaxOptimizer(user=self.user)
        # Clean up any existing tax lots/sales for our test user
        TaxLotSale.objects.filter(user=self.user).delete()
        TaxLot.objects.filter(user=self.user).delete()

    def tearDown(self):
        """Clean up after each test."""
        TaxLotSale.objects.filter(user=self.user).delete()
        TaxLot.objects.filter(user=self.user).delete()

    def _create_tax_lot(
        self,
        symbol: str,
        quantity: Decimal,
        cost_basis: Decimal,
        days_ago: int = 0,
        current_price: Decimal | None = None,
        is_closed: bool = False,
        acquisition_type: str = 'purchase',
    ) -> TaxLot:
        """Helper to create a TaxLot with realistic data."""
        acquired_at = timezone.now() - timedelta(days=days_ago)
        total_cost = quantity * cost_basis

        lot = TaxLot.objects.create(
            user=self.user,
            symbol=symbol.upper(),
            original_quantity=quantity,
            remaining_quantity=quantity if not is_closed else Decimal('0'),
            cost_basis_per_share=cost_basis,
            total_cost_basis=total_cost,
            acquired_at=acquired_at,
            acquisition_type=acquisition_type,
            is_closed=is_closed,
        )

        if current_price:
            lot.update_market_data(float(current_price))

        return lot

    def _create_tax_lot_sale(
        self,
        lot: TaxLot,
        quantity: Decimal,
        sale_price: Decimal,
        days_ago: int = 0,
        is_wash_sale: bool = False,
        wash_sale_disallowed: Decimal = Decimal('0'),
    ) -> TaxLotSale:
        """Helper to create a TaxLotSale with realistic data."""
        sold_at = timezone.now() - timedelta(days=days_ago)
        proceeds = quantity * sale_price
        cost_basis_sold = quantity * lot.cost_basis_per_share
        realized_gain = proceeds - cost_basis_sold

        sale = TaxLotSale.objects.create(
            user=self.user,
            tax_lot=lot,
            symbol=lot.symbol,
            quantity_sold=quantity,
            sale_price=sale_price,
            proceeds=proceeds,
            sold_at=sold_at,
            cost_basis_sold=cost_basis_sold,
            realized_gain=realized_gain,
            is_gain=realized_gain > 0,
            is_long_term=lot.is_long_term,
            is_wash_sale=is_wash_sale,
            wash_sale_disallowed=wash_sale_disallowed,
            lot_selection_method='fifo',
        )

        # Update lot remaining quantity
        lot.remaining_quantity -= quantity
        if lot.remaining_quantity <= 0:
            lot.is_closed = True
            lot.closed_at = sold_at
        lot.save()

        return sale

    # ==================== Initialization Tests ====================

    def test_initialization(self):
        """Test optimizer initialization with real user."""
        optimizer = TaxOptimizer(self.user)
        self.assertEqual(optimizer.user, self.user)

    def test_tax_rate_constants(self):
        """Test tax rate constants are defined correctly."""
        self.assertEqual(SHORT_TERM_TAX_RATE, Decimal('0.35'))
        self.assertEqual(LONG_TERM_TAX_RATE, Decimal('0.15'))
        self.assertEqual(WASH_SALE_WINDOW_DAYS, 30)

    def test_get_tax_optimizer_service_factory(self):
        """Test get_tax_optimizer_service factory function."""
        optimizer = get_tax_optimizer_service(self.user)
        self.assertIsInstance(optimizer, TaxOptimizer)
        self.assertEqual(optimizer.user, self.user)

    # ==================== Get All Lots Tests ====================

    def test_get_all_lots_empty(self):
        """Test get_all_lots returns empty list when no lots exist."""
        lots = self.optimizer.get_all_lots()
        self.assertEqual(lots, [])

    def test_get_all_lots_returns_open_lots(self):
        """Test get_all_lots returns only open lots by default."""
        # Create open and closed lots
        open_lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))
        closed_lot = self._create_tax_lot('GOOGL', Decimal('50'), Decimal('2500.00'), is_closed=True)

        lots = self.optimizer.get_all_lots()

        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0]['symbol'], 'AAPL')

    def test_get_all_lots_include_closed(self):
        """Test get_all_lots can include closed lots."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))
        self._create_tax_lot('GOOGL', Decimal('50'), Decimal('2500.00'), is_closed=True)

        lots = self.optimizer.get_all_lots(include_closed=True)

        self.assertEqual(len(lots), 2)

    def test_get_all_lots_with_symbol_filter(self):
        """Test get_all_lots filters by symbol correctly."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))
        self._create_tax_lot('AAPL', Decimal('50'), Decimal('155.00'))
        self._create_tax_lot('MSFT', Decimal('200'), Decimal('380.00'))

        lots = self.optimizer.get_all_lots(symbol='AAPL')

        self.assertEqual(len(lots), 2)
        self.assertTrue(all(lot['symbol'] == 'AAPL' for lot in lots))

    def test_get_all_lots_symbol_case_insensitive(self):
        """Test get_all_lots is case-insensitive for symbols."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))

        lots = self.optimizer.get_all_lots(symbol='aapl')

        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0]['symbol'], 'AAPL')

    def test_get_all_lots_user_isolation(self):
        """Test get_all_lots only returns current user's lots."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))

        # Create lot for other user
        TaxLot.objects.create(
            user=self.other_user,
            symbol='AAPL',
            original_quantity=Decimal('500'),
            remaining_quantity=Decimal('500'),
            cost_basis_per_share=Decimal('145.00'),
            total_cost_basis=Decimal('72500.00'),
            acquired_at=timezone.now(),
        )

        lots = self.optimizer.get_all_lots()

        self.assertEqual(len(lots), 1)
        self.assertEqual(lots[0]['remaining_quantity'], 100.0)

    # ==================== Get Lots By Symbol Tests ====================

    def test_get_lots_by_symbol_returns_summary(self):
        """Test get_lots_by_symbol returns complete summary."""
        lot1 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=400)
        lot2 = self._create_tax_lot('AAPL', Decimal('50'), Decimal('160.00'), days_ago=100)

        # Update market prices
        lot1.update_market_data(170.00)
        lot2.update_market_data(170.00)

        result = self.optimizer.get_lots_by_symbol('AAPL')

        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(len(result['lots']), 2)

        summary = result['summary']
        self.assertEqual(summary['total_lots'], 2)
        self.assertEqual(summary['total_quantity'], 150.0)
        # Average cost: (100*150 + 50*160) / 150 = 153.33
        self.assertAlmostEqual(summary['average_cost_basis'], 153.33, places=1)
        self.assertEqual(summary['total_cost_basis'], 23000.0)  # 15000 + 8000
        self.assertEqual(summary['total_market_value'], 25500.0)  # 150 * 170
        self.assertEqual(summary['total_unrealized_gain'], 2500.0)  # 25500 - 23000

    def test_get_lots_by_symbol_distinguishes_long_short_term(self):
        """Test get_lots_by_symbol correctly separates long and short term."""
        # Long term lot (held > 365 days)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=400)
        # Short term lot (held < 365 days)
        self._create_tax_lot('AAPL', Decimal('50'), Decimal('160.00'), days_ago=100)

        result = self.optimizer.get_lots_by_symbol('AAPL')
        summary = result['summary']

        self.assertEqual(summary['long_term_quantity'], 100.0)
        self.assertEqual(summary['short_term_quantity'], 50.0)

    def test_get_lots_by_symbol_empty(self):
        """Test get_lots_by_symbol with no matching lots."""
        result = self.optimizer.get_lots_by_symbol('AAPL')

        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['lots'], [])
        self.assertEqual(result['summary']['total_lots'], 0)
        self.assertEqual(result['summary']['total_quantity'], 0.0)

    # ==================== Lot Selection Tests (FIFO/LIFO/HIFO) ====================

    def test_select_lots_fifo_ordering(self):
        """Test FIFO selects oldest lots first."""
        # Create lots with different acquisition dates
        lot_old = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot_medium = self._create_tax_lot('AAPL', Decimal('100'), Decimal('155.00'), days_ago=50)
        lot_new = self._create_tax_lot('AAPL', Decimal('100'), Decimal('160.00'), days_ago=10)

        selected = self.optimizer._select_lots('AAPL', Decimal('150'), 'fifo')

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]['lot'].id, lot_old.id)
        self.assertEqual(selected[0]['quantity'], Decimal('100'))
        self.assertEqual(selected[1]['lot'].id, lot_medium.id)
        self.assertEqual(selected[1]['quantity'], Decimal('50'))

    def test_select_lots_lifo_ordering(self):
        """Test LIFO selects newest lots first."""
        lot_old = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot_medium = self._create_tax_lot('AAPL', Decimal('100'), Decimal('155.00'), days_ago=50)
        lot_new = self._create_tax_lot('AAPL', Decimal('100'), Decimal('160.00'), days_ago=10)

        selected = self.optimizer._select_lots('AAPL', Decimal('150'), 'lifo')

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]['lot'].id, lot_new.id)
        self.assertEqual(selected[0]['quantity'], Decimal('100'))
        self.assertEqual(selected[1]['lot'].id, lot_medium.id)
        self.assertEqual(selected[1]['quantity'], Decimal('50'))

    def test_select_lots_hifo_ordering(self):
        """Test HIFO selects highest cost basis lots first."""
        lot_low = self._create_tax_lot('AAPL', Decimal('100'), Decimal('140.00'), days_ago=80)
        lot_high = self._create_tax_lot('AAPL', Decimal('100'), Decimal('170.00'), days_ago=60)
        lot_medium = self._create_tax_lot('AAPL', Decimal('100'), Decimal('155.00'), days_ago=40)

        selected = self.optimizer._select_lots('AAPL', Decimal('150'), 'hifo')

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]['lot'].id, lot_high.id)
        self.assertEqual(selected[0]['quantity'], Decimal('100'))
        self.assertEqual(selected[1]['lot'].id, lot_medium.id)
        self.assertEqual(selected[1]['quantity'], Decimal('50'))

    def test_select_lots_partial_quantity(self):
        """Test lot selection handles partial lot quantities correctly."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))

        selected = self.optimizer._select_lots('AAPL', Decimal('50'), 'fifo')

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['quantity'], Decimal('50'))
        self.assertEqual(selected[0]['lot'].id, lot.id)

    def test_select_lots_insufficient_quantity(self):
        """Test lot selection when requested quantity exceeds available."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))

        selected = self.optimizer._select_lots('AAPL', Decimal('200'), 'fifo')

        self.assertEqual(len(selected), 1)
        # Should only return what's available
        self.assertEqual(selected[0]['quantity'], Decimal('100'))

    def test_select_lots_excludes_closed(self):
        """Test lot selection excludes closed lots."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), is_closed=True)
        open_lot = self._create_tax_lot('AAPL', Decimal('50'), Decimal('160.00'))

        selected = self.optimizer._select_lots('AAPL', Decimal('100'), 'fifo')

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['lot'].id, open_lot.id)
        self.assertEqual(selected[0]['quantity'], Decimal('50'))

    def test_select_lots_unknown_method_defaults_fifo(self):
        """Test unknown lot selection method defaults to FIFO."""
        lot_old = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot_new = self._create_tax_lot('AAPL', Decimal('100'), Decimal('160.00'), days_ago=10)

        selected = self.optimizer._select_lots('AAPL', Decimal('50'), 'unknown_method')

        # Should use FIFO as default (oldest first)
        self.assertEqual(selected[0]['lot'].id, lot_old.id)

    # ==================== Preview Sale Tax Impact Tests ====================

    def test_preview_sale_tax_impact_no_lots(self):
        """Test preview_sale_tax_impact with no available lots."""
        result = self.optimizer.preview_sale_tax_impact('AAPL', 100, 150.0)

        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('No available lots', result['error'])

    def test_preview_sale_tax_impact_profitable_long_term(self):
        """Test preview_sale_tax_impact for profitable long-term sale."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 50, 150.0, lot_selection='fifo'
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['quantity'], 50.0)

        summary = result['summary']
        self.assertEqual(summary['total_proceeds'], 7500.0)  # 50 * 150
        self.assertEqual(summary['total_cost_basis'], 5000.0)  # 50 * 100
        self.assertEqual(summary['total_gain_loss'], 2500.0)
        self.assertEqual(summary['long_term_gain_loss'], 2500.0)
        self.assertEqual(summary['short_term_gain_loss'], 0.0)

        # Long-term tax: 2500 * 0.15 = 375
        self.assertEqual(summary['estimated_long_term_tax'], 375.0)
        self.assertEqual(summary['estimated_short_term_tax'], 0.0)
        self.assertEqual(summary['total_estimated_tax'], 375.0)

    def test_preview_sale_tax_impact_profitable_short_term(self):
        """Test preview_sale_tax_impact for profitable short-term sale."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 50, 150.0, lot_selection='fifo'
        )

        self.assertTrue(result['success'])

        summary = result['summary']
        self.assertEqual(summary['total_gain_loss'], 2500.0)
        self.assertEqual(summary['short_term_gain_loss'], 2500.0)
        self.assertEqual(summary['long_term_gain_loss'], 0.0)

        # Short-term tax: 2500 * 0.35 = 875
        self.assertEqual(summary['estimated_short_term_tax'], 875.0)
        self.assertEqual(summary['estimated_long_term_tax'], 0.0)

    def test_preview_sale_tax_impact_loss(self):
        """Test preview_sale_tax_impact for a sale at a loss."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 50, 100.0, lot_selection='fifo'
        )

        self.assertTrue(result['success'])

        summary = result['summary']
        self.assertEqual(summary['total_proceeds'], 5000.0)  # 50 * 100
        self.assertEqual(summary['total_cost_basis'], 7500.0)  # 50 * 150
        self.assertEqual(summary['total_gain_loss'], -2500.0)

        # No tax on losses
        self.assertEqual(summary['total_estimated_tax'], 0.0)

    def test_preview_sale_tax_impact_mixed_term_lots(self):
        """Test preview_sale_tax_impact with both long and short term lots."""
        # Long term lot
        lot_long = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)
        # Short term lot
        lot_short = self._create_tax_lot('AAPL', Decimal('100'), Decimal('120.00'), days_ago=100)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 150, 150.0, lot_selection='fifo'
        )

        self.assertTrue(result['success'])

        summary = result['summary']
        # FIFO: 100 from long-term (gain: 5000) + 50 from short-term (gain: 1500)
        self.assertEqual(summary['total_proceeds'], 22500.0)  # 150 * 150
        self.assertEqual(summary['total_cost_basis'], 16000.0)  # 100*100 + 50*120
        self.assertEqual(summary['long_term_gain_loss'], 5000.0)  # 100 * (150-100)
        self.assertEqual(summary['short_term_gain_loss'], 1500.0)  # 50 * (150-120)

    def test_preview_sale_tax_impact_includes_lot_details(self):
        """Test preview_sale_tax_impact includes detailed lot breakdown."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 50, 150.0, lot_selection='fifo'
        )

        self.assertEqual(len(result['lots_to_sell']), 1)
        lot_detail = result['lots_to_sell'][0]

        self.assertEqual(lot_detail['lot_id'], lot.id)
        self.assertEqual(lot_detail['quantity'], 50.0)
        self.assertEqual(lot_detail['cost_basis_per_share'], 100.0)
        self.assertEqual(lot_detail['sale_price'], 150.0)
        self.assertEqual(lot_detail['gain_loss'], 2500.0)
        self.assertTrue(lot_detail['is_long_term'])

    def test_preview_sale_tax_impact_includes_wash_sale_warning(self):
        """Test preview_sale_tax_impact includes wash sale warning."""
        # Create a recent purchase (within 30 days)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=10)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 50, 140.0, lot_selection='fifo'
        )

        self.assertIn('wash_sale_warning', result)
        self.assertTrue(result['wash_sale_warning']['is_risky'])
        self.assertEqual(result['wash_sale_warning']['risk_level'], 'high')

    def test_preview_sale_tax_impact_compares_methods(self):
        """Test preview_sale_tax_impact includes method comparison."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('140.00'), days_ago=50)

        result = self.optimizer.preview_sale_tax_impact(
            'AAPL', 100, 150.0, lot_selection='fifo'
        )

        self.assertIn('alternative_methods', result)
        alternatives = result['alternative_methods']
        self.assertIn('fifo', alternatives)
        self.assertIn('lifo', alternatives)
        self.assertIn('hifo', alternatives)

    # ==================== Wash Sale Detection Tests ====================

    def test_check_wash_sale_risk_no_recent_purchases(self):
        """Test wash sale risk with no recent purchases."""
        # Create old lot (beyond 30-day window)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=60)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertFalse(result['is_risky'])
        self.assertEqual(result['risk_level'], 'none')
        self.assertEqual(result['recent_purchases'], [])

    def test_check_wash_sale_risk_with_recent_purchase(self):
        """Test wash sale risk detection with recent purchase."""
        # Create lot within 30-day window
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=15)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertTrue(result['is_risky'])
        self.assertEqual(result['risk_level'], 'high')
        self.assertEqual(len(result['recent_purchases']), 1)
        self.assertEqual(result['recent_purchases'][0]['days_ago'], 15)

    def test_check_wash_sale_risk_multiple_recent_purchases(self):
        """Test wash sale risk with multiple recent purchases."""
        self._create_tax_lot('AAPL', Decimal('50'), Decimal('150.00'), days_ago=10)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('155.00'), days_ago=20)
        self._create_tax_lot('AAPL', Decimal('75'), Decimal('160.00'), days_ago=5)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertTrue(result['is_risky'])
        self.assertEqual(result['risk_level'], 'high')
        # Should return up to 5 most recent purchases
        self.assertLessEqual(len(result['recent_purchases']), 5)

    def test_check_wash_sale_risk_boundary_30_days(self):
        """Test wash sale detection at exactly 30 days boundary."""
        # Lot at exactly 29 days should be in the window (within 30-day lookback)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=29)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        # 29 days ago is within the 30-day window
        self.assertTrue(result['is_risky'])

    def test_check_wash_sale_risk_beyond_30_days(self):
        """Test wash sale detection just beyond 30 days."""
        # Lot at 31 days should be outside window
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=31)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertFalse(result['is_risky'])
        self.assertEqual(result['risk_level'], 'none')

    def test_check_wash_sale_risk_case_insensitive(self):
        """Test wash sale risk is case-insensitive for symbol."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=10)

        result = self.optimizer.check_wash_sale_risk('aapl')

        self.assertTrue(result['is_risky'])
        self.assertEqual(result['symbol'], 'AAPL')

    def test_check_wash_sale_risk_includes_recommendation(self):
        """Test wash sale risk includes actionable recommendation."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=15)

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertIn('recommendation', result)
        self.assertIn('WASH SALE RISK', result['recommendation'])
        self.assertIn('Wait', result['recommendation'])

    def test_check_wash_sale_risk_user_isolation(self):
        """Test wash sale detection only considers current user's purchases."""
        # Create lot for other user (within window)
        TaxLot.objects.create(
            user=self.other_user,
            symbol='AAPL',
            original_quantity=Decimal('100'),
            remaining_quantity=Decimal('100'),
            cost_basis_per_share=Decimal('150.00'),
            total_cost_basis=Decimal('15000.00'),
            acquired_at=timezone.now() - timedelta(days=10),
        )

        result = self.optimizer.check_wash_sale_risk('AAPL')

        self.assertFalse(result['is_risky'])

    # ==================== Days Until Wash Safe Tests ====================

    def test_days_until_wash_safe_no_sales(self):
        """Test days_until_wash_safe with no recent sales."""
        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 0)

    def test_days_until_wash_safe_recent_sale(self):
        """Test days_until_wash_safe with recent sale."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=60)
        self._create_tax_lot_sale(lot, Decimal('50'), Decimal('140.00'), days_ago=15)

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 15)  # 30 - 15 = 15 days

    def test_days_until_wash_safe_past_window(self):
        """Test days_until_wash_safe when past wash sale window."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        self._create_tax_lot_sale(lot, Decimal('50'), Decimal('140.00'), days_ago=40)

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 0)  # Already past 30 days

    def test_days_until_wash_safe_exactly_30_days(self):
        """Test days_until_wash_safe at exactly 30 days."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=60)
        self._create_tax_lot_sale(lot, Decimal('50'), Decimal('140.00'), days_ago=30)

        result = self.optimizer._days_until_wash_safe('AAPL')

        self.assertEqual(result, 0)  # Exactly 30 days means safe

    # ==================== Loss Harvesting Tests ====================

    def test_get_harvesting_opportunities_empty(self):
        """Test get_harvesting_opportunities with no loss positions."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)

        result = self.optimizer.get_harvesting_opportunities()

        self.assertEqual(result, [])

    def test_get_harvesting_opportunities_finds_losses(self):
        """Test get_harvesting_opportunities finds positions with losses."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        # Update with lower current price to create unrealized loss
        lot.update_market_data(100.00)  # Loss of 50 per share = 5000 total

        result = self.optimizer.get_harvesting_opportunities(min_loss=100.0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertLess(result[0]['potential_loss'], 0)

    def test_get_harvesting_opportunities_respects_min_loss(self):
        """Test get_harvesting_opportunities respects minimum loss threshold."""
        lot1 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot1.update_market_data(140.00)  # Loss of $1000

        lot2 = self._create_tax_lot('MSFT', Decimal('10'), Decimal('380.00'), days_ago=100)
        lot2.update_market_data(375.00)  # Loss of $50 (below threshold)

        result = self.optimizer.get_harvesting_opportunities(min_loss=100.0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')

    def test_get_harvesting_opportunities_includes_tax_savings(self):
        """Test harvesting opportunities include estimated tax savings."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)  # Loss of $5000

        result = self.optimizer.get_harvesting_opportunities(min_loss=100.0)

        self.assertIn('tax_savings_estimate', result[0])
        self.assertIn('estimated_savings', result[0]['tax_savings_estimate'])

    def test_get_harvesting_opportunities_includes_wash_risk(self):
        """Test harvesting opportunities include wash sale risk."""
        # Create an older lot with loss
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)

        # Create a recent purchase that would trigger wash sale
        self._create_tax_lot('AAPL', Decimal('50'), Decimal('105.00'), days_ago=10)

        result = self.optimizer.get_harvesting_opportunities(min_loss=100.0)

        self.assertEqual(len(result), 1)
        self.assertIn('wash_sale_risk', result[0])
        self.assertTrue(result[0]['wash_sale_risk']['is_risky'])

    def test_get_harvesting_opportunities_respects_limit(self):
        """Test get_harvesting_opportunities respects result limit."""
        # Create multiple lots with losses
        for i in range(10):
            lot = self._create_tax_lot(f'SYM{i}', Decimal('100'), Decimal('150.00'), days_ago=100)
            lot.update_market_data(100.00)

        result = self.optimizer.get_harvesting_opportunities(min_loss=100.0, limit=5)

        self.assertLessEqual(len(result), 5)

    # ==================== Year Summary Tests ====================

    def test_get_year_summary_no_sales(self):
        """Test get_year_summary with no sales."""
        result = self.optimizer.get_year_summary()

        self.assertEqual(result['year'], timezone.now().year)
        self.assertEqual(result['total_sales'], 0)
        self.assertEqual(result['net_gain_loss'], 0.0)
        self.assertIn('disclaimer', result)

    def test_get_year_summary_with_gains(self):
        """Test get_year_summary with realized gains."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)
        self._create_tax_lot_sale(lot, Decimal('50'), Decimal('150.00'), days_ago=5)

        result = self.optimizer.get_year_summary()

        self.assertEqual(result['total_sales'], 1)
        self.assertGreater(result['net_gain_loss'], 0)
        self.assertGreater(result['long_term']['net'], 0)

    def test_get_year_summary_with_losses(self):
        """Test get_year_summary with realized losses."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        self._create_tax_lot_sale(lot, Decimal('50'), Decimal('100.00'), days_ago=5)

        result = self.optimizer.get_year_summary()

        self.assertLess(result['net_gain_loss'], 0)
        self.assertLess(result['short_term']['net'], 0)

    def test_get_year_summary_short_and_long_term(self):
        """Test get_year_summary correctly separates short and long term."""
        # Long term sale (gain)
        lot_long = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)
        self._create_tax_lot_sale(lot_long, Decimal('50'), Decimal('150.00'), days_ago=5)

        # Short term sale (loss)
        lot_short = self._create_tax_lot('MSFT', Decimal('100'), Decimal('400.00'), days_ago=100)
        self._create_tax_lot_sale(lot_short, Decimal('50'), Decimal('350.00'), days_ago=3)

        result = self.optimizer.get_year_summary()

        self.assertEqual(result['total_sales'], 2)
        self.assertGreater(result['long_term']['gains'], 0)
        self.assertGreater(result['short_term']['losses'], 0)

    def test_get_year_summary_loss_carryforward(self):
        """Test get_year_summary calculates loss carryforward correctly."""
        # Create a large loss
        lot = self._create_tax_lot('AAPL', Decimal('1000'), Decimal('150.00'), days_ago=100)
        # Sell at a significant loss
        self._create_tax_lot_sale(lot, Decimal('1000'), Decimal('100.00'), days_ago=5)
        # Loss: $50,000

        result = self.optimizer.get_year_summary()

        # Max deductible is $3,000 per year
        # Carryforward should be $50,000 - $3,000 = $47,000
        self.assertGreater(result['loss_carryforward'], 0)
        self.assertLess(result['net_gain_loss'], -3000)

    def test_get_year_summary_specific_year(self):
        """Test get_year_summary for a specific year."""
        result = self.optimizer.get_year_summary(year=2023)

        self.assertEqual(result['year'], 2023)

    def test_get_year_summary_by_symbol(self):
        """Test get_year_summary includes breakdown by symbol."""
        lot1 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        self._create_tax_lot_sale(lot1, Decimal('50'), Decimal('150.00'), days_ago=5)

        lot2 = self._create_tax_lot('MSFT', Decimal('50'), Decimal('350.00'), days_ago=100)
        self._create_tax_lot_sale(lot2, Decimal('25'), Decimal('400.00'), days_ago=3)

        result = self.optimizer.get_year_summary()

        self.assertIn('by_symbol', result)
        self.assertIn('AAPL', result['by_symbol'])
        self.assertIn('MSFT', result['by_symbol'])

    def test_get_year_summary_with_wash_sales(self):
        """Test get_year_summary handles wash sale adjustments."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        self._create_tax_lot_sale(
            lot, Decimal('50'), Decimal('100.00'), days_ago=5,
            is_wash_sale=True, wash_sale_disallowed=Decimal('1000.00')
        )

        result = self.optimizer.get_year_summary()

        self.assertGreater(result['wash_sale_adjustments'], 0)
        self.assertEqual(result['wash_sale_adjustments'], 1000.0)

    def test_get_year_summary_estimates_tax(self):
        """Test get_year_summary estimates tax liability correctly."""
        # Long term gain
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)
        self._create_tax_lot_sale(lot, Decimal('100'), Decimal('200.00'), days_ago=5)
        # Gain: $10,000 long-term

        result = self.optimizer.get_year_summary()

        # Long-term tax: 10000 * 0.15 = 1500
        self.assertEqual(result['estimated_tax_liability']['long_term_tax'], 1500.0)

    # ==================== Suggest Lot Selection Tests ====================

    def test_suggest_lot_selection_minimize_tax(self):
        """Test suggest_lot_selection for minimizing tax."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=50)

        result = self.optimizer.suggest_lot_selection('AAPL', 100, goal='minimize_tax')

        self.assertEqual(result['recommended_method'], 'hifo')
        self.assertIn('reason', result)
        self.assertIn('highest cost', result['reason'].lower())

    def test_suggest_lot_selection_maximize_loss(self):
        """Test suggest_lot_selection for maximizing harvestable loss."""
        lot1 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        lot1.update_market_data(80.00)
        lot2 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=50)
        lot2.update_market_data(80.00)

        result = self.optimizer.suggest_lot_selection('AAPL', 100, goal='maximize_loss')

        self.assertIn(result['recommended_method'], ['fifo', 'lifo'])
        self.assertIn('method_comparison', result)

    def test_suggest_lot_selection_long_term_priority(self):
        """Test suggest_lot_selection for long-term priority."""
        result = self.optimizer.suggest_lot_selection('AAPL', 100, goal='long_term_priority')

        self.assertEqual(result['recommended_method'], 'specific')
        self.assertIn('long-term', result['reason'].lower())

    def test_suggest_lot_selection_default(self):
        """Test suggest_lot_selection with unknown goal defaults to FIFO."""
        result = self.optimizer.suggest_lot_selection('AAPL', 100, goal='unknown_goal')

        self.assertEqual(result['recommended_method'], 'fifo')

    # ==================== Compare Lot Selection Methods Tests ====================

    def test_compare_lot_selection_methods(self):
        """Test _compare_lot_selection_methods returns all method results."""
        lot1 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        lot2 = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=50)
        lot1.update_market_data(120.00)
        lot2.update_market_data(120.00)

        result = self.optimizer._compare_lot_selection_methods('AAPL', 100, 120.0)

        self.assertIn('fifo', result)
        self.assertIn('lifo', result)
        self.assertIn('hifo', result)

        # FIFO (oldest first, $100 cost) should have higher gain than LIFO ($150 cost)
        self.assertGreater(result['fifo']['total_gain'], result['lifo']['total_gain'])

    def test_compare_lot_selection_methods_uses_current_price(self):
        """Test _compare_lot_selection_methods uses current price when sale_price not provided."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        lot.update_market_data(150.00)

        result = self.optimizer._compare_lot_selection_methods('AAPL', 50, None)

        # Should use current price of 150
        self.assertEqual(result['fifo']['total_proceeds'], 7500.0)  # 50 * 150

    # ==================== Helper Method Tests ====================

    def test_estimate_tax_savings_no_loss(self):
        """Test _estimate_tax_savings with no loss."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)
        lot.update_market_data(150.00)  # Gain, not loss
        lot.refresh_from_db()

        result = self.optimizer._estimate_tax_savings(lot)

        self.assertEqual(result['savings'], 0)

    def test_estimate_tax_savings_short_term_loss(self):
        """Test _estimate_tax_savings with short-term loss."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)  # Loss of $5000
        lot.refresh_from_db()

        result = self.optimizer._estimate_tax_savings(lot)

        self.assertEqual(result['holding_type'], 'short_term')
        self.assertGreater(result['estimated_savings'], 0)
        # Short-term rate is 35%
        self.assertEqual(result['tax_rate_applied'], 35.0)

    def test_estimate_tax_savings_long_term_loss(self):
        """Test _estimate_tax_savings with long-term loss."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=400)
        lot.update_market_data(100.00)  # Loss of $5000
        lot.refresh_from_db()

        result = self.optimizer._estimate_tax_savings(lot)

        self.assertEqual(result['holding_type'], 'long_term')
        self.assertGreater(result['estimated_savings'], 0)
        # Long-term rate is 15%
        self.assertEqual(result['tax_rate_applied'], 15.0)

    def test_find_similar_alternatives_known_symbol(self):
        """Test _find_similar_alternatives for known symbols."""
        alternatives = self.optimizer._find_similar_alternatives('SPY')

        self.assertIsInstance(alternatives, list)
        self.assertGreater(len(alternatives), 0)
        symbols = [alt['symbol'] for alt in alternatives]
        self.assertIn('VOO', symbols)

    def test_find_similar_alternatives_unknown_symbol(self):
        """Test _find_similar_alternatives for unknown symbol."""
        alternatives = self.optimizer._find_similar_alternatives('UNKNOWN_XYZ')

        self.assertEqual(alternatives, [])

    def test_check_strategy_intentions(self):
        """Test _check_strategy_intentions placeholder."""
        result = self.optimizer._check_strategy_intentions('AAPL')

        self.assertFalse(result['has_pending'])
        self.assertIn('strategies', result)

    def test_get_harvest_recommendation_high_risk(self):
        """Test _get_harvest_recommendation with high wash risk."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)
        lot.refresh_from_db()

        wash_risk = {'risk_level': 'high'}

        result = self.optimizer._get_harvest_recommendation(lot, wash_risk, days_until_safe=15)

        self.assertIn('WARNING', result)
        self.assertIn('wash sale', result.lower())

    def test_get_harvest_recommendation_medium_risk(self):
        """Test _get_harvest_recommendation with medium wash risk."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)

        wash_risk = {'risk_level': 'medium'}

        result = self.optimizer._get_harvest_recommendation(lot, wash_risk, days_until_safe=0)

        self.assertIn('CAUTION', result)

    def test_get_harvest_recommendation_no_risk_short_term(self):
        """Test _get_harvest_recommendation no risk short-term."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.update_market_data(100.00)
        lot.refresh_from_db()

        wash_risk = {'risk_level': 'none'}

        result = self.optimizer._get_harvest_recommendation(lot, wash_risk, days_until_safe=0)

        self.assertIn('OPPORTUNITY', result)
        self.assertIn('short-term', result)

    def test_get_harvest_recommendation_no_risk_long_term(self):
        """Test _get_harvest_recommendation no risk long-term."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=400)
        lot.update_market_data(100.00)
        lot.refresh_from_db()

        wash_risk = {'risk_level': 'none'}

        result = self.optimizer._get_harvest_recommendation(lot, wash_risk, days_until_safe=0)

        self.assertIn('OPPORTUNITY', result)
        self.assertIn('long-term', result)

    def test_get_wash_sale_recommendation_none(self):
        """Test _get_wash_sale_recommendation with no risk."""
        result = self.optimizer._get_wash_sale_recommendation('none', [])

        self.assertIn('Safe', result)

    def test_get_wash_sale_recommendation_high(self):
        """Test _get_wash_sale_recommendation with high risk."""
        recent_purchases = [{'days_ago': 10}]

        result = self.optimizer._get_wash_sale_recommendation('high', recent_purchases)

        self.assertIn('WASH SALE RISK', result)
        self.assertIn('21', result)  # 31 - 10 = 21 days to wait

    def test_get_wash_sale_recommendation_medium(self):
        """Test _get_wash_sale_recommendation with medium risk."""
        result = self.optimizer._get_wash_sale_recommendation('medium', [])

        self.assertIn('POTENTIAL WASH SALE', result)

    # ==================== Tax Calculation Accuracy Tests ====================

    def test_tax_calculation_accuracy_short_term(self):
        """Verify short-term tax calculations are accurate."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=100)

        result = self.optimizer.preview_sale_tax_impact('AAPL', 100, 200.0)

        # Gain: 100 * (200 - 100) = 10,000
        # Short-term tax: 10,000 * 0.35 = 3,500
        self.assertEqual(result['summary']['total_gain_loss'], 10000.0)
        self.assertEqual(result['summary']['estimated_short_term_tax'], 3500.0)
        self.assertEqual(result['summary']['effective_tax_rate'], 35.0)

    def test_tax_calculation_accuracy_long_term(self):
        """Verify long-term tax calculations are accurate."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)

        result = self.optimizer.preview_sale_tax_impact('AAPL', 100, 200.0)

        # Gain: 100 * (200 - 100) = 10,000
        # Long-term tax: 10,000 * 0.15 = 1,500
        self.assertEqual(result['summary']['total_gain_loss'], 10000.0)
        self.assertEqual(result['summary']['estimated_long_term_tax'], 1500.0)
        self.assertEqual(result['summary']['effective_tax_rate'], 15.0)

    def test_tax_calculation_mixed_lots(self):
        """Verify mixed short/long-term tax calculations."""
        # Long-term lot: 100 shares @ $100
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('100.00'), days_ago=400)
        # Short-term lot: 100 shares @ $150
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)

        # Sell 200 shares at $200 using FIFO
        result = self.optimizer.preview_sale_tax_impact('AAPL', 200, 200.0)

        # FIFO: Long-term lot first
        # Long-term gain: 100 * (200 - 100) = 10,000
        # Short-term gain: 100 * (200 - 150) = 5,000
        self.assertEqual(result['summary']['long_term_gain_loss'], 10000.0)
        self.assertEqual(result['summary']['short_term_gain_loss'], 5000.0)

        # Taxes:
        # Long-term: 10,000 * 0.15 = 1,500
        # Short-term: 5,000 * 0.35 = 1,750
        self.assertEqual(result['summary']['estimated_long_term_tax'], 1500.0)
        self.assertEqual(result['summary']['estimated_short_term_tax'], 1750.0)
        self.assertEqual(result['summary']['total_estimated_tax'], 3250.0)

    # ==================== Edge Cases ====================

    def test_lot_with_zero_remaining_quantity(self):
        """Test handling of lots with zero remaining quantity."""
        lot = self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'), days_ago=100)
        lot.remaining_quantity = Decimal('0')
        lot.is_closed = True
        lot.save()

        lots = self.optimizer.get_all_lots()

        self.assertEqual(len(lots), 0)

    def test_multiple_symbols(self):
        """Test handling multiple symbols correctly."""
        self._create_tax_lot('AAPL', Decimal('100'), Decimal('150.00'))
        self._create_tax_lot('MSFT', Decimal('50'), Decimal('380.00'))
        self._create_tax_lot('GOOGL', Decimal('25'), Decimal('2500.00'))

        all_lots = self.optimizer.get_all_lots()
        apple_lots = self.optimizer.get_all_lots(symbol='AAPL')

        self.assertEqual(len(all_lots), 3)
        self.assertEqual(len(apple_lots), 1)

    def test_decimal_precision(self):
        """Test that decimal precision is maintained."""
        lot = self._create_tax_lot(
            'AAPL',
            Decimal('100.123456'),
            Decimal('150.654321'),
            days_ago=100
        )

        result = self.optimizer.get_lots_by_symbol('AAPL')

        lot_data = result['lots'][0]
        self.assertAlmostEqual(lot_data['remaining_quantity'], 100.123456, places=6)
        self.assertAlmostEqual(lot_data['cost_basis_per_share'], 150.654321, places=6)

    def test_large_quantity_handling(self):
        """Test handling of large quantities."""
        lot = self._create_tax_lot(
            'AAPL',
            Decimal('1000000'),  # 1 million shares
            Decimal('150.00')
        )
        lot.update_market_data(200.00)

        result = self.optimizer.preview_sale_tax_impact('AAPL', 1000000, 200.0)

        self.assertTrue(result['success'])
        # Gain: 1,000,000 * 50 = 50,000,000
        self.assertEqual(result['summary']['total_gain_loss'], 50000000.0)
