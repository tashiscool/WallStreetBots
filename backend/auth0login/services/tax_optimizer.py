"""
Tax Optimization Service for WallStreetBots.

Provides tax lot management, loss harvesting opportunities,
wash sale detection, and pre-trade tax impact analysis.

DISCLAIMER: This is for informational purposes only and does not
constitute tax advice. Users should consult qualified tax professionals.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from django.contrib.auth.models import User
from django.db.models import Q, Sum, F
from django.utils import timezone

logger = logging.getLogger(__name__)

# Tax rates for estimation (2024 rates, simplified)
SHORT_TERM_TAX_RATE = Decimal('0.35')  # Ordinary income (assume high bracket)
LONG_TERM_TAX_RATE = Decimal('0.15')   # Long-term capital gains

# Wash sale window
WASH_SALE_WINDOW_DAYS = 30


class TaxOptimizer:
    """
    Service for tax-aware trading decisions.

    Features:
    - Loss harvesting opportunity detection
    - Wash sale risk assessment
    - Pre-trade tax impact preview
    - Lot selection optimization
    - Year-to-date tax summary
    """

    def __init__(self, user: User):
        """
        Initialize the tax optimizer for a specific user.

        Args:
            user: Django User instance
        """
        self.user = user

    def get_all_lots(
        self,
        symbol: Optional[str] = None,
        include_closed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all tax lots for the user.

        Args:
            symbol: Filter by symbol (optional)
            include_closed: Include fully sold lots

        Returns:
            List of tax lot dictionaries
        """
        from backend.tradingbot.models.models import TaxLot

        queryset = TaxLot.objects.filter(user=self.user)

        if symbol:
            queryset = queryset.filter(symbol=symbol.upper())

        if not include_closed:
            queryset = queryset.filter(is_closed=False)

        return [lot.to_dict() for lot in queryset.order_by('symbol', 'acquired_at')]

    def get_lots_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed lot breakdown for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Summary with lots and aggregated stats
        """
        from backend.tradingbot.models.models import TaxLot

        lots = TaxLot.objects.filter(
            user=self.user,
            symbol=symbol.upper(),
            is_closed=False
        ).order_by('acquired_at')

        lot_list = [lot.to_dict() for lot in lots]

        # Aggregate stats
        total_quantity = sum(Decimal(str(l['remaining_quantity'])) for l in lot_list)
        total_cost = sum(
            Decimal(str(l['remaining_quantity'])) * Decimal(str(l['cost_basis_per_share']))
            for l in lot_list
        )
        total_market_value = sum(
            Decimal(str(l['market_value'])) for l in lot_list
            if l['market_value'] is not None
        )
        total_unrealized = sum(
            Decimal(str(l['unrealized_gain'])) for l in lot_list
            if l['unrealized_gain'] is not None
        )

        avg_cost_basis = total_cost / total_quantity if total_quantity > 0 else Decimal('0')

        long_term_qty = sum(
            Decimal(str(l['remaining_quantity'])) for l in lot_list
            if l['is_long_term']
        )
        short_term_qty = total_quantity - long_term_qty

        return {
            'symbol': symbol.upper(),
            'lots': lot_list,
            'summary': {
                'total_lots': len(lot_list),
                'total_quantity': float(total_quantity),
                'average_cost_basis': float(avg_cost_basis),
                'total_cost_basis': float(total_cost),
                'total_market_value': float(total_market_value),
                'total_unrealized_gain': float(total_unrealized),
                'long_term_quantity': float(long_term_qty),
                'short_term_quantity': float(short_term_qty),
            }
        }

    def get_harvesting_opportunities(
        self,
        min_loss: float = 100.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find positions with losses that could be harvested.

        Args:
            min_loss: Minimum unrealized loss to consider (positive number)
            limit: Maximum number of opportunities to return

        Returns:
            List of harvesting opportunities with wash sale risk
        """
        from backend.tradingbot.models.models import TaxLot

        # Find lots with unrealized losses
        lots = TaxLot.objects.filter(
            user=self.user,
            is_closed=False,
            remaining_quantity__gt=0,
            unrealized_gain__lt=-Decimal(str(min_loss))
        ).order_by('unrealized_gain')[:limit]

        opportunities = []
        for lot in lots:
            wash_risk = self.check_wash_sale_risk(lot.symbol)
            tax_savings = self._estimate_tax_savings(lot)
            days_until_safe = self._days_until_wash_safe(lot.symbol)
            alternatives = self._find_similar_alternatives(lot.symbol)

            opportunities.append({
                'lot': lot.to_dict(),
                'symbol': lot.symbol,
                'potential_loss': float(lot.unrealized_gain),
                'tax_savings_estimate': tax_savings,
                'is_long_term': lot.is_long_term,
                'days_held': lot.days_held,
                'wash_sale_risk': wash_risk,
                'days_until_safe': days_until_safe,
                'similar_alternatives': alternatives,
                'recommendation': self._get_harvest_recommendation(
                    lot, wash_risk, days_until_safe
                ),
            })

        return opportunities

    def check_wash_sale_risk(self, symbol: str) -> Dict[str, Any]:
        """
        Check if selling a symbol would trigger wash sale rules.

        Wash sale: Cannot claim loss if you buy substantially identical
        securities within 30 days before or after the sale.

        Args:
            symbol: Trading symbol to check

        Returns:
            Wash sale risk assessment
        """
        from backend.tradingbot.models.models import TaxLot, TaxLotSale

        symbol = symbol.upper()
        now = timezone.now()
        window_start = now - timedelta(days=WASH_SALE_WINDOW_DAYS)

        # Check for recent purchases (last 30 days)
        recent_purchases = TaxLot.objects.filter(
            user=self.user,
            symbol=symbol,
            acquired_at__gte=window_start
        ).order_by('-acquired_at')

        recent_purchase_list = [{
            'acquired_at': p.acquired_at.isoformat(),
            'quantity': float(p.original_quantity),
            'cost_basis': float(p.cost_basis_per_share),
            'days_ago': (now - p.acquired_at).days,
        } for p in recent_purchases[:5]]

        # Check pending strategy buys (if any scheduled)
        pending_buys = self._check_strategy_intentions(symbol)

        # Determine risk level
        is_risky = len(recent_purchase_list) > 0 or pending_buys['has_pending']

        risk_level = 'none'
        if is_risky:
            if len(recent_purchase_list) > 0:
                risk_level = 'high'  # Definite wash sale if sold now
            elif pending_buys['has_pending']:
                risk_level = 'medium'  # Potential wash sale if strategies buy

        return {
            'symbol': symbol,
            'is_risky': is_risky,
            'risk_level': risk_level,
            'recent_purchases': recent_purchase_list,
            'pending_strategy_buys': pending_buys,
            'wash_sale_window_end': (
                now + timedelta(days=WASH_SALE_WINDOW_DAYS)
            ).isoformat(),
            'recommendation': self._get_wash_sale_recommendation(
                risk_level, recent_purchase_list
            ),
        }

    def preview_sale_tax_impact(
        self,
        symbol: str,
        quantity: float,
        sale_price: float,
        lot_selection: str = 'fifo'
    ) -> Dict[str, Any]:
        """
        Preview the tax impact of a proposed sale.

        Args:
            symbol: Trading symbol
            quantity: Number of shares to sell
            sale_price: Expected sale price per share
            lot_selection: Lot selection method ('fifo', 'lifo', 'hifo', 'specific')

        Returns:
            Tax impact preview
        """
        from backend.tradingbot.models.models import TaxLot

        symbol = symbol.upper()
        quantity = Decimal(str(quantity))
        sale_price = Decimal(str(sale_price))

        # Select lots based on method
        lots_to_sell = self._select_lots(symbol, quantity, lot_selection)

        if not lots_to_sell:
            return {
                'error': f'No available lots for {symbol}',
                'success': False,
            }

        # Calculate totals
        total_proceeds = Decimal('0')
        total_cost_basis = Decimal('0')
        short_term_gain = Decimal('0')
        long_term_gain = Decimal('0')
        lots_detail = []

        for lot_info in lots_to_sell:
            qty = lot_info['quantity']
            lot = lot_info['lot']

            proceeds = qty * sale_price
            cost = qty * lot.cost_basis_per_share
            gain = proceeds - cost

            total_proceeds += proceeds
            total_cost_basis += cost

            if lot.is_long_term:
                long_term_gain += gain
            else:
                short_term_gain += gain

            lots_detail.append({
                'lot_id': lot.id,
                'acquired_at': lot.acquired_at.isoformat(),
                'quantity': float(qty),
                'cost_basis_per_share': float(lot.cost_basis_per_share),
                'sale_price': float(sale_price),
                'proceeds': float(proceeds),
                'cost_basis': float(cost),
                'gain_loss': float(gain),
                'is_long_term': lot.is_long_term,
                'days_held': lot.days_held,
            })

        total_gain = short_term_gain + long_term_gain

        # Estimate tax
        short_term_tax = short_term_gain * SHORT_TERM_TAX_RATE if short_term_gain > 0 else Decimal('0')
        long_term_tax = long_term_gain * LONG_TERM_TAX_RATE if long_term_gain > 0 else Decimal('0')
        total_estimated_tax = short_term_tax + long_term_tax

        # Check wash sale risk
        wash_sale_warning = self.check_wash_sale_risk(symbol)

        return {
            'success': True,
            'symbol': symbol,
            'quantity': float(quantity),
            'sale_price': float(sale_price),
            'lot_selection_method': lot_selection,
            'lots_to_sell': lots_detail,
            'summary': {
                'total_proceeds': float(total_proceeds),
                'total_cost_basis': float(total_cost_basis),
                'total_gain_loss': float(total_gain),
                'short_term_gain_loss': float(short_term_gain),
                'long_term_gain_loss': float(long_term_gain),
                'estimated_short_term_tax': float(short_term_tax),
                'estimated_long_term_tax': float(long_term_tax),
                'total_estimated_tax': float(total_estimated_tax),
                'effective_tax_rate': float(
                    (total_estimated_tax / total_gain * 100) if total_gain > 0 else Decimal('0')
                ),
            },
            'wash_sale_warning': wash_sale_warning,
            'alternative_methods': self._compare_lot_selection_methods(
                symbol, quantity, sale_price
            ),
        }

    def suggest_lot_selection(
        self,
        symbol: str,
        quantity: float,
        goal: str = 'minimize_tax'
    ) -> Dict[str, Any]:
        """
        Suggest which lot selection method to use based on goal.

        Args:
            symbol: Trading symbol
            quantity: Number of shares to sell
            goal: Optimization goal
                - 'minimize_tax': Minimize current tax liability
                - 'maximize_loss': Maximize harvestable loss
                - 'long_term_priority': Prefer long-term lots
                - 'short_term_priority': Prefer short-term lots

        Returns:
            Recommended lot selection method with explanation
        """
        methods = ['fifo', 'lifo', 'hifo']
        comparisons = self._compare_lot_selection_methods(
            symbol, float(quantity), None
        )

        if goal == 'minimize_tax':
            # HIFO usually minimizes tax (highest cost = lowest gain)
            recommended = 'hifo'
            reason = "Selling highest cost basis lots first minimizes taxable gain"
        elif goal == 'maximize_loss':
            # Want lowest cost basis for maximum loss
            recommended = 'fifo' if comparisons.get('fifo', {}).get('total_gain', 0) < 0 else 'lifo'
            reason = "Selecting lots with lowest cost basis maximizes harvestable loss"
        elif goal == 'long_term_priority':
            recommended = 'specific'  # Would need specific lot selection
            reason = "Long-term gains are taxed at lower rates (15-20% vs ordinary income)"
        elif goal == 'short_term_priority':
            recommended = 'specific'
            reason = "Short-term losses offset ordinary income at higher rates"
        else:
            recommended = 'fifo'
            reason = "FIFO is the default IRS method and simplest to track"

        return {
            'symbol': symbol,
            'quantity': quantity,
            'goal': goal,
            'recommended_method': recommended,
            'reason': reason,
            'method_comparison': comparisons,
        }

    def get_year_summary(
        self,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get year-to-date realized gains/losses summary.

        Args:
            year: Tax year (defaults to current year)

        Returns:
            YTD tax summary
        """
        from backend.tradingbot.models.models import TaxLotSale

        if year is None:
            year = timezone.now().year

        year_start = timezone.make_aware(datetime(year, 1, 1))
        year_end = timezone.make_aware(datetime(year, 12, 31, 23, 59, 59))

        sales = TaxLotSale.objects.filter(
            user=self.user,
            sold_at__gte=year_start,
            sold_at__lte=year_end
        )

        # Aggregate by term
        short_term_gains = Decimal('0')
        short_term_losses = Decimal('0')
        long_term_gains = Decimal('0')
        long_term_losses = Decimal('0')
        wash_sale_adjustments = Decimal('0')
        total_proceeds = Decimal('0')
        total_cost_basis = Decimal('0')

        sales_by_symbol = {}

        for sale in sales:
            total_proceeds += sale.proceeds
            total_cost_basis += sale.cost_basis_sold
            wash_sale_adjustments += sale.wash_sale_disallowed

            gain = sale.realized_gain - sale.wash_sale_disallowed

            if sale.is_long_term:
                if gain >= 0:
                    long_term_gains += gain
                else:
                    long_term_losses += abs(gain)
            else:
                if gain >= 0:
                    short_term_gains += gain
                else:
                    short_term_losses += abs(gain)

            # Track by symbol
            if sale.symbol not in sales_by_symbol:
                sales_by_symbol[sale.symbol] = {
                    'proceeds': Decimal('0'),
                    'cost_basis': Decimal('0'),
                    'realized_gain': Decimal('0'),
                    'count': 0,
                }
            sales_by_symbol[sale.symbol]['proceeds'] += sale.proceeds
            sales_by_symbol[sale.symbol]['cost_basis'] += sale.cost_basis_sold
            sales_by_symbol[sale.symbol]['realized_gain'] += gain
            sales_by_symbol[sale.symbol]['count'] += 1

        # Net positions
        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        net_total = net_short_term + net_long_term

        # Estimate tax liability
        st_tax = net_short_term * SHORT_TERM_TAX_RATE if net_short_term > 0 else Decimal('0')
        lt_tax = net_long_term * LONG_TERM_TAX_RATE if net_long_term > 0 else Decimal('0')

        # Loss carryforward potential (max $3,000 per year)
        loss_carryforward = Decimal('0')
        if net_total < 0:
            deductible = min(abs(net_total), Decimal('3000'))
            loss_carryforward = abs(net_total) - deductible

        return {
            'year': year,
            'total_sales': sales.count(),
            'total_proceeds': float(total_proceeds),
            'total_cost_basis': float(total_cost_basis),
            'short_term': {
                'gains': float(short_term_gains),
                'losses': float(short_term_losses),
                'net': float(net_short_term),
            },
            'long_term': {
                'gains': float(long_term_gains),
                'losses': float(long_term_losses),
                'net': float(net_long_term),
            },
            'wash_sale_adjustments': float(wash_sale_adjustments),
            'net_gain_loss': float(net_total),
            'estimated_tax_liability': {
                'short_term_tax': float(st_tax),
                'long_term_tax': float(lt_tax),
                'total': float(st_tax + lt_tax),
            },
            'loss_carryforward': float(loss_carryforward),
            'by_symbol': {
                symbol: {
                    'proceeds': float(data['proceeds']),
                    'cost_basis': float(data['cost_basis']),
                    'realized_gain': float(data['realized_gain']),
                    'sale_count': data['count'],
                }
                for symbol, data in sorted(
                    sales_by_symbol.items(),
                    key=lambda x: x[1]['realized_gain'],
                    reverse=True
                )[:10]
            },
            'disclaimer': (
                "This is an estimate for informational purposes only. "
                "Actual tax liability may differ. Consult a tax professional."
            ),
        }

    def _select_lots(
        self,
        symbol: str,
        quantity: Decimal,
        method: str
    ) -> List[Dict[str, Any]]:
        """Select lots based on method."""
        from backend.tradingbot.models.models import TaxLot

        if method == 'fifo':
            ordering = 'acquired_at'
        elif method == 'lifo':
            ordering = '-acquired_at'
        elif method == 'hifo':
            ordering = '-cost_basis_per_share'
        else:
            ordering = 'acquired_at'  # Default to FIFO

        lots = TaxLot.objects.filter(
            user=self.user,
            symbol=symbol,
            is_closed=False,
            remaining_quantity__gt=0
        ).order_by(ordering)

        selected = []
        remaining = quantity

        for lot in lots:
            if remaining <= 0:
                break

            take = min(remaining, lot.remaining_quantity)
            selected.append({
                'lot': lot,
                'quantity': take,
            })
            remaining -= take

        return selected

    def _compare_lot_selection_methods(
        self,
        symbol: str,
        quantity: float,
        sale_price: Optional[float]
    ) -> Dict[str, Any]:
        """Compare different lot selection methods."""
        from backend.tradingbot.models.models import TaxLot

        qty = Decimal(str(quantity))
        methods = ['fifo', 'lifo', 'hifo']
        results = {}

        # Get current price if sale_price not provided
        if sale_price is None:
            lot = TaxLot.objects.filter(
                user=self.user,
                symbol=symbol.upper(),
                current_price__isnull=False
            ).first()
            sale_price = float(lot.current_price) if lot else 0

        price = Decimal(str(sale_price))

        for method in methods:
            lots = self._select_lots(symbol, qty, method)
            if not lots:
                continue

            total_cost = sum(
                info['quantity'] * info['lot'].cost_basis_per_share
                for info in lots
            )
            total_qty = sum(info['quantity'] for info in lots)
            proceeds = total_qty * price
            gain = proceeds - total_cost

            results[method] = {
                'total_cost_basis': float(total_cost),
                'total_proceeds': float(proceeds),
                'total_gain': float(gain),
                'avg_cost_basis': float(total_cost / total_qty) if total_qty > 0 else 0,
                'lot_count': len(lots),
            }

        return results

    def _estimate_tax_savings(self, lot) -> Dict[str, Any]:
        """Estimate tax savings from harvesting a loss."""
        if lot.unrealized_gain >= 0:
            return {'savings': 0, 'note': 'No loss to harvest'}

        loss = abs(lot.unrealized_gain)

        if lot.is_long_term:
            # Long-term loss can offset long-term gains
            savings = float(loss * LONG_TERM_TAX_RATE)
            rate = float(LONG_TERM_TAX_RATE * 100)
        else:
            # Short-term loss can offset ordinary income
            savings = float(loss * SHORT_TERM_TAX_RATE)
            rate = float(SHORT_TERM_TAX_RATE * 100)

        return {
            'potential_loss': float(loss),
            'estimated_savings': savings,
            'tax_rate_applied': rate,
            'holding_type': 'long_term' if lot.is_long_term else 'short_term',
        }

    def _days_until_wash_safe(self, symbol: str) -> int:
        """Calculate days until it's safe to buy back without wash sale."""
        from backend.tradingbot.models.models import TaxLotSale

        # Find most recent sale
        last_sale = TaxLotSale.objects.filter(
            user=self.user,
            symbol=symbol.upper()
        ).order_by('-sold_at').first()

        if not last_sale:
            return 0  # No recent sales, safe now

        days_since_sale = (timezone.now() - last_sale.sold_at).days

        if days_since_sale >= WASH_SALE_WINDOW_DAYS:
            return 0  # Already past wash sale window

        return WASH_SALE_WINDOW_DAYS - days_since_sale

    def _check_strategy_intentions(self, symbol: str) -> Dict[str, Any]:
        """Check if any strategies might buy this symbol soon."""
        # This would integrate with the strategy system
        # For now, return a placeholder
        return {
            'has_pending': False,
            'strategies': [],
            'note': 'Check strategy settings to ensure symbol is excluded',
        }

    def _find_similar_alternatives(self, symbol: str) -> List[Dict[str, Any]]:
        """Find similar securities to maintain exposure without wash sale."""
        # Mapping of common alternatives
        alternatives_map = {
            'SPY': [
                {'symbol': 'VOO', 'name': 'Vanguard S&P 500 ETF'},
                {'symbol': 'IVV', 'name': 'iShares Core S&P 500 ETF'},
                {'symbol': 'SPLG', 'name': 'SPDR Portfolio S&P 500 ETF'},
            ],
            'QQQ': [
                {'symbol': 'QQQM', 'name': 'Invesco NASDAQ 100 ETF'},
                {'symbol': 'VGT', 'name': 'Vanguard Information Technology ETF'},
            ],
            'IWM': [
                {'symbol': 'VB', 'name': 'Vanguard Small-Cap ETF'},
                {'symbol': 'SCHA', 'name': 'Schwab U.S. Small-Cap ETF'},
            ],
            'VTI': [
                {'symbol': 'ITOT', 'name': 'iShares Core S&P Total US Stock Market'},
                {'symbol': 'SPTM', 'name': 'SPDR Portfolio S&P 1500 ETF'},
            ],
        }

        return alternatives_map.get(symbol.upper(), [])

    def _get_harvest_recommendation(
        self,
        lot,
        wash_risk: Dict[str, Any],
        days_until_safe: int
    ) -> str:
        """Generate a recommendation for harvesting."""
        if wash_risk['risk_level'] == 'high':
            return (
                f"WARNING: Recent purchase detected. Selling now will trigger "
                f"wash sale. Wait {days_until_safe} days or do not repurchase."
            )
        elif wash_risk['risk_level'] == 'medium':
            return (
                "CAUTION: Active strategies may repurchase. Consider excluding "
                "symbol from strategies before harvesting."
            )
        else:
            loss = abs(float(lot.unrealized_gain))
            if lot.is_long_term:
                return (
                    f"OPPORTUNITY: ${loss:.2f} long-term loss available. "
                    f"Can offset long-term gains at 15% rate."
                )
            else:
                return (
                    f"OPPORTUNITY: ${loss:.2f} short-term loss available. "
                    f"Can offset ordinary income at up to 35% rate."
                )

    def _get_wash_sale_recommendation(
        self,
        risk_level: str,
        recent_purchases: List[Dict]
    ) -> str:
        """Generate wash sale avoidance recommendation."""
        if risk_level == 'none':
            return "Safe to sell. No wash sale risk detected."
        elif risk_level == 'high':
            days_ago = recent_purchases[0]['days_ago'] if recent_purchases else 0
            wait_days = 31 - days_ago
            return (
                f"WASH SALE RISK: Purchased {days_ago} days ago. "
                f"Wait {wait_days} more days before selling at a loss, "
                f"or do not repurchase within 30 days after sale."
            )
        else:
            return (
                "POTENTIAL WASH SALE: Automated strategies may trigger wash sale. "
                "Consider disabling strategies for this symbol before selling."
            )


def get_tax_optimizer_service(user: User) -> TaxOptimizer:
    """Factory function to get a TaxOptimizer instance."""
    return TaxOptimizer(user)
