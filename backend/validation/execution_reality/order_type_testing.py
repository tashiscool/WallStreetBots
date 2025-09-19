"""Order type policy testing and optimization.

Tests different order types (market, limit, IOC) to optimize execution
quality across different market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Supported order types for testing."""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    MARKETABLE_LIMIT = "marketable_limit"  # Limit at/through market


@dataclass
class OrderTest:
    """Configuration for order type testing."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_types: List[OrderType]
    market_conditions: Dict[str, float]
    test_duration_seconds: int = 60
    max_orders_per_type: int = 10


@dataclass
class OrderResult:
    """Result of a single order execution."""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    side: str
    quantity: int
    fill_quantity: int
    fill_price: Optional[float]
    market_price_at_order: float
    spread_at_order: float
    latency_ms: float
    filled: bool
    partial_fill: bool
    adverse_selection_bps: float
    order_id: str


class OrderTypeOptimizer:
    """Optimizes order type selection based on market conditions."""

    def __init__(self):
        self.test_results = []
        self.performance_matrix = {}  # (order_type, market_regime) -> performance metrics
        self.current_policies = {}  # symbol -> order_type

    def run_order_type_test(self, test_config: OrderTest) -> Dict[str, Any]:
        """Run comprehensive order type testing.

        Args:
            test_config: Test configuration

        Returns:
            Dictionary with test results and recommendations
        """
        logger.info(f"Starting order type test for {test_config.symbol}")

        test_results = {
            'symbol': test_config.symbol,
            'test_start': datetime.now(),
            'market_conditions': test_config.market_conditions,
            'order_results': [],
            'performance_summary': {},
            'recommendations': {}
        }

        # Run tests for each order type
        for order_type in test_config.order_types:
            logger.info(f"Testing {order_type.value} orders")

            try:
                type_results = self._test_single_order_type(
                    test_config, order_type
                )
                test_results['order_results'].extend(type_results)

            except Exception as e:
                logger.error(f"Failed to test {order_type.value}: {e}")
                continue

        # Analyze results
        test_results['performance_summary'] = self._analyze_performance(
            test_results['order_results']
        )

        test_results['recommendations'] = self._generate_recommendations(
            test_results['performance_summary'],
            test_config.market_conditions
        )

        # Store results for future optimization
        self.test_results.append(test_results)

        return test_results

    def _test_single_order_type(self, test_config: OrderTest,
                               order_type: OrderType) -> List[OrderResult]:
        """Test a single order type with multiple orders."""
        results = []

        for i in range(min(test_config.max_orders_per_type, 10)):
            try:
                # Simulate order placement and execution
                result = self._simulate_order_execution(
                    test_config, order_type, order_sequence=i
                )
                results.append(result)

                # Add delay between orders to avoid market impact
                if i < test_config.max_orders_per_type - 1:
                    delay = test_config.test_duration_seconds / test_config.max_orders_per_type
                    # In real implementation, this would be actual time delay
                    # For simulation, we just adjust the timestamp
                    pass

            except Exception as e:
                logger.error(f"Order {i} failed for {order_type.value}: {e}")
                continue

        return results

    def _simulate_order_execution(self, test_config: OrderTest,
                                 order_type: OrderType,
                                 order_sequence: int) -> OrderResult:
        """Simulate order execution based on order type and market conditions.

        This is a simulation - in production this would interface with actual broker APIs.
        """
        symbol = test_config.symbol
        side = test_config.side
        quantity = test_config.quantity
        market_conditions = test_config.market_conditions

        # Generate realistic market data
        base_price = 100.0  # Placeholder
        spread_bps = market_conditions.get('spread_bps', 5)
        volatility = market_conditions.get('volatility', 0.02)

        # Current bid/ask
        spread_dollars = base_price * spread_bps / 10000
        if side == 'buy':
            market_price = base_price + spread_dollars / 2  # Ask price
            opposite_price = base_price - spread_dollars / 2  # Bid price
        else:
            market_price = base_price - spread_dollars / 2  # Bid price
            opposite_price = base_price + spread_dollars / 2  # Ask price

        # Simulate latency (varies by order type)
        latency_ms = self._simulate_latency(order_type, market_conditions)

        # Simulate execution based on order type
        timestamp = datetime.now() + timedelta(seconds=order_sequence * 10)
        fill_result = self._simulate_fill_behavior(
            order_type, quantity, market_price, market_conditions
        )

        # Calculate adverse selection (price movement during latency)
        adverse_selection_bps = self._calculate_adverse_selection(
            latency_ms, volatility, market_conditions
        )

        return OrderResult(
            timestamp=timestamp,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            fill_quantity=fill_result['fill_quantity'],
            fill_price=fill_result['fill_price'],
            market_price_at_order=market_price,
            spread_at_order=spread_dollars,
            latency_ms=latency_ms,
            filled=fill_result['filled'],
            partial_fill=fill_result['partial_fill'],
            adverse_selection_bps=adverse_selection_bps,
            order_id=f"{symbol}_{order_type.value}_{order_sequence}_{timestamp.strftime('%H%M%S')}"
        )

    def _simulate_latency(self, order_type: OrderType,
                         market_conditions: Dict[str, float]) -> float:
        """Simulate order latency based on order type."""
        base_latency = {
            OrderType.MARKET: 80,
            OrderType.LIMIT: 50,
            OrderType.IOC: 90,
            OrderType.FOK: 95,
            OrderType.MARKETABLE_LIMIT: 85
        }

        latency = base_latency.get(order_type, 100)

        # Add market condition adjustments
        if market_conditions.get('volatility', 0) > 0.03:
            latency *= 1.2  # Higher latency in volatile markets

        if market_conditions.get('volume', 1000) < 500:
            latency *= 1.1  # Higher latency in low volume

        # Add random variation
        latency += np.random.normal(0, 10)

        return max(20, latency)  # Minimum 20ms

    def _simulate_fill_behavior(self, order_type: OrderType, quantity: int,
                               market_price: float,
                               market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Simulate order fill behavior based on order type."""
        volume_available = market_conditions.get('volume', 1000)
        spread_bps = market_conditions.get('spread_bps', 5)

        if order_type == OrderType.MARKET:
            # Market orders usually fill but may have slippage
            fill_probability = 0.98
            if np.random.random() < fill_probability:
                # Calculate slippage
                slippage_bps = spread_bps * 0.3 + np.random.exponential(2)
                slippage_dollars = market_price * slippage_bps / 10000
                fill_price = market_price + slippage_dollars

                # Check for partial fills in large orders
                if quantity > volume_available * 0.1:
                    fill_quantity = min(quantity, int(volume_available * 0.8))
                    partial_fill = fill_quantity < quantity
                else:
                    fill_quantity = quantity
                    partial_fill = False

                return {
                    'filled': True,
                    'fill_quantity': fill_quantity,
                    'fill_price': fill_price,
                    'partial_fill': partial_fill
                }
            else:
                return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

        elif order_type == OrderType.LIMIT:
            # Limit orders have lower fill probability but better price
            fill_probability = 0.7
            if np.random.random() < fill_probability:
                # Limit orders get better execution
                improvement_bps = spread_bps * 0.4
                improvement_dollars = market_price * improvement_bps / 10000
                fill_price = market_price - improvement_dollars

                # Lower chance of partial fills for limit orders
                if quantity > volume_available * 0.2:
                    fill_quantity = min(quantity, int(volume_available * 0.6))
                    partial_fill = fill_quantity < quantity
                else:
                    fill_quantity = quantity
                    partial_fill = False

                return {
                    'filled': True,
                    'fill_quantity': fill_quantity,
                    'fill_price': fill_price,
                    'partial_fill': partial_fill
                }
            else:
                return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

        elif order_type == OrderType.IOC:
            # IOC fills immediately or not at all
            fill_probability = 0.85
            if np.random.random() < fill_probability:
                # IOC gets reasonable execution but may have more slippage than limit
                slippage_bps = spread_bps * 0.2 + np.random.exponential(1)
                slippage_dollars = market_price * slippage_bps / 10000
                fill_price = market_price + slippage_dollars

                # Higher chance of partial fills for IOC
                if quantity > volume_available * 0.15:
                    fill_quantity = min(quantity, int(volume_available * 0.7))
                    partial_fill = fill_quantity < quantity
                else:
                    fill_quantity = quantity
                    partial_fill = False

                return {
                    'filled': True,
                    'fill_quantity': fill_quantity,
                    'fill_price': fill_price,
                    'partial_fill': partial_fill
                }
            else:
                return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

        elif order_type == OrderType.FOK:
            # FOK fills completely or not at all
            fill_probability = 0.75
            can_fill_completely = quantity <= volume_available * 0.5

            if np.random.random() < fill_probability and can_fill_completely:
                slippage_bps = spread_bps * 0.25
                slippage_dollars = market_price * slippage_bps / 10000
                fill_price = market_price + slippage_dollars

                return {
                    'filled': True,
                    'fill_quantity': quantity,  # FOK fills completely or not at all
                    'fill_price': fill_price,
                    'partial_fill': False
                }
            else:
                return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

        elif order_type == OrderType.MARKETABLE_LIMIT:
            # Marketable limit orders behave like aggressive limit orders
            fill_probability = 0.92
            if np.random.random() < fill_probability:
                # Good execution, similar to aggressive limit
                slippage_bps = spread_bps * 0.1
                slippage_dollars = market_price * slippage_bps / 10000
                fill_price = market_price + slippage_dollars

                if quantity > volume_available * 0.12:
                    fill_quantity = min(quantity, int(volume_available * 0.75))
                    partial_fill = fill_quantity < quantity
                else:
                    fill_quantity = quantity
                    partial_fill = False

                return {
                    'filled': True,
                    'fill_quantity': fill_quantity,
                    'fill_price': fill_price,
                    'partial_fill': partial_fill
                }
            else:
                return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

        # Default fallback
        return {'filled': False, 'fill_quantity': 0, 'fill_price': None, 'partial_fill': False}

    def _calculate_adverse_selection(self, latency_ms: float, volatility: float,
                                   market_conditions: Dict[str, float]) -> float:
        """Calculate adverse selection cost due to latency."""
        # Convert latency to fraction of day
        latency_fraction = latency_ms / (1000 * 60 * 60 * 24)

        # Estimate price movement during latency
        price_std = volatility * np.sqrt(latency_fraction)

        # Adverse selection is directional - market moves against us
        adverse_selection = np.random.normal(0, price_std) * 0.5  # 50% of movement is adverse

        return abs(adverse_selection) * 10000  # Convert to basis points

    def _analyze_performance(self, order_results: List[OrderResult]) -> Dict[str, Any]:
        """Analyze performance across different order types."""
        if not order_results:
            return {'error': 'No order results to analyze'}

        # Group results by order type
        results_by_type = {}
        for result in order_results:
            order_type = result.order_type.value
            if order_type not in results_by_type:
                results_by_type[order_type] = []
            results_by_type[order_type].append(result)

        # Calculate metrics for each order type
        performance_summary = {}
        for order_type, results in results_by_type.items():
            if not results:
                continue

            filled_results = [r for r in results if r.filled]

            metrics = {
                'total_orders': len(results),
                'filled_orders': len(filled_results),
                'fill_rate': len(filled_results) / len(results) if results else 0,
                'avg_latency_ms': np.mean([r.latency_ms for r in results]),
                'avg_adverse_selection_bps': np.mean([r.adverse_selection_bps for r in results]),
                'partial_fill_rate': np.mean([r.partial_fill for r in results]),
                'avg_fill_ratio': 0,
                'execution_quality_score': 0
            }

            if filled_results:
                # Calculate slippage for filled orders
                slippages = []
                for result in filled_results:
                    if result.fill_price and result.market_price_at_order:
                        if result.side == 'buy':
                            slippage_bps = (result.fill_price - result.market_price_at_order) / result.market_price_at_order * 10000
                        else:
                            slippage_bps = (result.market_price_at_order - result.fill_price) / result.market_price_at_order * 10000
                        slippages.append(slippage_bps)

                metrics['avg_slippage_bps'] = np.mean(slippages) if slippages else 0
                metrics['slippage_std_bps'] = np.std(slippages) if slippages else 0

                # Average fill ratio for filled orders
                fill_ratios = [r.fill_quantity / r.quantity for r in filled_results if r.quantity > 0]
                metrics['avg_fill_ratio'] = np.mean(fill_ratios) if fill_ratios else 0

                # Calculate execution quality score (0-1, higher is better)
                quality_score = 1.0
                quality_score -= min(metrics['avg_slippage_bps'] / 20, 0.3)  # Penalize slippage
                quality_score -= min(metrics['avg_latency_ms'] / 200, 0.2)  # Penalize latency
                quality_score -= min(metrics['partial_fill_rate'] * 0.5, 0.2)  # Penalize partial fills
                quality_score -= min(metrics['avg_adverse_selection_bps'] / 10, 0.2)  # Penalize adverse selection
                quality_score += (metrics['fill_rate'] - 0.8) * 0.5  # Bonus for high fill rate

                metrics['execution_quality_score'] = max(0, quality_score)

            performance_summary[order_type] = metrics

        return performance_summary

    def _generate_recommendations(self, performance_summary: Dict[str, Any],
                                market_conditions: Dict[str, float]) -> Dict[str, Any]:
        """Generate order type recommendations based on test results."""
        if not performance_summary:
            return {'error': 'No performance data to analyze'}

        recommendations = {
            'primary_recommendation': None,
            'alternative_recommendation': None,
            'market_condition_based': {},
            'reasoning': []
        }

        # Find best performing order type overall
        best_order_type = None
        best_score = -1

        for order_type, metrics in performance_summary.items():
            if 'execution_quality_score' in metrics:
                score = metrics['execution_quality_score']
                if score > best_score:
                    best_score = score
                    best_order_type = order_type

        if best_order_type:
            recommendations['primary_recommendation'] = best_order_type
            recommendations['reasoning'].append(
                f"{best_order_type} achieved highest execution quality score: {best_score:.3f}"
            )

        # Market condition based recommendations
        volatility = market_conditions.get('volatility', 0.02)
        spread_bps = market_conditions.get('spread_bps', 5)
        volume = market_conditions.get('volume', 1000)

        if volatility > 0.03:  # High volatility
            recommendations['market_condition_based']['high_volatility'] = 'ioc'
            recommendations['reasoning'].append(
                "High volatility detected - IOC recommended for immediate execution"
            )

        if spread_bps > 15:  # Wide spreads
            recommendations['market_condition_based']['wide_spreads'] = 'limit'
            recommendations['reasoning'].append(
                "Wide spreads detected - Limit orders recommended to avoid excessive slippage"
            )

        if volume < 500:  # Low volume
            recommendations['market_condition_based']['low_volume'] = 'limit'
            recommendations['reasoning'].append(
                "Low volume detected - Limit orders recommended to reduce market impact"
            )

        # Find alternative recommendation (second best)
        sorted_types = sorted(
            [(ot, metrics.get('execution_quality_score', 0)) for ot, metrics in performance_summary.items()],
            key=lambda x: x[1], reverse=True
        )

        if len(sorted_types) > 1:
            recommendations['alternative_recommendation'] = sorted_types[1][0]

        return recommendations

    def update_policies(self, symbol: str, test_results: Dict[str, Any]):
        """Update order type policies based on test results."""
        if 'recommendations' not in test_results:
            return

        recommendations = test_results['recommendations']
        primary_rec = recommendations.get('primary_recommendation')

        if primary_rec:
            self.current_policies[symbol] = OrderType(primary_rec)
            logger.info(f"Updated order type policy for {symbol}: {primary_rec}")

    def get_recommended_order_type(self, symbol: str,
                                 market_conditions: Dict[str, float]) -> OrderType:
        """Get recommended order type for current market conditions."""
        # Check if we have a current policy for this symbol
        if symbol in self.current_policies:
            base_recommendation = self.current_policies[symbol]
        else:
            base_recommendation = OrderType.LIMIT  # Default to limit orders

        # Override based on current market conditions
        volatility = market_conditions.get('volatility', 0.02)
        spread_bps = market_conditions.get('spread_bps', 5)

        if volatility > 0.04 and spread_bps < 10:
            return OrderType.IOC  # Fast execution in volatile but liquid markets

        if spread_bps > 20:
            return OrderType.LIMIT  # Patient execution in wide spread markets

        return base_recommendation

    def get_performance_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance history for order types."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_tests = [
            test for test in self.test_results
            if test['test_start'] >= cutoff_time
        ]

        if not recent_tests:
            return {'error': 'No recent test data available'}

        # Aggregate performance across all recent tests
        aggregated_performance = {}
        for test in recent_tests:
            if 'performance_summary' in test:
                for order_type, metrics in test['performance_summary'].items():
                    if order_type not in aggregated_performance:
                        aggregated_performance[order_type] = []
                    aggregated_performance[order_type].append(metrics)

        # Calculate averages
        summary = {}
        for order_type, metrics_list in aggregated_performance.items():
            if metrics_list:
                summary[order_type] = {
                    'avg_fill_rate': np.mean([m.get('fill_rate', 0) for m in metrics_list]),
                    'avg_latency_ms': np.mean([m.get('avg_latency_ms', 0) for m in metrics_list]),
                    'avg_slippage_bps': np.mean([m.get('avg_slippage_bps', 0) for m in metrics_list]),
                    'avg_quality_score': np.mean([m.get('execution_quality_score', 0) for m in metrics_list]),
                    'test_count': len(metrics_list)
                }

        return {
            'period_hours': hours,
            'tests_analyzed': len(recent_tests),
            'order_type_performance': summary
        }