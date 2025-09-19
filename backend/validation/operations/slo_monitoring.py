"""
Production SLO Monitoring & Error Budget Management
Implements service level objectives and error budget tracking for trading systems.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json


class SLOStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


@dataclass
class SLODefinition:
    name: str
    description: str
    target_percentage: float  # 99.9 = 99.9%
    measurement_window_hours: int  # Rolling window
    error_budget_window_hours: int  # Error budget calculation window
    alerting_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 0.8,  # 80% of error budget consumed
        'critical': 0.95  # 95% of error budget consumed
    })
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SLOMetrics:
    timestamp: datetime
    success_count: int
    total_count: int
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    error_details: Dict[str, int] = field(default_factory=dict)


class SLOCalculator:
    """Calculates SLO compliance and error budget consumption."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_availability(self, metrics: List[SLOMetrics],
                             window_hours: int) -> Tuple[float, Dict[str, Any]]:
        """Calculate availability percentage over time window."""
        if not metrics:
            return 0.0, {}

        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return 0.0, {}

        total_requests = sum(m.total_count for m in recent_metrics)
        successful_requests = sum(m.success_count for m in recent_metrics)

        if total_requests == 0:
            return 100.0, {'total_requests': 0}

        availability = (successful_requests / total_requests) * 100

        details = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': total_requests - successful_requests,
            'availability_percentage': availability,
            'measurement_period_hours': window_hours,
            'data_points': len(recent_metrics)
        }

        return availability, details

    def calculate_latency_slo(self, metrics: List[SLOMetrics],
                            percentile: str, threshold_ms: float,
                            window_hours: int) -> Tuple[float, Dict[str, Any]]:
        """Calculate latency SLO compliance."""
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return 0.0, {}

        latency_values = []
        for m in recent_metrics:
            if percentile == 'p50' and m.latency_p50 is not None:
                latency_values.append(m.latency_p50)
            elif percentile == 'p95' and m.latency_p95 is not None:
                latency_values.append(m.latency_p95)
            elif percentile == 'p99' and m.latency_p99 is not None:
                latency_values.append(m.latency_p99)

        if not latency_values:
            return 0.0, {}

        compliant_count = sum(1 for latency in latency_values if latency <= threshold_ms)
        compliance_percentage = (compliant_count / len(latency_values)) * 100

        details = {
            'compliance_percentage': compliance_percentage,
            'threshold_ms': threshold_ms,
            'measurements': len(latency_values),
            'compliant_measurements': compliant_count,
            f'avg_{percentile}_ms': np.mean(latency_values),
            f'max_{percentile}_ms': max(latency_values),
            f'min_{percentile}_ms': min(latency_values)
        }

        return compliance_percentage, details

    def calculate_error_budget(self, slo_def: SLODefinition,
                             current_availability: float,
                             window_hours: int) -> Dict[str, Any]:
        """Calculate error budget consumption."""
        # Error budget = (100 - SLO target) over the measurement window
        error_budget_percentage = 100 - slo_def.target_percentage

        # Current error rate
        current_error_rate = 100 - current_availability

        # Error budget consumption
        if error_budget_percentage > 0:
            budget_consumed_percentage = (current_error_rate / error_budget_percentage) * 100
        else:
            budget_consumed_percentage = 0 if current_error_rate == 0 else 100

        # Time-based calculations
        total_budget_minutes = (error_budget_percentage / 100) * window_hours * 60
        consumed_budget_minutes = (current_error_rate / 100) * window_hours * 60
        remaining_budget_minutes = max(0, total_budget_minutes - consumed_budget_minutes)

        # Burn rate (how fast we're consuming error budget)
        recent_window_hours = min(4, window_hours)  # Look at last 4 hours for burn rate
        burn_rate = self._calculate_burn_rate(current_availability,
                                            slo_def.target_percentage,
                                            recent_window_hours)

        # Time to exhaustion
        if burn_rate > 0 and remaining_budget_minutes > 0:
            hours_to_exhaustion = remaining_budget_minutes / 60 / burn_rate
        else:
            hours_to_exhaustion = float('inf')

        return {
            'error_budget_percentage': error_budget_percentage,
            'budget_consumed_percentage': min(100, budget_consumed_percentage),
            'remaining_budget_percentage': max(0, 100 - budget_consumed_percentage),
            'total_budget_minutes': total_budget_minutes,
            'consumed_budget_minutes': consumed_budget_minutes,
            'remaining_budget_minutes': remaining_budget_minutes,
            'current_burn_rate': burn_rate,
            'hours_to_exhaustion': hours_to_exhaustion,
            'status': self._get_budget_status(budget_consumed_percentage, slo_def)
        }

    def _calculate_burn_rate(self, current_availability: float,
                           target_availability: float, window_hours: int) -> float:
        """Calculate error budget burn rate (multiplier of normal consumption)."""
        normal_error_rate = 100 - target_availability
        current_error_rate = 100 - current_availability

        if normal_error_rate > 0:
            return current_error_rate / normal_error_rate
        return 0

    def _get_budget_status(self, consumed_percentage: float,
                          slo_def: SLODefinition) -> SLOStatus:
        """Determine SLO status based on error budget consumption."""
        if consumed_percentage >= 100:
            return SLOStatus.EXHAUSTED
        elif consumed_percentage >= slo_def.alerting_thresholds['critical'] * 100:
            return SLOStatus.CRITICAL
        elif consumed_percentage >= slo_def.alerting_thresholds['warning'] * 100:
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY


class TradingSLOMonitor:
    """Production SLO monitoring for trading systems."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calculator = SLOCalculator()
        self.metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.slo_definitions: Dict[str, SLODefinition] = {}
        self.alert_callbacks: List[callable] = []
        self._setup_trading_slos()

    def _setup_trading_slos(self):
        """Setup critical trading system SLOs."""

        # Order execution availability
        self.slo_definitions['order_execution'] = SLODefinition(
            name='order_execution',
            description='Order execution success rate',
            target_percentage=99.5,  # 99.5% success rate
            measurement_window_hours=24,
            error_budget_window_hours=168,  # Weekly error budget
            alerting_thresholds={'warning': 0.7, 'critical': 0.9}
        )

        # Market data feed availability
        self.slo_definitions['market_data'] = SLODefinition(
            name='market_data',
            description='Market data feed availability',
            target_percentage=99.9,  # 99.9% uptime
            measurement_window_hours=24,
            error_budget_window_hours=168,
            alerting_thresholds={'warning': 0.8, 'critical': 0.95}
        )

        # Risk engine latency
        self.slo_definitions['risk_engine_latency'] = SLODefinition(
            name='risk_engine_latency',
            description='Risk engine response time P95 < 100ms',
            target_percentage=95.0,  # 95% of requests under 100ms
            measurement_window_hours=4,
            error_budget_window_hours=24,
            alerting_thresholds={'warning': 0.8, 'critical': 0.9}
        )

        # Portfolio reconciliation
        self.slo_definitions['portfolio_reconciliation'] = SLODefinition(
            name='portfolio_reconciliation',
            description='Portfolio reconciliation success rate',
            target_percentage=99.0,  # 99% success rate
            measurement_window_hours=24,
            error_budget_window_hours=168,
            alerting_thresholds={'warning': 0.6, 'critical': 0.8}
        )

        # Strategy execution latency
        self.slo_definitions['strategy_execution'] = SLODefinition(
            name='strategy_execution',
            description='Strategy signal to order latency P99 < 500ms',
            target_percentage=99.0,  # 99% under 500ms
            measurement_window_hours=4,
            error_budget_window_hours=24,
            alerting_thresholds={'warning': 0.7, 'critical': 0.9}
        )

    def record_metrics(self, slo_name: str, metrics: SLOMetrics):
        """Record metrics for SLO calculation."""
        if slo_name not in self.slo_definitions:
            self.logger.warning(f"Unknown SLO: {slo_name}")
            return

        self.metrics_store[slo_name].append(metrics)
        self.logger.debug(f"Recorded metrics for {slo_name}: {metrics}")

    def get_slo_status(self, slo_name: str) -> Dict[str, Any]:
        """Get current SLO status and error budget."""
        if slo_name not in self.slo_definitions:
            raise ValueError(f"Unknown SLO: {slo_name}")

        slo_def = self.slo_definitions[slo_name]
        metrics = list(self.metrics_store[slo_name])

        # Calculate availability SLO
        availability, availability_details = self.calculator.calculate_availability(
            metrics, slo_def.measurement_window_hours
        )

        # Calculate error budget
        error_budget = self.calculator.calculate_error_budget(
            slo_def, availability, slo_def.error_budget_window_hours
        )

        # Check for latency SLOs
        latency_slos = {}
        if 'latency' in slo_name.lower():
            if 'risk_engine' in slo_name:
                threshold_ms = 100
                percentile = 'p95'
            elif 'strategy' in slo_name:
                threshold_ms = 500
                percentile = 'p99'
            else:
                threshold_ms = 200
                percentile = 'p95'

            latency_compliance, latency_details = self.calculator.calculate_latency_slo(
                metrics, percentile, threshold_ms, slo_def.measurement_window_hours
            )
            latency_slos[f'{percentile}_latency'] = {
                'compliance': latency_compliance,
                'details': latency_details
            }

        return {
            'slo_name': slo_name,
            'slo_definition': slo_def,
            'availability': {
                'percentage': availability,
                'details': availability_details
            },
            'error_budget': error_budget,
            'latency_slos': latency_slos,
            'overall_status': error_budget['status'],
            'timestamp': datetime.now()
        }

    def get_all_slo_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all SLOs."""
        return {name: self.get_slo_status(name) for name in self.slo_definitions.keys()}

    def add_alert_callback(self, callback: callable):
        """Add callback for SLO alerts."""
        self.alert_callbacks.append(callback)

    async def monitor_slos(self, check_interval_seconds: int = 60):
        """Continuous SLO monitoring with alerting."""
        self.logger.info("Starting SLO monitoring")

        while True:
            try:
                all_status = self.get_all_slo_status()

                for slo_name, status in all_status.items():
                    slo_status = status['overall_status']
                    error_budget = status['error_budget']

                    # Check for alerts
                    if slo_status in [SLOStatus.WARNING, SLOStatus.CRITICAL, SLOStatus.EXHAUSTED]:
                        alert_data = {
                            'slo_name': slo_name,
                            'status': slo_status,
                            'error_budget_consumed': error_budget['budget_consumed_percentage'],
                            'hours_to_exhaustion': error_budget['hours_to_exhaustion'],
                            'burn_rate': error_budget['current_burn_rate'],
                            'timestamp': datetime.now()
                        }

                        # Send alerts
                        for callback in self.alert_callbacks:
                            try:
                                await callback(alert_data)
                            except Exception as e:
                                self.logger.error(f"Alert callback failed: {e}")

                await asyncio.sleep(check_interval_seconds)

            except Exception as e:
                self.logger.error(f"SLO monitoring error: {e}")
                await asyncio.sleep(check_interval_seconds)


class ErrorBudgetManager:
    """Manages error budget allocation and spending decisions."""

    def __init__(self, slo_monitor: TradingSLOMonitor):
        self.slo_monitor = slo_monitor
        self.logger = logging.getLogger(__name__)
        self.budget_policies: Dict[str, Dict[str, Any]] = {}
        self._setup_budget_policies()

    def _setup_budget_policies(self):
        """Setup error budget spending policies."""
        self.budget_policies = {
            'order_execution': {
                'max_burn_rate': 2.0,  # Max 2x normal error rate
                'deployment_gate_threshold': 0.8,  # Block deploys at 80% budget consumed
                'load_shedding_threshold': 0.9  # Start load shedding at 90%
            },
            'market_data': {
                'max_burn_rate': 1.5,
                'deployment_gate_threshold': 0.9,
                'load_shedding_threshold': 0.95
            },
            'risk_engine_latency': {
                'max_burn_rate': 3.0,  # Can tolerate higher burn for latency
                'deployment_gate_threshold': 0.7,
                'load_shedding_threshold': 0.85
            }
        }

    def should_block_deployment(self, slo_name: str) -> Tuple[bool, str]:
        """Check if deployment should be blocked due to error budget."""
        status = self.slo_monitor.get_slo_status(slo_name)
        error_budget = status['error_budget']

        if slo_name not in self.budget_policies:
            return False, "No policy defined"

        policy = self.budget_policies[slo_name]
        consumed_percentage = error_budget['budget_consumed_percentage'] / 100

        if consumed_percentage >= policy['deployment_gate_threshold']:
            return True, f"Error budget {consumed_percentage:.1%} >= threshold {policy['deployment_gate_threshold']:.1%}"

        burn_rate = error_budget['current_burn_rate']
        if burn_rate > policy['max_burn_rate']:
            return True, f"Burn rate {burn_rate:.1f}x > max {policy['max_burn_rate']}x"

        return False, "Deployment approved"

    def should_enable_load_shedding(self, slo_name: str) -> Tuple[bool, str]:
        """Check if load shedding should be enabled."""
        status = self.slo_monitor.get_slo_status(slo_name)
        error_budget = status['error_budget']

        if slo_name not in self.budget_policies:
            return False, "No policy defined"

        policy = self.budget_policies[slo_name]
        consumed_percentage = error_budget['budget_consumed_percentage'] / 100

        if consumed_percentage >= policy['load_shedding_threshold']:
            return True, f"Error budget {consumed_percentage:.1%} >= threshold {policy['load_shedding_threshold']:.1%}"

        return False, "Load shedding not required"

    def get_budget_recommendations(self) -> Dict[str, Dict[str, Any]]:
        """Get error budget recommendations for all SLOs."""
        recommendations = {}

        for slo_name in self.slo_monitor.slo_definitions.keys():
            status = self.slo_monitor.get_slo_status(slo_name)

            block_deployment, deployment_reason = self.should_block_deployment(slo_name)
            enable_load_shedding, load_shedding_reason = self.should_enable_load_shedding(slo_name)

            recommendations[slo_name] = {
                'current_status': status['overall_status'],
                'error_budget_consumed': status['error_budget']['budget_consumed_percentage'],
                'burn_rate': status['error_budget']['current_burn_rate'],
                'hours_to_exhaustion': status['error_budget']['hours_to_exhaustion'],
                'block_deployment': block_deployment,
                'deployment_reason': deployment_reason,
                'enable_load_shedding': enable_load_shedding,
                'load_shedding_reason': load_shedding_reason,
                'timestamp': datetime.now()
            }

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def example_alert_handler(alert_data):
        print(f"ALERT: {alert_data['slo_name']} - {alert_data['status'].value}")
        print(f"Error budget: {alert_data['error_budget_consumed']:.1f}% consumed")
        print(f"Burn rate: {alert_data['burn_rate']:.1f}x")

    async def simulate_trading_day():
        monitor = TradingSLOMonitor()
        budget_manager = ErrorBudgetManager(monitor)
        monitor.add_alert_callback(example_alert_handler)

        # Simulate some metrics
        now = datetime.now()

        # Good order execution
        monitor.record_metrics('order_execution', SLOMetrics(
            timestamp=now,
            success_count=995,
            total_count=1000,
            latency_p95=50.0
        ))

        # Market data with some issues
        monitor.record_metrics('market_data', SLOMetrics(
            timestamp=now,
            success_count=980,
            total_count=1000
        ))

        # Risk engine latency issues
        monitor.record_metrics('risk_engine_latency', SLOMetrics(
            timestamp=now,
            success_count=900,
            total_count=1000,
            latency_p95=150.0
        ))

        # Get status
        all_status = monitor.get_all_slo_status()
        for slo_name, status in all_status.items():
            print(f"\n{slo_name.upper()}:")
            print(f"  Availability: {status['availability']['percentage']:.2f}%")
            print(f"  Error Budget: {status['error_budget']['budget_consumed_percentage']:.1f}% consumed")
            print(f"  Status: {status['overall_status'].value}")

        # Check budget recommendations
        recommendations = budget_manager.get_budget_recommendations()
        print("\nBUDGET RECOMMENDATIONS:")
        for slo_name, rec in recommendations.items():
            print(f"{slo_name}: Deploy={not rec['block_deployment']}, LoadShed={rec['enable_load_shedding']}")

    asyncio.run(simulate_trading_day())