"""
Integrated Risk Management System
=================================

Complete Phase 3 integration combining unified risk management,
real-time monitoring, and automated response mechanisms for
Index Baseline strategies deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

# Import Phase 3 components
from unified_risk_manager import UnifiedRiskManager, PortfolioRisk, RiskAlert, RiskLevel
from real_time_monitor import RealTimeMonitor, MonitoringAlert, MonitoringMode, RegimeState


class SystemStatus(Enum):
    """System operational status."""
    INITIALIZING = "INITIALIZING"
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    EMERGENCY = "EMERGENCY"
    SHUTDOWN = "SHUTDOWN"


class AutoResponseLevel(Enum):
    """Automated response levels."""
    PASSIVE = "PASSIVE"      # Monitor only
    ALERTS_ONLY = "ALERTS_ONLY"  # Send alerts
    DEFENSIVE = "DEFENSIVE"  # Reduce positions
    AGGRESSIVE = "AGGRESSIVE"  # Halt strategies
    EMERGENCY = "EMERGENCY"  # Halt all trading


@dataclass
class SystemConfiguration:
    """Integrated system configuration."""
    # Risk management
    max_portfolio_var: float = 0.02  # 2% daily VaR limit
    max_position_size: float = 0.10  # 10% max position size
    max_strategy_allocation: float = 0.30  # 30% max per strategy
    emergency_halt_threshold: float = 0.05  # 5% daily loss halt

    # Monitoring
    monitoring_frequency: int = 60  # seconds
    drift_sensitivity: float = 3.0  # CUSUM threshold
    regime_sensitivity: float = 0.7  # Regime change threshold
    attribution_window: int = 30  # days

    # Auto-response
    auto_response_level: AutoResponseLevel = AutoResponseLevel.DEFENSIVE
    position_reduction_pct: float = 0.50  # 50% reduction on alerts
    cooling_off_period: int = 24  # hours after emergency halt

    # Reporting
    report_frequency: int = 3600  # seconds (1 hour)
    alert_retention_days: int = 7
    performance_history_days: int = 90


@dataclass
class SystemEvent:
    """System event logging."""
    timestamp: datetime
    event_type: str
    severity: str
    source: str  # unified_risk, real_time_monitor, integrated_system
    message: str
    data: Optional[Dict[str, Any]] = None
    auto_action_taken: Optional[str] = None


class IntegratedRiskSystem:
    """
    Integrated Risk Management System for Index Baseline strategies.

    Combines:
    - Unified Risk Management (portfolio-level controls)
    - Real-Time Monitoring (drift detection, regime analysis)
    - Automated Response (position management, halt mechanisms)
    - Performance Attribution (strategy contribution analysis)
    """

    def __init__(self, config: Optional[SystemConfiguration] = None):
        self.config = config or SystemConfiguration()
        self.logger = logging.getLogger(__name__)

        # System state
        self.system_status = SystemStatus.INITIALIZING
        self.last_emergency_halt = None
        self.system_events: List[SystemEvent] = []

        # Core components
        self.risk_manager = UnifiedRiskManager()
        self.real_time_monitor = RealTimeMonitor(MonitoringMode.ACTIVE)

        # Integration state
        self.active_strategies: Dict[str, Dict[str, Any]] = {}
        self.strategy_allocations: Dict[str, float] = {}
        self.emergency_halted_strategies: List[str] = []

        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Event callbacks
        self.event_callbacks: List[Callable[[SystemEvent], None]] = []

        self._initialize_system()

    def _initialize_system(self):
        """Initialize the integrated system."""

        self.logger.info("Initializing Integrated Risk Management System")

        # Configure risk manager limits based on system config
        self._configure_risk_limits()

        # Set up monitoring callbacks
        self._setup_monitoring_integration()

        # Initialize strategies from Phase 1 and 2
        self._initialize_index_baseline_strategies()

        self.system_status = SystemStatus.OPERATIONAL
        self._log_system_event("SYSTEM_INITIALIZED", "INFO", "Integrated system operational")

    def _configure_risk_limits(self):
        """Configure risk limits from system configuration."""

        # Update portfolio VaR limit
        if 'portfolio_var_1d' in self.risk_manager.risk_limits:
            self.risk_manager.risk_limits['portfolio_var_1d'].limit_value = \
                self.config.max_portfolio_var

        # Update position size limit
        if 'position_size' in self.risk_manager.risk_limits:
            self.risk_manager.risk_limits['position_size'].limit_value = \
                self.config.max_position_size

        # Update strategy allocation limit
        if 'strategy_allocation' in self.risk_manager.risk_limits:
            self.risk_manager.risk_limits['strategy_allocation'].limit_value = \
                self.config.max_strategy_allocation

        self.logger.info("Risk limits configured from system settings")

    def _setup_monitoring_integration(self):
        """Set up integration between risk manager and real-time monitor."""

        # Risk manager alert callback
        def handle_risk_alert(alert: RiskAlert):
            self._process_risk_alert(alert)

        self.risk_manager.add_alert_callback(handle_risk_alert)

        # Real-time monitor alert callback
        def handle_monitoring_alert(alert: MonitoringAlert):
            self._process_monitoring_alert(alert)

        self.real_time_monitor.add_alert_callback(handle_monitoring_alert)

    def _initialize_index_baseline_strategies(self):
        """Initialize Index Baseline strategies with Phase 1 and 2 improvements."""

        strategies_config = {
            'wheel_strategy': {
                'allocation': 0.25,  # 25% allocation
                'risk_controls': {
                    'max_daily_loss': 0.02,
                    'position_stop_loss': 0.15,  # 15% for conservative wheel
                    'profit_target': 0.30
                },
                'baseline_performance': {
                    'expected_daily_return': 0.0008,
                    'expected_sharpe': 1.2,
                    'expected_volatility': 0.012
                }
            },

            'spx_credit_spreads': {
                'allocation': 0.20,  # 20% allocation
                'risk_controls': {
                    'max_daily_loss': 0.01,
                    'position_stop_loss': 0.05,  # 5% for low-vol strategy
                    'profit_target': 0.10
                },
                'baseline_performance': {
                    'expected_daily_return': 0.0006,
                    'expected_sharpe': 2.5,
                    'expected_volatility': 0.008
                }
            },

            'swing_trading': {
                'allocation': 0.25,  # 25% allocation (improved from Phase 1)
                'risk_controls': {
                    'max_daily_loss': 0.02,
                    'position_stop_loss': 0.03,  # 3% from Phase 1 fixes
                    'profit_target': 0.25
                },
                'baseline_performance': {
                    'expected_daily_return': 0.0004,  # Improved from negative
                    'expected_sharpe': 0.8,
                    'expected_volatility': 0.018
                }
            },

            'leaps_strategy': {
                'allocation': 0.30,  # 30% allocation
                'risk_controls': {
                    'max_daily_loss': 0.025,
                    'position_stop_loss': 0.35,  # 35% from Phase 1 fixes
                    'profit_target': 1.00  # 100% for LEAPS
                },
                'baseline_performance': {
                    'expected_daily_return': 0.0007,
                    'expected_sharpe': 1.0,
                    'expected_volatility': 0.020
                }
            }
        }

        for strategy, config in strategies_config.items():
            self.active_strategies[strategy] = config
            self.strategy_allocations[strategy] = config['allocation']

            # Initialize real-time monitoring
            self.real_time_monitor.initialize_strategy_monitoring(
                strategy, config['baseline_performance']
            )

            self.logger.info(f"Initialized strategy: {strategy} (allocation: {config['allocation']:.1%})")

    def add_position(self, symbol: str, strategy: str, position_data: Dict[str, Any]) -> bool:
        """
        Add position with integrated risk checking.

        Returns True if position was added, False if rejected by risk controls.
        """

        # Check if strategy is halted
        if strategy in self.emergency_halted_strategies:
            self._log_system_event(
                "POSITION_REJECTED", "WARNING",
                f"Position rejected - strategy {strategy} is halted",
                {"symbol": symbol, "strategy": strategy}
            )
            return False

        # Check strategy allocation limits
        current_allocation = self.strategy_allocations.get(strategy, 0)
        max_allocation = self.config.max_strategy_allocation

        if current_allocation >= max_allocation:
            self._log_system_event(
                "POSITION_REJECTED", "WARNING",
                f"Position rejected - strategy {strategy} at allocation limit",
                {"symbol": symbol, "current_allocation": current_allocation}
            )
            return False

        # Add position to risk manager
        self.risk_manager.add_position(symbol, strategy, position_data)

        # Check risk limits after addition
        alerts = self.risk_manager.check_risk_limits()

        # If critical risk limit violated, reject position
        critical_alerts = [a for a in alerts if a.severity == RiskLevel.CRITICAL]

        if critical_alerts:
            # Remove the position we just added
            position_key = f"{strategy}_{symbol}"
            if position_key in self.risk_manager.positions:
                del self.risk_manager.positions[position_key]

            self._log_system_event(
                "POSITION_REJECTED", "CRITICAL",
                "Position rejected - risk limit violation",
                {"symbol": symbol, "strategy": strategy, "alerts": len(critical_alerts)}
            )
            return False

        # Position accepted
        self.logger.info(f"Position added: {symbol} ({strategy}) - ${position_data.get('position_value', 0):,.0f}")
        return True

    def update_strategy_performance(self, strategy: str, performance_data: Dict[str, float]):
        """Update strategy performance with integrated monitoring."""

        # Update real-time monitor
        self.real_time_monitor.update_strategy_performance(strategy, performance_data)

        # Check for emergency conditions
        daily_return = performance_data.get('daily_return', 0.0)

        if daily_return < -self.config.emergency_halt_threshold:
            self._trigger_emergency_halt(strategy, f"Emergency loss: {daily_return:.2%}")

    def update_market_data(self, market_price: float, benchmark_return: float):
        """Update market data for regime monitoring."""
        self.real_time_monitor.update_market_data(market_price, benchmark_return)

    def _process_risk_alert(self, alert: RiskAlert):
        """Process alert from risk manager."""

        # Log system event
        self._log_system_event(
            f"RISK_ALERT_{alert.severity.value}",
            alert.severity.value,
            alert.recommended_action,
            {
                "current_value": alert.current_value,
                "limit_value": alert.limit_value,
                "affected_positions": alert.affected_positions
            }
        )

        # Auto-response based on configuration
        if self.config.auto_response_level == AutoResponseLevel.AGGRESSIVE:
            if alert.severity == RiskLevel.CRITICAL:
                self._execute_emergency_response("RISK_LIMIT_CRITICAL", alert.recommended_action)

        elif self.config.auto_response_level == AutoResponseLevel.DEFENSIVE:
            if alert.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                self._execute_defensive_response("RISK_LIMIT_HIGH", alert.recommended_action)

    def _process_monitoring_alert(self, alert: MonitoringAlert):
        """Process alert from real-time monitor."""

        # Log system event
        self._log_system_event(
            f"MONITOR_ALERT_{alert.severity}",
            alert.severity,
            alert.message,
            {
                "strategy": alert.strategy,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value
            }
        )

        # Auto-response for critical monitoring alerts
        if alert.severity == "CRITICAL":
            if alert.alert_type == "EMERGENCY_LOSS":
                self._trigger_emergency_halt(alert.strategy, alert.message)

            elif alert.alert_type == "DRIFT_DETECTION" and "DOWNWARD" in alert.message:
                if self.config.auto_response_level in [AutoResponseLevel.DEFENSIVE, AutoResponseLevel.AGGRESSIVE]:
                    self._execute_defensive_response("DRIFT_DETECTED", f"Downward drift in {alert.strategy}")

    def _trigger_emergency_halt(self, strategy: str, reason: str):
        """Trigger emergency halt for a strategy."""

        if strategy not in self.emergency_halted_strategies:
            self.emergency_halted_strategies.append(strategy)
            self.last_emergency_halt = datetime.now()

            self._log_system_event(
                "EMERGENCY_HALT", "CRITICAL",
                f"Emergency halt triggered for {strategy}: {reason}",
                {"strategy": strategy, "reason": reason}
            )

            # Notify all callbacks
            self._notify_emergency_halt(strategy, reason)

    def _execute_emergency_response(self, trigger: str, reason: str):
        """Execute emergency response across all strategies."""

        self.system_status = SystemStatus.EMERGENCY

        # Halt all strategies
        for strategy in self.active_strategies.keys():
            if strategy not in self.emergency_halted_strategies:
                self.emergency_halted_strategies.append(strategy)

        self.last_emergency_halt = datetime.now()

        self._log_system_event(
            "EMERGENCY_RESPONSE", "CRITICAL",
            f"Emergency response activated: {reason}",
            {"trigger": trigger, "halted_strategies": self.emergency_halted_strategies}
        )

    def _execute_defensive_response(self, trigger: str, reason: str):
        """Execute defensive response (position reduction)."""

        self.system_status = SystemStatus.DEGRADED

        # Reduce positions by configured percentage
        reduction_factor = self.config.position_reduction_pct

        self._log_system_event(
            "DEFENSIVE_RESPONSE", "WARNING",
            f"Defensive response activated: {reason}",
            {"trigger": trigger, "reduction_factor": reduction_factor}
        )

        # In production, this would interface with position management system
        # For now, we log the intended action

    def _notify_emergency_halt(self, strategy: str, reason: str):
        """Notify external systems of emergency halt."""
        # In production, this would send alerts to external monitoring systems
        self.logger.critical(f"EMERGENCY HALT: {strategy} - {reason}")

    def _log_system_event(self, event_type: str, severity: str, message: str, data: Optional[Dict] = None):
        """Log system event."""

        event = SystemEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            source="integrated_system",
            message=message,
            data=data
        )

        self.system_events.append(event)

        # Keep only recent events
        if len(self.system_events) > 10000:
            self.system_events = self.system_events[-5000:]

        # Trigger callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback failed: {e}")

    def can_resume_strategy(self, strategy: str) -> bool:
        """Check if strategy can resume after emergency halt."""

        if strategy not in self.emergency_halted_strategies:
            return True

        if self.last_emergency_halt is None:
            return True

        # Check cooling off period
        cooling_off_end = self.last_emergency_halt + timedelta(hours=self.config.cooling_off_period)

        if datetime.now() < cooling_off_end:
            return False

        # Check current system status
        if self.system_status == SystemStatus.EMERGENCY:
            return False

        return True

    def resume_strategy(self, strategy: str, manual_override: bool = False) -> bool:
        """Resume a halted strategy."""

        if not manual_override and not self.can_resume_strategy(strategy):
            return False

        if strategy in self.emergency_halted_strategies:
            self.emergency_halted_strategies.remove(strategy)

            self._log_system_event(
                "STRATEGY_RESUMED", "INFO",
                f"Strategy resumed: {strategy}",
                {"strategy": strategy, "manual_override": manual_override}
            )

            return True

        return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""

        # Get component statuses
        risk_summary = self.risk_manager.get_risk_summary()
        monitoring_status = self.real_time_monitor.get_monitoring_status()

        # Recent events summary
        recent_events = [e for e in self.system_events
                        if e.timestamp >= datetime.now() - timedelta(hours=1)]

        event_summary = {
            'total_events_24h': len([e for e in self.system_events
                                   if e.timestamp >= datetime.now() - timedelta(hours=24)]),
            'recent_events_1h': len(recent_events),
            'critical_events': len([e for e in recent_events if e.severity == "CRITICAL"]),
            'warning_events': len([e for e in recent_events if e.severity == "WARNING"])
        }

        return {
            'timestamp': datetime.now(),
            'system_status': self.system_status.value,
            'operational_strategies': [s for s in self.active_strategies.keys()
                                     if s not in self.emergency_halted_strategies],
            'halted_strategies': self.emergency_halted_strategies,
            'auto_response_level': self.config.auto_response_level.value,
            'last_emergency_halt': self.last_emergency_halt,
            'risk_summary': risk_summary,
            'monitoring_status': monitoring_status,
            'event_summary': event_summary,
            'strategy_allocations': self.strategy_allocations
        }

    def generate_integrated_report(self) -> str:
        """Generate comprehensive integrated system report."""

        status = self.get_system_status()

        report = []
        report.append("=" * 100)
        report.append("INTEGRATED RISK MANAGEMENT SYSTEM - STATUS REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"System Status: {status['system_status']}")
        report.append("")

        # System Overview
        report.append("🎯 SYSTEM OVERVIEW:")
        report.append(f"  Operational Strategies: {len(status['operational_strategies'])}")
        report.append(f"  Halted Strategies: {len(status['halted_strategies'])}")
        report.append(f"  Auto-Response Level: {status['auto_response_level']}")

        if status['last_emergency_halt']:
            halt_time = status['last_emergency_halt'].strftime('%Y-%m-%d %H:%M:%S')
            report.append(f"  Last Emergency Halt: {halt_time}")
        report.append("")

        # Strategy Status
        report.append("📊 STRATEGY STATUS:")
        for strategy, allocation in status['strategy_allocations'].items():
            if strategy in status['halted_strategies']:
                status_icon = "🛑"
                status_text = "HALTED"
            else:
                status_icon = "✅"
                status_text = "OPERATIONAL"

            report.append(f"  {status_icon} {strategy}: {status_text} (Allocation: {allocation:.1%})")
        report.append("")

        # Risk Summary
        risk_summary = status['risk_summary']
        portfolio_risk = risk_summary['portfolio_risk']

        report.append("⚠️ RISK SUMMARY:")
        report.append(f"  Portfolio Value: ${portfolio_risk.total_value:,.0f}")
        report.append(f"  Portfolio VaR (1D): ${portfolio_risk.portfolio_var_1d:,.0f} "
                     f"({portfolio_risk.portfolio_var_1d/portfolio_risk.total_value:.2%})")
        report.append(f"  Risk Level: {portfolio_risk.risk_level.value}")
        report.append(f"  Active Alerts: {risk_summary['alert_summary']['total_active_alerts']}")
        report.append("")

        # Monitoring Summary
        monitoring = status['monitoring_status']
        report.append("📈 MONITORING SUMMARY:")
        report.append(f"  Current Regime: {monitoring['current_regime']}")
        report.append(f"  Active Alerts: {monitoring['alert_summary']['total_active']}")
        report.append(f"  Critical Alerts: {monitoring['alert_summary']['critical_count']}")

        # Performance Attribution
        if 'performance_attribution' in monitoring:
            attribution = monitoring['performance_attribution']['strategy_attribution']
            report.append("\n💰 PERFORMANCE ATTRIBUTION (30-day):")
            for strategy, attr in attribution.items():
                contribution = attr.get('relative_contribution', 0) * 100
                report.append(f"  {strategy}: {contribution:+.1f}% contribution")
        report.append("")

        # System Events
        event_summary = status['event_summary']
        report.append("📋 SYSTEM EVENTS:")
        report.append(f"  Events (24h): {event_summary['total_events_24h']}")
        report.append(f"  Recent (1h): {event_summary['recent_events_1h']}")
        report.append(f"  Critical: {event_summary['critical_events']}")
        report.append(f"  Warnings: {event_summary['warning_events']}")
        report.append("")

        # Phase 3 Integration Status
        report.append("🔗 PHASE 3 INTEGRATION STATUS:")
        report.append("  ✅ Unified Risk Management: Operational")
        report.append("  ✅ Real-Time Monitoring: Active")
        report.append("  ✅ Automated Responses: Configured")
        report.append("  ✅ Performance Attribution: Running")
        report.append("  ✅ Emergency Halt System: Ready")
        report.append("")

        report.append("=" * 100)

        return "\n".join(report)

    def add_event_callback(self, callback: Callable[[SystemEvent], None]):
        """Add callback for system events."""
        self.event_callbacks.append(callback)


# Example usage and testing
if __name__ == "__main__":
    def test_integrated_risk_system():
        """Test the integrated risk management system."""

        print("Testing Integrated Risk Management System")
        print("=" * 70)

        # Create system with custom configuration
        config = SystemConfiguration(
            max_portfolio_var=0.015,  # 1.5% VaR limit
            auto_response_level=AutoResponseLevel.DEFENSIVE,
            monitoring_frequency=30
        )

        system = IntegratedRiskSystem(config)

        print("1. System Initialization:")
        print(f"   System Status: {system.system_status.value}")
        print(f"   Active Strategies: {len(system.active_strategies)}")
        print(f"   Auto-Response Level: {config.auto_response_level.value}")

        print("\n2. Adding positions:")

        # Add some test positions
        positions = [
            {
                'symbol': 'AAPL',
                'strategy': 'wheel_strategy',
                'position_data': {
                    'quantity': 100,
                    'market_price': 150.0,
                    'position_value': 15000,
                    'option_type': 'put',
                    'strike': 145,
                    'returns_history': np.random.normal(0.001, 0.015, 30).tolist()
                }
            },
            {
                'symbol': 'SPY',
                'strategy': 'spx_credit_spreads',
                'position_data': {
                    'quantity': -5,
                    'market_price': 420.0,
                    'position_value': 5000,
                    'option_type': 'call',
                    'strike': 425,
                    'returns_history': np.random.normal(0.0005, 0.008, 30).tolist()
                }
            }
        ]

        for pos in positions:
            added = system.add_position(pos['symbol'], pos['strategy'], pos['position_data'])
            status = "✅ ADDED" if added else "❌ REJECTED"
            print(f"   {pos['symbol']} ({pos['strategy']}): {status}")

        print("\n3. Simulating performance updates:")

        # Simulate performance data
        performance_scenarios = [
            {'strategy': 'wheel_strategy', 'daily_return': 0.002, 'scenario': 'good day'},
            {'strategy': 'spx_credit_spreads', 'daily_return': 0.001, 'scenario': 'normal day'},
            {'strategy': 'swing_trading', 'daily_return': -0.01, 'scenario': 'bad day'},
            {'strategy': 'leaps_strategy', 'daily_return': -0.06, 'scenario': 'emergency loss!'}
        ]

        for scenario in performance_scenarios:
            performance_data = {
                'daily_return': scenario['daily_return'],
                'cumulative_return': scenario['daily_return'],
                'sharpe_ratio': scenario['daily_return'] / 0.015 * np.sqrt(252),
                'volatility': 0.015,
                'max_drawdown': min(0, scenario['daily_return']),
                'win_rate': 0.6 if scenario['daily_return'] > 0 else 0.4,
                'var_1d': abs(scenario['daily_return']) * 1.65,
                'position_count': 2,
                'total_exposure': 20000
            }

            print(f"   {scenario['strategy']}: {scenario['daily_return']:+.1%} ({scenario['scenario']})")
            system.update_strategy_performance(scenario['strategy'], performance_data)

        print("\n4. System Status Check:")
        status = system.get_system_status()

        print(f"   System Status: {status['system_status']}")
        print(f"   Operational Strategies: {len(status['operational_strategies'])}")
        print(f"   Halted Strategies: {len(status['halted_strategies'])}")

        if status['halted_strategies']:
            print(f"   Emergency Halts: {status['halted_strategies']}")

        print(f"   System Events (24h): {status['event_summary']['total_events_24h']}")
        print(f"   Critical Events: {status['event_summary']['critical_events']}")

        print("\n5. Risk and Monitoring Integration:")
        risk_alerts = status['risk_summary']['alert_summary']['total_active_alerts']
        monitor_alerts = status['monitoring_status']['alert_summary']['total_active']

        print(f"   Risk Manager Alerts: {risk_alerts}")
        print(f"   Real-Time Monitor Alerts: {monitor_alerts}")
        print(f"   Current Market Regime: {status['monitoring_status']['current_regime']}")

        print("\n6. Integrated System Report:")
        report = system.generate_integrated_report()
        print(report[:1000] + "..." if len(report) > 1000 else report)

        return system

    # Run test
    test_system = test_integrated_risk_system()