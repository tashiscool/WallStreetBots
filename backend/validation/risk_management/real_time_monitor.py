"""
Real-Time Risk Monitoring System
================================

Advanced monitoring system with CUSUM drift detection, regime change
detection, performance attribution, and automated response mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import the drift monitor from validation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from execution_reality.drift_monitor import LiveDriftMonitor, DriftAlert, CUSUMDriftDetector
    DRIFT_MONITOR_AVAILABLE = True
except ImportError:
    DRIFT_MONITOR_AVAILABLE = False
    print("Drift monitor not available, implementing simplified version")


class MonitoringMode(Enum):
    """Monitoring mode enumeration."""
    PASSIVE = "PASSIVE"  # Monitor only, no actions
    ACTIVE = "ACTIVE"    # Monitor and alert
    DEFENSIVE = "DEFENSIVE"  # Monitor, alert, and auto-reduce positions
    HALT = "HALT"        # Halt all trading


class RegimeState(Enum):
    """Market regime states."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    VOLATILE = "VOLATILE"
    CALM = "CALM"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: datetime
    strategy: str
    daily_return: float
    cumulative_return: float
    sharpe_ratio: float
    volatility: float
    max_drawdown: float
    win_rate: float
    var_1d: float
    position_count: int
    total_exposure: float


@dataclass
class RegimeDetection:
    """Regime change detection result."""
    timestamp: datetime
    current_regime: RegimeState
    previous_regime: RegimeState
    confidence: float
    regime_duration: timedelta
    change_detected: bool
    volatility_regime: str  # "low", "medium", "high"
    trend_strength: float


@dataclass
class MonitoringAlert:
    """Real-time monitoring alert."""
    timestamp: datetime
    alert_type: str
    severity: str  # INFO, WARNING, CRITICAL
    strategy: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    auto_action_taken: Optional[str]
    requires_manual_intervention: bool


class CUSUMDetector:
    """Enhanced CUSUM detector for drift monitoring."""

    def __init__(self, k: float = 0.5, h: float = 3.0):
        self.k = k  # Allowance parameter
        self.h = h  # Decision threshold
        self.s_pos = 0.0  # Positive CUSUM
        self.s_neg = 0.0  # Negative CUSUM
        self.values = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)

    def update(self, value: float, timestamp: Optional[datetime] = None) -> Tuple[bool, str]:
        """Update CUSUM and return (alarm_triggered, direction)."""

        if timestamp is None:
            timestamp = datetime.now()

        self.values.append(value)
        self.timestamps.append(timestamp)

        # Update CUSUM statistics
        self.s_pos = max(0, self.s_pos + value - self.k)
        self.s_neg = min(0, self.s_neg + value + self.k)

        # Check for alarms
        if self.s_pos > self.h:
            return True, "UPWARD_DRIFT"
        elif self.s_neg < -self.h:
            return True, "DOWNWARD_DRIFT"
        else:
            return False, "NO_DRIFT"

    def reset(self):
        """Reset CUSUM statistics."""
        self.s_pos = 0.0
        self.s_neg = 0.0

    def get_status(self) -> Dict[str, float]:
        """Get current CUSUM status."""
        return {
            's_pos': self.s_pos,
            's_neg': self.s_neg,
            'positive_alarm': self.s_pos > self.h,
            'negative_alarm': self.s_neg < -self.h
        }


class RegimeDetector:
    """Market regime detection using volatility and trend analysis."""

    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.price_history = deque(maxlen=lookback_window * 2)
        self.return_history = deque(maxlen=lookback_window)
        self.current_regime = RegimeState.UNKNOWN
        self.regime_start_time = datetime.now()

    def update(self, price: float, timestamp: Optional[datetime] = None) -> RegimeDetection:
        """Update regime detection with new price data."""

        if timestamp is None:
            timestamp = datetime.now()

        self.price_history.append((timestamp, price))

        # Calculate return if we have previous price
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2][1]
            daily_return = (price - prev_price) / prev_price
            self.return_history.append(daily_return)

        # Need sufficient history for regime detection
        if len(self.return_history) < self.lookback_window:
            return RegimeDetection(
                timestamp=timestamp,
                current_regime=RegimeState.UNKNOWN,
                previous_regime=RegimeState.UNKNOWN,
                confidence=0.0,
                regime_duration=timedelta(),
                change_detected=False,
                volatility_regime="unknown",
                trend_strength=0.0
            )

        # Analyze recent returns
        recent_returns = np.array(list(self.return_history)[-self.lookback_window:])

        # Calculate regime indicators
        volatility = np.std(recent_returns, ddof=1)
        trend = np.mean(recent_returns)

        # Trend strength (consistency of direction)
        positive_returns = np.sum(recent_returns > 0)
        trend_strength = abs(positive_returns / len(recent_returns) - 0.5) * 2

        # Volatility regimes
        if volatility < 0.01:  # < 1% daily vol
            vol_regime = "low"
        elif volatility < 0.025:  # < 2.5% daily vol
            vol_regime = "medium"
        else:
            vol_regime = "high"

        # Determine regime
        previous_regime = self.current_regime

        if volatility > 0.04:  # Very high volatility
            new_regime = RegimeState.CRISIS
        elif volatility > 0.025:
            new_regime = RegimeState.VOLATILE
        elif trend > 0.002 and trend_strength > 0.6:  # Strong uptrend
            new_regime = RegimeState.TRENDING_UP
        elif trend < -0.002 and trend_strength > 0.6:  # Strong downtrend
            new_regime = RegimeState.TRENDING_DOWN
        elif volatility < 0.01:  # Low volatility
            new_regime = RegimeState.CALM
        else:
            new_regime = RegimeState.VOLATILE

        # Check for regime change
        change_detected = new_regime != self.current_regime

        if change_detected:
            self.regime_start_time = timestamp
            self.current_regime = new_regime

        regime_duration = timestamp - self.regime_start_time

        # Calculate confidence (how clear the regime signal is)
        confidence_factors = []

        # Volatility clarity
        vol_boundaries = [0.01, 0.025, 0.04]
        vol_distance = min(abs(volatility - b) for b in vol_boundaries)
        vol_confidence = min(1.0, vol_distance * 100)  # Scale factor
        confidence_factors.append(vol_confidence)

        # Trend clarity
        trend_confidence = trend_strength
        confidence_factors.append(trend_confidence)

        overall_confidence = np.mean(confidence_factors)

        return RegimeDetection(
            timestamp=timestamp,
            current_regime=new_regime,
            previous_regime=previous_regime,
            confidence=overall_confidence,
            regime_duration=regime_duration,
            change_detected=change_detected,
            volatility_regime=vol_regime,
            trend_strength=trend_strength
        )


class PerformanceAttributor:
    """Performance attribution analysis."""

    def __init__(self):
        self.strategy_snapshots: Dict[str, List[PerformanceSnapshot]] = {}

    def add_snapshot(self, snapshot: PerformanceSnapshot):
        """Add performance snapshot for a strategy."""

        if snapshot.strategy not in self.strategy_snapshots:
            self.strategy_snapshots[snapshot.strategy] = []

        self.strategy_snapshots[snapshot.strategy].append(snapshot)

        # Keep only recent snapshots (last 1000)
        if len(self.strategy_snapshots[snapshot.strategy]) > 1000:
            self.strategy_snapshots[snapshot.strategy] = \
                self.strategy_snapshots[snapshot.strategy][-1000:]

    def calculate_attribution(self, period_days: int = 30) -> Dict[str, Any]:
        """Calculate performance attribution over specified period."""

        cutoff_time = datetime.now() - timedelta(days=period_days)
        attribution = {}

        total_portfolio_return = 0.0
        total_portfolio_exposure = 0.0

        for strategy, snapshots in self.strategy_snapshots.items():
            # Filter to period
            period_snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]

            if not period_snapshots:
                continue

            # Calculate strategy contribution
            latest = period_snapshots[-1]
            earliest = period_snapshots[0]

            strategy_return = latest.cumulative_return - earliest.cumulative_return
            avg_exposure = np.mean([s.total_exposure for s in period_snapshots])

            # Weight by average exposure
            weighted_contribution = strategy_return * avg_exposure

            attribution[strategy] = {
                'return': strategy_return,
                'exposure': avg_exposure,
                'contribution': weighted_contribution,
                'sharpe': latest.sharpe_ratio,
                'volatility': latest.volatility,
                'max_drawdown': latest.max_drawdown,
                'win_rate': latest.win_rate
            }

            total_portfolio_return += weighted_contribution
            total_portfolio_exposure += avg_exposure

        # Calculate relative contributions
        if total_portfolio_exposure > 0:
            for strategy in attribution:
                attribution[strategy]['relative_contribution'] = \
                    attribution[strategy]['contribution'] / total_portfolio_return \
                    if total_portfolio_return != 0 else 0

        return {
            'period_days': period_days,
            'total_return': total_portfolio_return,
            'strategy_attribution': attribution,
            'timestamp': datetime.now()
        }


class RealTimeMonitor:
    """
    Real-time monitoring system for Index Baseline strategies.

    Provides:
    - CUSUM drift detection
    - Market regime monitoring
    - Performance attribution
    - Automated risk responses
    """

    def __init__(self, monitoring_mode: MonitoringMode = MonitoringMode.ACTIVE):
        self.logger = logging.getLogger(__name__)
        self.monitoring_mode = monitoring_mode

        # Monitoring components
        self.drift_detectors: Dict[str, CUSUMDetector] = {}
        self.regime_detector = RegimeDetector()
        self.performance_attributor = PerformanceAttributor()

        # Monitoring state
        self.active_alerts: List[MonitoringAlert] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Strategy performance tracking
        self.strategy_performance: Dict[str, deque] = {}
        self.benchmark_performance = deque(maxlen=1000)

        # Alert callbacks
        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []

        # Auto-response configuration
        self.auto_response_enabled = True
        self.emergency_halt_threshold = 0.05  # 5% daily loss triggers halt

    def initialize_strategy_monitoring(self, strategy: str, baseline_performance: Dict[str, float]):
        """Initialize monitoring for a strategy."""

        # Create CUSUM detector
        self.drift_detectors[strategy] = CUSUMDetector(k=0.5, h=3.0)

        # Initialize performance tracking
        self.strategy_performance[strategy] = deque(maxlen=1000)

        self.logger.info(f"Initialized monitoring for strategy: {strategy}")

    def update_strategy_performance(self, strategy: str, performance_data: Dict[str, float]):
        """Update strategy performance data."""

        timestamp = datetime.now()

        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            strategy=strategy,
            daily_return=performance_data.get('daily_return', 0.0),
            cumulative_return=performance_data.get('cumulative_return', 0.0),
            sharpe_ratio=performance_data.get('sharpe_ratio', 0.0),
            volatility=performance_data.get('volatility', 0.0),
            max_drawdown=performance_data.get('max_drawdown', 0.0),
            win_rate=performance_data.get('win_rate', 0.5),
            var_1d=performance_data.get('var_1d', 0.0),
            position_count=performance_data.get('position_count', 0),
            total_exposure=performance_data.get('total_exposure', 0.0)
        )

        # Update performance tracking
        self.strategy_performance[strategy].append(snapshot)
        self.performance_attributor.add_snapshot(snapshot)

        # Update drift detector
        if strategy in self.drift_detectors:
            drift_detected, drift_direction = self.drift_detectors[strategy].update(
                performance_data.get('daily_return', 0.0), timestamp
            )

            if drift_detected:
                self._handle_drift_detection(strategy, drift_direction, performance_data)

        # Check for performance alerts
        self._check_performance_alerts(strategy, performance_data)

    def update_market_data(self, market_price: float, benchmark_return: float):
        """Update market data for regime detection."""

        # Update regime detector
        regime_detection = self.regime_detector.update(market_price)

        if regime_detection.change_detected:
            self._handle_regime_change(regime_detection)

        # Update benchmark performance
        self.benchmark_performance.append({
            'timestamp': datetime.now(),
            'return': benchmark_return,
            'price': market_price
        })

    def _handle_drift_detection(self, strategy: str, drift_direction: str, performance_data: Dict[str, float]):
        """Handle drift detection alert."""

        severity = "CRITICAL" if "DOWNWARD" in drift_direction else "WARNING"

        alert = MonitoringAlert(
            timestamp=datetime.now(),
            alert_type="DRIFT_DETECTION",
            severity=severity,
            strategy=strategy,
            message=f"Performance drift detected in {strategy}: {drift_direction}",
            metric_name="daily_return_drift",
            current_value=performance_data.get('daily_return', 0.0),
            threshold_value=0.0,
            auto_action_taken=None,
            requires_manual_intervention=True
        )

        self._process_alert(alert)

        # Auto-response for downward drift
        if "DOWNWARD" in drift_direction and self.auto_response_enabled:
            if self.monitoring_mode in [MonitoringMode.DEFENSIVE, MonitoringMode.HALT]:
                self._execute_auto_response(strategy, "REDUCE_EXPOSURE", "Drift detected")

    def _handle_regime_change(self, regime_detection: RegimeDetection):
        """Handle market regime change."""

        severity = "WARNING"
        if regime_detection.current_regime == RegimeState.CRISIS:
            severity = "CRITICAL"

        alert = MonitoringAlert(
            timestamp=regime_detection.timestamp,
            alert_type="REGIME_CHANGE",
            severity=severity,
            strategy="ALL",
            message=f"Market regime changed: {regime_detection.previous_regime.value} → {regime_detection.current_regime.value}",
            metric_name="regime_state",
            current_value=regime_detection.confidence,
            threshold_value=0.7,
            auto_action_taken=None,
            requires_manual_intervention=regime_detection.current_regime == RegimeState.CRISIS
        )

        self._process_alert(alert)

        # Auto-response for crisis regime
        if (regime_detection.current_regime == RegimeState.CRISIS and
            self.monitoring_mode == MonitoringMode.DEFENSIVE):

            for strategy in self.strategy_performance.keys():
                self._execute_auto_response(strategy, "HALT_STRATEGY", "Crisis regime detected")

    def _check_performance_alerts(self, strategy: str, performance_data: Dict[str, float]):
        """Check for performance-based alerts."""

        alerts = []

        # Check daily return threshold
        daily_return = performance_data.get('daily_return', 0.0)
        if daily_return < -self.emergency_halt_threshold:
            alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="EMERGENCY_LOSS",
                severity="CRITICAL",
                strategy=strategy,
                message=f"Emergency loss threshold exceeded: {daily_return:.2%}",
                metric_name="daily_return",
                current_value=daily_return,
                threshold_value=-self.emergency_halt_threshold,
                auto_action_taken=None,
                requires_manual_intervention=True
            ))

        # Check drawdown threshold
        max_drawdown = abs(performance_data.get('max_drawdown', 0.0))
        if max_drawdown > 0.25:  # 25% drawdown threshold
            alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="HIGH_DRAWDOWN",
                severity="WARNING",
                strategy=strategy,
                message=f"High drawdown detected: {max_drawdown:.1%}",
                metric_name="max_drawdown",
                current_value=max_drawdown,
                threshold_value=0.25,
                auto_action_taken=None,
                requires_manual_intervention=False
            ))

        # Check Sharpe ratio deterioration
        sharpe_ratio = performance_data.get('sharpe_ratio', 0.0)
        if sharpe_ratio < 0.0:
            alerts.append(MonitoringAlert(
                timestamp=datetime.now(),
                alert_type="NEGATIVE_SHARPE",
                severity="WARNING",
                strategy=strategy,
                message=f"Negative Sharpe ratio: {sharpe_ratio:.2f}",
                metric_name="sharpe_ratio",
                current_value=sharpe_ratio,
                threshold_value=0.0,
                auto_action_taken=None,
                requires_manual_intervention=False
            ))

        # Process all alerts
        for alert in alerts:
            self._process_alert(alert)

    def _process_alert(self, alert: MonitoringAlert):
        """Process and distribute alert."""

        self.active_alerts.append(alert)

        # Remove old alerts (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [a for a in self.active_alerts if a.timestamp >= cutoff_time]

        # Log alert
        self.logger.warning(f"Monitoring Alert: {alert.alert_type} - {alert.message}")

        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

        # Auto-response for critical alerts
        if (alert.severity == "CRITICAL" and
            self.auto_response_enabled and
            self.monitoring_mode in [MonitoringMode.DEFENSIVE, MonitoringMode.HALT]):

            if alert.alert_type == "EMERGENCY_LOSS":
                self._execute_auto_response(alert.strategy, "HALT_STRATEGY", alert.message)

    def _execute_auto_response(self, strategy: str, action: str, reason: str):
        """Execute automated response action."""

        self.logger.warning(f"Executing auto-response: {action} for {strategy} - {reason}")

        # In production, this would interface with the trading system
        # For now, we log the action and update the alert

        action_descriptions = {
            "REDUCE_EXPOSURE": "Reduced position sizes by 50%",
            "HALT_STRATEGY": "Halted all new positions",
            "EMERGENCY_HALT": "Emergency halt of all trading"
        }

        description = action_descriptions.get(action, action)

        # Update the last alert with auto-action
        if self.active_alerts:
            self.active_alerts[-1].auto_action_taken = description

    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""

        # Recent alerts summary
        recent_alerts = [a for a in self.active_alerts
                        if a.timestamp >= datetime.now() - timedelta(hours=1)]

        alert_summary = {
            'total_active': len(self.active_alerts),
            'recent_count': len(recent_alerts),
            'critical_count': len([a for a in self.active_alerts if a.severity == "CRITICAL"]),
            'warning_count': len([a for a in self.active_alerts if a.severity == "WARNING"])
        }

        # Drift detector status
        drift_status = {}
        for strategy, detector in self.drift_detectors.items():
            drift_status[strategy] = detector.get_status()

        # Current regime
        current_regime = self.regime_detector.current_regime.value
        regime_duration = datetime.now() - self.regime_detector.regime_start_time

        # Performance attribution
        attribution = self.performance_attributor.calculate_attribution(30)  # 30-day

        return {
            'timestamp': datetime.now(),
            'monitoring_mode': self.monitoring_mode.value,
            'is_active': self.is_monitoring,
            'alert_summary': alert_summary,
            'drift_status': drift_status,
            'current_regime': current_regime,
            'regime_duration_hours': regime_duration.total_seconds() / 3600,
            'performance_attribution': attribution,
            'strategies_monitored': list(self.strategy_performance.keys()),
            'auto_response_enabled': self.auto_response_enabled
        }

    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""

        status = self.get_monitoring_status()

        report = []
        report.append("=" * 80)
        report.append("REAL-TIME MONITORING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Monitoring Mode: {status['monitoring_mode']}")
        report.append("")

        # Alert Summary
        alert_summary = status['alert_summary']
        report.append("🚨 ALERT SUMMARY:")
        report.append(f"  Active Alerts: {alert_summary['total_active']}")
        report.append(f"  Recent (1h): {alert_summary['recent_count']}")
        report.append(f"  Critical: {alert_summary['critical_count']}")
        report.append(f"  Warnings: {alert_summary['warning_count']}")
        report.append("")

        # Market Regime
        report.append("📊 MARKET REGIME:")
        report.append(f"  Current Regime: {status['current_regime']}")
        report.append(f"  Duration: {status['regime_duration_hours']:.1f} hours")
        report.append("")

        # Drift Detection Status
        report.append("📈 DRIFT DETECTION:")
        for strategy, drift_info in status['drift_status'].items():
            pos_alarm = "🚨" if drift_info['positive_alarm'] else "✅"
            neg_alarm = "🚨" if drift_info['negative_alarm'] else "✅"
            report.append(f"  {strategy}: Pos{pos_alarm} Neg{neg_alarm}")
            report.append(f"    S+: {drift_info['s_pos']:.2f}, S-: {drift_info['s_neg']:.2f}")
        report.append("")

        # Performance Attribution
        attribution = status['performance_attribution']['strategy_attribution']
        if attribution:
            report.append("💰 PERFORMANCE ATTRIBUTION (30-day):")
            for strategy, attr in attribution.items():
                contribution = attr.get('relative_contribution', 0) * 100
                report.append(f"  {strategy}: {contribution:+.1f}% contribution")
                report.append(f"    Return: {attr['return']:+.2%}, Sharpe: {attr['sharpe']:.2f}")

        report.append("=" * 80)

        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    def test_real_time_monitor():
        """Test the real-time monitoring system."""

        print("Testing Real-Time Monitoring System")
        print("=" * 60)

        # Create monitor
        monitor = RealTimeMonitor(MonitoringMode.ACTIVE)

        # Initialize strategies
        strategies = ["wheel_strategy", "spx_credit_spreads", "swing_trading", "leaps_strategy"]

        print("1. Initializing strategy monitoring:")
        for strategy in strategies:
            baseline_perf = {
                'expected_return': 0.001,
                'expected_sharpe': 1.0,
                'expected_volatility': 0.015
            }
            monitor.initialize_strategy_monitoring(strategy, baseline_perf)
            print(f"   Initialized {strategy}")

        print("\n2. Simulating performance updates:")

        # Simulate some performance data
        np.random.seed(42)

        for day in range(10):
            # Update market data
            market_price = 4200 + np.random.normal(0, 20)
            benchmark_return = np.random.normal(0.0005, 0.01)
            monitor.update_market_data(market_price, benchmark_return)

            # Update each strategy
            for i, strategy in enumerate(strategies):
                # Different performance characteristics
                base_return = [0.0008, 0.0006, 0.0004, 0.0007][i]  # Different base returns
                volatility = [0.012, 0.008, 0.018, 0.020][i]       # Different volatilities

                # Add some drift on day 5 for swing_trading
                if strategy == "swing_trading" and day >= 5:
                    base_return -= 0.002  # Introduce negative drift

                daily_return = np.random.normal(base_return, volatility)

                performance_data = {
                    'daily_return': daily_return,
                    'cumulative_return': daily_return * (day + 1),  # Simplified
                    'sharpe_ratio': base_return / volatility * np.sqrt(252),
                    'volatility': volatility,
                    'max_drawdown': min(0, daily_return * -2),  # Simplified
                    'win_rate': 0.55 if daily_return > 0 else 0.45,
                    'var_1d': abs(daily_return) * 1.65,  # Approximate
                    'position_count': np.random.randint(1, 5),
                    'total_exposure': np.random.randint(10000, 50000)
                }

                monitor.update_strategy_performance(strategy, performance_data)

            print(f"   Day {day + 1}: Updated all strategies")

        print("\n3. Monitoring status:")
        status = monitor.get_monitoring_status()

        print(f"   Active alerts: {status['alert_summary']['total_active']}")
        print(f"   Critical alerts: {status['alert_summary']['critical_count']}")
        print(f"   Current regime: {status['current_regime']}")
        print(f"   Strategies monitored: {len(status['strategies_monitored'])}")

        # Check for drift detection
        print("\n4. Drift detection results:")
        for strategy, drift_status in status['drift_status'].items():
            if drift_status['positive_alarm'] or drift_status['negative_alarm']:
                alarm_type = "Positive" if drift_status['positive_alarm'] else "Negative"
                print(f"   🚨 {strategy}: {alarm_type} drift detected!")
            else:
                print(f"   ✅ {strategy}: No drift detected")

        print("\n5. Performance attribution:")
        attribution = status['performance_attribution']['strategy_attribution']
        for strategy, attr in attribution.items():
            contribution = attr.get('relative_contribution', 0) * 100
            print(f"   {strategy}: {contribution:+.1f}% contribution (Return: {attr['return']:+.2%})")

        print("\n6. Monitoring report:")
        report = monitor.generate_monitoring_report()
        print(report[:800] + "..." if len(report) > 800 else report)

        return monitor

    # Run test
    test_monitor = test_real_time_monitor()