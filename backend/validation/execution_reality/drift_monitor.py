"""
Live Drift Monitoring with CUSUM
Monitors for drift between live performance and backtest expectations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    timestamp: datetime
    drift_type: str
    severity: str  # 'warning', 'critical'
    drift_magnitude: float
    description: str
    recommended_action: str


class CUSUMDriftDetector:
    """CUSUM (Cumulative Sum) drift detection."""

    def __init__(self, k: float = 0.0, h: float = 3.0, reset_on_alarm: bool = True):
        """
        Initialize CUSUM detector.

        Args:
            k: Allowance parameter (slack value)
            h: Decision interval (threshold for alarm)
            reset_on_alarm: Whether to reset CUSUM after alarm
        """
        self.k = k
        self.h = h
        self.reset_on_alarm = reset_on_alarm
        self.gp = 0.0  # Positive CUSUM
        self.gn = 0.0  # Negative CUSUM
        self.logger = logging.getLogger(__name__)

    def update(self, x: float) -> bool:
        """
        Update CUSUM with new observation.

        Args:
            x: New observation (e.g., daily edge = live_ret - modeled_ret)

        Returns:
            bool: True if alarm triggered
        """
        # Update positive CUSUM (detects upward drift)
        self.gp = max(0.0, self.gp + (x - self.k))

        # Update negative CUSUM (detects downward drift)
        self.gn = min(0.0, self.gn + (x + self.k))

        # Check for alarm
        alarm = (self.gp > self.h) or (self.gn < -self.h)

        if alarm and self.reset_on_alarm:
            self.gp = 0.0
            self.gn = 0.0

        return alarm

    def get_current_state(self) -> Dict[str, float]:
        """Get current CUSUM state."""
        return {
            'gp': self.gp,
            'gn': self.gn,
            'positive_alarm': self.gp > self.h,
            'negative_alarm': self.gn < -self.h
        }

    def reset(self):
        """Reset CUSUM state."""
        self.gp = 0.0
        self.gn = 0.0


class LiveDriftMonitor:
    """
    Monitors drift between live strategy performance and backtest expectations.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # CUSUM detectors for different metrics
        self.detectors = {
            'returns': CUSUMDriftDetector(k=0.0, h=3.0),
            'sharpe': CUSUMDriftDetector(k=0.0, h=2.5),
            'win_rate': CUSUMDriftDetector(k=0.0, h=2.0),
            'drawdown': CUSUMDriftDetector(k=0.0, h=2.0)
        }

        # Store recent observations
        self.live_data = deque(maxlen=1000)
        self.backtest_expectations = {}
        self.alerts: List[DriftAlert] = []

    def set_backtest_expectations(self, expectations: Dict[str, float]):
        """
        Set expected performance metrics from backtest.

        Args:
            expectations: Dict with keys like 'daily_return', 'sharpe_ratio', 'win_rate', etc.
        """
        self.backtest_expectations = expectations.copy()
        self.logger.info(f"Set backtest expectations: {expectations}")

    def update_live_performance(self, live_metrics: Dict[str, float]) -> List[DriftAlert]:
        """
        Update with new live performance metrics and check for drift.

        Args:
            live_metrics: Dict with current live metrics

        Returns:
            List of any new drift alerts
        """
        timestamp = datetime.now()
        self.live_data.append({
            'timestamp': timestamp,
            'metrics': live_metrics.copy()
        })

        new_alerts = []

        try:
            # Check each metric for drift
            if 'daily_return' in live_metrics and 'daily_return' in self.backtest_expectations:
                drift = live_metrics['daily_return'] - self.backtest_expectations['daily_return']
                alarm = self.detectors['returns'].update(drift)

                if alarm:
                    alert = self._create_drift_alert(
                        'returns', drift, timestamp,
                        f"Daily return drift detected: live={live_metrics['daily_return']:.4f}, "
                        f"expected={self.backtest_expectations['daily_return']:.4f}"
                    )
                    new_alerts.append(alert)

            # Sharpe ratio drift (calculate rolling Sharpe if enough data)
            if len(self.live_data) >= 30:
                live_sharpe = self._calculate_rolling_sharpe(30)
                expected_sharpe = self.backtest_expectations.get('sharpe_ratio', 0)

                if live_sharpe is not None:
                    sharpe_drift = live_sharpe - expected_sharpe
                    alarm = self.detectors['sharpe'].update(sharpe_drift)

                    if alarm:
                        alert = self._create_drift_alert(
                            'sharpe', sharpe_drift, timestamp,
                            f"Sharpe ratio drift detected: live={live_sharpe:.2f}, expected={expected_sharpe:.2f}"
                        )
                        new_alerts.append(alert)

            # Win rate drift
            if len(self.live_data) >= 20:
                live_win_rate = self._calculate_rolling_win_rate(20)
                expected_win_rate = self.backtest_expectations.get('win_rate', 0.5)

                if live_win_rate is not None:
                    win_rate_drift = live_win_rate - expected_win_rate
                    alarm = self.detectors['win_rate'].update(win_rate_drift)

                    if alarm:
                        alert = self._create_drift_alert(
                            'win_rate', win_rate_drift, timestamp,
                            f"Win rate drift detected: live={live_win_rate:.2%}, expected={expected_win_rate:.2%}"
                        )
                        new_alerts.append(alert)

            # Add alerts to history
            self.alerts.extend(new_alerts)

            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(days=30)
            self.alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

            return new_alerts

        except Exception as e:
            self.logger.error(f"Drift monitoring update failed: {e}")
            return []

    def _create_drift_alert(self, metric_name: str, drift_magnitude: float,
                           timestamp: datetime, description: str) -> DriftAlert:
        """Create a drift alert."""
        # Determine severity based on magnitude
        abs_drift = abs(drift_magnitude)

        if metric_name == 'returns':
            if abs_drift > 0.002:  # 20 bps daily
                severity = 'critical'
                action = "Halt trading and investigate immediately"
            elif abs_drift > 0.001:  # 10 bps daily
                severity = 'warning'
                action = "Reduce position sizes by 50% and monitor closely"
            else:
                severity = 'info'
                action = "Continue monitoring"
        elif metric_name == 'sharpe':
            if abs_drift > 1.0:
                severity = 'critical'
                action = "Review strategy parameters and risk management"
            elif abs_drift > 0.5:
                severity = 'warning'
                action = "Increase monitoring frequency"
            else:
                severity = 'info'
                action = "Continue monitoring"
        elif metric_name == 'win_rate':
            if abs_drift > 0.2:  # 20 percentage points
                severity = 'critical'
                action = "Check execution quality and market conditions"
            elif abs_drift > 0.1:  # 10 percentage points
                severity = 'warning'
                action = "Review recent trades for execution issues"
            else:
                severity = 'info'
                action = "Continue monitoring"
        else:
            severity = 'warning'
            action = "Investigate metric deviation"

        return DriftAlert(
            timestamp=timestamp,
            drift_type=metric_name,
            severity=severity,
            drift_magnitude=drift_magnitude,
            description=description,
            recommended_action=action
        )

    def _calculate_rolling_sharpe(self, window: int) -> Optional[float]:
        """Calculate rolling Sharpe ratio from recent live data."""
        try:
            if len(self.live_data) < window:
                return None

            recent_returns = []
            for i in range(-window, 0):
                if 'daily_return' in self.live_data[i]['metrics']:
                    recent_returns.append(self.live_data[i]['metrics']['daily_return'])

            if len(recent_returns) < window:
                return None

            returns_array = np.array(recent_returns)
            std = returns_array.std()
            if std < 1e-10:  # Use threshold for near-zero volatility
                return 0.0

            return float(np.sqrt(252) * returns_array.mean() / std)

        except Exception as e:
            self.logger.error(f"Rolling Sharpe calculation failed: {e}")
            return None

    def _calculate_rolling_win_rate(self, window: int) -> Optional[float]:
        """Calculate rolling win rate from recent live data."""
        try:
            if len(self.live_data) < window:
                return None

            recent_returns = []
            for i in range(-window, 0):
                if 'daily_return' in self.live_data[i]['metrics']:
                    recent_returns.append(self.live_data[i]['metrics']['daily_return'])

            if len(recent_returns) < window:
                return None

            wins = sum(1 for r in recent_returns if r > 0)
            return float(wins / len(recent_returns))

        except Exception as e:
            self.logger.error(f"Rolling win rate calculation failed: {e}")
            return None

    def get_drift_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get summary of drift monitoring over recent period."""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

        # Count alerts by type and severity
        alert_counts = {
            'total': len(recent_alerts),
            'critical': len([a for a in recent_alerts if a.severity == 'critical']),
            'warning': len([a for a in recent_alerts if a.severity == 'warning']),
            'by_type': {}
        }

        for alert in recent_alerts:
            if alert.drift_type not in alert_counts['by_type']:
                alert_counts['by_type'][alert.drift_type] = 0
            alert_counts['by_type'][alert.drift_type] += 1

        # Current CUSUM states
        cusum_states = {}
        for name, detector in self.detectors.items():
            cusum_states[name] = detector.get_current_state()

        # Recent performance vs expectations
        performance_summary = {}
        if len(self.live_data) >= 10:
            live_sharpe = self._calculate_rolling_sharpe(min(30, len(self.live_data)))
            live_win_rate = self._calculate_rolling_win_rate(min(20, len(self.live_data)))

            performance_summary = {
                'live_sharpe': live_sharpe,
                'expected_sharpe': self.backtest_expectations.get('sharpe_ratio'),
                'live_win_rate': live_win_rate,
                'expected_win_rate': self.backtest_expectations.get('win_rate'),
                'days_of_data': len(self.live_data)
            }

        return {
            'alert_counts': alert_counts,
            'cusum_states': cusum_states,
            'performance_summary': performance_summary,
            'recent_alerts': recent_alerts[-5:] if recent_alerts else [],  # Last 5 alerts
            'monitoring_status': self._get_monitoring_status(recent_alerts)
        }

    def _get_monitoring_status(self, recent_alerts: List[DriftAlert]) -> str:
        """Determine overall monitoring status."""
        if not recent_alerts:
            return "HEALTHY"

        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        warning_alerts = [a for a in recent_alerts if a.severity == 'warning']

        if critical_alerts:
            return "CRITICAL"
        elif len(warning_alerts) >= 3:
            return "WARNING"
        elif warning_alerts:
            return "CAUTION"
        else:
            return "HEALTHY"

    def should_halt_trading(self) -> Tuple[bool, str]:
        """Check if trading should be halted due to drift."""
        recent_critical = [a for a in self.alerts[-10:] if a.severity == 'critical']

        if len(recent_critical) >= 2:
            return True, f"Multiple critical drift alerts: {[a.drift_type for a in recent_critical]}"

        # Check current CUSUM states
        for name, detector in self.detectors.items():
            state = detector.get_current_state()
            if state['positive_alarm'] and state['gp'] > 5.0:
                return True, f"Severe positive drift in {name}: CUSUM={state['gp']:.2f}"
            if state['negative_alarm'] and state['gn'] < -5.0:
                return True, f"Severe negative drift in {name}: CUSUM={state['gn']:.2f}"

        return False, "No halt conditions met"

    def reset_all_detectors(self):
        """Reset all CUSUM detectors."""
        for detector in self.detectors.values():
            detector.reset()
        self.logger.info("Reset all drift detectors")

    def get_recent_drift_metrics(self, window: int = 30) -> Dict[str, float]:
        """Get recent drift metrics for analysis."""
        if len(self.live_data) < window:
            return {}

        metrics = {}

        # Calculate various drift metrics
        if self.backtest_expectations:
            # Return drift
            recent_returns = [d['metrics'].get('daily_return', 0) for d in list(self.live_data)[-window:]]
            expected_return = self.backtest_expectations.get('daily_return', 0)

            if recent_returns:
                live_mean = np.mean(recent_returns)
                metrics['return_drift'] = live_mean - expected_return
                metrics['return_drift_std'] = np.std([r - expected_return for r in recent_returns])

            # Volatility drift
            if recent_returns and len(recent_returns) > 1:
                live_vol = np.std(recent_returns)
                expected_vol = self.backtest_expectations.get('volatility', live_vol)
                metrics['volatility_drift'] = live_vol - expected_vol

        return metrics


# Example usage and testing
if __name__ == "__main__":
    def test_drift_monitoring():
        print("=== Live Drift Monitoring Demo ===")

        monitor = LiveDriftMonitor()

        # Set backtest expectations
        expectations = {
            'daily_return': 0.0005,  # 5 bps daily
            'sharpe_ratio': 1.2,
            'win_rate': 0.55,
            'volatility': 0.015
        }
        monitor.set_backtest_expectations(expectations)

        print(f"Set expectations: {expectations}")

        # Simulate live trading with gradual drift
        np.random.seed(42)

        print("\n=== Simulating Live Trading ===")
        for day in range(50):
            # Simulate drift starting on day 20
            drift_factor = 0.0
            if day >= 20:
                drift_factor = (day - 20) * 0.0001  # Gradual negative drift

            # Simulate daily performance
            base_return = expectations['daily_return']
            noise = np.random.normal(0, 0.01)
            live_return = base_return - drift_factor + noise

            live_metrics = {
                'daily_return': live_return,
                'timestamp': datetime.now()
            }

            # Update monitor
            alerts = monitor.update_live_performance(live_metrics)

            # Print alerts
            if alerts:
                for alert in alerts:
                    print(f"Day {day+1} ALERT: {alert.drift_type} - {alert.severity}")
                    print(f"  {alert.description}")
                    print(f"  Action: {alert.recommended_action}")

            # Check halt condition
            should_halt, reason = monitor.should_halt_trading()
            if should_halt:
                print(f"Day {day+1} HALT TRADING: {reason}")
                break

        # Get summary
        summary = monitor.get_drift_summary()
        print("\n=== Drift Summary ===")
        print(f"Total alerts: {summary['alert_counts']['total']}")
        print(f"Critical alerts: {summary['alert_counts']['critical']}")
        print(f"Warning alerts: {summary['alert_counts']['warning']}")
        print(f"Monitoring status: {summary['monitoring_status']}")

        # Performance comparison
        perf = summary['performance_summary']
        if perf:
            print("\nPerformance Comparison:")
            if perf.get('live_sharpe') and perf.get('expected_sharpe'):
                print(f"  Sharpe: Live={perf['live_sharpe']:.2f}, Expected={perf['expected_sharpe']:.2f}")
            if perf.get('live_win_rate') and perf.get('expected_win_rate'):
                print(f"  Win Rate: Live={perf['live_win_rate']:.1%}, Expected={perf['expected_win_rate']:.1%}")

        # CUSUM states
        print("\nCUSUM States:")
        for name, state in summary['cusum_states'].items():
            print(f"  {name}: GP={state['gp']:.2f}, GN={state['gn']:.2f}")

    test_drift_monitoring()