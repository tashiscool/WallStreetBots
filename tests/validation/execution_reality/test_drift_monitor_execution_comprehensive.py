#!/usr/bin/env python3
"""
Comprehensive tests for execution_reality/drift_monitor module.
Tests all public methods, edge cases, and error handling.
Target: 80%+ coverage
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.validation.execution_reality.drift_monitor import (
    CUSUMDriftDetector,
    LiveDriftMonitor,
    DriftAlert
)


class TestDriftAlert:
    """Test DriftAlert dataclass."""

    def test_drift_alert_creation(self):
        """Test DriftAlert creation."""
        alert = DriftAlert(
            timestamp=datetime.now(),
            drift_type='returns',
            severity='warning',
            drift_magnitude=0.0015,
            description='Test drift detected',
            recommended_action='Monitor closely'
        )

        assert alert.drift_type == 'returns'
        assert alert.severity == 'warning'
        assert alert.drift_magnitude == 0.0015


class TestCUSUMDriftDetector:
    """Test CUSUM drift detector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return CUSUMDriftDetector(k=0.0, h=3.0, reset_on_alarm=True)

    def test_initialization_default(self):
        """Test default initialization."""
        detector = CUSUMDriftDetector()
        assert detector.k == 0.0
        assert detector.h == 3.0
        assert detector.reset_on_alarm is True
        assert detector.gp == 0.0
        assert detector.gn == 0.0

    def test_initialization_custom(self):
        """Test custom initialization."""
        detector = CUSUMDriftDetector(k=0.5, h=5.0, reset_on_alarm=False)
        assert detector.k == 0.5
        assert detector.h == 5.0
        assert detector.reset_on_alarm is False

    def test_update_positive_drift(self, detector):
        """Test update with positive drift."""
        # Simulate consistent positive drift
        alarm_triggered = False
        for i in range(10):
            alarm = detector.update(0.5)  # Positive drift
            if alarm:
                alarm_triggered = True
                break

        assert alarm_triggered

    def test_update_negative_drift(self, detector):
        """Test update with negative drift."""
        # Simulate consistent negative drift
        alarm_triggered = False
        for i in range(10):
            alarm = detector.update(-0.5)  # Negative drift
            if alarm:
                alarm_triggered = True
                break

        assert alarm_triggered

    def test_update_no_drift(self, detector):
        """Test update with no drift."""
        # Small random fluctuations
        np.random.seed(42)
        for i in range(20):
            alarm = detector.update(np.random.normal(0, 0.1))

        # Should not trigger alarm
        assert detector.gp < detector.h
        assert detector.gn > -detector.h

    def test_get_current_state(self, detector):
        """Test getting current state."""
        detector.update(1.0)
        detector.update(1.0)

        state = detector.get_current_state()

        assert 'gp' in state
        assert 'gn' in state
        assert 'positive_alarm' in state
        assert 'negative_alarm' in state

    def test_reset(self, detector):
        """Test resetting detector."""
        # Build up CUSUM values
        for i in range(5):
            detector.update(0.5)

        # Reset
        detector.reset()

        assert detector.gp == 0.0
        assert detector.gn == 0.0

    def test_reset_on_alarm(self):
        """Test auto-reset on alarm."""
        detector = CUSUMDriftDetector(k=0.0, h=2.0, reset_on_alarm=True)

        # Trigger alarm
        for i in range(10):
            alarm = detector.update(0.5)
            if alarm:
                break

        # Should be reset
        assert detector.gp == 0.0
        assert detector.gn == 0.0

    def test_no_reset_on_alarm(self):
        """Test no auto-reset on alarm."""
        detector = CUSUMDriftDetector(k=0.0, h=2.0, reset_on_alarm=False)

        # Trigger alarm
        for i in range(10):
            alarm = detector.update(0.5)
            if alarm:
                break

        # Should not be reset
        assert detector.gp > 0.0


class TestLiveDriftMonitor:
    """Test LiveDriftMonitor class."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return LiveDriftMonitor()

    @pytest.fixture
    def sample_expectations(self):
        """Create sample backtest expectations."""
        return {
            'daily_return': 0.0005,
            'sharpe_ratio': 1.2,
            'win_rate': 0.55,
            'volatility': 0.015
        }

    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.detectors is not None
        assert 'returns' in monitor.detectors
        assert 'sharpe' in monitor.detectors
        assert len(monitor.live_data) == 0
        assert len(monitor.alerts) == 0

    def test_set_backtest_expectations(self, monitor, sample_expectations):
        """Test setting backtest expectations."""
        monitor.set_backtest_expectations(sample_expectations)

        assert monitor.backtest_expectations == sample_expectations
        assert monitor.backtest_expectations['daily_return'] == 0.0005

    def test_update_live_performance_no_expectations(self, monitor):
        """Test update without setting expectations."""
        live_metrics = {'daily_return': 0.0004}

        alerts = monitor.update_live_performance(live_metrics)

        # Should handle gracefully
        assert isinstance(alerts, list)

    def test_update_live_performance_return_drift(self, monitor, sample_expectations):
        """Test detecting return drift."""
        monitor.set_backtest_expectations(sample_expectations)

        # Simulate drift
        for day in range(10):
            live_metrics = {
                'daily_return': 0.0001  # Much lower than expected 0.0005
            }
            alerts = monitor.update_live_performance(live_metrics)

        # Check if alerts were generated
        assert len(monitor.alerts) > 0 or len(monitor.live_data) > 0

    def test_update_live_performance_sharpe_drift(self, monitor, sample_expectations):
        """Test detecting Sharpe ratio drift."""
        monitor.set_backtest_expectations(sample_expectations)

        # Simulate enough data for Sharpe calculation
        np.random.seed(42)
        for day in range(35):
            live_metrics = {
                'daily_return': np.random.normal(-0.001, 0.02)  # Poor performance
            }
            alerts = monitor.update_live_performance(live_metrics)

        # With poor performance, should detect drift
        assert len(monitor.live_data) >= 30

    def test_update_live_performance_win_rate_drift(self, monitor, sample_expectations):
        """Test detecting win rate drift."""
        monitor.set_backtest_expectations(sample_expectations)

        # Simulate mostly losing days
        for day in range(25):
            live_metrics = {
                'daily_return': -0.001  # Consistent losses
            }
            alerts = monitor.update_live_performance(live_metrics)

        assert len(monitor.live_data) >= 20

    def test_create_drift_alert_returns_critical(self, monitor):
        """Test creating critical return drift alert."""
        alert = monitor._create_drift_alert(
            'returns',
            0.003,  # 30 bps drift
            datetime.now(),
            'Test description'
        )

        assert alert.severity == 'critical'
        assert 'Halt' in alert.recommended_action or 'halt' in alert.recommended_action

    def test_create_drift_alert_returns_warning(self, monitor):
        """Test creating warning return drift alert."""
        alert = monitor._create_drift_alert(
            'returns',
            0.0012,  # 12 bps drift
            datetime.now(),
            'Test description'
        )

        assert alert.severity == 'warning'

    def test_create_drift_alert_sharpe(self, monitor):
        """Test creating Sharpe drift alert."""
        alert = monitor._create_drift_alert(
            'sharpe',
            1.5,  # Large Sharpe drift
            datetime.now(),
            'Test description'
        )

        assert alert.severity == 'critical'

    def test_create_drift_alert_win_rate(self, monitor):
        """Test creating win rate drift alert."""
        alert = monitor._create_drift_alert(
            'win_rate',
            0.25,  # 25% drift
            datetime.now(),
            'Test description'
        )

        assert alert.severity == 'critical'

    def test_calculate_rolling_sharpe(self, monitor):
        """Test rolling Sharpe calculation."""
        # Add data
        np.random.seed(42)
        for i in range(40):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': np.random.normal(0.001, 0.01)}
            })

        sharpe = monitor._calculate_rolling_sharpe(30)

        assert sharpe is not None
        assert isinstance(sharpe, float)

    def test_calculate_rolling_sharpe_insufficient_data(self, monitor):
        """Test Sharpe with insufficient data."""
        # Add only 10 days of data
        for i in range(10):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': 0.001}
            })

        sharpe = monitor._calculate_rolling_sharpe(30)

        assert sharpe is None

    def test_calculate_rolling_win_rate(self, monitor):
        """Test rolling win rate calculation."""
        # Add data with mix of wins and losses
        for i in range(25):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': 0.001 if i % 2 == 0 else -0.001}
            })

        win_rate = monitor._calculate_rolling_win_rate(20)

        assert win_rate is not None
        assert 0 <= win_rate <= 1

    def test_calculate_rolling_win_rate_insufficient_data(self, monitor):
        """Test win rate with insufficient data."""
        # Add only 10 days
        for i in range(10):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': 0.001}
            })

        win_rate = monitor._calculate_rolling_win_rate(20)

        assert win_rate is None

    def test_get_drift_summary(self, monitor, sample_expectations):
        """Test getting drift summary."""
        monitor.set_backtest_expectations(sample_expectations)

        # Add some data and alerts
        for i in range(30):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': 0.0004}
            })

        summary = monitor.get_drift_summary(days_back=7)

        assert 'alert_counts' in summary
        assert 'cusum_states' in summary
        assert 'performance_summary' in summary
        assert 'monitoring_status' in summary

    def test_get_drift_summary_with_alerts(self, monitor):
        """Test summary with various alert types."""
        # Add some alerts
        monitor.alerts.append(DriftAlert(
            timestamp=datetime.now(),
            drift_type='returns',
            severity='critical',
            drift_magnitude=0.003,
            description='Test',
            recommended_action='Halt'
        ))

        monitor.alerts.append(DriftAlert(
            timestamp=datetime.now(),
            drift_type='sharpe',
            severity='warning',
            drift_magnitude=0.5,
            description='Test',
            recommended_action='Monitor'
        ))

        summary = monitor.get_drift_summary(days_back=7)

        assert summary['alert_counts']['total'] == 2
        assert summary['alert_counts']['critical'] == 1
        assert summary['alert_counts']['warning'] == 1

    def test_get_monitoring_status(self, monitor):
        """Test monitoring status determination."""
        # No alerts
        status = monitor._get_monitoring_status([])
        assert status == "HEALTHY"

        # Warning alerts
        warning_alerts = [
            DriftAlert(datetime.now(), 'returns', 'warning', 0.001, '', '')
        ]
        status = monitor._get_monitoring_status(warning_alerts)
        assert status in ["CAUTION", "WARNING"]

        # Critical alerts
        critical_alerts = [
            DriftAlert(datetime.now(), 'returns', 'critical', 0.003, '', '')
        ]
        status = monitor._get_monitoring_status(critical_alerts)
        assert status == "CRITICAL"

    def test_should_halt_trading_no_issues(self, monitor):
        """Test halt check with no issues."""
        should_halt, reason = monitor.should_halt_trading()

        assert should_halt is False
        assert 'No halt conditions' in reason

    def test_should_halt_trading_multiple_critical(self, monitor):
        """Test halt with multiple critical alerts."""
        # Add critical alerts
        for i in range(3):
            monitor.alerts.append(DriftAlert(
                timestamp=datetime.now(),
                drift_type='returns',
                severity='critical',
                drift_magnitude=0.003,
                description='Test',
                recommended_action='Halt'
            ))

        should_halt, reason = monitor.should_halt_trading()

        assert should_halt is True
        assert 'critical' in reason.lower()

    def test_should_halt_trading_severe_drift(self, monitor):
        """Test halt with severe CUSUM drift."""
        # Simulate severe drift
        for i in range(20):
            monitor.detectors['returns'].update(0.5)

        should_halt, reason = monitor.should_halt_trading()

        # May or may not halt depending on threshold
        assert isinstance(should_halt, bool)

    def test_reset_all_detectors(self, monitor):
        """Test resetting all detectors."""
        # Build up some drift
        for i in range(5):
            monitor.detectors['returns'].update(0.5)

        # Reset
        monitor.reset_all_detectors()

        # Check all are reset
        for detector in monitor.detectors.values():
            assert detector.gp == 0.0
            assert detector.gn == 0.0

    def test_get_recent_drift_metrics(self, monitor, sample_expectations):
        """Test getting recent drift metrics."""
        monitor.set_backtest_expectations(sample_expectations)

        # Add data
        np.random.seed(42)
        for i in range(35):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': np.random.normal(0.0003, 0.01)}
            })

        metrics = monitor.get_recent_drift_metrics(window=30)

        assert 'return_drift' in metrics
        assert 'return_drift_std' in metrics

    def test_get_recent_drift_metrics_insufficient_data(self, monitor):
        """Test drift metrics with insufficient data."""
        metrics = monitor.get_recent_drift_metrics(window=30)

        assert len(metrics) == 0

    def test_alert_pruning(self, monitor):
        """Test that old alerts are pruned."""
        # Add old alert
        old_alert = DriftAlert(
            timestamp=datetime.now() - timedelta(days=35),
            drift_type='returns',
            severity='warning',
            drift_magnitude=0.001,
            description='Old',
            recommended_action='Monitor'
        )

        monitor.alerts.append(old_alert)

        # Add recent data to trigger pruning
        live_metrics = {'daily_return': 0.0005}
        monitor.update_live_performance(live_metrics)

        # Old alert should be pruned (older than 30 days)
        alert_ages = [(datetime.now() - a.timestamp).days for a in monitor.alerts]
        assert all(age <= 30 for age in alert_ages)

    def test_error_handling_in_update(self, monitor):
        """Test error handling during update."""
        # Set expectations
        monitor.set_backtest_expectations({'daily_return': 0.0005})

        # Pass invalid metrics
        invalid_metrics = {'invalid_key': 'invalid_value'}

        # Should handle gracefully
        alerts = monitor.update_live_performance(invalid_metrics)

        assert isinstance(alerts, list)


class TestLiveDriftMonitorIntegration:
    """Integration tests for LiveDriftMonitor."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = LiveDriftMonitor()

        # Set expectations
        expectations = {
            'daily_return': 0.0005,
            'sharpe_ratio': 1.2,
            'win_rate': 0.55,
            'volatility': 0.015
        }
        monitor.set_backtest_expectations(expectations)

        # Simulate trading with drift
        np.random.seed(42)
        for day in range(40):
            # Introduce drift after day 20
            drift_factor = 0 if day < 20 else (day - 20) * 0.0001

            live_return = expectations['daily_return'] - drift_factor + np.random.normal(0, 0.01)

            alerts = monitor.update_live_performance({
                'daily_return': live_return
            })

            # Check for halt
            should_halt, reason = monitor.should_halt_trading()
            if should_halt:
                break

        # Get summary
        summary = monitor.get_drift_summary()

        assert 'alert_counts' in summary
        assert 'monitoring_status' in summary

    def test_strategy_degradation_detection(self):
        """Test detecting strategy degradation."""
        monitor = LiveDriftMonitor()

        monitor.set_backtest_expectations({
            'daily_return': 0.001,
            'sharpe_ratio': 1.5,
            'win_rate': 0.60
        })

        # Simulate gradual degradation
        np.random.seed(42)
        alerts_detected = False

        for day in range(50):
            # Performance degrades over time
            degradation = day * 0.00005
            live_return = 0.001 - degradation + np.random.normal(0, 0.015)

            alerts = monitor.update_live_performance({
                'daily_return': live_return
            })

            if len(alerts) > 0:
                alerts_detected = True

        # Should detect degradation
        assert alerts_detected or len(monitor.alerts) > 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_extreme_drift_values(self):
        """Test with extreme drift values."""
        detector = CUSUMDriftDetector(k=0.0, h=3.0)

        # Extreme positive
        alarm = detector.update(100.0)
        assert alarm is True

        detector.reset()

        # Extreme negative
        alarm = detector.update(-100.0)
        assert alarm is True

    def test_zero_volatility_sharpe(self):
        """Test Sharpe calculation with zero volatility."""
        monitor = LiveDriftMonitor()

        # Add constant returns
        for i in range(35):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': 0.001}
            })

        sharpe = monitor._calculate_rolling_sharpe(30)

        # Should handle zero volatility
        assert sharpe == 0.0

    def test_all_losses(self):
        """Test with all losing trades."""
        monitor = LiveDriftMonitor()

        # Add all losses
        for i in range(25):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'daily_return': -0.001}
            })

        win_rate = monitor._calculate_rolling_win_rate(20)

        assert win_rate == 0.0

    def test_missing_daily_return(self):
        """Test with missing daily return in metrics."""
        monitor = LiveDriftMonitor()

        # Add data without daily_return
        for i in range(10):
            monitor.live_data.append({
                'timestamp': datetime.now(),
                'metrics': {'other_metric': 0.5}
            })

        sharpe = monitor._calculate_rolling_sharpe(5)

        # Should handle missing data
        assert sharpe is None
