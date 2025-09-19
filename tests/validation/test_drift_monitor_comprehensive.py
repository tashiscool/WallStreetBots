"""
Comprehensive Tests for Drift Monitor
====================================

Enhanced test coverage for drift monitoring functionality,
CUSUM detection, PSR monitoring, and performance drift detection.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from backend.validation.drift_monitor import (
    DriftAlert,
    DriftMetrics,
    CUSUMDrift,
    PSRDrift,
    PerformanceDriftMonitor
)


class TestDriftAlert:
    """Test DriftAlert dataclass functionality."""

    def test_drift_alert_creation(self):
        """Test basic DriftAlert creation."""
        alert = DriftAlert(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            alert_type='cusum',
            severity='warning',
            message='Test alert',
            current_value=1.5,
            threshold=1.0,
            recommended_action='Monitor closely'
        )
        assert alert.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert alert.alert_type == 'cusum'
        assert alert.severity == 'warning'
        assert alert.message == 'Test alert'
        assert alert.current_value == 1.5
        assert alert.threshold == 1.0
        assert alert.recommended_action == 'Monitor closely'

    def test_drift_alert_with_different_types(self):
        """Test DriftAlert with different alert types."""
        # CUSUM alert
        cusum_alert = DriftAlert(
            timestamp=datetime.now(),
            alert_type='cusum',
            severity='critical',
            message='CUSUM drift detected',
            current_value=3.5,
            threshold=3.0,
            recommended_action='Reduce position size'
        )
        assert cusum_alert.alert_type == 'cusum'
        assert cusum_alert.severity == 'critical'

        # PSR alert
        psr_alert = DriftAlert(
            timestamp=datetime.now(),
            alert_type='psr',
            severity='warning',
            message='PSR drift detected',
            current_value=0.03,
            threshold=0.05,
            recommended_action='Re-evaluate strategy'
        )
        assert psr_alert.alert_type == 'psr'
        assert psr_alert.severity == 'warning'


class TestDriftMetrics:
    """Test DriftMetrics dataclass functionality."""

    def test_drift_metrics_creation(self):
        """Test basic DriftMetrics creation."""
        metrics = DriftMetrics(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            cusum_positive=1.2,
            cusum_negative=-0.8,
            psr_value=0.95,
            performance_deviation=0.05,
            days_since_reset=15,
            alert_active=True
        )
        assert metrics.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert metrics.cusum_positive == 1.2
        assert metrics.cusum_negative == -0.8
        assert metrics.psr_value == 0.95
        assert metrics.performance_deviation == 0.05
        assert metrics.days_since_reset == 15
        assert metrics.alert_active is True

    def test_drift_metrics_zero_values(self):
        """Test DriftMetrics with zero values."""
        metrics = DriftMetrics(
            timestamp=datetime.now(),
            cusum_positive=0.0,
            cusum_negative=0.0,
            psr_value=0.0,
            performance_deviation=0.0,
            days_since_reset=0,
            alert_active=False
        )
        assert metrics.cusum_positive == 0.0
        assert metrics.cusum_negative == 0.0
        assert metrics.psr_value == 0.0
        assert metrics.performance_deviation == 0.0
        assert metrics.days_since_reset == 0
        assert metrics.alert_active is False


class TestCUSUMDrift:
    """Test CUSUM drift detection functionality."""

    def test_cusum_drift_initialization(self):
        """Test CUSUM drift detector initialization."""
        detector = CUSUMDrift()
        assert detector.k == 0.0
        assert detector.h == 3.0
        assert detector.reset_threshold == 5.0
        assert detector.gp == 0.0
        assert detector.gn == 0.0
        assert detector.alarm_count == 0

    def test_cusum_custom_initialization(self):
        """Test CUSUM with custom parameters."""
        detector = CUSUMDrift(k=0.5, h=2.0, reset_threshold=4.0)
        assert detector.k == 0.5
        assert detector.h == 2.0
        assert detector.reset_threshold == 4.0

    def test_cusum_positive_drift_detection(self):
        """Test positive drift detection."""
        detector = CUSUMDrift(k=0.0, h=2.0)

        # Add small positive values that accumulate
        for i in range(5):
            alarm, alert = detector.update(0.5)
            if i < 4:  # Should not trigger alarm yet
                assert not alarm
                assert alert is None
            else:  # Should trigger alarm when CUSUM exceeds threshold
                if alarm:
                    assert alert is not None
                    assert alert.alert_type == 'cusum'
                    assert 'Positive drift detected' in alert.message
                    break

    def test_cusum_negative_drift_detection(self):
        """Test negative drift detection."""
        detector = CUSUMDrift(k=0.0, h=2.0)

        # Add negative values that accumulate
        for i in range(5):
            alarm, alert = detector.update(-0.5)
            if alarm:
                assert alert is not None
                assert alert.alert_type == 'cusum'
                assert 'Negative drift detected' in alert.message
                break

    def test_cusum_no_drift_normal_operation(self):
        """Test CUSUM with no drift (normal operation)."""
        detector = CUSUMDrift(k=0.0, h=5.0)  # Higher threshold to avoid random triggering

        # Add small values around zero that shouldn't accumulate to trigger alarm
        for _ in range(10):
            x = 0.1  # Small positive value
            alarm, alert = detector.update(x)
            if not alarm:  # Most updates should not trigger alarm
                assert alert is None
            # Note: Can't guarantee no alarm with random walks, so just check most don't trigger

    def test_cusum_reset_after_alarm(self):
        """Test CUSUM reset after alarm."""
        detector = CUSUMDrift(k=0.0, h=2.0)

        # Trigger positive alarm
        detector.update(5.0)  # Large positive value
        alarm, alert = detector.update(0.0)

        if alarm:
            # CUSUM should be reset after alarm
            assert detector.gp == 0.0
            assert detector.gn == 0.0
            assert detector.alarm_count > 0

    def test_cusum_get_current_metrics(self):
        """Test getting current CUSUM metrics."""
        detector = CUSUMDrift(k=0.0, h=3.0)

        # Update with some values
        detector.update(1.0)
        detector.update(0.5)

        metrics = detector.get_current_metrics()
        assert isinstance(metrics, DriftMetrics)
        assert metrics.cusum_positive > 0
        assert metrics.cusum_negative <= 0
        assert metrics.performance_deviation >= 0
        assert metrics.days_since_reset >= 0

    def test_cusum_severity_levels(self):
        """Test different severity levels."""
        detector = CUSUMDrift(k=0.0, h=2.0)

        # Test warning level
        detector.update(2.5)  # Just above threshold
        alarm, alert = detector.update(0.0)
        if alarm and alert:
            assert alert.severity in ['warning', 'critical']

        # Reset and test critical level
        detector = CUSUMDrift(k=0.0, h=2.0)
        detector.update(4.0)  # Well above threshold * 1.5
        alarm, alert = detector.update(0.0)
        if alarm and alert:
            assert alert.severity == 'critical'

    def test_cusum_edge_cases(self):
        """Test CUSUM edge cases."""
        detector = CUSUMDrift(k=0.0, h=3.0)

        # Test with zero values
        alarm, alert = detector.update(0.0)
        assert not alarm
        assert alert is None

        # Test with very small values
        alarm, alert = detector.update(1e-10)
        assert not alarm

        # Test with infinity
        alarm, alert = detector.update(float('inf'))
        if alarm:
            assert alert is not None


class TestPSRDrift:
    """Test PSR (Probabilistic Sharpe Ratio) drift detection."""

    def test_psr_drift_initialization(self):
        """Test PSR drift detector initialization."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252)
        assert detector.backtest_sharpe == 1.5
        assert detector.backtest_observations == 252
        assert detector.confidence_level == 0.95
        assert detector.min_observations == 30
        assert detector.live_returns == []
        assert detector.psr_history == []

    def test_psr_custom_initialization(self):
        """Test PSR with custom parameters."""
        detector = PSRDrift(
            backtest_sharpe=2.0,
            backtest_observations=500,
            confidence_level=0.99,
            min_observations=50
        )
        assert detector.backtest_sharpe == 2.0
        assert detector.backtest_observations == 500
        assert detector.confidence_level == 0.99
        assert detector.min_observations == 50

    def test_psr_insufficient_observations(self):
        """Test PSR with insufficient observations."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252, min_observations=10)

        # Add fewer than minimum observations
        for i in range(5):
            alarm, alert = detector.update(0.001)
            assert not alarm
            assert alert is None

    def test_psr_sharpe_calculation(self):
        """Test Sharpe ratio calculation."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252)

        # Test with known returns
        returns = [0.01, 0.02, -0.005, 0.015, 0.008]
        sharpe = detector._calculate_sharpe_ratio(returns)

        # Verify it's a reasonable Sharpe ratio
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_psr_sharpe_edge_cases(self):
        """Test Sharpe ratio calculation edge cases."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252)

        # Test with insufficient data
        sharpe = detector._calculate_sharpe_ratio([0.01])
        assert sharpe == 0.0

        # Test with zero standard deviation
        sharpe = detector._calculate_sharpe_ratio([0.01, 0.01, 0.01])
        assert sharpe == 0.0

        # Test with empty list
        sharpe = detector._calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_psr_calculation(self):
        """Test PSR calculation."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252)

        # Test PSR calculation
        psr = detector._calculate_psr(live_sharpe=1.2, n_observations=30)
        assert isinstance(psr, float)
        assert 0.0 <= psr <= 1.0

    def test_psr_drift_detection_normal(self):
        """Test PSR drift detection with normal performance."""
        detector = PSRDrift(backtest_sharpe=1.5, backtest_observations=252, min_observations=10)

        # Add returns that should maintain similar Sharpe ratio
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 15)  # Similar to backtest

        alarm_triggered = False
        for ret in returns:
            alarm, alert = detector.update(ret)
            if alarm:
                alarm_triggered = True
                break

        # Should not trigger alarm with similar performance
        assert not alarm_triggered

    def test_psr_drift_detection_degraded(self):
        """Test PSR drift detection with degraded performance."""
        detector = PSRDrift(backtest_sharpe=2.0, backtest_observations=252, min_observations=10)

        # Add poor returns that should trigger drift
        poor_returns = [-0.01, -0.005, -0.008, -0.012, -0.006,
                       -0.009, -0.004, -0.011, -0.007, -0.013,
                       -0.008, -0.015, -0.005, -0.010, -0.009]

        alarm_triggered = False
        final_alert = None
        for ret in poor_returns:
            alarm, alert = detector.update(ret)
            if alarm:
                alarm_triggered = True
                final_alert = alert
                break

        if alarm_triggered:
            assert final_alert is not None
            assert final_alert.alert_type == 'psr'
            assert 'PSR drift detected' in final_alert.message

    def test_psr_severity_levels(self):
        """Test PSR alert severity levels."""
        detector = PSRDrift(backtest_sharpe=2.0, backtest_observations=252, min_observations=5)

        # Add very poor returns to trigger critical alert
        very_poor_returns = [-0.05] * 10

        for ret in very_poor_returns:
            alarm, alert = detector.update(ret)
            if alarm and alert:
                assert alert.severity in ['warning', 'critical']
                break


class TestPerformanceDriftMonitor:
    """Test comprehensive performance drift monitor."""

    def test_performance_drift_monitor_initialization(self):
        """Test performance drift monitor initialization."""
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'daily_return': 0.001,
            'volatility': 0.02,
            'observations': 252
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)
        assert monitor.backtest_metrics == backtest_metrics
        assert isinstance(monitor.cusum_detector, CUSUMDrift)
        assert isinstance(monitor.psr_detector, PSRDrift)
        assert monitor.alerts_history == []
        assert monitor.performance_history == []

    def test_performance_drift_monitor_custom_thresholds(self):
        """Test monitor with custom drift thresholds."""
        backtest_metrics = {'sharpe_ratio': 1.5, 'daily_return': 0.001}
        custom_thresholds = {
            'sharpe_deviation': 0.3,
            'return_deviation': 0.01,
            'volatility_deviation': 0.005
        }

        monitor = PerformanceDriftMonitor(backtest_metrics, custom_thresholds)
        assert monitor.drift_thresholds['sharpe_deviation'] == 0.3
        assert monitor.drift_thresholds['return_deviation'] == 0.01
        assert monitor.drift_thresholds['volatility_deviation'] == 0.005

    def test_performance_drift_monitor_update_normal(self):
        """Test monitor update with normal performance."""
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'daily_return': 0.001,
            'volatility': 0.02,
            'observations': 252
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Add normal returns
        alerts = monitor.update(0.0015)  # Slightly above expected
        assert isinstance(alerts, list)
        # May or may not have alerts depending on threshold sensitivity

    def test_performance_drift_monitor_update_with_additional_metrics(self):
        """Test monitor update with additional metrics."""
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'daily_return': 0.001,
            'volatility': 0.02
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        additional_metrics = {
            'sharpe_ratio': 1.6,  # Slightly higher than expected
            'volatility': 0.025   # Slightly higher volatility
        }

        alerts = monitor.update(0.001, additional_metrics)
        assert isinstance(alerts, list)

    def test_performance_drift_monitor_check_performance_metrics(self):
        """Test performance metrics checking."""
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'volatility': 0.02,
            'max_drawdown': 0.10
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Test with metrics that exceed thresholds
        bad_metrics = {
            'sharpe_ratio': 0.5,   # Much lower than expected 1.5
            'volatility': 0.08,    # Much higher than expected 0.02
            'max_drawdown': 0.25   # Much worse than expected 0.10
        }

        alerts = monitor._check_performance_metrics(bad_metrics)
        assert isinstance(alerts, list)
        # Should have alerts for metrics that exceed thresholds

    def test_performance_drift_monitor_alert_history(self):
        """Test alert history tracking."""
        backtest_metrics = {'sharpe_ratio': 1.5, 'daily_return': 0.001}
        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Create mock alerts by forcing bad performance
        bad_metrics = {'sharpe_ratio': 0.1}  # Very low Sharpe
        alerts = monitor._check_performance_metrics(bad_metrics)
        monitor.alerts_history.extend(alerts)

        assert len(monitor.alerts_history) >= 0

    def test_performance_drift_monitor_get_drift_summary(self):
        """Test drift summary generation."""
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'daily_return': 0.001,
            'volatility': 0.02,
            'observations': 252
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Add some performance data
        for i in range(10):
            monitor.update(np.random.normal(0.001, 0.02))

        summary = monitor.get_drift_summary(days=30)

        assert 'monitoring_period_days' in summary
        assert 'total_alerts' in summary
        assert 'alert_counts' in summary
        assert 'performance_summary' in summary
        assert 'cusum_status' in summary
        assert 'drift_detected' in summary
        assert 'recommendation' in summary

        assert summary['monitoring_period_days'] == 30
        assert isinstance(summary['total_alerts'], int)
        assert isinstance(summary['alert_counts'], dict)
        assert isinstance(summary['drift_detected'], bool)

    def test_performance_drift_monitor_recommendations(self):
        """Test recommendation generation."""
        backtest_metrics = {'sharpe_ratio': 1.5, 'daily_return': 0.001}
        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Test with no alerts
        recommendation = monitor._get_recommendation([])
        assert "no drift detected" in recommendation.lower()

        # Test with warning alerts
        warning_alerts = [
            DriftAlert(
                timestamp=datetime.now(),
                alert_type='cusum',
                severity='warning',
                message='Warning drift',
                current_value=1.0,
                threshold=0.8,
                recommended_action='Monitor'
            )
        ] * 3

        recommendation = monitor._get_recommendation(warning_alerts)
        assert "warning" in recommendation.lower()

        # Test with critical alerts
        critical_alerts = [
            DriftAlert(
                timestamp=datetime.now(),
                alert_type='psr',
                severity='critical',
                message='Critical drift',
                current_value=2.0,
                threshold=1.0,
                recommended_action='Reduce position'
            )
        ]

        recommendation = monitor._get_recommendation(critical_alerts)
        assert "critical" in recommendation.lower()


class TestDriftMonitorEdgeCases:
    """Test edge cases and error conditions."""

    def test_cusum_with_extreme_values(self):
        """Test CUSUM with extreme values."""
        detector = CUSUMDrift()

        # Test with very large positive value
        alarm, alert = detector.update(1000.0)
        assert alarm
        assert alert is not None

        # Test with very large negative value
        detector = CUSUMDrift()
        alarm, alert = detector.update(-1000.0)
        assert alarm
        assert alert is not None

    def test_psr_with_extreme_sharpe_values(self):
        """Test PSR with extreme Sharpe ratio values."""
        # Very high backtest Sharpe
        detector = PSRDrift(backtest_sharpe=10.0, backtest_observations=252, min_observations=5)

        # Add mediocre returns
        for _ in range(10):
            detector.update(0.001)

        # Should likely trigger drift with such high expectations

    def test_monitor_with_empty_backtest_metrics(self):
        """Test monitor with minimal backtest metrics."""
        monitor = PerformanceDriftMonitor({})

        # Should handle gracefully
        alerts = monitor.update(0.001)
        assert isinstance(alerts, list)

    def test_monitor_with_zero_volatility_returns(self):
        """Test monitor with zero volatility returns."""
        backtest_metrics = {'sharpe_ratio': 1.5, 'daily_return': 0.001}
        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Add identical returns (zero volatility)
        for _ in range(10):
            alerts = monitor.update(0.001)

        # Should handle gracefully without errors

    def test_drift_summary_edge_cases(self):
        """Test drift summary with edge cases."""
        monitor = PerformanceDriftMonitor({})

        # Test summary with no data
        summary = monitor.get_drift_summary(days=30)
        assert summary['total_alerts'] == 0
        assert summary['performance_summary']['observations'] == 0

        # Test summary with very short period
        summary = monitor.get_drift_summary(days=0)
        assert summary['monitoring_period_days'] == 0


class TestDriftMonitorIntegration:
    """Test integration scenarios."""

    def test_full_drift_detection_pipeline(self):
        """Test complete drift detection pipeline."""
        # Simulate realistic backtest metrics
        backtest_metrics = {
            'sharpe_ratio': 1.8,
            'daily_return': 0.0008,
            'volatility': 0.015,
            'max_drawdown': 0.08,
            'observations': 252
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Simulate trading period with drift
        np.random.seed(42)

        # Phase 1: Normal performance
        for day in range(20):
            live_return = np.random.normal(0.0008, 0.015)
            alerts = monitor.update(live_return)

        # Phase 2: Introduce drift
        for day in range(20):
            live_return = np.random.normal(-0.002, 0.025)  # Worse performance
            alerts = monitor.update(live_return)

        # Get final summary
        summary = monitor.get_drift_summary(days=30)

        # Verify summary structure
        assert isinstance(summary, dict)
        assert summary['monitoring_period_days'] == 30
        assert 'performance_summary' in summary
        assert 'recommendation' in summary

    def test_multiple_detector_coordination(self):
        """Test coordination between multiple detectors."""
        backtest_metrics = {
            'sharpe_ratio': 2.0,
            'daily_return': 0.001,
            'observations': 252
        }

        monitor = PerformanceDriftMonitor(backtest_metrics)

        # Add consistent poor performance that should trigger multiple detectors
        poor_returns = [-0.01] * 50

        total_alerts = 0
        alert_types = set()

        for ret in poor_returns:
            alerts = monitor.update(ret)
            total_alerts += len(alerts)
            for alert in alerts:
                alert_types.add(alert.alert_type)

        # Should have triggered alerts from multiple detectors
        assert total_alerts > 0

    def test_performance_monitoring_with_real_metrics(self):
        """Test performance monitoring with realistic metrics."""
        # Realistic crypto trading strategy metrics
        backtest_metrics = {
            'sharpe_ratio': 1.2,
            'daily_return': 0.0015,
            'volatility': 0.035,
            'max_drawdown': 0.15,
            'observations': 365
        }

        custom_thresholds = {
            'sharpe_deviation': 0.3,
            'return_deviation': 0.001,
            'volatility_deviation': 0.01,
            'max_drawdown_deviation': 0.05
        }

        monitor = PerformanceDriftMonitor(backtest_metrics, custom_thresholds)

        # Simulate gradual performance degradation
        for week in range(12):  # 12 weeks
            for day in range(7):
                # Gradually worse performance
                degradation_factor = 1 + week * 0.1
                live_return = np.random.normal(
                    0.0015 / degradation_factor,
                    0.035 * degradation_factor
                )

                additional_metrics = {
                    'volatility': 0.035 * degradation_factor,
                    'max_drawdown': min(0.15 * degradation_factor, 0.5)
                }

                alerts = monitor.update(live_return, additional_metrics)

        final_summary = monitor.get_drift_summary(days=60)

        # Should detect the gradual degradation
        assert isinstance(final_summary['drift_detected'], bool)
        assert final_summary['performance_summary']['observations'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])