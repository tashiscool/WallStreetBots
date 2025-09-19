"""
Live Drift Detection and Monitoring
Implements CUSUM and PSR (Probabilistic Sharpe Ratio) for detecting
when live performance deviates from backtest expectations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Drift detection alert."""
    timestamp: datetime
    alert_type: str  # 'cusum', 'psr', 'performance'
    severity: str    # 'warning', 'critical'
    message: str
    current_value: float
    threshold: float
    recommended_action: str


@dataclass
class DriftMetrics:
    """Current drift metrics."""
    timestamp: datetime
    cusum_positive: float
    cusum_negative: float
    psr_value: float
    performance_deviation: float
    days_since_reset: int
    alert_active: bool


class CUSUMDrift:
    """
    CUSUM (Cumulative Sum) drift detector for monitoring
    when live performance deviates from expected.
    """
    
    def __init__(self, k: float = 0.0, h: float = 3.0, reset_threshold: float = 5.0):
        """
        Initialize CUSUM detector.
        
        Args:
            k: Reference value (typically 0 for zero-mean)
            h: Decision threshold
            reset_threshold: Threshold for resetting after alarm
        """
        self.k = k
        self.h = h
        self.reset_threshold = reset_threshold
        self.gp = 0.0  # Positive CUSUM
        self.gn = 0.0  # Negative CUSUM
        self.last_reset = datetime.now()
        self.alarm_count = 0
        
    def update(self, x: float) -> Tuple[bool, DriftAlert]:
        """
        Update CUSUM with new observation.
        
        Args:
            x: New observation (e.g., daily edge: live_ret - modeled_ret)
            
        Returns:
            Tuple of (alarm_triggered, alert_object)
        """
        # Update CUSUM statistics
        self.gp = max(0.0, self.gp + (x - self.k))
        self.gn = min(0.0, self.gn + (x + self.k))
        
        # Check for alarm
        alarm = (self.gp > self.h) or (self.gn < -self.h)
        
        alert = None
        if alarm:
            self.alarm_count += 1
            
            # Determine severity and message
            if abs(self.gp) > abs(self.gn):
                severity = 'critical' if self.gp > self.h * 1.5 else 'warning'
                message = f"Positive drift detected: CUSUM+ = {self.gp:.3f} > {self.h}"
                recommended_action = "Reduce position size by 50% and investigate"
            else:
                severity = 'critical' if self.gn < -self.h * 1.5 else 'warning'
                message = f"Negative drift detected: CUSUM- = {self.gn:.3f} < {-self.h}"
                recommended_action = "Reduce position size by 50% and investigate"
            
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type='cusum',
                severity=severity,
                message=message,
                current_value=max(abs(self.gp), abs(self.gn)),
                threshold=self.h,
                recommended_action=recommended_action
            )
            
            # Reset CUSUM after alarm
            self.gp = self.gn = 0.0
            self.last_reset = datetime.now()
            
        return alarm, alert
    
    def get_current_metrics(self) -> DriftMetrics:
        """Get current CUSUM metrics."""
        days_since_reset = (datetime.now() - self.last_reset).days
        
        return DriftMetrics(
            timestamp=datetime.now(),
            cusum_positive=self.gp,
            cusum_negative=self.gn,
            psr_value=0.0,  # Not applicable for CUSUM
            performance_deviation=max(abs(self.gp), abs(self.gn)),
            days_since_reset=days_since_reset,
            alert_active=self.gp > self.h * 0.8 or self.gn < -self.h * 0.8
        )


class PSRDrift:
    """
    Probabilistic Sharpe Ratio (PSR) drift detector.
    Monitors when live Sharpe ratio deviates significantly from backtest.
    """
    
    def __init__(self, backtest_sharpe: float, backtest_observations: int,
                 confidence_level: float = 0.95, min_observations: int = 30):
        """
        Initialize PSR detector.
        
        Args:
            backtest_sharpe: Expected Sharpe ratio from backtest
            backtest_observations: Number of observations in backtest
            confidence_level: Confidence level for PSR calculation
            min_observations: Minimum observations before PSR calculation
        """
        self.backtest_sharpe = backtest_sharpe
        self.backtest_observations = backtest_observations
        self.confidence_level = confidence_level
        self.min_observations = min_observations
        
        self.live_returns = []
        self.psr_history = []
        
    def update(self, live_return: float) -> Tuple[bool, Optional[DriftAlert]]:
        """
        Update PSR with new live return.
        
        Args:
            live_return: New live return observation
            
        Returns:
            Tuple of (alarm_triggered, alert_object)
        """
        self.live_returns.append(live_return)
        
        if len(self.live_returns) < self.min_observations:
            return False, None
            
        # Calculate live Sharpe ratio
        live_sharpe = self._calculate_sharpe_ratio(self.live_returns)
        
        # Calculate PSR
        psr = self._calculate_psr(live_sharpe, len(self.live_returns))
        self.psr_history.append(psr)
        
        # Check for significant deviation
        threshold = 1 - self.confidence_level
        alarm = psr < threshold
        
        alert = None
        if alarm:
            severity = 'critical' if psr < threshold * 0.5 else 'warning'
            message = f"PSR drift detected: PSR = {psr:.3f} < {threshold:.3f}"
            recommended_action = "Reduce position size and re-evaluate strategy"
            
            alert = DriftAlert(
                timestamp=datetime.now(),
                alert_type='psr',
                severity=severity,
                message=message,
                current_value=psr,
                threshold=threshold,
                recommended_action=recommended_action
            )
            
        return alarm, alert
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0:
            return 0.0
            
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_psr(self, live_sharpe: float, n_observations: int) -> float:
        """
        Calculate Probabilistic Sharpe Ratio.
        
        PSR = Prob(SR_live > SR_backtest | observed data)
        """
        # Estimate moments of Sharpe ratio distribution
        # Using approximation from Bailey & López de Prado
        
        # Skewness and kurtosis estimates (simplified)
        skew = 0.0  # Assume normal for simplicity
        kurt = 3.0  # Normal kurtosis
        
        # Calculate PSR using normal approximation
        z_score = (live_sharpe - self.backtest_sharpe) / np.sqrt(1 + skew * live_sharpe + (kurt - 1) / 4 * live_sharpe**2)
        
        # Convert to probability
        psr = stats.norm.cdf(z_score)
        
        return psr


class PerformanceDriftMonitor:
    """
    Comprehensive performance drift monitoring system.
    Combines multiple drift detection methods.
    """
    
    def __init__(self, backtest_metrics: Dict[str, float], 
                 drift_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize performance drift monitor.
        
        Args:
            backtest_metrics: Expected metrics from backtest
            drift_thresholds: Custom drift thresholds
        """
        self.backtest_metrics = backtest_metrics
        self.drift_thresholds = drift_thresholds or {
            'sharpe_deviation': 0.5,
            'return_deviation': 0.02,
            'volatility_deviation': 0.01,
            'max_drawdown_deviation': 0.05
        }
        
        # Initialize detectors
        self.cusum_detector = CUSUMDrift()
        self.psr_detector = PSRDrift(
            backtest_sharpe=backtest_metrics.get('sharpe_ratio', 1.0),
            backtest_observations=backtest_metrics.get('observations', 252)
        )
        
        self.alerts_history = []
        self.performance_history = []
        
    def update(self, live_return: float, additional_metrics: Optional[Dict[str, float]] = None) -> List[DriftAlert]:
        """
        Update all drift detectors with new live data.
        
        Args:
            live_return: New live return
            additional_metrics: Additional performance metrics
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        
        # Update CUSUM (using return deviation from expected)
        expected_return = self.backtest_metrics.get('daily_return', 0.0)
        return_deviation = live_return - expected_return
        
        cusum_alarm, cusum_alert = self.cusum_detector.update(return_deviation)
        if cusum_alarm and cusum_alert:
            alerts.append(cusum_alert)
            
        # Update PSR
        psr_alarm, psr_alert = self.psr_detector.update(live_return)
        if psr_alarm and psr_alert:
            alerts.append(psr_alert)
            
        # Check additional performance metrics
        if additional_metrics:
            perf_alerts = self._check_performance_metrics(additional_metrics)
            alerts.extend(perf_alerts)
            
        # Store alerts and performance data
        self.alerts_history.extend(alerts)
        self.performance_history.append({
            'timestamp': datetime.now(),
            'live_return': live_return,
            'return_deviation': return_deviation,
            'alerts_count': len(alerts)
        })
        
        return alerts
    
    def _check_performance_metrics(self, metrics: Dict[str, float]) -> List[DriftAlert]:
        """Check additional performance metrics for drift."""
        alerts = []
        
        for metric_name, current_value in metrics.items():
            expected_value = self.backtest_metrics.get(metric_name, 0.0)
            threshold = self.drift_thresholds.get(f'{metric_name}_deviation', 0.1)
            
            deviation = abs(current_value - expected_value)
            
            if deviation > threshold:
                severity = 'critical' if deviation > threshold * 2 else 'warning'
                message = f"{metric_name} drift: {current_value:.4f} vs expected {expected_value:.4f}"
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    alert_type='performance',
                    severity=severity,
                    message=message,
                    current_value=current_value,
                    threshold=expected_value + threshold,
                    recommended_action=f"Monitor {metric_name} closely"
                )
                alerts.append(alert)
                
        return alerts
    
    def get_drift_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_alerts = [a for a in self.alerts_history if a.timestamp >= cutoff_date]
        recent_performance = [p for p in self.performance_history if p['timestamp'] >= cutoff_date]
        
        # Count alerts by type and severity
        alert_counts = {}
        for alert in recent_alerts:
            key = f"{alert.alert_type}_{alert.severity}"
            alert_counts[key] = alert_counts.get(key, 0) + 1
            
        # Calculate performance statistics
        if recent_performance:
            returns = [p['live_return'] for p in recent_performance]
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            sharpe = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0
        else:
            avg_return = volatility = sharpe = 0
            
        return {
            'monitoring_period_days': days,
            'total_alerts': len(recent_alerts),
            'alert_counts': alert_counts,
            'performance_summary': {
                'avg_daily_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'observations': len(recent_performance)
            },
            'cusum_status': self.cusum_detector.get_current_metrics(),
            'drift_detected': len(recent_alerts) > 0,
            'recommendation': self._get_recommendation(recent_alerts)
        }
    
    def _get_recommendation(self, recent_alerts: List[DriftAlert]) -> str:
        """Get recommendation based on recent alerts."""
        if not recent_alerts:
            return "Continue monitoring - no drift detected"
            
        critical_alerts = [a for a in recent_alerts if a.severity == 'critical']
        if critical_alerts:
            return "CRITICAL: Reduce position size by 50% and investigate immediately"
            
        warning_alerts = [a for a in recent_alerts if a.severity == 'warning']
        if len(warning_alerts) >= 3:
            return "WARNING: Multiple drift alerts - consider reducing position size"
            
        return "Monitor closely - drift alerts detected"


# Example usage and testing
if __name__ == "__main__":
    def test_drift_monitoring():
        """Test the drift monitoring system."""
        print("=== Drift Monitoring Test ===")
        
        # Simulate backtest metrics
        backtest_metrics = {
            'sharpe_ratio': 1.5,
            'daily_return': 0.001,
            'volatility': 0.02,
            'max_drawdown': 0.10,
            'observations': 252
        }
        
        # Initialize monitor
        monitor = PerformanceDriftMonitor(backtest_metrics)
        
        # Simulate live trading with some drift
        np.random.seed(42)
        
        print("Simulating live trading...")
        for day in range(50):
            # Simulate returns - start normal, then introduce drift
            if day < 30:
                live_return = np.random.normal(0.001, 0.02)
            else:
                # Introduce negative drift
                live_return = np.random.normal(-0.002, 0.025)
                
            alerts = monitor.update(live_return)
            
            if alerts:
                print(f"Day {day}: {len(alerts)} alerts")
                for alert in alerts:
                    print(f"  {alert.severity.upper()}: {alert.message}")
        
        # Get summary
        summary = monitor.get_drift_summary(days=30)
        print("\nDrift Summary:")
        print(f"Total alerts: {summary['total_alerts']}")
        print(f"Alert counts: {summary['alert_counts']}")
        print(f"Recommendation: {summary['recommendation']}")
    
    test_drift_monitoring()



