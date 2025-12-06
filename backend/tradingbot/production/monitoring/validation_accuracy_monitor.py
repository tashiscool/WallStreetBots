"""Validation Accuracy Metrics Monitoring

Monitors and reports on signal validation accuracy over time.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationAccuracyMetrics:
    """Validation accuracy metrics for a strategy."""
    strategy_name: str
    timestamp: datetime
    total_validations: int
    correct_predictions: int
    accuracy_rate: float
    precision: float  # True positives / (True positives + False positives)
    recall: float  # True positives / (True positives + False negatives)
    f1_score: float
    average_prediction_error: float
    false_positive_rate: float
    false_negative_rate: float
    strength_score_correlation: float  # Correlation between strength score and actual outcome


@dataclass
class ValidationAccuracyReport:
    """Comprehensive validation accuracy report."""
    timestamp: datetime
    strategies: Dict[str, ValidationAccuracyMetrics]
    overall_metrics: Dict[str, float]
    trends: Dict[str, str]  # 'IMPROVING', 'DEGRADING', 'STABLE'
    recommendations: List[str]
    alerts: List[str]


class ValidationAccuracyMonitor:
    """Monitor validation accuracy metrics over time."""
    
    def __init__(self, lookback_days: int = 30):
        """Initialize monitor.
        
        Args:
            lookback_days: Number of days to look back for metrics
        """
        self.lookback_days = lookback_days
        self.logger = logging.getLogger(__name__)
        
        # Store validation records
        self.validation_records: Dict[str, List[Dict[str, Any]]] = {}
        
        # Metrics history
        self.metrics_history: Dict[str, List[ValidationAccuracyMetrics]] = {}
        
    def record_validation(
        self,
        strategy_name: str,
        validation_result: Dict[str, Any],
        actual_outcome: Dict[str, Any]
    ):
        """Record a validation result and its actual outcome.
        
        Args:
            strategy_name: Name of the strategy
            validation_result: Original validation result
            actual_outcome: Actual trade outcome with 'win', 'return', etc.
        """
        try:
            if strategy_name not in self.validation_records:
                self.validation_records[strategy_name] = []
            
            record = {
                'timestamp': datetime.now(),
                'validation_result': validation_result,
                'actual_outcome': actual_outcome,
                'predicted_action': validation_result.get('recommended_action', 'execute'),
                'strength_score': validation_result.get('strength_score', 0),
                'actual_win': actual_outcome.get('win', False),
                'actual_return': actual_outcome.get('return', 0.0),
                'was_correct': self._calculate_correctness(validation_result, actual_outcome)
            }
            
            self.validation_records[strategy_name].append(record)
            
            # Keep only recent records
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            self.validation_records[strategy_name] = [
                r for r in self.validation_records[strategy_name]
                if r['timestamp'] >= cutoff_date
            ]
            
        except Exception as e:
            self.logger.error(f"Error recording validation: {e}")
    
    def _calculate_correctness(
        self,
        validation_result: Dict[str, Any],
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """Calculate if validation prediction was correct."""
        predicted_action = validation_result.get('recommended_action', 'execute')
        actual_win = actual_outcome.get('win', False)
        
        # Correct if: (predicted execute and won) or (predicted reject and lost)
        if predicted_action == 'execute':
            return actual_win
        else:
            return not actual_win
    
    def calculate_metrics(self, strategy_name: str) -> Optional[ValidationAccuracyMetrics]:
        """Calculate accuracy metrics for a strategy."""
        try:
            if strategy_name not in self.validation_records:
                return None
            
            records = self.validation_records[strategy_name]
            
            if len(records) < 5:  # Need minimum samples
                return None
            
            # Calculate basic metrics
            total = len(records)
            correct = sum(1 for r in records if r['was_correct'])
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate precision and recall
            true_positives = sum(
                1 for r in records
                if r['predicted_action'] == 'execute' and r['actual_win']
            )
            false_positives = sum(
                1 for r in records
                if r['predicted_action'] == 'execute' and not r['actual_win']
            )
            false_negatives = sum(
                1 for r in records
                if r['predicted_action'] != 'execute' and r['actual_win']
            )
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate prediction error
            prediction_errors = [
                abs(r['strength_score'] - (100 if r['actual_win'] else 0))
                for r in records
            ]
            avg_prediction_error = np.mean(prediction_errors) if prediction_errors else 0.0
            
            # Calculate false positive/negative rates
            false_positive_rate = false_positives / total if total > 0 else 0.0
            false_negative_rate = false_negatives / total if total > 0 else 0.0
            
            # Calculate strength score correlation
            strength_scores = [r['strength_score'] for r in records]
            actual_wins = [1 if r['actual_win'] else 0 for r in records]
            if len(strength_scores) > 1:
                correlation = float(np.corrcoef(strength_scores, actual_wins)[0, 1])
            else:
                correlation = 0.0
            
            metrics = ValidationAccuracyMetrics(
                strategy_name=strategy_name,
                timestamp=datetime.now(),
                total_validations=total,
                correct_predictions=correct,
                accuracy_rate=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                average_prediction_error=avg_prediction_error,
                false_positive_rate=false_positive_rate,
                false_negative_rate=false_negative_rate,
                strength_score_correlation=correlation
            )
            
            # Store in history
            if strategy_name not in self.metrics_history:
                self.metrics_history[strategy_name] = []
            self.metrics_history[strategy_name].append(metrics)
            
            # Keep only recent history
            if len(self.metrics_history[strategy_name]) > 100:
                self.metrics_history[strategy_name] = self.metrics_history[strategy_name][-100:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {strategy_name}: {e}")
            return None
    
    def generate_report(self) -> ValidationAccuracyReport:
        """Generate comprehensive validation accuracy report."""
        try:
            strategies_metrics = {}
            all_accuracy_rates = []
            all_f1_scores = []
            
            # Calculate metrics for each strategy
            for strategy_name in self.validation_records.keys():
                metrics = self.calculate_metrics(strategy_name)
                if metrics:
                    strategies_metrics[strategy_name] = metrics
                    all_accuracy_rates.append(metrics.accuracy_rate)
                    all_f1_scores.append(metrics.f1_score)
            
            # Calculate overall metrics
            overall_metrics = {
                'average_accuracy': np.mean(all_accuracy_rates) if all_accuracy_rates else 0.0,
                'average_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0,
                'total_validations': sum(
                    m.total_validations for m in strategies_metrics.values()
                ),
                'strategies_monitored': len(strategies_metrics)
            }
            
            # Calculate trends
            trends = {}
            for strategy_name, metrics in strategies_metrics.items():
                trend = self._calculate_trend(strategy_name)
                trends[strategy_name] = trend
            
            # Generate recommendations
            recommendations = self._generate_recommendations(strategies_metrics, overall_metrics)
            
            # Generate alerts
            alerts = self._generate_alerts(strategies_metrics, overall_metrics)
            
            return ValidationAccuracyReport(
                timestamp=datetime.now(),
                strategies=strategies_metrics,
                overall_metrics=overall_metrics,
                trends=trends,
                recommendations=recommendations,
                alerts=alerts
            )
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return ValidationAccuracyReport(
                timestamp=datetime.now(),
                strategies={},
                overall_metrics={},
                trends={},
                recommendations=[],
                alerts=[f"Error generating report: {e}"]
            )
    
    def _calculate_trend(self, strategy_name: str) -> str:
        """Calculate trend for a strategy's accuracy."""
        if strategy_name not in self.metrics_history:
            return 'UNKNOWN'
        
        history = self.metrics_history[strategy_name]
        if len(history) < 3:
            return 'INSUFFICIENT_DATA'
        
        # Compare recent vs older accuracy
        recent = history[-3:]
        older = history[-6:-3] if len(history) >= 6 else history[:3]
        
        recent_avg = np.mean([m.accuracy_rate for m in recent])
        older_avg = np.mean([m.accuracy_rate for m in older]) if older else recent_avg
        
        if recent_avg > older_avg + 0.05:
            return 'IMPROVING'
        elif recent_avg < older_avg - 0.05:
            return 'DEGRADING'
        else:
            return 'STABLE'
    
    def _generate_recommendations(
        self,
        strategies_metrics: Dict[str, ValidationAccuracyMetrics],
        overall_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        # Overall accuracy check
        if overall_metrics['average_accuracy'] < 0.60:
            recommendations.append(
                "Overall validation accuracy is below 60%. Consider reviewing validation criteria."
            )
        
        # Strategy-specific recommendations
        for name, metrics in strategies_metrics.items():
            if metrics.accuracy_rate < 0.55:
                recommendations.append(
                    f"{name}: Accuracy is {metrics.accuracy_rate:.1%}. Consider tightening validation criteria."
                )
            
            if metrics.false_positive_rate > 0.20:
                recommendations.append(
                    f"{name}: High false positive rate ({metrics.false_positive_rate:.1%}). "
                    "Too many bad signals are being accepted."
                )
            
            if metrics.false_negative_rate > 0.30:
                recommendations.append(
                    f"{name}: High false negative rate ({metrics.false_negative_rate:.1%}). "
                    "Too many good signals are being rejected."
                )
            
            if abs(metrics.strength_score_correlation) < 0.3:
                recommendations.append(
                    f"{name}: Weak correlation between strength score and outcome. "
                    "Consider improving strength score calculation."
                )
        
        return recommendations
    
    def _generate_alerts(
        self,
        strategies_metrics: Dict[str, ValidationAccuracyMetrics],
        overall_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate alerts for critical issues."""
        alerts = []
        
        # Critical accuracy issues
        if overall_metrics['average_accuracy'] < 0.50:
            alerts.append(
                f"CRITICAL: Overall validation accuracy is {overall_metrics['average_accuracy']:.1%} - "
                "validation system may be broken"
            )
        
        # Strategy-specific alerts
        for name, metrics in strategies_metrics.items():
            if metrics.accuracy_rate < 0.45:
                alerts.append(
                    f"CRITICAL: {name} validation accuracy is {metrics.accuracy_rate:.1%} - "
                    "consider disabling strategy"
                )
        
        return alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        report = self.generate_report()
        
        return {
            'timestamp': report.timestamp.isoformat(),
            'overall_metrics': report.overall_metrics,
            'strategies': {
                name: {
                    'accuracy_rate': metrics.accuracy_rate,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'total_validations': metrics.total_validations,
                    'trend': report.trends.get(name, 'UNKNOWN')
                }
                for name, metrics in report.strategies.items()
            },
            'recommendations': report.recommendations,
            'alerts': report.alerts
        }

