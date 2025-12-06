"""Walk-Forward Backtesting Framework

Implements walk-forward analysis to validate strategies without overfitting.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from decimal import Decimal
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """A single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_days: int
    test_days: int


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    average_trade_return: float
    volatility: float
    calmar_ratio: float  # Return / Max Drawdown


@dataclass
class WalkForwardResult:
    """Result of walk-forward backtest."""
    window: WalkForwardWindow
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics
    out_of_sample_performance: float  # Test Sharpe / Train Sharpe
    stability_score: float  # Consistency of performance


@dataclass
class WalkForwardReport:
    """Comprehensive walk-forward report."""
    timestamp: datetime
    total_windows: int
    results: List[WalkForwardResult]
    overall_metrics: Dict[str, float]
    stability_analysis: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]


class WalkForwardBacktester:
    """Walk-forward backtesting framework."""
    
    def __init__(
        self,
        train_window_days: int = 90,
        test_window_days: int = 30,
        step_days: int = 30
    ):
        """Initialize walk-forward backtester.
        
        Args:
            train_window_days: Days in training window
            test_window_days: Days in test window
            step_days: Days to step forward each iteration
        """
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.logger = logging.getLogger(__name__)
        
    def run_walk_forward(
        self,
        strategy_function: Callable,
        historical_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> WalkForwardReport:
        """Run walk-forward backtest.
        
        Args:
            strategy_function: Function that takes (data, params) and returns trades
            historical_data: Historical price/volume data
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_params: Strategy parameters
            
        Returns:
            WalkForwardReport
        """
        try:
            self.logger.info("Starting walk-forward backtest")
            
            if strategy_params is None:
                strategy_params = {}
            
            # Generate walk-forward windows
            windows = self._generate_windows(start_date, end_date)
            
            results = []
            all_test_sharpes = []
            all_train_sharpes = []
            
            for window in windows:
                try:
                    # Split data
                    train_data = self._get_data_in_range(
                        historical_data, window.train_start, window.train_end
                    )
                    test_data = self._get_data_in_range(
                        historical_data, window.test_start, window.test_end
                    )
                    
                    if len(train_data) < 10 or len(test_data) < 5:
                        continue
                    
                    # Run backtest on training data
                    train_trades = strategy_function(train_data, strategy_params)
                    train_metrics = self._calculate_metrics(train_trades, train_data)
                    
                    # Run backtest on test data (out-of-sample)
                    test_trades = strategy_function(test_data, strategy_params)
                    test_metrics = self._calculate_metrics(test_trades, test_data)
                    
                    # Calculate out-of-sample performance
                    oos_performance = (
                        test_metrics.sharpe_ratio / train_metrics.sharpe_ratio
                        if train_metrics.sharpe_ratio > 0 else 0.0
                    )
                    
                    # Calculate stability (how consistent is performance)
                    stability = self._calculate_stability(train_metrics, test_metrics)
                    
                    result = WalkForwardResult(
                        window=window,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        out_of_sample_performance=oos_performance,
                        stability_score=stability
                    )
                    
                    results.append(result)
                    all_test_sharpes.append(test_metrics.sharpe_ratio)
                    all_train_sharpes.append(train_metrics.sharpe_ratio)
                    
                except Exception as e:
                    self.logger.warning(f"Error in window {window}: {e}")
                    continue
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(results)
            
            # Stability analysis
            stability_analysis = self._analyze_stability(results)
            
            # Generate recommendations
            recommendations, warnings = self._generate_recommendations(
                results, overall_metrics, stability_analysis
            )
            
            report = WalkForwardReport(
                timestamp=datetime.now(),
                total_windows=len(results),
                results=results,
                overall_metrics=overall_metrics,
                stability_analysis=stability_analysis,
                recommendations=recommendations,
                warnings=warnings
            )
            
            self.logger.info(f"Walk-forward complete: {len(results)} windows tested")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward backtest: {e}")
            return WalkForwardReport(
                timestamp=datetime.now(),
                total_windows=0,
                results=[],
                overall_metrics={},
                stability_analysis={},
                recommendations=[],
                warnings=[f"Backtest failed: {e}"]
            )
    
    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward windows."""
        windows = []
        current_date = start_date
        
        while current_date < end_date:
            train_start = current_date
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window_days)
            
            if test_end > end_date:
                break
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_days=self.train_window_days,
                test_days=self.test_window_days
            )
            
            windows.append(window)
            
            # Step forward
            current_date += timedelta(days=self.step_days)
        
        return windows
    
    def _get_data_in_range(
        self,
        data: pd.DataFrame,
        start: datetime,
        end: datetime
    ) -> pd.DataFrame:
        """Get data in date range."""
        if 'date' in data.columns:
            mask = (data['date'] >= start) & (data['date'] <= end)
            return data[mask].copy()
        elif data.index.name == 'date' or isinstance(data.index, pd.DatetimeIndex):
            return data[(data.index >= start) & (data.index <= end)].copy()
        else:
            # Assume data is already filtered
            return data
    
    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> BacktestMetrics:
        """Calculate backtest metrics from trades."""
        if not trades:
            return BacktestMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                average_trade_return=0.0,
                volatility=0.0,
                calmar_ratio=0.0
            )
        
        # Calculate returns
        returns = [t.get('return', 0.0) for t in trades]
        returns_series = pd.Series(returns)
        
        # Basic metrics
        total_return = float(returns_series.sum())
        win_rate = float((returns_series > 0).sum() / len(returns_series))
        total_trades = len(trades)
        average_trade_return = float(returns_series.mean())
        volatility = float(returns_series.std() * np.sqrt(252)) if len(returns_series) > 1 else 0.0
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = float((average_trade_return * 252) / volatility)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(abs(drawdown.min()))
        
        # Profit factor
        wins = returns_series[returns_series > 0]
        losses = returns_series[returns_series < 0]
        profit_factor = (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) > 0 and losses.sum() < 0 else 0.0
        )
        
        # Calmar ratio
        calmar_ratio = (
            total_return / max_drawdown
            if max_drawdown > 0 else 0.0
        )
        
        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            average_trade_return=average_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio
        )
    
    def _calculate_stability(
        self,
        train_metrics: BacktestMetrics,
        test_metrics: BacktestMetrics
    ) -> float:
        """Calculate stability score (how consistent is performance)."""
        # Stability is based on how similar test performance is to train
        sharpe_diff = abs(test_metrics.sharpe_ratio - train_metrics.sharpe_ratio)
        sharpe_avg = (abs(train_metrics.sharpe_ratio) + abs(test_metrics.sharpe_ratio)) / 2
        
        if sharpe_avg > 0:
            stability = 1.0 - min(1.0, sharpe_diff / (sharpe_avg + 0.1))
        else:
            stability = 0.0
        
        return float(stability)
    
    def _calculate_overall_metrics(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, float]:
        """Calculate overall metrics across all windows."""
        if not results:
            return {}
        
        test_sharpes = [r.test_metrics.sharpe_ratio for r in results]
        train_sharpes = [r.train_metrics.sharpe_ratio for r in results]
        oos_performances = [r.out_of_sample_performance for r in results]
        stabilities = [r.stability_score for r in results]
        
        return {
            'average_test_sharpe': float(np.mean(test_sharpes)),
            'average_train_sharpe': float(np.mean(train_sharpes)),
            'average_oos_performance': float(np.mean(oos_performances)),
            'average_stability': float(np.mean(stabilities)),
            'std_test_sharpe': float(np.std(test_sharpes)),
            'min_test_sharpe': float(np.min(test_sharpes)),
            'max_test_sharpe': float(np.max(test_sharpes)),
            'positive_oos_windows': sum(1 for oos in oos_performances if oos > 0.8),
            'total_windows': len(results)
        }
    
    def _analyze_stability(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, Any]:
        """Analyze stability of results."""
        if not results:
            return {}
        
        stabilities = [r.stability_score for r in results]
        oos_performances = [r.out_of_sample_performance for r in results]
        
        return {
            'average_stability': float(np.mean(stabilities)),
            'stability_std': float(np.std(stabilities)),
            'consistent_windows': sum(1 for s in stabilities if s > 0.7),
            'degradation_windows': sum(1 for oos in oos_performances if oos < 0.5),
            'improvement_windows': sum(1 for oos in oos_performances if oos > 1.2)
        }
    
    def _generate_recommendations(
        self,
        results: List[WalkForwardResult],
        overall_metrics: Dict[str, float],
        stability_analysis: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Generate recommendations and warnings."""
        recommendations = []
        warnings = []
        
        # Check out-of-sample performance
        avg_oos = overall_metrics.get('average_oos_performance', 0)
        if avg_oos < 0.7:
            warnings.append(
                f"Poor out-of-sample performance ({avg_oos:.2f}). "
                "Strategy may be overfitted."
            )
        elif avg_oos > 0.9:
            recommendations.append(
                f"Good out-of-sample performance ({avg_oos:.2f}). "
                "Strategy appears robust."
            )
        
        # Check stability
        avg_stability = overall_metrics.get('average_stability', 0)
        if avg_stability < 0.6:
            warnings.append(
                f"Low stability ({avg_stability:.2f}). "
                "Performance is inconsistent across windows."
            )
        
        # Check for degradation
        degradation_windows = stability_analysis.get('degradation_windows', 0)
        if degradation_windows > len(results) * 0.3:
            warnings.append(
                f"{degradation_windows} windows show performance degradation. "
                "Strategy may not be robust."
            )
        
        # Check Sharpe ratio consistency
        std_sharpe = overall_metrics.get('std_test_sharpe', 0)
        avg_sharpe = overall_metrics.get('average_test_sharpe', 0)
        if std_sharpe > abs(avg_sharpe) * 0.5:
            warnings.append(
                "High variance in Sharpe ratios across windows. "
                "Strategy performance is inconsistent."
            )
        
        return recommendations, warnings

