#!/usr / bin / env python3
"""Advanced Analytics Module
Comprehensive performance analytics including Sharpe ratio, max drawdown, and risk - adjusted metrics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float
    annualized_return: float
    volatility: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float

    # Trading metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Advanced metrics
    information_ratio: float
    treynor_ratio: float
    alpha: float
    beta: float
    tracking_error: float

    # Period info
    period_start: datetime
    period_end: datetime
    trading_days: int

    # Additional stats
    best_day: float
    worst_day: float
    positive_days: int
    negative_days: int

    # Recovery metrics
    recovery_factor: float
    ulcer_index: float
    sterling_ratio: float


@dataclass
class DrawdownPeriod:
    """Drawdown period analysis."""

    start_date: datetime
    end_date: datetime
    recovery_date: datetime | None
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: int | None
    is_recovered: bool


class AdvancedAnalytics:
    """Advanced Analytics Engine.

    Calculates comprehensive performance metrics including:
    - Sharpe ratio, Sortino ratio, Calmar ratio
    - Maximum drawdown analysis with recovery periods
    - Value at Risk (VaR) and Conditional VaR
    - Alpha, Beta, Information ratio
    - Win rate, profit factor, recovery metrics
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize analytics engine.

        Args:
            risk_free_rate: Annual risk - free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_metrics(
        self,
        returns: list[float] | np.ndarray | pd.Series,
        benchmark_returns: list[float] | np.ndarray | pd.Series | None = None,
        portfolio_values: list[float] | np.ndarray | pd.Series | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            returns: Portfolio returns (daily)
            benchmark_returns: Benchmark returns for relative metrics
            portfolio_values: Portfolio values for drawdown analysis
            start_date: Period start date
            end_date: Period end date

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        try:
            # Convert inputs to numpy arrays
            returns = np.array(returns)
            if benchmark_returns is not None:
                benchmark_returns = np.array(benchmark_returns)
            if portfolio_values is not None:
                portfolio_values = np.array(portfolio_values)

            # Handle empty returns
            if len(returns) == 0:
                return self._create_empty_metrics()

            # Calculate basic metrics
            total_return = self._calculate_total_return(returns)
            annualized_return = self._calculate_annualized_return(returns)
            volatility = self._calculate_volatility(returns)

            # Risk - adjusted ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, portfolio_values)

            # Drawdown analysis
            max_drawdown = self._calculate_max_drawdown(
                portfolio_values or self._returns_to_values(returns)
            )

            # Value at Risk
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            cvar_95 = self._calculate_cvar(returns, 0.95)

            # Trading metrics
            win_rate = self._calculate_win_rate(returns)
            avg_win, avg_loss = self._calculate_avg_win_loss(returns)
            profit_factor = self._calculate_profit_factor(returns)

            # Relative metrics (if benchmark provided)
            alpha, beta, information_ratio, treynor_ratio, tracking_error = (
                self._calculate_relative_metrics(returns, benchmark_returns)
            )

            # Period info
            trading_days = len(returns)
            period_start = start_date or datetime.now() - timedelta(days=trading_days)
            period_end = end_date or datetime.now()

            # Additional stats
            best_day = float(np.max(returns)) if len(returns) > 0 else 0.0
            worst_day = float(np.min(returns)) if len(returns) > 0 else 0.0
            positive_days = int(np.sum(returns > 0))
            negative_days = int(np.sum(returns < 0))

            # Recovery metrics
            recovery_factor = self._calculate_recovery_factor(returns, max_drawdown)
            ulcer_index = self._calculate_ulcer_index(
                portfolio_values or self._returns_to_values(returns)
            )
            sterling_ratio = self._calculate_sterling_ratio(returns, max_drawdown)

            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio,
                alpha=alpha,
                beta=beta,
                tracking_error=tracking_error,
                period_start=period_start,
                period_end=period_end,
                trading_days=trading_days,
                best_day=best_day,
                worst_day=worst_day,
                positive_days=positive_days,
                negative_days=negative_days,
                recovery_factor=recovery_factor,
                ulcer_index=ulcer_index,
                sterling_ratio=sterling_ratio,
            )

        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return self._create_empty_metrics()

    def analyze_drawdown_periods(
        self,
        portfolio_values: list[float] | np.ndarray | pd.Series,
        dates: list[datetime] | None = None,
    ) -> list[DrawdownPeriod]:
        """Analyze all drawdown periods.

        Args:
            portfolio_values: Portfolio values over time
            dates: Corresponding dates (optional)

        Returns:
            List of DrawdownPeriod objects
        """
        try:
            values = np.array(portfolio_values)
            if len(values) < 2:
                return []

            # Generate dates if not provided
            if dates is None:
                dates = [
                    datetime.now() - timedelta(days=len(values) - i - 1)
                    for i in range(len(values))
                ]

            drawdown_periods = []
            peak = values[0]
            peak_idx = 0
            in_drawdown = False
            start_idx = 0

            for i, value in enumerate(values):
                if value > peak:
                    # New peak reached
                    if in_drawdown:
                        # End of drawdown period - add to list
                        trough_value = np.min(values[start_idx:i])
                        trough_idx = start_idx + np.argmin(values[start_idx:i])

                        drawdown_pct = (peak - trough_value) / peak
                        duration = i - start_idx
                        recovery_days = i - trough_idx

                        drawdown_periods.append(
                            DrawdownPeriod(
                                start_date=dates[start_idx],
                                end_date=dates[trough_idx],
                                recovery_date=dates[i],
                                peak_value=peak,
                                trough_value=trough_value,
                                drawdown_pct=drawdown_pct,
                                duration_days=duration,
                                recovery_days=recovery_days,
                                is_recovered=True,
                            )
                        )

                    peak = value
                    peak_idx = i
                    in_drawdown = False
                # Value below peak
                elif not in_drawdown:
                    # Start of new drawdown
                    in_drawdown = True
                    start_idx = peak_idx

            # Handle ongoing drawdown
            if in_drawdown:
                trough_value = np.min(values[start_idx:])
                trough_idx = start_idx + np.argmin(values[start_idx:])

                drawdown_pct = (peak - trough_value) / peak
                duration = len(values) - start_idx

                drawdown_periods.append(
                    DrawdownPeriod(
                        start_date=dates[start_idx],
                        end_date=dates[trough_idx],
                        recovery_date=None,
                        peak_value=peak,
                        trough_value=trough_value,
                        drawdown_pct=drawdown_pct,
                        duration_days=duration,
                        recovery_days=None,
                        is_recovered=False,
                    )
                )

            return sorted(drawdown_periods, key=lambda x: x.drawdown_pct, reverse=True)

        except Exception as e:
            self.logger.error(f"Error analyzing drawdown periods: {e}")
            return []

    def _calculate_total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return float(np.prod(1 + returns) - 1)

    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252  # Assuming 252 trading days per year
        return float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        return float(np.std(returns) * np.sqrt(252))

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        return float(np.mean(excess_returns) * 252 / (np.std(returns) * np.sqrt(252)))

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        return float(np.mean(excess_returns) * 252 / downside_deviation)

    def _calculate_calmar_ratio(
        self, returns: np.ndarray, portfolio_values: np.ndarray | None
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        ann_return = self._calculate_annualized_return(returns)

        if portfolio_values is not None:
            max_dd = self._calculate_max_drawdown(portfolio_values)
        else:
            values = self._returns_to_values(returns)
            max_dd = self._calculate_max_drawdown(values)

        return float(ann_return / max_dd) if max_dd > 0 else 0.0

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) < 2:
            return 0.0

        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return float(max_drawdown)

    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return float(-np.percentile(returns, (1 - confidence_level) * 100))

    def _calculate_cvar(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0

        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= -var]

        return float(-np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0

    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.0
        return float(np.sum(returns > 0) / len(returns))

    def _calculate_avg_win_loss(self, returns: np.ndarray) -> tuple[float, float]:
        """Calculate average win and loss."""
        if len(returns) == 0:
            return 0.0, 0.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        return avg_win, avg_loss

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0

        return float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

    def _calculate_relative_metrics(
        self, returns: np.ndarray, benchmark_returns: np.ndarray | None
    ) -> tuple[float, float, float, float, float]:
        """Calculate relative metrics vs benchmark."""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return 0.0, 1.0, 0.0, 0.0, 0.0

        # Align arrays
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Calculate beta
        if np.var(benchmark_returns) > 0:
            beta = float(
                np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            )
        else:
            beta = 1.0

        # Calculate alpha
        portfolio_return = self._calculate_annualized_return(returns)
        benchmark_return = self._calculate_annualized_return(benchmark_returns)
        alpha = float(
            portfolio_return
            - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        )

        # Calculate tracking error
        tracking_error = float(np.std(returns - benchmark_returns) * np.sqrt(252))

        # Calculate information ratio
        excess_return = np.mean(returns - benchmark_returns) * 252
        information_ratio = (
            float(excess_return / tracking_error) if tracking_error > 0 else 0.0
        )

        # Calculate Treynor ratio
        excess_portfolio_return = portfolio_return - self.risk_free_rate
        treynor_ratio = float(excess_portfolio_return / beta) if beta != 0 else 0.0

        return alpha, beta, information_ratio, treynor_ratio, tracking_error

    def _calculate_recovery_factor(
        self, returns: np.ndarray, max_drawdown: float
    ) -> float:
        """Calculate recovery factor."""
        total_return = self._calculate_total_return(returns)
        return float(total_return / max_drawdown) if max_drawdown > 0 else 0.0

    def _calculate_ulcer_index(self, portfolio_values: np.ndarray) -> float:
        """Calculate Ulcer Index (measure of downside risk)."""
        if len(portfolio_values) < 2:
            return 0.0

        peak = portfolio_values[0]
        squared_drawdowns = []

        for value in portfolio_values:
            peak = max(peak, value)
            drawdown = (peak - value) / peak * 100
            squared_drawdowns.append(drawdown**2)

        return float(np.sqrt(np.mean(squared_drawdowns)))

    def _calculate_sterling_ratio(
        self, returns: np.ndarray, max_drawdown: float
    ) -> float:
        """Calculate Sterling ratio."""
        ann_return = self._calculate_annualized_return(returns)
        return float(ann_return / max_drawdown) if max_drawdown > 0 else 0.0

    def _returns_to_values(
        self, returns: np.ndarray, initial_value: float = 10000.0
    ) -> np.ndarray:
        """Convert returns to portfolio values."""
        values = [initial_value]
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        return np.array(values[1:])  # Exclude initial value

    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object."""
        now = datetime.now()
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            alpha=0.0,
            beta=1.0,
            tracking_error=0.0,
            period_start=now,
            period_end=now,
            trading_days=0,
            best_day=0.0,
            worst_day=0.0,
            positive_days=0,
            negative_days=0,
            recovery_factor=0.0,
            ulcer_index=0.0,
            sterling_ratio=0.0,
        )

    def generate_analytics_report(self, metrics: PerformanceMetrics) -> str:
        """Generate formatted analytics report."""
        report = f"""
📊 ADVANCED PERFORMANCE ANALYTICS REPORT
{" = " * 50}

📈 RETURNS & RISK METRICS:
  Total Return:         {metrics.total_return: > 8.2%}
  Annualized Return:    {metrics.annualized_return: > 8.2%}
  Volatility:           {metrics.volatility: > 8.2%}

🏆 RISK - ADJUSTED RATIOS:
  Sharpe Ratio:         {metrics.sharpe_ratio: > 8.2f}
  Sortino Ratio:        {metrics.sortino_ratio: > 8.2f}
  Calmar Ratio:         {metrics.calmar_ratio: > 8.2f}
  Information Ratio:    {metrics.information_ratio: > 8.2f}

📉 DOWNSIDE PROTECTION:
  Max Drawdown:         {metrics.max_drawdown: > 8.2%}
  VaR (95%):           {metrics.var_95: > 8.2%}
  VaR (99%):           {metrics.var_99: > 8.2%}
  CVaR (95%):          {metrics.cvar_95: > 8.2%}
  Ulcer Index:          {metrics.ulcer_index: > 8.2f}

📊 TRADING PERFORMANCE:
  Win Rate:             {metrics.win_rate: > 8.2%}
  Average Win:          {metrics.avg_win: > 8.2%}
  Average Loss:         {metrics.avg_loss: > 8.2%}
  Profit Factor:        {metrics.profit_factor: > 8.2f}

📅 PERIOD STATISTICS:
  Period:               {metrics.period_start.strftime("%Y-%m-%d")} to {metrics.period_end.strftime("%Y-%m-%d")}
  Trading Days:         {metrics.trading_days: > 8d}
  Best Day:             {metrics.best_day: > 8.2%}
  Worst Day:            {metrics.worst_day: > 8.2%}
  Positive Days:        {metrics.positive_days: > 8d} ({metrics.positive_days / max(metrics.trading_days, 1): > 5.1%})
  Negative Days:        {metrics.negative_days: > 8d} ({metrics.negative_days / max(metrics.trading_days, 1): > 5.1%})

🔄 RECOVERY METRICS:
  Recovery Factor:      {metrics.recovery_factor: > 8.2f}
  Sterling Ratio:       {metrics.sterling_ratio: > 8.2f}

📊 RELATIVE PERFORMANCE:
  Alpha:                {metrics.alpha: > 8.2%}
  Beta:                 {metrics.beta: > 8.2f}
  Tracking Error:       {metrics.tracking_error: > 8.2%}
  Treynor Ratio:        {metrics.treynor_ratio: > 8.2f}

{" = " * 50}
"""
        return report


# Convenience function for quick analysis
def analyze_performance(
    returns: list[float] | np.ndarray,
    benchmark_returns: list[float] | np.ndarray | None = None,
    portfolio_values: list[float] | np.ndarray | None = None,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """Quick performance analysis.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns (optional)
        portfolio_values: Portfolio values (optional)
        risk_free_rate: Risk - free rate (default 2%)

    Returns:
        PerformanceMetrics object
    """
    analytics = AdvancedAnalytics(risk_free_rate)
    return analytics.calculate_comprehensive_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        portfolio_values=portfolio_values,
    )
