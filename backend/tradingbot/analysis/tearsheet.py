from __future__ import annotations

"""
Interactive Performance Tearsheet Generator.

Ported from QuantConnect/LEAN and Nautilus Trader patterns.
Generates comprehensive HTML tearsheets with Plotly visualizations.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Install with: pip install plotly")

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False


class ThemeType(Enum):
    """Available themes for tearsheets."""
    LIGHT = "plotly_white"
    DARK = "plotly_dark"
    NAUTILUS = "nautilus"
    NAUTILUS_DARK = "nautilus_dark"


@dataclass
class ThemeColors:
    """Color palette for a theme."""
    background: str = "#ffffff"
    text: str = "#000000"
    primary: str = "#1f77b4"
    positive: str = "#00c805"
    negative: str = "#ff2d21"
    neutral: str = "#808080"
    grid: str = "#e0e0e0"
    paper: str = "#ffffff"


THEMES = {
    ThemeType.LIGHT: ThemeColors(
        background="#ffffff",
        text="#000000",
        primary="#1f77b4",
        positive="#00c805",
        negative="#ff2d21",
        neutral="#808080",
        grid="#e0e0e0",
        paper="#ffffff",
    ),
    ThemeType.DARK: ThemeColors(
        background="#1e1e1e",
        text="#ffffff",
        primary="#00d4ff",
        positive="#00ff88",
        negative="#ff4444",
        neutral="#888888",
        grid="#333333",
        paper="#2d2d2d",
    ),
    ThemeType.NAUTILUS: ThemeColors(
        background="#fafafa",
        text="#333333",
        primary="#0077cc",
        positive="#28a745",
        negative="#dc3545",
        neutral="#6c757d",
        grid="#dee2e6",
        paper="#ffffff",
    ),
    ThemeType.NAUTILUS_DARK: ThemeColors(
        background="#0d1117",
        text="#c9d1d9",
        primary="#58a6ff",
        positive="#3fb950",
        negative="#f85149",
        neutral="#8b949e",
        grid="#21262d",
        paper="#161b22",
    ),
}


@dataclass
class TearsheetConfig:
    """Configuration for tearsheet generation."""
    title: str = "Strategy Performance Tearsheet"
    benchmark_symbol: str = "SPY"
    include_benchmark: bool = True
    theme: ThemeType = ThemeType.LIGHT
    height_per_row: int = 300
    charts_to_include: List[str] = field(default_factory=lambda: [
        "equity_curve",
        "drawdown",
        "monthly_returns",
        "returns_distribution",
        "rolling_sharpe",
        "yearly_returns",
        "trade_analysis",
    ])
    show_trades: bool = True
    rolling_window: int = 252  # Trading days for rolling metrics


@dataclass
class PerformanceMetrics:
    """Calculated performance metrics."""
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_duration: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "max_drawdown_duration": f"{self.max_drawdown_duration} days",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "volatility": f"{self.volatility:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "avg_win": f"${self.avg_win:,.2f}",
            "avg_loss": f"${self.avg_loss:,.2f}",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "best_trade": f"${self.best_trade:,.2f}",
            "worst_trade": f"${self.worst_trade:,.2f}",
            "avg_trade_duration": self.avg_trade_duration,
        }


class TearsheetGenerator:
    """
    Generates interactive performance tearsheets.

    Features:
    - Equity curve with benchmark comparison
    - Drawdown analysis
    - Monthly/yearly returns heatmaps
    - Returns distribution
    - Rolling Sharpe ratio
    - Trade analysis
    """

    def __init__(
        self,
        config: Optional[TearsheetConfig] = None,
    ):
        """
        Initialize tearsheet generator.

        Args:
            config: Tearsheet configuration
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required. Install with: pip install plotly")

        self.config = config or TearsheetConfig()
        self.theme = THEMES[self.config.theme]

    def calculate_metrics(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from returns series.

        Args:
            returns: Daily returns series
            trades: Trade history DataFrame
            risk_free_rate: Annual risk-free rate

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if returns.empty:
            return metrics

        # Basic returns metrics
        cumulative = (1 + returns).cumprod()
        metrics.total_return = cumulative.iloc[-1] - 1

        # CAGR
        years = len(returns) / 252
        if years > 0 and cumulative.iloc[-1] > 0:
            metrics.cagr = (cumulative.iloc[-1] ** (1 / years)) - 1

        # Volatility (annualized)
        metrics.volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252
        if returns.std() > 0:
            metrics.sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252)

        # Maximum Drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        metrics.max_drawdown = drawdown.min()

        # Max Drawdown Duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        if drawdown_periods:
            metrics.max_drawdown_duration = max(drawdown_periods)

        # Calmar Ratio
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = metrics.cagr / abs(metrics.max_drawdown)

        # Trade-based metrics
        if trades is not None and not trades.empty:
            pnl_col = 'pnl' if 'pnl' in trades.columns else 'profit'
            if pnl_col in trades.columns:
                trade_pnl = trades[pnl_col]
                metrics.total_trades = len(trade_pnl)
                metrics.winning_trades = len(trade_pnl[trade_pnl > 0])
                metrics.losing_trades = len(trade_pnl[trade_pnl < 0])

                if metrics.total_trades > 0:
                    metrics.win_rate = metrics.winning_trades / metrics.total_trades

                winning = trade_pnl[trade_pnl > 0]
                losing = trade_pnl[trade_pnl < 0]

                if len(winning) > 0:
                    metrics.avg_win = winning.mean()
                    metrics.best_trade = winning.max()

                if len(losing) > 0:
                    metrics.avg_loss = abs(losing.mean())
                    metrics.worst_trade = losing.min()

                # Profit Factor
                total_wins = winning.sum() if len(winning) > 0 else 0
                total_losses = abs(losing.sum()) if len(losing) > 0 else 0
                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses

        return metrics

    def create_equity_curve(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> go.Figure:
        """Create equity curve chart."""
        cumulative = (1 + returns).cumprod() * 100  # Start at 100

        fig = go.Figure()

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            name="Strategy",
            line={"color": self.theme.primary, "width": 2},
            hovertemplate="Date: %{x}<br>Value: $%{y:.2f}<extra></extra>",
        ))

        # Benchmark
        if benchmark_returns is not None and self.config.include_benchmark:
            bench_cumulative = (1 + benchmark_returns).cumprod() * 100
            fig.add_trace(go.Scatter(
                x=bench_cumulative.index,
                y=bench_cumulative.values,
                name=self.config.benchmark_symbol,
                line={"color": self.theme.neutral, "width": 1, "dash": "dash"},
                hovertemplate="Date: %{x}<br>Value: $%{y:.2f}<extra></extra>",
            ))

        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template=self.config.theme.value,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            xaxis={
                "rangeselector": {
                    "buttons": [
                        {"count": 1, "label": "1M", "step": "month"},
                        {"count": 6, "label": "6M", "step": "month"},
                        {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                        {"count": 1, "label": "1Y", "step": "year"},
                        {"step": "all", "label": "ALL"},
                    ]
                },
                "rangeslider": {"visible": True},
            },
        )

        return fig

    def create_drawdown_chart(
        self,
        returns: pd.Series,
    ) -> go.Figure:
        """Create drawdown chart."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max * 100

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            name="Drawdown",
            line={"color": self.theme.negative, "width": 1},
            fillcolor=f"rgba({int(self.theme.negative[1:3], 16)}, "
                      f"{int(self.theme.negative[3:5], 16)}, "
                      f"{int(self.theme.negative[5:7], 16)}, 0.3)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template=self.config.theme.value,
            hovermode="x unified",
        )

        return fig

    def create_monthly_returns_heatmap(
        self,
        returns: pd.Series,
    ) -> go.Figure:
        """Create monthly returns heatmap."""
        # Resample to monthly
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values * 100,
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        # Color scale
        colorscale = [
            [0, self.theme.negative],
            [0.5, "#ffffff" if self.config.theme in [ThemeType.LIGHT, ThemeType.NAUTILUS] else "#333333"],
            [1, self.theme.positive],
        ]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Monthly Returns (%)",
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.config.theme.value,
        )

        return fig

    def create_returns_distribution(
        self,
        returns: pd.Series,
    ) -> go.Figure:
        """Create returns distribution histogram."""
        fig = go.Figure()

        # Separate positive and negative returns
        positive = returns[returns >= 0] * 100
        negative = returns[returns < 0] * 100

        fig.add_trace(go.Histogram(
            x=positive,
            name="Positive",
            marker_color=self.theme.positive,
            opacity=0.7,
            nbinsx=50,
        ))

        fig.add_trace(go.Histogram(
            x=negative,
            name="Negative",
            marker_color=self.theme.negative,
            opacity=0.7,
            nbinsx=50,
        ))

        # Add median line
        median = returns.median() * 100
        fig.add_vline(
            x=median,
            line_dash="dash",
            line_color=self.theme.primary,
            annotation_text=f"Median: {median:.2f}%",
        )

        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template=self.config.theme.value,
            barmode="overlay",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )

        return fig

    def create_rolling_sharpe(
        self,
        returns: pd.Series,
        window: Optional[int] = None,
    ) -> go.Figure:
        """Create rolling Sharpe ratio chart."""
        window = window or self.config.rolling_window

        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name=f"{window}-Day Rolling Sharpe",
            line={"color": self.theme.primary, "width": 2},
            hovertemplate="Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
        ))

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color=self.theme.neutral)

        # Add 1.0 and 2.0 reference lines
        fig.add_hline(y=1.0, line_dash="dot", line_color=self.theme.positive,
                     annotation_text="Good (1.0)")
        fig.add_hline(y=2.0, line_dash="dot", line_color=self.theme.positive,
                     annotation_text="Excellent (2.0)")

        fig.update_layout(
            title=f"Rolling Sharpe Ratio ({window}-Day Window)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template=self.config.theme.value,
            hovermode="x unified",
        )

        return fig

    def create_yearly_returns(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> go.Figure:
        """Create yearly returns bar chart."""
        yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100

        fig = go.Figure()

        # Strategy bars
        colors = [self.theme.positive if r >= 0 else self.theme.negative for r in yearly.values]
        fig.add_trace(go.Bar(
            x=yearly.index.year,
            y=yearly.values,
            name="Strategy",
            marker_color=colors,
            hovertemplate="Year: %{x}<br>Return: %{y:.2f}%<extra></extra>",
        ))

        # Benchmark bars
        if benchmark_returns is not None and self.config.include_benchmark:
            bench_yearly = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
            fig.add_trace(go.Bar(
                x=bench_yearly.index.year,
                y=bench_yearly.values,
                name=self.config.benchmark_symbol,
                marker_color=self.theme.neutral,
                opacity=0.5,
                hovertemplate="Year: %{x}<br>Return: %{y:.2f}%<extra></extra>",
            ))

        fig.update_layout(
            title="Yearly Returns",
            xaxis_title="Year",
            yaxis_title="Return (%)",
            template=self.config.theme.value,
            barmode="group",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        )

        return fig

    def create_trade_analysis(
        self,
        trades: pd.DataFrame,
    ) -> go.Figure:
        """Create trade analysis chart."""
        if trades.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trades to display", showarrow=False)
            return fig

        pnl_col = 'pnl' if 'pnl' in trades.columns else 'profit'
        if pnl_col not in trades.columns:
            fig = go.Figure()
            fig.add_annotation(text="No P&L data available", showarrow=False)
            return fig

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("P&L per Trade", "Cumulative P&L", "Win/Loss Distribution", "Trade Duration"),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "histogram"}]],
        )

        # P&L per trade
        colors = [self.theme.positive if p > 0 else self.theme.negative for p in trades[pnl_col]]
        fig.add_trace(go.Bar(
            x=list(range(len(trades))),
            y=trades[pnl_col],
            marker_color=colors,
            name="Trade P&L",
        ), row=1, col=1)

        # Cumulative P&L
        cumulative_pnl = trades[pnl_col].cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(len(trades))),
            y=cumulative_pnl,
            mode="lines",
            name="Cumulative P&L",
            line={"color": self.theme.primary},
        ), row=1, col=2)

        # Win/Loss pie
        wins = len(trades[trades[pnl_col] > 0])
        losses = len(trades[trades[pnl_col] <= 0])
        fig.add_trace(go.Pie(
            labels=["Wins", "Losses"],
            values=[wins, losses],
            marker_colors=[self.theme.positive, self.theme.negative],
        ), row=2, col=1)

        # Trade duration histogram (if available)
        if 'duration' in trades.columns:
            fig.add_trace(go.Histogram(
                x=trades['duration'],
                marker_color=self.theme.primary,
                name="Duration",
            ), row=2, col=2)

        fig.update_layout(
            title="Trade Analysis",
            template=self.config.theme.value,
            showlegend=False,
            height=600,
        )

        return fig

    def generate_tearsheet(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
    ) -> str:
        """
        Generate complete interactive tearsheet as HTML.

        Args:
            returns: Daily returns series
            benchmark_returns: Benchmark daily returns
            trades: Trade history DataFrame
            strategy_name: Name for display

        Returns:
            HTML string with complete tearsheet
        """
        # Calculate metrics
        metrics = self.calculate_metrics(returns, trades)

        # Generate charts
        charts = {}

        if "equity_curve" in self.config.charts_to_include:
            charts["equity_curve"] = self.create_equity_curve(returns, benchmark_returns)

        if "drawdown" in self.config.charts_to_include:
            charts["drawdown"] = self.create_drawdown_chart(returns)

        if "monthly_returns" in self.config.charts_to_include:
            charts["monthly_returns"] = self.create_monthly_returns_heatmap(returns)

        if "returns_distribution" in self.config.charts_to_include:
            charts["returns_distribution"] = self.create_returns_distribution(returns)

        if "rolling_sharpe" in self.config.charts_to_include:
            charts["rolling_sharpe"] = self.create_rolling_sharpe(returns)

        if "yearly_returns" in self.config.charts_to_include:
            charts["yearly_returns"] = self.create_yearly_returns(returns, benchmark_returns)

        if "trade_analysis" in self.config.charts_to_include and trades is not None:
            charts["trade_analysis"] = self.create_trade_analysis(trades)

        # Build HTML
        html = self._build_html(strategy_name, metrics, charts)

        return html

    def _build_html(
        self,
        strategy_name: str,
        metrics: PerformanceMetrics,
        charts: Dict[str, go.Figure],
    ) -> str:
        """Build complete HTML document."""
        theme_bg = self.theme.background
        theme_text = self.theme.text
        theme_primary = self.theme.primary
        theme_positive = self.theme.positive
        theme_negative = self.theme.negative

        # Convert charts to HTML divs
        chart_divs = []
        for fig in charts.values():
            chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
            chart_divs.append(f'<div class="chart-container">{chart_html}</div>')

        metrics_dict = metrics.to_dict()

        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{self.config.title} - {strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: {theme_bg};
            color: {theme_text};
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid {theme_primary};
            margin-bottom: 30px;
        }}
        .header h1 {{ color: {theme_primary}; margin-bottom: 10px; }}
        .header .timestamp {{ color: {self.theme.neutral}; font-size: 14px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: {self.theme.paper};
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card .label {{ font-size: 12px; color: {self.theme.neutral}; }}
        .metric-card .value {{ font-size: 20px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: {theme_positive}; }}
        .negative {{ color: {theme_negative}; }}
        .chart-container {{
            background: {self.theme.paper};
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }}
        @media (min-width: 1200px) {{
            .charts-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.title}</h1>
        <div class="strategy-name">{strategy_name}</div>
        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="label">Total Return</div>
            <div class="value {'positive' if metrics.total_return >= 0 else 'negative'}">{metrics_dict['total_return']}</div>
        </div>
        <div class="metric-card">
            <div class="label">CAGR</div>
            <div class="value {'positive' if metrics.cagr >= 0 else 'negative'}">{metrics_dict['cagr']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sharpe Ratio</div>
            <div class="value">{metrics_dict['sharpe_ratio']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Sortino Ratio</div>
            <div class="value">{metrics_dict['sortino_ratio']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{metrics_dict['max_drawdown']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Volatility</div>
            <div class="value">{metrics_dict['volatility']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Win Rate</div>
            <div class="value">{metrics_dict['win_rate']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Profit Factor</div>
            <div class="value">{metrics_dict['profit_factor']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Total Trades</div>
            <div class="value">{metrics_dict['total_trades']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Best Trade</div>
            <div class="value positive">{metrics_dict['best_trade']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Worst Trade</div>
            <div class="value negative">{metrics_dict['worst_trade']}</div>
        </div>
        <div class="metric-card">
            <div class="label">Calmar Ratio</div>
            <div class="value">{metrics_dict['calmar_ratio']}</div>
        </div>
    </div>

    <div class="charts-grid">
        {"".join(chart_divs)}
    </div>
</body>
</html>'''

        return html

    def save_tearsheet(
        self,
        returns: pd.Series,
        output_path: str,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
    ) -> str:
        """
        Generate and save tearsheet to file.

        Args:
            returns: Daily returns series
            output_path: Path to save HTML file
            benchmark_returns: Benchmark daily returns
            trades: Trade history DataFrame
            strategy_name: Name for display

        Returns:
            Path to saved file
        """
        html = self.generate_tearsheet(
            returns=returns,
            benchmark_returns=benchmark_returns,
            trades=trades,
            strategy_name=strategy_name,
        )

        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"Tearsheet saved to {output_path}")
        return output_path


def create_tearsheet_generator(
    theme: str = "light",
    include_benchmark: bool = True,
) -> TearsheetGenerator:
    """
    Factory function for tearsheet generator.

    Args:
        theme: Theme name (light, dark, nautilus, nautilus_dark)
        include_benchmark: Whether to include benchmark comparison

    Returns:
        Configured TearsheetGenerator
    """
    theme_map = {
        "light": ThemeType.LIGHT,
        "dark": ThemeType.DARK,
        "nautilus": ThemeType.NAUTILUS,
        "nautilus_dark": ThemeType.NAUTILUS_DARK,
    }

    config = TearsheetConfig(
        theme=theme_map.get(theme.lower(), ThemeType.LIGHT),
        include_benchmark=include_benchmark,
    )

    return TearsheetGenerator(config=config)
