"""
PDF Report Generator.

Generates professional PDF performance reports from trading data.
Uses TearsheetGenerator for charts, weasyprint for PDF rendering.
"""

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import kaleido  # noqa
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class PDFReportConfig:
    """Configuration for PDF report sections."""
    title: str = "Trading Performance Report"
    subtitle: str = ""
    cover_page: bool = True
    executive_summary: bool = True
    equity_curve: bool = True
    drawdown_chart: bool = True
    monthly_heatmap: bool = True
    returns_distribution: bool = True
    rolling_sharpe: bool = True
    trade_log: bool = True
    trade_log_max_rows: int = 50
    risk_metrics: bool = True
    tax_summary: bool = False
    page_size: str = "A4"  # A4 or Letter
    orientation: str = "portrait"
    theme: str = "light"  # light or dark


class PDFReportGenerator:
    """Generate professional PDF performance reports."""

    def __init__(self, config: Optional[PDFReportConfig] = None):
        self.config = config or PDFReportConfig()

    def generate(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_name: str = "Strategy",
        output_path: Optional[str] = None,
    ) -> bytes:
        """
        Generate a PDF report.

        Args:
            returns: Daily returns series (index=dates, values=pct returns)
            trades: DataFrame with columns: symbol, side, qty, entry_price, exit_price, pnl, entry_date, exit_date
            benchmark_returns: Benchmark daily returns for comparison
            strategy_name: Name of the strategy
            output_path: Optional file path to save PDF

        Returns:
            PDF bytes
        """
        # Build HTML content
        html = self._build_html(returns, trades, benchmark_returns, strategy_name)

        # Convert to PDF
        if WEASYPRINT_AVAILABLE:
            pdf_bytes = weasyprint.HTML(string=html).write_pdf()
        else:
            # Return HTML as bytes if weasyprint not available
            pdf_bytes = html.encode('utf-8')
            logger.warning("weasyprint not available, returning HTML instead of PDF")

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pdf_bytes)

        return pdf_bytes

    def _build_html(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame],
        benchmark_returns: Optional[pd.Series],
        strategy_name: str,
    ) -> str:
        """Build HTML report content."""
        sections = []

        # Calculate metrics
        metrics = self._calculate_metrics(returns, benchmark_returns)

        # CSS
        css = self._get_css()

        # Cover page
        if self.config.cover_page:
            sections.append(self._cover_page(strategy_name, returns, metrics))

        # Executive summary
        if self.config.executive_summary:
            sections.append(self._executive_summary(metrics, strategy_name))

        # Charts section
        chart_sections = []

        if self.config.equity_curve and PLOTLY_AVAILABLE:
            chart_sections.append(self._equity_curve_section(returns, benchmark_returns))

        if self.config.drawdown_chart and PLOTLY_AVAILABLE:
            chart_sections.append(self._drawdown_section(returns))

        if self.config.monthly_heatmap and PLOTLY_AVAILABLE:
            chart_sections.append(self._monthly_heatmap_section(returns))

        if self.config.returns_distribution and PLOTLY_AVAILABLE:
            chart_sections.append(self._returns_distribution_section(returns))

        if self.config.rolling_sharpe and PLOTLY_AVAILABLE:
            chart_sections.append(self._rolling_sharpe_section(returns))

        sections.extend(chart_sections)

        # Trade log
        if self.config.trade_log and trades is not None and len(trades) > 0:
            sections.append(self._trade_log_section(trades))

        # Risk metrics
        if self.config.risk_metrics:
            sections.append(self._risk_metrics_section(metrics))

        # Assemble HTML
        body = '\n'.join(sections)
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.config.title}</title>
    <style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""
        return html

    def _calculate_metrics(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate performance metrics from returns series."""
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

        # Annualized return
        n_days = len(returns)
        ann_factor = 252 / max(n_days, 1)
        ann_return = (1 + total_return) ** ann_factor - 1 if n_days > 0 else 0

        # Volatility
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(252)

        # Sharpe ratio
        sharpe = (returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0

        # Sortino ratio
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Win rate (daily)
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        # Best/worst days
        best_day = returns.max() if len(returns) > 0 else 0
        worst_day = returns.min() if len(returns) > 0 else 0

        # VaR 95%
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0

        metrics = {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
            'var_95': var_95,
            'total_days': total_days,
            'start_date': returns.index[0] if len(returns) > 0 else None,
            'end_date': returns.index[-1] if len(returns) > 0 else None,
        }

        if benchmark is not None and len(benchmark) > 0:
            bench_cum = (1 + benchmark).cumprod()
            bench_return = bench_cum.iloc[-1] - 1
            metrics['benchmark_return'] = bench_return
            metrics['excess_return'] = total_return - bench_return
            # Beta
            aligned = pd.concat([returns, benchmark], axis=1, join='inner')
            if len(aligned) > 1:
                cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                beta = cov[0][1] / cov[1][1] if cov[1][1] != 0 else 0
                alpha = ann_return - beta * (benchmark.mean() * 252)
                metrics['beta'] = beta
                metrics['alpha'] = alpha

        return metrics

    def _fig_to_base64(self, fig) -> str:
        """Convert Plotly figure to base64 PNG."""
        if KALEIDO_AVAILABLE:
            img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
            return base64.b64encode(img_bytes).decode('utf-8')
        else:
            # Fallback: return empty placeholder
            return ""

    def _cover_page(self, strategy_name: str, returns: pd.Series, metrics: Dict) -> str:
        start = metrics.get('start_date', '')
        end = metrics.get('end_date', '')
        total_ret = metrics.get('total_return', 0)
        return f"""
        <div class="cover-page">
            <h1>{self.config.title}</h1>
            <h2>{strategy_name}</h2>
            <p class="subtitle">{self.config.subtitle or f"Period: {start} to {end}"}</p>
            <div class="cover-stats">
                <div class="stat">
                    <span class="stat-value {'positive' if total_ret >= 0 else 'negative'}">{total_ret:.1%}</span>
                    <span class="stat-label">Total Return</span>
                </div>
                <div class="stat">
                    <span class="stat-value">{metrics.get('sharpe_ratio', 0):.2f}</span>
                    <span class="stat-label">Sharpe Ratio</span>
                </div>
                <div class="stat">
                    <span class="stat-value negative">{metrics.get('max_drawdown', 0):.1%}</span>
                    <span class="stat-label">Max Drawdown</span>
                </div>
            </div>
            <p class="generated-at">Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
        </div>
        """

    def _executive_summary(self, metrics: Dict, strategy_name: str) -> str:
        rows = [
            ('Total Return', f"{metrics.get('total_return', 0):.2%}"),
            ('Annualized Return', f"{metrics.get('ann_return', 0):.2%}"),
            ('Annualized Volatility', f"{metrics.get('ann_volatility', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
            ('Win Rate (Daily)', f"{metrics.get('win_rate', 0):.1%}"),
            ('Best Day', f"{metrics.get('best_day', 0):.2%}"),
            ('Worst Day', f"{metrics.get('worst_day', 0):.2%}"),
            ('VaR (95%)', f"{metrics.get('var_95', 0):.2%}"),
            ('Trading Days', str(metrics.get('total_days', 0))),
        ]

        if 'benchmark_return' in metrics:
            rows.extend([
                ('Benchmark Return', f"{metrics.get('benchmark_return', 0):.2%}"),
                ('Excess Return', f"{metrics.get('excess_return', 0):.2%}"),
                ('Beta', f"{metrics.get('beta', 0):.2f}"),
                ('Alpha', f"{metrics.get('alpha', 0):.2%}"),
            ])

        table_rows = '\n'.join(
            f'<tr><td>{name}</td><td>{val}</td></tr>' for name, val in rows
        )

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <table class="metrics-table">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """

    def _equity_curve_section(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> str:
        cumulative = (1 + returns).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values, name='Strategy', line=dict(color='#1f77b4')))
        if benchmark is not None:
            bench_cum = (1 + benchmark).cumprod()
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, name='Benchmark', line=dict(color='#ff7f0e', dash='dash')))
        fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Growth of $1', template='plotly_white', height=400)
        img = self._fig_to_base64(fig)
        if img:
            return f'<div class="section"><h2>Equity Curve</h2><img src="data:image/png;base64,{img}" class="chart"></div>'
        return '<div class="section"><h2>Equity Curve</h2><p>Chart unavailable (kaleido not installed)</p></div>'

    def _drawdown_section(self, returns: pd.Series) -> str:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', fillcolor='rgba(255,0,0,0.2)', line=dict(color='red')))
        fig.update_layout(title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown', template='plotly_white', height=300)
        img = self._fig_to_base64(fig)
        if img:
            return f'<div class="section"><h2>Drawdown</h2><img src="data:image/png;base64,{img}" class="chart"></div>'
        return '<div class="section"><h2>Drawdown</h2><p>Chart unavailable</p></div>'

    def _monthly_heatmap_section(self, returns: pd.Series) -> str:
        # Build monthly returns matrix
        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly) == 0:
            return '<div class="section"><h2>Monthly Returns</h2><p>Insufficient data</p></div>'

        # Create pivot table: year x month
        df = pd.DataFrame({'return': monthly})
        df['year'] = df.index.year
        df['month'] = df.index.month
        pivot = df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values * 100,
            x=pivot.columns,
            y=[str(y) for y in pivot.index],
            colorscale='RdYlGn',
            text=[[f'{v:.1f}%' if not np.isnan(v) else '' for v in row] for row in pivot.values * 100],
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        fig.update_layout(title='Monthly Returns (%)', template='plotly_white', height=max(200, len(pivot) * 40 + 100))
        img = self._fig_to_base64(fig)
        if img:
            return f'<div class="section"><h2>Monthly Returns</h2><img src="data:image/png;base64,{img}" class="chart"></div>'
        return '<div class="section"><h2>Monthly Returns</h2><p>Chart unavailable</p></div>'

    def _returns_distribution_section(self, returns: pd.Series) -> str:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns.values * 100, nbinsx=50, marker_color='#1f77b4'))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(title='Returns Distribution', xaxis_title='Daily Return (%)', yaxis_title='Frequency', template='plotly_white', height=300)
        img = self._fig_to_base64(fig)
        if img:
            return f'<div class="section"><h2>Returns Distribution</h2><img src="data:image/png;base64,{img}" class="chart"></div>'
        return '<div class="section"><h2>Returns Distribution</h2><p>Chart unavailable</p></div>'

    def _rolling_sharpe_section(self, returns: pd.Series) -> str:
        if len(returns) < 63:
            return '<div class="section"><h2>Rolling Sharpe Ratio</h2><p>Insufficient data (need 63+ days)</p></div>'
        rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, line=dict(color='#1f77b4')))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Target")
        fig.update_layout(title='Rolling Sharpe Ratio (63-day)', xaxis_title='Date', yaxis_title='Sharpe Ratio', template='plotly_white', height=300)
        img = self._fig_to_base64(fig)
        if img:
            return f'<div class="section"><h2>Rolling Sharpe Ratio</h2><img src="data:image/png;base64,{img}" class="chart"></div>'
        return '<div class="section"><h2>Rolling Sharpe Ratio</h2><p>Chart unavailable</p></div>'

    def _trade_log_section(self, trades: pd.DataFrame) -> str:
        max_rows = self.config.trade_log_max_rows
        display_trades = trades.head(max_rows)

        # Build table
        cols = [c for c in ['symbol', 'side', 'qty', 'entry_price', 'exit_price', 'pnl', 'entry_date', 'exit_date'] if c in display_trades.columns]

        header = ''.join(f'<th>{c.replace("_", " ").title()}</th>' for c in cols)
        rows = []
        for _, row in display_trades.iterrows():
            cells = []
            for c in cols:
                val = row.get(c, '')
                if c == 'pnl' and val is not None:
                    css_class = 'positive' if float(val) >= 0 else 'negative'
                    cells.append(f'<td class="{css_class}">${float(val):,.2f}</td>')
                elif c in ('entry_price', 'exit_price') and val is not None:
                    cells.append(f'<td>${float(val):,.2f}</td>')
                else:
                    cells.append(f'<td>{val}</td>')
            rows.append(f'<tr>{"".join(cells)}</tr>')

        total_note = f'<p class="note">Showing {min(max_rows, len(trades))} of {len(trades)} trades</p>' if len(trades) > max_rows else ''

        return f"""
        <div class="section">
            <h2>Trade Log</h2>
            <table class="trade-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
            {total_note}
        </div>
        """

    def _risk_metrics_section(self, metrics: Dict) -> str:
        rows = [
            ('Value at Risk (95%)', f"{metrics.get('var_95', 0):.2%}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
            ('Annualized Volatility', f"{metrics.get('ann_volatility', 0):.2%}"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
        ]
        if 'beta' in metrics:
            rows.append(('Beta', f"{metrics.get('beta', 0):.2f}"))

        table_rows = '\n'.join(f'<tr><td>{name}</td><td>{val}</td></tr>' for name, val in rows)

        return f"""
        <div class="section">
            <h2>Risk Metrics</h2>
            <table class="metrics-table">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """

    def _get_css(self) -> str:
        """Return CSS styles for the report."""
        return """
            @page { size: A4; margin: 2cm; }
            body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #333; line-height: 1.6; }
            .cover-page { text-align: center; page-break-after: always; padding-top: 200px; }
            .cover-page h1 { font-size: 36px; color: #1a1a2e; margin-bottom: 10px; }
            .cover-page h2 { font-size: 24px; color: #16213e; font-weight: 300; }
            .cover-page .subtitle { color: #666; font-size: 16px; margin-top: 20px; }
            .cover-stats { display: flex; justify-content: center; gap: 60px; margin-top: 60px; }
            .stat { text-align: center; }
            .stat-value { display: block; font-size: 32px; font-weight: 700; }
            .stat-label { display: block; font-size: 14px; color: #666; margin-top: 5px; }
            .generated-at { color: #999; font-size: 12px; margin-top: 80px; }
            .section { page-break-inside: avoid; margin-bottom: 30px; }
            .section h2 { color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; font-size: 20px; }
            .metrics-table { width: 100%; border-collapse: collapse; }
            .metrics-table th, .metrics-table td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
            .metrics-table th { background: #f5f5f5; font-weight: 600; }
            .trade-table { width: 100%; border-collapse: collapse; font-size: 12px; }
            .trade-table th, .trade-table td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #eee; }
            .trade-table th { background: #f5f5f5; font-weight: 600; }
            .chart { width: 100%; max-width: 800px; display: block; margin: 10px auto; }
            .positive { color: #00c805; }
            .negative { color: #ff2d21; }
            .note { color: #999; font-size: 12px; font-style: italic; }
        """
