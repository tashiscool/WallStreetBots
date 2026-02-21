"""
Options Payoff Visualization.

Generates interactive payoff diagrams and Greeks dashboards for option strategies.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import plotly.io as pio
    KALEIDO_AVAILABLE = True
    try:
        pio.kaleido.scope
    except Exception:
        KALEIDO_AVAILABLE = False
except (ImportError, AttributeError):
    KALEIDO_AVAILABLE = False


from .exotic_spreads import (
    OptionSpread, SpreadLeg, LegType, SpreadType
)


def _leg_pnl_at_price(leg: SpreadLeg, price: float) -> float:
    """Calculate P&L for a single option leg at a given underlying price at expiry."""
    strike = float(leg.strike)
    premium = float(leg.premium) if leg.premium else 0.0
    contracts = leg.contracts  # Positive for long, negative for short
    multiplier = 100  # Standard option multiplier

    if leg.leg_type in (LegType.LONG_CALL, LegType.SHORT_CALL):
        # Call payoff
        intrinsic = max(0, price - strike)
    else:
        # Put payoff
        intrinsic = max(0, strike - price)

    if leg.leg_type in (LegType.LONG_CALL, LegType.LONG_PUT):
        # Long: pay premium, receive intrinsic
        pnl = (intrinsic - premium) * abs(contracts) * multiplier
    else:
        # Short: receive premium, pay intrinsic
        pnl = (premium - intrinsic) * abs(contracts) * multiplier

    return pnl


def _black_scholes_value(
    price: float,
    strike: float,
    days_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    is_call: bool,
) -> float:
    """Simple Black-Scholes option value for pre-expiry curves."""
    if days_to_expiry <= 0:
        if is_call:
            return max(0, price - strike)
        return max(0, strike - price)

    t = days_to_expiry / 365.0
    if volatility <= 0 or t <= 0:
        if is_call:
            return max(0, price - strike)
        return max(0, strike - price)

    try:
        from math import log, sqrt, exp, erf

        d1 = (log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * t) / (volatility * sqrt(t))
        d2 = d1 - volatility * sqrt(t)

        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + erf(x / sqrt(2)))

        if is_call:
            return price * norm_cdf(d1) - strike * exp(-risk_free_rate * t) * norm_cdf(d2)
        else:
            return strike * exp(-risk_free_rate * t) * norm_cdf(-d2) - price * norm_cdf(-d1)
    except (ValueError, ZeroDivisionError):
        if is_call:
            return max(0, price - strike)
        return max(0, strike - price)


def _leg_value_before_expiry(
    leg: SpreadLeg,
    price: float,
    days_to_expiry: float,
    volatility: float = 0.30,
    risk_free_rate: float = 0.05,
) -> float:
    """Calculate P&L for a leg at a given price before expiry."""
    strike = float(leg.strike)
    premium = float(leg.premium) if leg.premium else 0.0
    is_call = leg.leg_type in (LegType.LONG_CALL, LegType.SHORT_CALL)
    multiplier = 100

    current_value = _black_scholes_value(price, strike, days_to_expiry, volatility, risk_free_rate, is_call)

    if leg.leg_type in (LegType.LONG_CALL, LegType.LONG_PUT):
        pnl = (current_value - premium) * abs(leg.contracts) * multiplier
    else:
        pnl = (premium - current_value) * abs(leg.contracts) * multiplier

    return pnl


@dataclass
class PayoffDiagramConfig:
    """Configuration for payoff diagram."""
    price_range_pct: float = 0.30  # ±30% from current price
    num_points: int = 200
    pre_expiry_dte: List[int] = field(default_factory=lambda: [30, 15, 7, 1])
    volatility: float = 0.30
    risk_free_rate: float = 0.05
    show_breakevens: bool = True
    show_max_profit_loss: bool = True
    chart_height: int = 500
    chart_width: int = 900


class PayoffDiagramGenerator:
    """Generate payoff diagrams for option spreads."""

    def __init__(self, config: Optional[PayoffDiagramConfig] = None):
        self.config = config or PayoffDiagramConfig()

    def generate(
        self,
        spread: OptionSpread,
        current_price: float,
        days_to_expiry: int = 30,
        output_format: str = "html",
    ) -> str:
        """
        Generate payoff diagram.

        Args:
            spread: The option spread to visualize
            current_price: Current underlying price
            days_to_expiry: Days to expiration
            output_format: 'html' for interactive HTML, 'png_base64' for static image

        Returns:
            HTML string or base64 PNG string
        """
        if not PLOTLY_AVAILABLE:
            return '<p>Plotly not available. Install with: pip install plotly</p>'

        # Calculate price range
        low = current_price * (1 - self.config.price_range_pct)
        high = current_price * (1 + self.config.price_range_pct)
        prices = np.linspace(low, high, self.config.num_points)

        fig = go.Figure()

        # Expiry P&L line
        expiry_pnl = np.array([
            sum(_leg_pnl_at_price(leg, p) for leg in spread.legs)
            for p in prices
        ])

        fig.add_trace(go.Scatter(
            x=prices, y=expiry_pnl,
            name='At Expiry',
            line={"color": '#1f77b4', "width": 3},
        ))

        # Pre-expiry curves
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, dte in enumerate(self.config.pre_expiry_dte):
            if dte >= days_to_expiry:
                continue
            pre_pnl = np.array([
                sum(_leg_value_before_expiry(
                    leg, p, dte,
                    self.config.volatility,
                    self.config.risk_free_rate,
                ) for leg in spread.legs)
                for p in prices
            ])
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=prices, y=pre_pnl,
                name=f'{dte} DTE',
                line={"color": color, "width": 1.5, "dash": 'dash'},
            ))

        # Zero line
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)

        # Current price vertical line
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray",
                      annotation_text=f"Current: ${current_price:.2f}")

        # Breakeven points
        if self.config.show_breakevens:
            breakevens = spread.get_breakeven_points()
            for be in breakevens:
                be_float = float(be)
                if low <= be_float <= high:
                    fig.add_vline(x=be_float, line_dash="dash", line_color="orange",
                                  annotation_text=f"BE: ${be_float:.2f}")

        # Max profit/loss annotations
        if self.config.show_max_profit_loss:
            max_profit = spread.get_max_profit()
            max_loss = spread.get_max_loss()
            annotations = []
            if max_profit is not None:
                mp = float(max_profit)
                annotations.append(f"Max Profit: ${mp:,.0f}")
            if max_loss is not None:
                ml = float(max_loss)
                annotations.append(f"Max Loss: ${ml:,.0f}")
            if annotations:
                fig.add_annotation(
                    text="<br>".join(annotations),
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font={"size": 12},
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                )

        # Shade profit/loss zones
        profit_mask = expiry_pnl > 0
        loss_mask = expiry_pnl < 0

        fig.add_trace(go.Scatter(
            x=prices[profit_mask], y=expiry_pnl[profit_mask],
            fill='tozeroy', fillcolor='rgba(0,200,5,0.1)',
            line={"width": 0}, showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=prices[loss_mask], y=expiry_pnl[loss_mask],
            fill='tozeroy', fillcolor='rgba(255,45,33,0.1)',
            line={"width": 0}, showlegend=False, hoverinfo='skip',
        ))

        # Layout
        fig.update_layout(
            title=f'{spread.spread_type.value.replace("_", " ").title()} - {spread.ticker}',
            xaxis_title='Underlying Price ($)',
            yaxis_title='P&L ($)',
            template='plotly_white',
            height=self.config.chart_height,
            width=self.config.chart_width,
            hovermode='x unified',
            legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        )

        if output_format == 'png_base64' and KALEIDO_AVAILABLE:
            import base64
            img_bytes = fig.to_image(format="png", width=self.config.chart_width, height=self.config.chart_height, scale=2)
            return base64.b64encode(img_bytes).decode('utf-8')

        return fig.to_html(include_plotlyjs='cdn', full_html=False)


class GreeksDashboard:
    """Generate Greeks visualization dashboard."""

    def generate(
        self,
        spread: OptionSpread,
        current_price: float,
        days_to_expiry: int = 30,
        volatility: float = 0.30,
        risk_free_rate: float = 0.05,
        output_format: str = "html",
    ) -> str:
        """
        Generate Greeks dashboard with multiple panels.

        Args:
            spread: The option spread
            current_price: Current underlying price
            days_to_expiry: Days to expiration
            volatility: Implied volatility
            risk_free_rate: Risk-free rate
            output_format: 'html' or 'png_base64'

        Returns:
            HTML string or base64 PNG
        """
        if not PLOTLY_AVAILABLE:
            return '<p>Plotly not available</p>'

        price_range_pct = 0.20
        low = current_price * (1 - price_range_pct)
        high = current_price * (1 + price_range_pct)
        prices = np.linspace(low, high, 100)

        # Calculate Greeks across price range
        deltas = []
        gammas = []
        thetas = []
        vegas = []

        for p in prices:
            total_delta = 0
            total_gamma = 0
            total_theta = 0
            total_vega = 0

            for leg in spread.legs:
                strike = float(leg.strike)
                is_call = leg.leg_type in (LegType.LONG_CALL, LegType.SHORT_CALL)
                is_long = leg.leg_type in (LegType.LONG_CALL, LegType.LONG_PUT)
                sign = 1 if is_long else -1
                contracts = abs(leg.contracts)

                d, g, t, v = self._calculate_greeks(
                    p, strike, days_to_expiry, volatility, risk_free_rate, is_call
                )
                total_delta += d * sign * contracts
                total_gamma += g * sign * contracts
                total_theta += t * sign * contracts
                total_vega += v * sign * contracts

            deltas.append(total_delta)
            gammas.append(total_gamma)
            thetas.append(total_theta)
            vegas.append(total_vega)

        # Create multi-panel figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta ($/day)', 'Vega ($/1% IV)'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        # Delta
        fig.add_trace(go.Scatter(x=prices, y=deltas, name='Delta', line={"color": '#1f77b4'}), row=1, col=1)
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="lightgray", row=1, col=1)

        # Gamma
        fig.add_trace(go.Scatter(x=prices, y=gammas, name='Gamma', line={"color": '#ff7f0e'}), row=1, col=2)
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray", row=1, col=2)

        # Theta
        fig.add_trace(go.Scatter(x=prices, y=thetas, name='Theta', line={"color": '#2ca02c'}), row=2, col=1)
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray", row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="lightgray", row=2, col=1)

        # Vega
        fig.add_trace(go.Scatter(x=prices, y=vegas, name='Vega', line={"color": '#d62728'}), row=2, col=2)
        fig.add_vline(x=current_price, line_dash="dot", line_color="gray", row=2, col=2)

        fig.update_layout(
            title=f'Greeks Dashboard - {spread.ticker} {spread.spread_type.value.replace("_", " ").title()}',
            template='plotly_white',
            height=700,
            width=1000,
            showlegend=False,
        )

        if output_format == 'png_base64' and KALEIDO_AVAILABLE:
            import base64
            img_bytes = fig.to_image(format="png", width=1000, height=700, scale=2)
            return base64.b64encode(img_bytes).decode('utf-8')

        return fig.to_html(include_plotlyjs='cdn', full_html=False)

    def _calculate_greeks(
        self,
        price: float,
        strike: float,
        days_to_expiry: int,
        volatility: float,
        risk_free_rate: float,
        is_call: bool,
    ) -> Tuple[float, float, float, float]:
        """Calculate delta, gamma, theta, vega for a single option."""
        t = days_to_expiry / 365.0
        if t <= 0 or volatility <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        try:
            from math import log, sqrt, exp, erf, pi

            def norm_cdf(x):
                return 0.5 * (1 + erf(x / sqrt(2)))

            def norm_pdf(x):
                return exp(-0.5 * x * x) / sqrt(2 * pi)

            d1 = (log(price / strike) + (risk_free_rate + 0.5 * volatility ** 2) * t) / (volatility * sqrt(t))
            d2 = d1 - volatility * sqrt(t)

            # Delta
            if is_call:
                delta = norm_cdf(d1)
            else:
                delta = norm_cdf(d1) - 1

            # Gamma (same for calls and puts)
            gamma = norm_pdf(d1) / (price * volatility * sqrt(t))

            # Theta (per day)
            first_term = -(price * norm_pdf(d1) * volatility) / (2 * sqrt(t))
            if is_call:
                theta = (first_term - risk_free_rate * strike * exp(-risk_free_rate * t) * norm_cdf(d2)) / 365
            else:
                theta = (first_term + risk_free_rate * strike * exp(-risk_free_rate * t) * norm_cdf(-d2)) / 365

            # Vega (per 1% move in IV)
            vega = price * sqrt(t) * norm_pdf(d1) / 100

            return (delta, gamma, theta * 100, vega)  # Scale theta to dollar terms per contract
        except (ValueError, ZeroDivisionError):
            return (0.0, 0.0, 0.0, 0.0)


def generate_pnl_heatmap(
    spread: OptionSpread,
    current_price: float,
    days_to_expiry: int = 30,
    volatility: float = 0.30,
    output_format: str = "html",
) -> str:
    """
    Generate P&L heatmap (price x implied volatility).

    Returns HTML or base64 PNG.
    """
    if not PLOTLY_AVAILABLE:
        return '<p>Plotly not available</p>'

    # Price range
    prices = np.linspace(current_price * 0.8, current_price * 1.2, 50)
    # IV range
    ivs = np.linspace(max(0.05, volatility * 0.5), volatility * 2.0, 30)

    pnl_matrix = np.zeros((len(ivs), len(prices)))

    for i, iv in enumerate(ivs):
        for j, p in enumerate(prices):
            total = sum(
                _leg_value_before_expiry(leg, p, days_to_expiry, iv, 0.05)
                for leg in spread.legs
            )
            pnl_matrix[i, j] = total

    fig = go.Figure(data=go.Heatmap(
        z=pnl_matrix,
        x=[f'${p:.0f}' for p in prices],
        y=[f'{iv:.0%}' for iv in ivs],
        colorscale='RdYlGn',
        colorbar={"title": 'P&L ($)'},
    ))

    fig.update_layout(
        title=f'P&L Heatmap - {spread.ticker} (Price x IV)',
        xaxis_title='Underlying Price',
        yaxis_title='Implied Volatility',
        template='plotly_white',
        height=500,
        width=800,
    )

    if output_format == 'png_base64' and KALEIDO_AVAILABLE:
        import base64
        img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')

    return fig.to_html(include_plotlyjs='cdn', full_html=False)
