"""
Plot Configurator for Trading Strategies.

Ported from freqtrade's plot configuration system.
Allows users to configure indicators, overlays, and chart layouts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime


class PlotType(Enum):
    """Types of plots available."""
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    AREA = "area"
    CANDLESTICK = "candlestick"
    OHLC = "ohlc"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"


class PlotLocation(Enum):
    """Where to place the plot."""
    MAIN = "main"  # On price chart
    SUBPLOT1 = "subplot1"  # First subplot below
    SUBPLOT2 = "subplot2"  # Second subplot
    SUBPLOT3 = "subplot3"  # Third subplot
    OVERLAY = "overlay"  # Overlay on main


class IndicatorCategory(Enum):
    """Categories of technical indicators."""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CUSTOM = "custom"


@dataclass
class PlotIndicator:
    """Configuration for a single indicator plot."""
    name: str
    column: str
    plot_type: PlotType = PlotType.LINE
    location: PlotLocation = PlotLocation.MAIN
    color: str = "#2196F3"
    fill: Optional[str] = None
    line_width: int = 1
    opacity: float = 1.0
    secondary_y: bool = False
    category: IndicatorCategory = IndicatorCategory.CUSTOM
    enabled: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "column": self.column,
            "plot_type": self.plot_type.value,
            "location": self.location.value,
            "color": self.color,
            "fill": self.fill,
            "line_width": self.line_width,
            "opacity": self.opacity,
            "secondary_y": self.secondary_y,
            "category": self.category.value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotIndicator":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            column=data["column"],
            plot_type=PlotType(data.get("plot_type", "line")),
            location=PlotLocation(data.get("location", "main")),
            color=data.get("color", "#2196F3"),
            fill=data.get("fill"),
            line_width=data.get("line_width", 1),
            opacity=data.get("opacity", 1.0),
            secondary_y=data.get("secondary_y", False),
            category=IndicatorCategory(data.get("category", "custom")),
            enabled=data.get("enabled", True),
        )


@dataclass
class TradeMarker:
    """Configuration for trade entry/exit markers."""
    show_entries: bool = True
    show_exits: bool = True
    entry_color: str = "#4CAF50"
    exit_color: str = "#F44336"
    marker_size: int = 12
    show_profit_loss: bool = True

    def to_dict(self) -> dict:
        return {
            "show_entries": self.show_entries,
            "show_exits": self.show_exits,
            "entry_color": self.entry_color,
            "exit_color": self.exit_color,
            "marker_size": self.marker_size,
            "show_profit_loss": self.show_profit_loss,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeMarker":
        return cls(**data)


@dataclass
class PlotConfig:
    """Complete plot configuration."""
    title: str = "Strategy Analysis"
    height: int = 800
    width: Optional[int] = None  # None = responsive
    theme: str = "dark"
    show_volume: bool = True
    show_trades: bool = True
    indicators: list[PlotIndicator] = field(default_factory=list)
    trade_markers: TradeMarker = field(default_factory=TradeMarker)
    subplot_heights: list[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "height": self.height,
            "width": self.width,
            "theme": self.theme,
            "show_volume": self.show_volume,
            "show_trades": self.show_trades,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "trade_markers": self.trade_markers.to_dict(),
            "subplot_heights": self.subplot_heights,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlotConfig":
        return cls(
            title=data.get("title", "Strategy Analysis"),
            height=data.get("height", 800),
            width=data.get("width"),
            theme=data.get("theme", "dark"),
            show_volume=data.get("show_volume", True),
            show_trades=data.get("show_trades", True),
            indicators=[PlotIndicator.from_dict(i) for i in data.get("indicators", [])],
            trade_markers=TradeMarker.from_dict(data.get("trade_markers", {})),
            subplot_heights=data.get("subplot_heights", [0.6, 0.2, 0.2]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PlotConfig":
        return cls.from_dict(json.loads(json_str))


# Predefined indicator configurations
BUILTIN_INDICATORS = {
    # Moving Averages
    "sma_20": PlotIndicator(
        name="SMA 20",
        column="sma_20",
        color="#FF9800",
        category=IndicatorCategory.TREND,
    ),
    "sma_50": PlotIndicator(
        name="SMA 50",
        column="sma_50",
        color="#9C27B0",
        category=IndicatorCategory.TREND,
    ),
    "sma_200": PlotIndicator(
        name="SMA 200",
        column="sma_200",
        color="#E91E63",
        category=IndicatorCategory.TREND,
    ),
    "ema_12": PlotIndicator(
        name="EMA 12",
        column="ema_12",
        color="#00BCD4",
        category=IndicatorCategory.TREND,
    ),
    "ema_26": PlotIndicator(
        name="EMA 26",
        column="ema_26",
        color="#009688",
        category=IndicatorCategory.TREND,
    ),

    # Bollinger Bands
    "bb_upper": PlotIndicator(
        name="BB Upper",
        column="bb_upper",
        color="#607D8B",
        opacity=0.7,
        category=IndicatorCategory.VOLATILITY,
    ),
    "bb_lower": PlotIndicator(
        name="BB Lower",
        column="bb_lower",
        color="#607D8B",
        fill="tonexty",
        opacity=0.3,
        category=IndicatorCategory.VOLATILITY,
    ),
    "bb_middle": PlotIndicator(
        name="BB Middle",
        column="bb_middle",
        color="#607D8B",
        line_width=1,
        category=IndicatorCategory.VOLATILITY,
    ),

    # RSI
    "rsi": PlotIndicator(
        name="RSI",
        column="rsi",
        color="#9C27B0",
        location=PlotLocation.SUBPLOT1,
        category=IndicatorCategory.MOMENTUM,
    ),

    # MACD
    "macd": PlotIndicator(
        name="MACD",
        column="macd",
        color="#2196F3",
        location=PlotLocation.SUBPLOT2,
        category=IndicatorCategory.MOMENTUM,
    ),
    "macd_signal": PlotIndicator(
        name="MACD Signal",
        column="macd_signal",
        color="#FF9800",
        location=PlotLocation.SUBPLOT2,
        category=IndicatorCategory.MOMENTUM,
    ),
    "macd_histogram": PlotIndicator(
        name="MACD Histogram",
        column="macd_histogram",
        color="#4CAF50",
        plot_type=PlotType.BAR,
        location=PlotLocation.SUBPLOT2,
        category=IndicatorCategory.MOMENTUM,
    ),

    # Volume indicators
    "volume_sma": PlotIndicator(
        name="Volume SMA",
        column="volume_sma",
        color="#FF9800",
        location=PlotLocation.SUBPLOT1,
        category=IndicatorCategory.VOLUME,
    ),
    "obv": PlotIndicator(
        name="OBV",
        column="obv",
        color="#4CAF50",
        location=PlotLocation.SUBPLOT2,
        category=IndicatorCategory.VOLUME,
    ),

    # Volatility
    "atr": PlotIndicator(
        name="ATR",
        column="atr",
        color="#FF5722",
        location=PlotLocation.SUBPLOT2,
        category=IndicatorCategory.VOLATILITY,
    ),

    # Stochastic
    "stoch_k": PlotIndicator(
        name="Stoch %K",
        column="stoch_k",
        color="#2196F3",
        location=PlotLocation.SUBPLOT1,
        category=IndicatorCategory.MOMENTUM,
    ),
    "stoch_d": PlotIndicator(
        name="Stoch %D",
        column="stoch_d",
        color="#FF9800",
        location=PlotLocation.SUBPLOT1,
        category=IndicatorCategory.MOMENTUM,
    ),
}


class PlotConfigurator:
    """
    Interactive plot configurator for trading strategies.

    Allows users to customize chart layouts, indicators, and trade markers.
    """

    # Theme configurations
    THEMES = {
        "dark": {
            "bg_color": "#1e1e1e",
            "paper_color": "#252525",
            "grid_color": "#404040",
            "text_color": "#e0e0e0",
            "up_color": "#26a69a",
            "down_color": "#ef5350",
        },
        "light": {
            "bg_color": "#ffffff",
            "paper_color": "#f5f5f5",
            "grid_color": "#e0e0e0",
            "text_color": "#333333",
            "up_color": "#26a69a",
            "down_color": "#ef5350",
        },
        "midnight": {
            "bg_color": "#0d1117",
            "paper_color": "#161b22",
            "grid_color": "#30363d",
            "text_color": "#c9d1d9",
            "up_color": "#3fb950",
            "down_color": "#f85149",
        },
        "terminal": {
            "bg_color": "#000000",
            "paper_color": "#0a0a0a",
            "grid_color": "#1a1a1a",
            "text_color": "#00ff00",
            "up_color": "#00ff00",
            "down_color": "#ff0000",
        },
    }

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize plot configurator."""
        self.config = config or PlotConfig()
        self._data: Optional[pd.DataFrame] = None
        self._trades: Optional[pd.DataFrame] = None

    def set_data(self, data: pd.DataFrame) -> "PlotConfigurator":
        """Set OHLCV data."""
        self._data = data.copy()
        return self

    def set_trades(self, trades: pd.DataFrame) -> "PlotConfigurator":
        """Set trade data."""
        self._trades = trades.copy()
        return self

    def add_indicator(self, indicator: PlotIndicator) -> "PlotConfigurator":
        """Add an indicator to the plot."""
        self.config.indicators.append(indicator)
        return self

    def add_builtin_indicator(self, name: str) -> "PlotConfigurator":
        """Add a built-in indicator by name."""
        if name in BUILTIN_INDICATORS:
            self.config.indicators.append(BUILTIN_INDICATORS[name])
        return self

    def remove_indicator(self, name: str) -> "PlotConfigurator":
        """Remove an indicator by name."""
        self.config.indicators = [
            ind for ind in self.config.indicators if ind.name != name
        ]
        return self

    def toggle_indicator(self, name: str, enabled: bool) -> "PlotConfigurator":
        """Toggle indicator visibility."""
        for ind in self.config.indicators:
            if ind.name == name:
                ind.enabled = enabled
                break
        return self

    def set_theme(self, theme: str) -> "PlotConfigurator":
        """Set the chart theme."""
        if theme in self.THEMES:
            self.config.theme = theme
        return self

    def configure_trade_markers(
        self,
        show_entries: bool = True,
        show_exits: bool = True,
        entry_color: str = "#4CAF50",
        exit_color: str = "#F44336",
        marker_size: int = 12,
        show_profit_loss: bool = True,
    ) -> "PlotConfigurator":
        """Configure trade markers."""
        self.config.trade_markers = TradeMarker(
            show_entries=show_entries,
            show_exits=show_exits,
            entry_color=entry_color,
            exit_color=exit_color,
            marker_size=marker_size,
            show_profit_loss=show_profit_loss,
        )
        return self

    def _get_theme_colors(self) -> dict:
        """Get current theme colors."""
        return self.THEMES.get(self.config.theme, self.THEMES["dark"])

    def _count_subplots(self) -> int:
        """Count number of subplots needed."""
        locations = set()
        if self.config.show_volume:
            locations.add(PlotLocation.SUBPLOT1)

        for ind in self.config.indicators:
            if ind.enabled and ind.location != PlotLocation.MAIN:
                locations.add(ind.location)

        return len(locations) + 1  # +1 for main plot

    def _create_figure(self) -> go.Figure:
        """Create the figure with subplots."""
        theme = self._get_theme_colors()
        num_subplots = self._count_subplots()

        # Determine subplot heights
        if num_subplots == 1:
            row_heights = [1.0]
        elif num_subplots == 2:
            row_heights = [0.7, 0.3]
        elif num_subplots == 3:
            row_heights = [0.6, 0.2, 0.2]
        else:
            main_height = 0.5
            sub_height = 0.5 / (num_subplots - 1)
            row_heights = [main_height] + [sub_height] * (num_subplots - 1)

        # Create subplot specs
        specs = [[{"secondary_y": True}] for _ in range(num_subplots)]

        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            specs=specs,
        )

        # Apply theme
        fig.update_layout(
            title=self.config.title,
            height=self.config.height,
            width=self.config.width,
            paper_bgcolor=theme["paper_color"],
            plot_bgcolor=theme["bg_color"],
            font=dict(color=theme["text_color"]),
            showlegend=True,
            legend=dict(
                bgcolor=theme["paper_color"],
                bordercolor=theme["grid_color"],
            ),
            xaxis=dict(
                gridcolor=theme["grid_color"],
                zerolinecolor=theme["grid_color"],
            ),
            yaxis=dict(
                gridcolor=theme["grid_color"],
                zerolinecolor=theme["grid_color"],
            ),
        )

        return fig

    def _add_candlesticks(self, fig: go.Figure) -> None:
        """Add candlestick chart to main plot."""
        if self._data is None:
            return

        theme = self._get_theme_colors()

        fig.add_trace(
            go.Candlestick(
                x=self._data.index,
                open=self._data["open"],
                high=self._data["high"],
                low=self._data["low"],
                close=self._data["close"],
                name="Price",
                increasing_line_color=theme["up_color"],
                decreasing_line_color=theme["down_color"],
            ),
            row=1,
            col=1,
        )

    def _add_volume(self, fig: go.Figure, row: int) -> None:
        """Add volume bars."""
        if self._data is None or "volume" not in self._data.columns:
            return

        theme = self._get_theme_colors()

        # Color volume bars based on price direction
        colors = [
            theme["up_color"] if close >= open_ else theme["down_color"]
            for close, open_ in zip(self._data["close"], self._data["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=self._data.index,
                y=self._data["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
            ),
            row=row,
            col=1,
        )

    def _add_indicators(self, fig: go.Figure) -> None:
        """Add configured indicators to the plot."""
        if self._data is None:
            return

        # Map locations to row numbers
        location_rows = {
            PlotLocation.MAIN: 1,
            PlotLocation.OVERLAY: 1,
            PlotLocation.SUBPLOT1: 2,
            PlotLocation.SUBPLOT2: 3,
            PlotLocation.SUBPLOT3: 4,
        }

        for ind in self.config.indicators:
            if not ind.enabled:
                continue

            if ind.column not in self._data.columns:
                continue

            row = location_rows.get(ind.location, 1)

            # Create trace based on plot type
            if ind.plot_type == PlotType.LINE:
                trace = go.Scatter(
                    x=self._data.index,
                    y=self._data[ind.column],
                    name=ind.name,
                    mode="lines",
                    line=dict(color=ind.color, width=ind.line_width),
                    opacity=ind.opacity,
                    fill=ind.fill,
                )
            elif ind.plot_type == PlotType.SCATTER:
                trace = go.Scatter(
                    x=self._data.index,
                    y=self._data[ind.column],
                    name=ind.name,
                    mode="markers",
                    marker=dict(color=ind.color, size=5),
                    opacity=ind.opacity,
                )
            elif ind.plot_type == PlotType.BAR:
                # Color bars based on positive/negative
                colors = [
                    "#4CAF50" if v >= 0 else "#F44336"
                    for v in self._data[ind.column]
                ]
                trace = go.Bar(
                    x=self._data.index,
                    y=self._data[ind.column],
                    name=ind.name,
                    marker_color=colors,
                    opacity=ind.opacity,
                )
            elif ind.plot_type == PlotType.AREA:
                trace = go.Scatter(
                    x=self._data.index,
                    y=self._data[ind.column],
                    name=ind.name,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=ind.color, width=ind.line_width),
                    opacity=ind.opacity,
                )
            elif ind.plot_type == PlotType.HISTOGRAM:
                trace = go.Histogram(
                    x=self._data[ind.column],
                    name=ind.name,
                    marker_color=ind.color,
                    opacity=ind.opacity,
                )
            else:
                continue

            fig.add_trace(trace, row=row, col=1, secondary_y=ind.secondary_y)

    def _add_trade_markers(self, fig: go.Figure) -> None:
        """Add trade entry/exit markers."""
        if self._trades is None or not self.config.show_trades:
            return

        markers = self.config.trade_markers

        if markers.show_entries and "entry_time" in self._trades.columns:
            # Entry markers
            fig.add_trace(
                go.Scatter(
                    x=self._trades["entry_time"],
                    y=self._trades["entry_price"],
                    mode="markers",
                    name="Entry",
                    marker=dict(
                        symbol="triangle-up",
                        size=markers.marker_size,
                        color=markers.entry_color,
                    ),
                    text=self._trades.get("symbol", ""),
                    hovertemplate="Entry<br>Price: %{y:.2f}<br>%{text}",
                ),
                row=1,
                col=1,
            )

        if markers.show_exits and "exit_time" in self._trades.columns:
            # Exit markers - color by profit/loss
            if markers.show_profit_loss and "profit" in self._trades.columns:
                colors = [
                    markers.entry_color if p >= 0 else markers.exit_color
                    for p in self._trades["profit"]
                ]
            else:
                colors = markers.exit_color

            fig.add_trace(
                go.Scatter(
                    x=self._trades["exit_time"],
                    y=self._trades["exit_price"],
                    mode="markers",
                    name="Exit",
                    marker=dict(
                        symbol="triangle-down",
                        size=markers.marker_size,
                        color=colors,
                    ),
                    text=[
                        f"P/L: ${p:.2f}" if "profit" in self._trades.columns else ""
                        for p in self._trades.get("profit", [0] * len(self._trades))
                    ],
                    hovertemplate="Exit<br>Price: %{y:.2f}<br>%{text}",
                ),
                row=1,
                col=1,
            )

    def _add_reference_lines(self, fig: go.Figure) -> None:
        """Add reference lines for oscillators."""
        # RSI reference lines (30 and 70)
        for ind in self.config.indicators:
            if ind.enabled and ind.name == "RSI":
                row = 2 if ind.location == PlotLocation.SUBPLOT1 else 3
                fig.add_hline(
                    y=70, line_dash="dash", line_color="red",
                    opacity=0.5, row=row, col=1
                )
                fig.add_hline(
                    y=30, line_dash="dash", line_color="green",
                    opacity=0.5, row=row, col=1
                )
                fig.add_hline(
                    y=50, line_dash="dot", line_color="gray",
                    opacity=0.3, row=row, col=1
                )

    def generate_plot(self) -> go.Figure:
        """Generate the complete plot."""
        fig = self._create_figure()

        # Add main chart
        self._add_candlesticks(fig)

        # Add volume
        if self.config.show_volume:
            self._add_volume(fig, row=2)

        # Add indicators
        self._add_indicators(fig)

        # Add trade markers
        self._add_trade_markers(fig)

        # Add reference lines
        self._add_reference_lines(fig)

        # Configure axes
        fig.update_xaxes(rangeslider_visible=False)

        return fig

    def to_html(self, include_plotlyjs: bool = True) -> str:
        """Generate HTML representation."""
        fig = self.generate_plot()
        return fig.to_html(include_plotlyjs=include_plotlyjs)

    def save_html(self, filepath: str) -> None:
        """Save plot to HTML file."""
        html = self.to_html()
        with open(filepath, "w") as f:
            f.write(html)

    def save_config(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.config.to_json())

    @classmethod
    def load_config(cls, filepath: str) -> "PlotConfigurator":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config = PlotConfig.from_json(f.read())
        return cls(config)

    def get_available_indicators(self) -> dict[str, list[str]]:
        """Get available built-in indicators by category."""
        result = {}
        for name, ind in BUILTIN_INDICATORS.items():
            category = ind.category.value
            if category not in result:
                result[category] = []
            result[category].append(name)
        return result


def create_plot_configurator(
    data: Optional[pd.DataFrame] = None,
    trades: Optional[pd.DataFrame] = None,
    config: Optional[PlotConfig] = None,
) -> PlotConfigurator:
    """Factory function to create a PlotConfigurator."""
    configurator = PlotConfigurator(config)
    if data is not None:
        configurator.set_data(data)
    if trades is not None:
        configurator.set_trades(trades)
    return configurator


# Preset configurations
def get_momentum_preset() -> PlotConfig:
    """Get preset for momentum trading."""
    config = PlotConfig(title="Momentum Strategy")
    config.indicators = [
        BUILTIN_INDICATORS["ema_12"],
        BUILTIN_INDICATORS["ema_26"],
        BUILTIN_INDICATORS["rsi"],
        BUILTIN_INDICATORS["macd"],
        BUILTIN_INDICATORS["macd_signal"],
        BUILTIN_INDICATORS["macd_histogram"],
    ]
    return config


def get_trend_following_preset() -> PlotConfig:
    """Get preset for trend following."""
    config = PlotConfig(title="Trend Following Strategy")
    config.indicators = [
        BUILTIN_INDICATORS["sma_20"],
        BUILTIN_INDICATORS["sma_50"],
        BUILTIN_INDICATORS["sma_200"],
        BUILTIN_INDICATORS["atr"],
    ]
    return config


def get_mean_reversion_preset() -> PlotConfig:
    """Get preset for mean reversion."""
    config = PlotConfig(title="Mean Reversion Strategy")
    config.indicators = [
        BUILTIN_INDICATORS["bb_upper"],
        BUILTIN_INDICATORS["bb_middle"],
        BUILTIN_INDICATORS["bb_lower"],
        BUILTIN_INDICATORS["rsi"],
        BUILTIN_INDICATORS["stoch_k"],
        BUILTIN_INDICATORS["stoch_d"],
    ]
    return config


def get_volume_analysis_preset() -> PlotConfig:
    """Get preset for volume analysis."""
    config = PlotConfig(title="Volume Analysis")
    config.indicators = [
        BUILTIN_INDICATORS["sma_20"],
        BUILTIN_INDICATORS["volume_sma"],
        BUILTIN_INDICATORS["obv"],
    ]
    return config
