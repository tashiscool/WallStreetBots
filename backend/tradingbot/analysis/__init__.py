"""Technical analysis and pattern recognition modules."""

from .tearsheet import (
    ThemeType,
    ThemeColors,
    THEMES,
    TearsheetConfig,
    PerformanceMetrics,
    TearsheetGenerator,
    create_tearsheet_generator,
)

from .plot_configurator import (
    PlotType,
    PlotLocation,
    IndicatorCategory,
    PlotIndicator,
    TradeMarker,
    PlotConfig,
    BUILTIN_INDICATORS,
    PlotConfigurator,
    create_plot_configurator,
    get_momentum_preset,
    get_trend_following_preset,
    get_mean_reversion_preset,
    get_volume_analysis_preset,
)

__all__ = [
    # Tearsheet
    "ThemeType",
    "ThemeColors",
    "THEMES",
    "TearsheetConfig",
    "PerformanceMetrics",
    "TearsheetGenerator",
    "create_tearsheet_generator",
    # Plot Configurator
    "PlotType",
    "PlotLocation",
    "IndicatorCategory",
    "PlotIndicator",
    "TradeMarker",
    "PlotConfig",
    "BUILTIN_INDICATORS",
    "PlotConfigurator",
    "create_plot_configurator",
    "get_momentum_preset",
    "get_trend_following_preset",
    "get_mean_reversion_preset",
    "get_volume_analysis_preset",
]
