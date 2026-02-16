"""Technical analysis and pattern recognition modules."""

try:
    from .tearsheet import (
        ThemeType,
        ThemeColors,
        THEMES,
        TearsheetConfig,
        PerformanceMetrics,
        TearsheetGenerator,
        create_tearsheet_generator,
    )
    TEARSHEET_AVAILABLE = True
except ImportError:
    TEARSHEET_AVAILABLE = False

try:
    from .pdf_report import (
        PDFReportConfig,
        PDFReportGenerator,
    )
    PDF_REPORT_AVAILABLE = True
except ImportError:
    PDF_REPORT_AVAILABLE = False

try:
    from .report_templates import (
        ReportTemplates,
    )
    REPORT_TEMPLATES_AVAILABLE = True
except ImportError:
    REPORT_TEMPLATES_AVAILABLE = False

try:
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
    PLOT_CONFIGURATOR_AVAILABLE = True
except ImportError:
    PLOT_CONFIGURATOR_AVAILABLE = False

__all__ = [
    "TEARSHEET_AVAILABLE",
    "PDF_REPORT_AVAILABLE",
    "REPORT_TEMPLATES_AVAILABLE",
    "PLOT_CONFIGURATOR_AVAILABLE",
]

if TEARSHEET_AVAILABLE:
    __all__.extend([
        "ThemeType",
        "ThemeColors",
        "THEMES",
        "TearsheetConfig",
        "PerformanceMetrics",
        "TearsheetGenerator",
        "create_tearsheet_generator",
    ])

if PDF_REPORT_AVAILABLE:
    __all__.extend([
        "PDFReportConfig",
        "PDFReportGenerator",
    ])

if REPORT_TEMPLATES_AVAILABLE:
    __all__.append("ReportTemplates")

if PLOT_CONFIGURATOR_AVAILABLE:
    __all__.extend([
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
    ])
