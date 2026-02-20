"""
UI Components Module.

Theme management and parameter configuration for trading UI.
"""

from .theme_manager import (
    ThemeMode,
    ColorPalette,
    ThemeColors,
    Typography,
    Spacing,
    Theme,
    ThemeManager,
    LIGHT_THEME,
    DARK_THEME,
    MIDNIGHT_THEME,
    TERMINAL_THEME,
    THEMES,
    theme_manager,
    get_theme_manager,
    create_theme_manager,
    get_current_theme,
    set_theme,
    toggle_dark_mode,
    get_css_variables,
    get_plotly_template,
)

from .parameter_config import (
    ParameterType,
    ParameterCategory,
    ParameterConstraint,
    OptimizationRange,
    StrategyParameter,
    ParameterPreset,
    StrategyParameterSchema,
    ParameterConfigBuilder,
    create_momentum_strategy_schema,
    create_mean_reversion_schema,
    register_schema,
    get_schema,
    list_schemas,
)

__all__ = [
    "DARK_THEME",
    "LIGHT_THEME",
    "MIDNIGHT_THEME",
    "TERMINAL_THEME",
    "THEMES",
    "ColorPalette",
    "OptimizationRange",
    "ParameterCategory",
    "ParameterConfigBuilder",
    "ParameterConstraint",
    "ParameterPreset",
    # Parameter Config
    "ParameterType",
    "Spacing",
    "StrategyParameter",
    "StrategyParameterSchema",
    "Theme",
    "ThemeColors",
    "ThemeManager",
    # Theme Manager
    "ThemeMode",
    "Typography",
    "create_mean_reversion_schema",
    "create_momentum_strategy_schema",
    "create_theme_manager",
    "get_css_variables",
    "get_current_theme",
    "get_plotly_template",
    "get_schema",
    "get_theme_manager",
    "list_schemas",
    "register_schema",
    "set_theme",
    "theme_manager",
    "toggle_dark_mode",
]
