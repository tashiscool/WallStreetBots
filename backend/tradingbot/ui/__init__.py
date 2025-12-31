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
    # Theme Manager
    "ThemeMode",
    "ColorPalette",
    "ThemeColors",
    "Typography",
    "Spacing",
    "Theme",
    "ThemeManager",
    "LIGHT_THEME",
    "DARK_THEME",
    "MIDNIGHT_THEME",
    "TERMINAL_THEME",
    "THEMES",
    "theme_manager",
    "get_theme_manager",
    "create_theme_manager",
    "get_current_theme",
    "set_theme",
    "toggle_dark_mode",
    "get_css_variables",
    "get_plotly_template",
    # Parameter Config
    "ParameterType",
    "ParameterCategory",
    "ParameterConstraint",
    "OptimizationRange",
    "StrategyParameter",
    "ParameterPreset",
    "StrategyParameterSchema",
    "ParameterConfigBuilder",
    "create_momentum_strategy_schema",
    "create_mean_reversion_schema",
    "register_schema",
    "get_schema",
    "list_schemas",
]
