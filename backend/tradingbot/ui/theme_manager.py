"""
Theme Manager for UI Dark/Light Mode Support.

Ported from nautilus_trader's theming system.
Provides consistent theming across all UI components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"  # Follow system preference


@dataclass
class ColorPalette:
    """Color palette definition."""
    # Primary colors
    primary: str = "#2196F3"
    primary_light: str = "#64B5F6"
    primary_dark: str = "#1976D2"

    # Secondary colors
    secondary: str = "#FF9800"
    secondary_light: str = "#FFB74D"
    secondary_dark: str = "#F57C00"

    # Accent colors
    accent: str = "#9C27B0"

    # Semantic colors
    success: str = "#4CAF50"
    warning: str = "#FF9800"
    error: str = "#F44336"
    info: str = "#2196F3"

    # Trading colors
    profit: str = "#4CAF50"
    loss: str = "#F44336"
    neutral: str = "#607D8B"
    buy: str = "#26A69A"
    sell: str = "#EF5350"

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "primary_light": self.primary_light,
            "primary_dark": self.primary_dark,
            "secondary": self.secondary,
            "secondary_light": self.secondary_light,
            "secondary_dark": self.secondary_dark,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
            "profit": self.profit,
            "loss": self.loss,
            "neutral": self.neutral,
            "buy": self.buy,
            "sell": self.sell,
        }


@dataclass
class ThemeColors:
    """Theme-specific color definitions."""
    # Background colors
    background: str = "#ffffff"
    background_secondary: str = "#f5f5f5"
    surface: str = "#ffffff"
    surface_elevated: str = "#ffffff"

    # Text colors
    text_primary: str = "#212121"
    text_secondary: str = "#757575"
    text_disabled: str = "#9e9e9e"
    text_hint: str = "#9e9e9e"
    text_inverse: str = "#ffffff"

    # Border colors
    border: str = "#e0e0e0"
    border_light: str = "#f5f5f5"
    border_dark: str = "#bdbdbd"
    divider: str = "#e0e0e0"

    # Chart colors
    chart_background: str = "#ffffff"
    chart_grid: str = "#e0e0e0"
    chart_text: str = "#333333"
    chart_crosshair: str = "#666666"

    # Interactive states
    hover: str = "rgba(0, 0, 0, 0.04)"
    focus: str = "rgba(33, 150, 243, 0.12)"
    selected: str = "rgba(33, 150, 243, 0.08)"
    pressed: str = "rgba(0, 0, 0, 0.12)"

    # Shadows
    shadow_sm: str = "0 1px 2px rgba(0, 0, 0, 0.05)"
    shadow_md: str = "0 4px 6px rgba(0, 0, 0, 0.1)"
    shadow_lg: str = "0 10px 15px rgba(0, 0, 0, 0.1)"

    def to_dict(self) -> dict:
        return {
            "background": self.background,
            "background_secondary": self.background_secondary,
            "surface": self.surface,
            "surface_elevated": self.surface_elevated,
            "text_primary": self.text_primary,
            "text_secondary": self.text_secondary,
            "text_disabled": self.text_disabled,
            "text_hint": self.text_hint,
            "text_inverse": self.text_inverse,
            "border": self.border,
            "border_light": self.border_light,
            "border_dark": self.border_dark,
            "divider": self.divider,
            "chart_background": self.chart_background,
            "chart_grid": self.chart_grid,
            "chart_text": self.chart_text,
            "chart_crosshair": self.chart_crosshair,
            "hover": self.hover,
            "focus": self.focus,
            "selected": self.selected,
            "pressed": self.pressed,
            "shadow_sm": self.shadow_sm,
            "shadow_md": self.shadow_md,
            "shadow_lg": self.shadow_lg,
        }


@dataclass
class Typography:
    """Typography configuration."""
    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_family_mono: str = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"

    # Font sizes
    font_size_xs: str = "0.75rem"
    font_size_sm: str = "0.875rem"
    font_size_base: str = "1rem"
    font_size_lg: str = "1.125rem"
    font_size_xl: str = "1.25rem"
    font_size_2xl: str = "1.5rem"
    font_size_3xl: str = "1.875rem"
    font_size_4xl: str = "2.25rem"

    # Font weights
    font_weight_light: int = 300
    font_weight_normal: int = 400
    font_weight_medium: int = 500
    font_weight_semibold: int = 600
    font_weight_bold: int = 700

    # Line heights
    line_height_tight: float = 1.25
    line_height_normal: float = 1.5
    line_height_relaxed: float = 1.75

    def to_dict(self) -> dict:
        return {
            "font_family": self.font_family,
            "font_family_mono": self.font_family_mono,
            "font_size_xs": self.font_size_xs,
            "font_size_sm": self.font_size_sm,
            "font_size_base": self.font_size_base,
            "font_size_lg": self.font_size_lg,
            "font_size_xl": self.font_size_xl,
            "font_size_2xl": self.font_size_2xl,
            "font_size_3xl": self.font_size_3xl,
            "font_size_4xl": self.font_size_4xl,
            "font_weight_light": self.font_weight_light,
            "font_weight_normal": self.font_weight_normal,
            "font_weight_medium": self.font_weight_medium,
            "font_weight_semibold": self.font_weight_semibold,
            "font_weight_bold": self.font_weight_bold,
            "line_height_tight": self.line_height_tight,
            "line_height_normal": self.line_height_normal,
            "line_height_relaxed": self.line_height_relaxed,
        }


@dataclass
class Spacing:
    """Spacing configuration."""
    space_0: str = "0"
    space_1: str = "0.25rem"
    space_2: str = "0.5rem"
    space_3: str = "0.75rem"
    space_4: str = "1rem"
    space_5: str = "1.25rem"
    space_6: str = "1.5rem"
    space_8: str = "2rem"
    space_10: str = "2.5rem"
    space_12: str = "3rem"
    space_16: str = "4rem"
    space_20: str = "5rem"
    space_24: str = "6rem"

    # Border radius
    radius_none: str = "0"
    radius_sm: str = "0.125rem"
    radius_base: str = "0.25rem"
    radius_md: str = "0.375rem"
    radius_lg: str = "0.5rem"
    radius_xl: str = "0.75rem"
    radius_2xl: str = "1rem"
    radius_full: str = "9999px"

    def to_dict(self) -> dict:
        return {
            "space_0": self.space_0,
            "space_1": self.space_1,
            "space_2": self.space_2,
            "space_3": self.space_3,
            "space_4": self.space_4,
            "space_5": self.space_5,
            "space_6": self.space_6,
            "space_8": self.space_8,
            "space_10": self.space_10,
            "space_12": self.space_12,
            "space_16": self.space_16,
            "space_20": self.space_20,
            "space_24": self.space_24,
            "radius_none": self.radius_none,
            "radius_sm": self.radius_sm,
            "radius_base": self.radius_base,
            "radius_md": self.radius_md,
            "radius_lg": self.radius_lg,
            "radius_xl": self.radius_xl,
            "radius_2xl": self.radius_2xl,
            "radius_full": self.radius_full,
        }


@dataclass
class Theme:
    """Complete theme definition."""
    name: str
    mode: ThemeMode
    palette: ColorPalette = field(default_factory=ColorPalette)
    colors: ThemeColors = field(default_factory=ThemeColors)
    typography: Typography = field(default_factory=Typography)
    spacing: Spacing = field(default_factory=Spacing)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode.value,
            "palette": self.palette.to_dict(),
            "colors": self.colors.to_dict(),
            "typography": self.typography.to_dict(),
            "spacing": self.spacing.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_css_variables(self) -> str:
        """Generate CSS custom properties."""
        css_vars = []

        # Palette colors
        for key, value in self.palette.to_dict().items():
            css_vars.append(f"  --color-{key.replace('_', '-')}: {value};")

        # Theme colors
        for key, value in self.colors.to_dict().items():
            css_vars.append(f"  --{key.replace('_', '-')}: {value};")

        # Typography
        for key, value in self.typography.to_dict().items():
            css_vars.append(f"  --{key.replace('_', '-')}: {value};")

        # Spacing
        for key, value in self.spacing.to_dict().items():
            css_vars.append(f"  --{key.replace('_', '-')}: {value};")

        return ":root {\n" + "\n".join(css_vars) + "\n}"


# Predefined themes
LIGHT_THEME = Theme(
    name="Light",
    mode=ThemeMode.LIGHT,
    palette=ColorPalette(),
    colors=ThemeColors(
        background="#ffffff",
        background_secondary="#f5f5f5",
        surface="#ffffff",
        surface_elevated="#ffffff",
        text_primary="#212121",
        text_secondary="#757575",
        text_disabled="#9e9e9e",
        text_hint="#9e9e9e",
        text_inverse="#ffffff",
        border="#e0e0e0",
        border_light="#f5f5f5",
        border_dark="#bdbdbd",
        divider="#e0e0e0",
        chart_background="#ffffff",
        chart_grid="#e0e0e0",
        chart_text="#333333",
        chart_crosshair="#666666",
        hover="rgba(0, 0, 0, 0.04)",
        focus="rgba(33, 150, 243, 0.12)",
        selected="rgba(33, 150, 243, 0.08)",
        pressed="rgba(0, 0, 0, 0.12)",
        shadow_sm="0 1px 2px rgba(0, 0, 0, 0.05)",
        shadow_md="0 4px 6px rgba(0, 0, 0, 0.1)",
        shadow_lg="0 10px 15px rgba(0, 0, 0, 0.1)",
    ),
)

DARK_THEME = Theme(
    name="Dark",
    mode=ThemeMode.DARK,
    palette=ColorPalette(
        profit="#26A69A",
        loss="#EF5350",
    ),
    colors=ThemeColors(
        background="#121212",
        background_secondary="#1e1e1e",
        surface="#1e1e1e",
        surface_elevated="#252525",
        text_primary="#e0e0e0",
        text_secondary="#a0a0a0",
        text_disabled="#6b6b6b",
        text_hint="#6b6b6b",
        text_inverse="#121212",
        border="#333333",
        border_light="#404040",
        border_dark="#1a1a1a",
        divider="#333333",
        chart_background="#1e1e1e",
        chart_grid="#333333",
        chart_text="#e0e0e0",
        chart_crosshair="#808080",
        hover="rgba(255, 255, 255, 0.04)",
        focus="rgba(33, 150, 243, 0.12)",
        selected="rgba(33, 150, 243, 0.16)",
        pressed="rgba(255, 255, 255, 0.12)",
        shadow_sm="0 1px 2px rgba(0, 0, 0, 0.3)",
        shadow_md="0 4px 6px rgba(0, 0, 0, 0.4)",
        shadow_lg="0 10px 15px rgba(0, 0, 0, 0.5)",
    ),
)

MIDNIGHT_THEME = Theme(
    name="Midnight",
    mode=ThemeMode.DARK,
    palette=ColorPalette(
        primary="#58A6FF",
        primary_light="#79B8FF",
        primary_dark="#388BFD",
        secondary="#F78166",
        profit="#3FB950",
        loss="#F85149",
        buy="#3FB950",
        sell="#F85149",
    ),
    colors=ThemeColors(
        background="#0D1117",
        background_secondary="#161B22",
        surface="#161B22",
        surface_elevated="#21262D",
        text_primary="#C9D1D9",
        text_secondary="#8B949E",
        text_disabled="#484F58",
        text_hint="#484F58",
        text_inverse="#0D1117",
        border="#30363D",
        border_light="#21262D",
        border_dark="#484F58",
        divider="#21262D",
        chart_background="#0D1117",
        chart_grid="#21262D",
        chart_text="#C9D1D9",
        chart_crosshair="#484F58",
        hover="rgba(56, 139, 253, 0.1)",
        focus="rgba(56, 139, 253, 0.3)",
        selected="rgba(56, 139, 253, 0.15)",
        pressed="rgba(56, 139, 253, 0.2)",
        shadow_sm="0 1px 0 rgba(27, 31, 35, 0.04)",
        shadow_md="0 3px 6px rgba(1, 4, 9, 0.15)",
        shadow_lg="0 8px 24px rgba(1, 4, 9, 0.15)",
    ),
)

TERMINAL_THEME = Theme(
    name="Terminal",
    mode=ThemeMode.DARK,
    palette=ColorPalette(
        primary="#00FF00",
        primary_light="#33FF33",
        primary_dark="#00CC00",
        secondary="#00FFFF",
        accent="#FF00FF",
        success="#00FF00",
        warning="#FFFF00",
        error="#FF0000",
        info="#00FFFF",
        profit="#00FF00",
        loss="#FF0000",
        neutral="#808080",
        buy="#00FF00",
        sell="#FF0000",
    ),
    colors=ThemeColors(
        background="#000000",
        background_secondary="#0a0a0a",
        surface="#0a0a0a",
        surface_elevated="#141414",
        text_primary="#00FF00",
        text_secondary="#00CC00",
        text_disabled="#006600",
        text_hint="#006600",
        text_inverse="#000000",
        border="#003300",
        border_light="#004400",
        border_dark="#002200",
        divider="#003300",
        chart_background="#000000",
        chart_grid="#003300",
        chart_text="#00FF00",
        chart_crosshair="#00FF00",
        hover="rgba(0, 255, 0, 0.1)",
        focus="rgba(0, 255, 0, 0.2)",
        selected="rgba(0, 255, 0, 0.15)",
        pressed="rgba(0, 255, 0, 0.25)",
        shadow_sm="0 0 2px rgba(0, 255, 0, 0.3)",
        shadow_md="0 0 4px rgba(0, 255, 0, 0.4)",
        shadow_lg="0 0 8px rgba(0, 255, 0, 0.5)",
    ),
    typography=Typography(
        font_family="'Courier New', Courier, monospace",
        font_family_mono="'Courier New', Courier, monospace",
    ),
)

# Available themes registry
THEMES = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
    "midnight": MIDNIGHT_THEME,
    "terminal": TERMINAL_THEME,
}


class ThemeManager:
    """
    Theme manager for handling theme state and preferences.

    Supports light/dark mode, system preference following,
    and custom themes.
    """

    def __init__(self, default_theme: str = "dark"):
        self._themes = dict(THEMES)
        self._current_theme_name = default_theme
        self._mode = ThemeMode.DARK if default_theme == "dark" else ThemeMode.LIGHT
        self._listeners: list[callable] = []

    @property
    def current_theme(self) -> Theme:
        """Get the current theme."""
        return self._themes.get(self._current_theme_name, DARK_THEME)

    @property
    def current_theme_name(self) -> str:
        """Get the current theme name."""
        return self._current_theme_name

    @property
    def mode(self) -> ThemeMode:
        """Get the current theme mode."""
        return self._mode

    def set_theme(self, theme_name: str) -> bool:
        """Set the current theme by name."""
        if theme_name not in self._themes:
            logger.warning(f"Theme '{theme_name}' not found")
            return False

        self._current_theme_name = theme_name
        self._mode = self._themes[theme_name].mode
        self._notify_listeners()
        return True

    def set_mode(self, mode: ThemeMode) -> None:
        """Set the theme mode (light/dark/system)."""
        self._mode = mode

        if mode == ThemeMode.LIGHT:
            self.set_theme("light")
        elif mode == ThemeMode.DARK:
            self.set_theme("dark")
        # ThemeMode.SYSTEM would follow OS preference

    def toggle_mode(self) -> ThemeMode:
        """Toggle between light and dark mode."""
        if self._mode == ThemeMode.LIGHT:
            self.set_mode(ThemeMode.DARK)
        else:
            self.set_mode(ThemeMode.LIGHT)
        return self._mode

    def register_theme(self, name: str, theme: Theme) -> None:
        """Register a custom theme."""
        self._themes[name] = theme

    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return list(self._themes.keys())

    def get_theme(self, name: str) -> Optional[Theme]:
        """Get a theme by name."""
        return self._themes.get(name)

    def add_listener(self, callback: callable) -> None:
        """Add a theme change listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: callable) -> None:
        """Remove a theme change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of theme change."""
        for listener in self._listeners:
            try:
                listener(self.current_theme)
            except Exception as e:
                logger.error(f"Theme listener error: {e}")

    def get_css_variables(self) -> str:
        """Get CSS variables for current theme."""
        return self.current_theme.to_css_variables()

    def get_plotly_template(self) -> dict:
        """Get Plotly template for current theme."""
        theme = self.current_theme
        colors = theme.colors

        return {
            "layout": {
                "paper_bgcolor": colors.surface,
                "plot_bgcolor": colors.chart_background,
                "font": {
                    "family": theme.typography.font_family,
                    "color": colors.text_primary,
                },
                "title": {
                    "font": {"color": colors.text_primary}
                },
                "xaxis": {
                    "gridcolor": colors.chart_grid,
                    "linecolor": colors.border,
                    "tickcolor": colors.text_secondary,
                    "tickfont": {"color": colors.text_secondary},
                    "title": {"font": {"color": colors.text_primary}},
                },
                "yaxis": {
                    "gridcolor": colors.chart_grid,
                    "linecolor": colors.border,
                    "tickcolor": colors.text_secondary,
                    "tickfont": {"color": colors.text_secondary},
                    "title": {"font": {"color": colors.text_primary}},
                },
                "legend": {
                    "bgcolor": colors.surface,
                    "bordercolor": colors.border,
                    "font": {"color": colors.text_primary},
                },
                "colorway": [
                    theme.palette.primary,
                    theme.palette.secondary,
                    theme.palette.accent,
                    theme.palette.success,
                    theme.palette.warning,
                    theme.palette.error,
                    theme.palette.info,
                ],
            },
            "data": {
                "candlestick": [{
                    "increasing": {
                        "line": {"color": theme.palette.profit},
                        "fillcolor": theme.palette.profit,
                    },
                    "decreasing": {
                        "line": {"color": theme.palette.loss},
                        "fillcolor": theme.palette.loss,
                    },
                }],
                "bar": [{
                    "marker": {"color": theme.palette.primary},
                }],
                "scatter": [{
                    "marker": {"color": theme.palette.primary},
                }],
            },
        }

    def to_json(self) -> str:
        """Serialize current state to JSON."""
        return json.dumps({
            "current_theme": self._current_theme_name,
            "mode": self._mode.value,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "ThemeManager":
        """Create ThemeManager from JSON."""
        data = json.loads(json_str)
        manager = cls(default_theme=data.get("current_theme", "dark"))
        return manager


# Global theme manager instance
theme_manager = ThemeManager()


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager."""
    return theme_manager


def create_theme_manager(default_theme: str = "dark") -> ThemeManager:
    """Factory function to create a new theme manager."""
    return ThemeManager(default_theme=default_theme)


# Helper functions for common operations
def get_current_theme() -> Theme:
    """Get the current theme."""
    return theme_manager.current_theme


def set_theme(theme_name: str) -> bool:
    """Set the current theme."""
    return theme_manager.set_theme(theme_name)


def toggle_dark_mode() -> ThemeMode:
    """Toggle dark mode."""
    return theme_manager.toggle_mode()


def get_css_variables() -> str:
    """Get CSS variables for current theme."""
    return theme_manager.get_css_variables()


def get_plotly_template() -> dict:
    """Get Plotly template for current theme."""
    return theme_manager.get_plotly_template()
