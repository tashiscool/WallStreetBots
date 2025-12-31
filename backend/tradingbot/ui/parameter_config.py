"""
Strategy Parameter Configuration UI.

Visual parameter editor for trading strategies.
Supports parameter validation, presets, and optimization ranges.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of strategy parameters."""
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    SYMBOL = "symbol"
    SYMBOL_LIST = "symbol_list"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    TIMEFRAME = "timeframe"
    JSON = "json"


class ParameterCategory(Enum):
    """Categories for grouping parameters."""
    GENERAL = "general"
    ENTRY = "entry"
    EXIT = "exit"
    RISK = "risk"
    POSITION_SIZING = "position_sizing"
    INDICATORS = "indicators"
    FILTERS = "filters"
    TIMING = "timing"
    ADVANCED = "advanced"


@dataclass
class ParameterConstraint:
    """Constraints for parameter validation."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    pattern: Optional[str] = None  # Regex for string validation
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[list[Any]] = None
    required: bool = True
    depends_on: Optional[str] = None  # Parameter name this depends on
    depends_value: Optional[Any] = None  # Required value of dependency

    def validate(self, value: Any, param_type: ParameterType) -> tuple[bool, Optional[str]]:
        """Validate a value against constraints."""
        if value is None:
            if self.required:
                return False, "Value is required"
            return True, None

        if param_type in (ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.PERCENTAGE):
            try:
                num_value = float(value)
                if self.min_value is not None and num_value < self.min_value:
                    return False, f"Value must be >= {self.min_value}"
                if self.max_value is not None and num_value > self.max_value:
                    return False, f"Value must be <= {self.max_value}"
                if self.step is not None:
                    # Check if value is a valid step
                    if self.min_value is not None:
                        offset = (num_value - self.min_value) / self.step
                        if abs(offset - round(offset)) > 1e-9:
                            return False, f"Value must be in steps of {self.step}"
            except (TypeError, ValueError):
                return False, "Invalid numeric value"

        if param_type == ParameterType.STRING:
            if not isinstance(value, str):
                return False, "Value must be a string"
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"Minimum length is {self.min_length}"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"Maximum length is {self.max_length}"
            if self.pattern:
                import re
                if not re.match(self.pattern, value):
                    return False, f"Value must match pattern: {self.pattern}"

        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, f"Value must be one of: {self.allowed_values}"

        return True, None


@dataclass
class OptimizationRange:
    """Range for parameter optimization."""
    min_value: float
    max_value: float
    step: float
    log_scale: bool = False

    def get_values(self) -> list[float]:
        """Generate list of values in range."""
        import numpy as np
        if self.log_scale:
            return list(np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                int((np.log10(self.max_value) - np.log10(self.min_value)) / np.log10(1 + self.step / self.min_value)) + 1
            ))
        else:
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += self.step
            return values

    def to_dict(self) -> dict:
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "log_scale": self.log_scale,
        }


@dataclass
class StrategyParameter:
    """Definition of a strategy parameter."""
    name: str
    display_name: str
    param_type: ParameterType
    default_value: Any
    description: str = ""
    category: ParameterCategory = ParameterCategory.GENERAL
    constraint: ParameterConstraint = field(default_factory=ParameterConstraint)
    optimization_range: Optional[OptimizationRange] = None
    ui_hints: dict = field(default_factory=dict)
    tooltip: Optional[str] = None
    unit: Optional[str] = None  # e.g., "%", "$", "days"
    options: Optional[list[dict]] = None  # For SELECT type: [{"value": x, "label": y}]
    visible: bool = True
    editable: bool = True

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a parameter value."""
        return self.constraint.validate(value, self.param_type)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.param_type.value,
            "default_value": self.default_value,
            "description": self.description,
            "category": self.category.value,
            "constraint": {
                "min_value": self.constraint.min_value,
                "max_value": self.constraint.max_value,
                "step": self.constraint.step,
                "allowed_values": self.constraint.allowed_values,
                "required": self.constraint.required,
            },
            "optimization_range": self.optimization_range.to_dict() if self.optimization_range else None,
            "ui_hints": self.ui_hints,
            "tooltip": self.tooltip,
            "unit": self.unit,
            "options": self.options,
            "visible": self.visible,
            "editable": self.editable,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyParameter":
        """Create parameter from dictionary."""
        constraint = ParameterConstraint(
            min_value=data.get("constraint", {}).get("min_value"),
            max_value=data.get("constraint", {}).get("max_value"),
            step=data.get("constraint", {}).get("step"),
            allowed_values=data.get("constraint", {}).get("allowed_values"),
            required=data.get("constraint", {}).get("required", True),
        )

        opt_range = None
        if data.get("optimization_range"):
            opt_data = data["optimization_range"]
            opt_range = OptimizationRange(
                min_value=opt_data["min_value"],
                max_value=opt_data["max_value"],
                step=opt_data["step"],
                log_scale=opt_data.get("log_scale", False),
            )

        return cls(
            name=data["name"],
            display_name=data["display_name"],
            param_type=ParameterType(data["type"]),
            default_value=data["default_value"],
            description=data.get("description", ""),
            category=ParameterCategory(data.get("category", "general")),
            constraint=constraint,
            optimization_range=opt_range,
            ui_hints=data.get("ui_hints", {}),
            tooltip=data.get("tooltip"),
            unit=data.get("unit"),
            options=data.get("options"),
            visible=data.get("visible", True),
            editable=data.get("editable", True),
        )


@dataclass
class ParameterPreset:
    """Preset configuration for a strategy."""
    name: str
    description: str
    values: dict[str, Any]
    is_default: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    author: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "values": self.values,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat(),
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterPreset":
        return cls(
            name=data["name"],
            description=data["description"],
            values=data["values"],
            is_default=data.get("is_default", False),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            author=data.get("author"),
        )


class StrategyParameterSchema:
    """
    Schema definition for strategy parameters.

    Defines all parameters, their types, constraints,
    and UI configuration for a strategy.
    """

    def __init__(
        self,
        strategy_name: str,
        strategy_version: str = "1.0.0",
        description: str = "",
    ):
        self.strategy_name = strategy_name
        self.strategy_version = strategy_version
        self.description = description
        self._parameters: dict[str, StrategyParameter] = {}
        self._presets: dict[str, ParameterPreset] = {}
        self._validators: dict[str, Callable[[dict], tuple[bool, Optional[str]]]] = {}

    def add_parameter(self, param: StrategyParameter) -> "StrategyParameterSchema":
        """Add a parameter to the schema."""
        self._parameters[param.name] = param
        return self

    def add_preset(self, preset: ParameterPreset) -> "StrategyParameterSchema":
        """Add a preset to the schema."""
        self._presets[preset.name] = preset
        return self

    def add_validator(
        self,
        name: str,
        validator: Callable[[dict], tuple[bool, Optional[str]]],
    ) -> "StrategyParameterSchema":
        """Add a cross-parameter validator."""
        self._validators[name] = validator
        return self

    def get_parameter(self, name: str) -> Optional[StrategyParameter]:
        """Get a parameter by name."""
        return self._parameters.get(name)

    def get_parameters(self) -> list[StrategyParameter]:
        """Get all parameters."""
        return list(self._parameters.values())

    def get_parameters_by_category(self) -> dict[ParameterCategory, list[StrategyParameter]]:
        """Get parameters grouped by category."""
        result: dict[ParameterCategory, list[StrategyParameter]] = {}
        for param in self._parameters.values():
            if param.category not in result:
                result[param.category] = []
            result[param.category].append(param)
        return result

    def get_presets(self) -> list[ParameterPreset]:
        """Get all presets."""
        return list(self._presets.values())

    def get_default_preset(self) -> Optional[ParameterPreset]:
        """Get the default preset."""
        for preset in self._presets.values():
            if preset.is_default:
                return preset
        return None

    def get_default_values(self) -> dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: param.default_value
            for name, param in self._parameters.items()
        }

    def validate(self, values: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate parameter values."""
        errors = []

        # Validate individual parameters
        for name, param in self._parameters.items():
            value = values.get(name)

            # Check dependency
            if param.constraint.depends_on:
                dep_value = values.get(param.constraint.depends_on)
                if dep_value != param.constraint.depends_value:
                    continue  # Skip validation if dependency not met

            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(f"{param.display_name}: {error}")

        # Run cross-parameter validators
        for validator_name, validator in self._validators.items():
            try:
                is_valid, error = validator(values)
                if not is_valid:
                    errors.append(error or f"Validation failed: {validator_name}")
            except Exception as e:
                errors.append(f"Validation error in {validator_name}: {e}")

        return len(errors) == 0, errors

    def apply_preset(self, preset_name: str, values: dict[str, Any]) -> dict[str, Any]:
        """Apply a preset to current values."""
        preset = self._presets.get(preset_name)
        if preset:
            return {**values, **preset.values}
        return values

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "description": self.description,
            "parameters": [p.to_dict() for p in self._parameters.values()],
            "presets": [p.to_dict() for p in self._presets.values()],
        }

    def to_json(self) -> str:
        """Convert schema to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyParameterSchema":
        """Create schema from dictionary."""
        schema = cls(
            strategy_name=data["strategy_name"],
            strategy_version=data.get("strategy_version", "1.0.0"),
            description=data.get("description", ""),
        )

        for param_data in data.get("parameters", []):
            schema.add_parameter(StrategyParameter.from_dict(param_data))

        for preset_data in data.get("presets", []):
            schema.add_preset(ParameterPreset.from_dict(preset_data))

        return schema

    @classmethod
    def from_json(cls, json_str: str) -> "StrategyParameterSchema":
        """Create schema from JSON."""
        return cls.from_dict(json.loads(json_str))


class ParameterConfigBuilder:
    """Builder for creating parameter configurations."""

    def __init__(self, strategy_name: str):
        self._schema = StrategyParameterSchema(strategy_name)

    def version(self, version: str) -> "ParameterConfigBuilder":
        """Set strategy version."""
        self._schema.strategy_version = version
        return self

    def description(self, desc: str) -> "ParameterConfigBuilder":
        """Set description."""
        self._schema.description = desc
        return self

    def add_integer(
        self,
        name: str,
        display_name: str,
        default: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        step: int = 1,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add an integer parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.INTEGER,
            default_value=default,
            constraint=ParameterConstraint(
                min_value=min_value,
                max_value=max_value,
                step=step,
            ),
            **kwargs,
        ))
        return self

    def add_float(
        self,
        name: str,
        display_name: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: float = 0.01,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a float parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.FLOAT,
            default_value=default,
            constraint=ParameterConstraint(
                min_value=min_value,
                max_value=max_value,
                step=step,
            ),
            **kwargs,
        ))
        return self

    def add_percentage(
        self,
        name: str,
        display_name: str,
        default: float,
        min_value: float = 0,
        max_value: float = 100,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a percentage parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.PERCENTAGE,
            default_value=default,
            unit="%",
            constraint=ParameterConstraint(
                min_value=min_value,
                max_value=max_value,
            ),
            **kwargs,
        ))
        return self

    def add_boolean(
        self,
        name: str,
        display_name: str,
        default: bool,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a boolean parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.BOOLEAN,
            default_value=default,
            **kwargs,
        ))
        return self

    def add_select(
        self,
        name: str,
        display_name: str,
        options: list[dict],
        default: Any,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a select parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.SELECT,
            default_value=default,
            options=options,
            constraint=ParameterConstraint(
                allowed_values=[o["value"] for o in options],
            ),
            **kwargs,
        ))
        return self

    def add_symbol_list(
        self,
        name: str,
        display_name: str,
        default: list[str],
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a symbol list parameter."""
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.SYMBOL_LIST,
            default_value=default,
            **kwargs,
        ))
        return self

    def add_timeframe(
        self,
        name: str,
        display_name: str,
        default: str,
        **kwargs,
    ) -> "ParameterConfigBuilder":
        """Add a timeframe parameter."""
        timeframe_options = [
            {"value": "1m", "label": "1 Minute"},
            {"value": "5m", "label": "5 Minutes"},
            {"value": "15m", "label": "15 Minutes"},
            {"value": "30m", "label": "30 Minutes"},
            {"value": "1h", "label": "1 Hour"},
            {"value": "4h", "label": "4 Hours"},
            {"value": "1d", "label": "1 Day"},
            {"value": "1w", "label": "1 Week"},
        ]
        self._schema.add_parameter(StrategyParameter(
            name=name,
            display_name=display_name,
            param_type=ParameterType.TIMEFRAME,
            default_value=default,
            options=timeframe_options,
            constraint=ParameterConstraint(
                allowed_values=[o["value"] for o in timeframe_options],
            ),
            **kwargs,
        ))
        return self

    def add_preset(
        self,
        name: str,
        description: str,
        values: dict[str, Any],
        is_default: bool = False,
    ) -> "ParameterConfigBuilder":
        """Add a preset."""
        self._schema.add_preset(ParameterPreset(
            name=name,
            description=description,
            values=values,
            is_default=is_default,
        ))
        return self

    def build(self) -> StrategyParameterSchema:
        """Build the schema."""
        return self._schema


# Example strategy configurations
def create_momentum_strategy_schema() -> StrategyParameterSchema:
    """Create schema for momentum strategy."""
    return (
        ParameterConfigBuilder("Momentum Strategy")
        .version("1.0.0")
        .description("Momentum-based trading strategy using RSI and moving averages")

        # General parameters
        .add_symbol_list(
            "symbols", "Symbols", ["AAPL", "MSFT", "GOOGL"],
            category=ParameterCategory.GENERAL,
            description="Symbols to trade",
        )
        .add_timeframe(
            "timeframe", "Timeframe", "1h",
            category=ParameterCategory.GENERAL,
        )

        # Entry parameters
        .add_integer(
            "rsi_period", "RSI Period", 14,
            min_value=5, max_value=50,
            category=ParameterCategory.ENTRY,
            description="Period for RSI calculation",
        )
        .add_float(
            "rsi_oversold", "RSI Oversold", 30,
            min_value=10, max_value=40,
            category=ParameterCategory.ENTRY,
            description="RSI level for oversold condition",
        )
        .add_float(
            "rsi_overbought", "RSI Overbought", 70,
            min_value=60, max_value=90,
            category=ParameterCategory.ENTRY,
            description="RSI level for overbought condition",
        )

        # Exit parameters
        .add_percentage(
            "take_profit", "Take Profit", 5,
            category=ParameterCategory.EXIT,
            description="Take profit percentage",
        )
        .add_percentage(
            "stop_loss", "Stop Loss", 2,
            category=ParameterCategory.EXIT,
            description="Stop loss percentage",
        )
        .add_boolean(
            "use_trailing_stop", "Use Trailing Stop", True,
            category=ParameterCategory.EXIT,
        )

        # Risk parameters
        .add_percentage(
            "max_position_size", "Max Position Size", 10,
            category=ParameterCategory.RISK,
            description="Maximum position size as % of portfolio",
        )
        .add_integer(
            "max_positions", "Max Positions", 5,
            min_value=1, max_value=20,
            category=ParameterCategory.RISK,
        )

        # Presets
        .add_preset(
            "conservative", "Conservative",
            {"rsi_period": 21, "take_profit": 3, "stop_loss": 1.5, "max_position_size": 5},
        )
        .add_preset(
            "aggressive", "Aggressive",
            {"rsi_period": 7, "take_profit": 10, "stop_loss": 5, "max_position_size": 20},
        )
        .add_preset(
            "default", "Default",
            {},
            is_default=True,
        )

        .build()
    )


def create_mean_reversion_schema() -> StrategyParameterSchema:
    """Create schema for mean reversion strategy."""
    return (
        ParameterConfigBuilder("Mean Reversion Strategy")
        .version("1.0.0")
        .description("Mean reversion strategy using Bollinger Bands")

        .add_symbol_list(
            "symbols", "Symbols", ["SPY"],
            category=ParameterCategory.GENERAL,
        )
        .add_integer(
            "bb_period", "BB Period", 20,
            min_value=10, max_value=50,
            category=ParameterCategory.INDICATORS,
        )
        .add_float(
            "bb_std", "BB Std Dev", 2.0,
            min_value=1.0, max_value=3.0,
            step=0.1,
            category=ParameterCategory.INDICATORS,
        )
        .add_percentage(
            "entry_deviation", "Entry Deviation", 2,
            category=ParameterCategory.ENTRY,
            description="Minimum deviation from mean for entry",
        )

        .build()
    )


# Schema registry
_schema_registry: dict[str, StrategyParameterSchema] = {}


def register_schema(schema: StrategyParameterSchema) -> None:
    """Register a schema globally."""
    _schema_registry[schema.strategy_name] = schema


def get_schema(strategy_name: str) -> Optional[StrategyParameterSchema]:
    """Get a registered schema."""
    return _schema_registry.get(strategy_name)


def list_schemas() -> list[str]:
    """List all registered schemas."""
    return list(_schema_registry.keys())


# Register built-in schemas
register_schema(create_momentum_strategy_schema())
register_schema(create_mean_reversion_schema())
