"""
Hyperoptable Parameters

Define strategy parameters that can be optimized via hyperopt.
Inspired by freqtrade's parameter system.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, List, Optional, Union


class HyperoptSpace(Enum):
    """Parameter optimization spaces."""
    BUY = "buy"
    SELL = "sell"
    ENTRY = "entry"
    EXIT = "exit"
    ROI = "roi"
    STOPLOSS = "stoploss"
    TRAILING = "trailing"
    PROTECTION = "protection"
    CUSTOM = "custom"


@dataclass
class BaseParameter:
    """Base class for all hyperoptable parameters."""
    default: Any
    space: HyperoptSpace = HyperoptSpace.CUSTOM
    optimize: bool = True
    load: bool = True  # Whether to load from saved params
    name: Optional[str] = None  # Set automatically

    def __post_init__(self):
        self._value = self.default

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, val: Any):
        self._value = val

    def get_optuna_distribution(self, trial, name: str) -> Any:
        """Override in subclasses to provide Optuna distribution."""
        raise NotImplementedError


@dataclass
class IntParameter(BaseParameter):
    """Integer parameter for hyperopt optimization."""
    low: int = 0
    high: int = 100
    default: int = 50
    step: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.low <= self.default <= self.high:
            raise ValueError(f"Default {self.default} not in range [{self.low}, {self.high}]")

    def get_optuna_distribution(self, trial, name: str) -> int:
        """Get Optuna integer distribution."""
        return trial.suggest_int(name, self.low, self.high, step=self.step)


@dataclass
class RealParameter(BaseParameter):
    """Float parameter for hyperopt optimization."""
    low: float = 0.0
    high: float = 1.0
    default: float = 0.5
    decimals: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not self.low <= self.default <= self.high:
            raise ValueError(f"Default {self.default} not in range [{self.low}, {self.high}]")

    def get_optuna_distribution(self, trial, name: str) -> float:
        """Get Optuna float distribution."""
        value = trial.suggest_float(name, self.low, self.high)
        return round(value, self.decimals)


@dataclass
class DecimalParameter(BaseParameter):
    """Decimal parameter for precise monetary calculations."""
    low: Decimal = Decimal("0.0")
    high: Decimal = Decimal("1.0")
    default: Decimal = Decimal("0.5")
    decimals: int = 4

    def __post_init__(self):
        super().__post_init__()
        # Convert to Decimal if needed
        if not isinstance(self.low, Decimal):
            self.low = Decimal(str(self.low))
        if not isinstance(self.high, Decimal):
            self.high = Decimal(str(self.high))
        if not isinstance(self.default, Decimal):
            self.default = Decimal(str(self.default))

    def get_optuna_distribution(self, trial, name: str) -> Decimal:
        """Get Optuna distribution and convert to Decimal."""
        value = trial.suggest_float(name, float(self.low), float(self.high))
        return Decimal(str(round(value, self.decimals)))


@dataclass
class CategoricalParameter(BaseParameter):
    """Categorical parameter for discrete choices."""
    categories: List[Any] = field(default_factory=list)
    default: Any = None

    def __post_init__(self):
        super().__post_init__()
        if self.default is None and self.categories:
            self.default = self.categories[0]
        if self.default not in self.categories and self.categories:
            raise ValueError(f"Default {self.default} not in categories {self.categories}")

    def get_optuna_distribution(self, trial, name: str) -> Any:
        """Get Optuna categorical distribution."""
        return trial.suggest_categorical(name, self.categories)


@dataclass
class BooleanParameter(BaseParameter):
    """Boolean parameter."""
    default: bool = False

    def get_optuna_distribution(self, trial, name: str) -> bool:
        """Get Optuna boolean distribution."""
        return trial.suggest_categorical(name, [True, False])


def get_hyperoptable_parameters(strategy_class) -> dict:
    """
    Extract all hyperoptable parameters from a strategy class.

    Returns:
        dict: {param_name: BaseParameter} for all optimizable parameters
    """
    params = {}
    for name in dir(strategy_class):
        if name.startswith('_'):
            continue
        attr = getattr(strategy_class, name, None)
        if isinstance(attr, BaseParameter) and attr.optimize:
            attr.name = name
            params[name] = attr
    return params


def apply_parameters(strategy_instance, params: dict) -> None:
    """
    Apply optimized parameters to a strategy instance.

    Args:
        strategy_instance: The strategy object
        params: dict of {param_name: value} to apply
    """
    for name, value in params.items():
        if hasattr(strategy_instance, name):
            attr = getattr(strategy_instance, name)
            if isinstance(attr, BaseParameter):
                attr.value = value
            else:
                setattr(strategy_instance, name, value)


def get_parameter_values(strategy_instance) -> dict:
    """
    Get current values of all hyperoptable parameters.

    Returns:
        dict: {param_name: current_value}
    """
    values = {}
    for name in dir(strategy_instance):
        if name.startswith('_'):
            continue
        attr = getattr(strategy_instance, name, None)
        if isinstance(attr, BaseParameter):
            values[name] = attr.value
    return values
