"""
Optimization Module.

Hyperparameter optimization and strategy tuning.
"""

from .hyperopt import (
    HyperoptEngine,
    HyperoptSpace,
    HyperoptResult,
    HyperoptSummary,
    HyperParameter,
    ParameterType,
    OptimizationMethod,
    ObjectiveFunction,
    BacktestObjective,
    HyperoptSpaceBuilder,
    create_hyperopt_engine,
)

__all__ = [
    # Engine
    "HyperoptEngine",
    "create_hyperopt_engine",
    # Space definition
    "HyperoptSpace",
    "HyperParameter",
    "ParameterType",
    "HyperoptSpaceBuilder",
    # Results
    "HyperoptResult",
    "HyperoptSummary",
    # Objectives
    "ObjectiveFunction",
    "BacktestObjective",
    # Methods
    "OptimizationMethod",
]
