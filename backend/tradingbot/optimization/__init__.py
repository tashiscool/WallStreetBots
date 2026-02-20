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
    "BacktestObjective",
    "HyperParameter",
    # Engine
    "HyperoptEngine",
    # Results
    "HyperoptResult",
    # Space definition
    "HyperoptSpace",
    "HyperoptSpaceBuilder",
    "HyperoptSummary",
    # Objectives
    "ObjectiveFunction",
    # Methods
    "OptimizationMethod",
    "ParameterType",
    "create_hyperopt_engine",
]
