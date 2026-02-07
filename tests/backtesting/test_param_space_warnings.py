"""Tests for parameter space cardinality and overfitting warnings."""

import pytest
from backend.tradingbot.backtesting.optimization_service import (
    OptimizationConfig,
    OptimizationResult,
    OptimizationService,
    ParameterRange,
)
from datetime import date


def _make_config(**kwargs):
    defaults = dict(
        strategy_name='test',
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
        n_trials=50,
        parameter_ranges=[],
    )
    defaults.update(kwargs)
    return OptimizationConfig(**defaults)


class TestCalculateParameterSpaceSize:
    def test_integer_range(self):
        config = _make_config(parameter_ranges=[
            ParameterRange('x', 1, 10, step=1, param_type='int'),
        ])
        assert OptimizationService._calculate_parameter_space_size(config) == 10

    def test_float_range_with_step(self):
        config = _make_config(parameter_ranges=[
            ParameterRange('x', 0.0, 1.0, step=0.1, param_type='float'),
        ])
        assert OptimizationService._calculate_parameter_space_size(config) == 11

    def test_categorical(self):
        config = _make_config(parameter_ranges=[
            ParameterRange('x', 0, 0, param_type='categorical', choices=['a', 'b', 'c']),
        ])
        assert OptimizationService._calculate_parameter_space_size(config) == 3

    def test_multiple_ranges_product(self):
        config = _make_config(parameter_ranges=[
            ParameterRange('x', 1, 5, step=1, param_type='int'),   # 5
            ParameterRange('y', 0, 1, step=0.5, param_type='float'),  # 3
        ])
        assert OptimizationService._calculate_parameter_space_size(config) == 15

    def test_empty_ranges(self):
        config = _make_config(parameter_ranges=[])
        assert OptimizationService._calculate_parameter_space_size(config) == 0

    def test_continuous_no_step(self):
        """Continuous param without step → estimated as 1000."""
        config = _make_config(parameter_ranges=[
            ParameterRange('x', 0.0, 1.0, param_type='float'),
        ])
        assert OptimizationService._calculate_parameter_space_size(config) == 1000


class TestCheckOverfittingRisk:
    def test_over_explored_warning(self):
        config = _make_config(
            n_trials=100,
            parameter_ranges=[
                ParameterRange('x', 1, 10, step=1, param_type='int'),  # space = 10
            ],
        )
        warnings = OptimizationService._check_overfitting_risk(config)
        assert any('Over-explored' in w for w in warnings)

    def test_more_params_than_data(self):
        config = _make_config(
            parameter_ranges=[
                ParameterRange('x', 0.0, 1.0, param_type='float'),  # 1000
            ],
        )
        warnings = OptimizationService._check_overfitting_risk(config, n_observations=100)
        assert any('More parameter combos' in w for w in warnings)

    def test_too_many_degrees_of_freedom(self):
        config = _make_config(
            parameter_ranges=[
                ParameterRange(f'p{i}', 0, 1, step=0.5, param_type='float')
                for i in range(8)  # > 7
            ],
        )
        warnings = OptimizationService._check_overfitting_risk(config)
        assert any('parameter dimensions' in w for w in warnings)

    def test_no_warnings_for_small_space(self):
        config = _make_config(
            n_trials=3,
            parameter_ranges=[
                ParameterRange('x', 1, 10, step=1, param_type='int'),  # 10 values, eff_dof=9
            ],
        )
        warnings = OptimizationService._check_overfitting_risk(config, n_observations=1000)
        assert len(warnings) == 0


class TestOptimizationResultFields:
    def test_result_has_new_fields(self):
        result = OptimizationResult(
            run_id='test',
            config=_make_config(),
            best_params={},
            best_value=0.0,
            best_metrics={},
            all_trials=[],
            parameter_importance={},
            parameter_space_size=100,
            overfitting_warnings=['test warning'],
        )
        assert result.parameter_space_size == 100
        assert 'test warning' in result.overfitting_warnings

    def test_result_defaults(self):
        result = OptimizationResult(
            run_id='test',
            config=_make_config(),
            best_params={},
            best_value=0.0,
            best_metrics={},
            all_trials=[],
            parameter_importance={},
        )
        assert result.parameter_space_size is None
        assert result.overfitting_warnings == []
