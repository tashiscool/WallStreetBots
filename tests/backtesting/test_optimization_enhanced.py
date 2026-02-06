"""Tests for Enhanced Optimization Service."""
import pytest

from backend.tradingbot.backtesting.loss_functions import (
    BaseLossFunction, SharpeLoss, SortinoLoss, CalmarLoss,
    ProfitLoss, MaxDrawdownLoss, CustomWeightedLoss, get_loss_function,
    LOSS_FUNCTIONS,
)
from backend.tradingbot.backtesting.optimization_service import (
    OptimizationObjective, SamplerType, PrunerType,
    ParameterRange, OptimizationConfig, OptimizationService,
)


# --- Loss Functions ---

class TestSharpeLoss:
    def test_returns_negative_sharpe(self):
        loss = SharpeLoss()
        result = loss.calculate({"sharpe_ratio": 1.5})
        assert result == -1.5

    def test_defaults_to_zero(self):
        loss = SharpeLoss()
        result = loss.calculate({})
        assert result == 0.0


class TestSortinoLoss:
    def test_returns_negative_sortino(self):
        loss = SortinoLoss()
        assert loss.calculate({"sortino_ratio": 2.0}) == -2.0


class TestCalmarLoss:
    def test_returns_negative_calmar(self):
        loss = CalmarLoss()
        assert loss.calculate({"calmar_ratio": 3.0}) == -3.0


class TestProfitLoss:
    def test_returns_negative_total_return(self):
        loss = ProfitLoss()
        assert loss.calculate({"total_return_pct": 15.0}) == -15.0


class TestMaxDrawdownLoss:
    def test_returns_drawdown_directly(self):
        loss = MaxDrawdownLoss()
        assert loss.calculate({"max_drawdown_pct": 12.5}) == 12.5

    def test_defaults_to_100(self):
        loss = MaxDrawdownLoss()
        assert loss.calculate({}) == 100.0


class TestCustomWeightedLoss:
    def test_weighted_combination(self):
        loss = CustomWeightedLoss({
            "sharpe_ratio": 0.5,
            "total_return_pct": 0.5,
        })
        metrics = {"sharpe_ratio": 2.0, "total_return_pct": 10.0}
        result = loss.calculate(metrics)
        # score = 0.5*2.0 + 0.5*10.0 = 6.0, loss = -6.0
        assert result == -6.0

    def test_handles_missing_metrics(self):
        loss = CustomWeightedLoss({"sharpe_ratio": 1.0, "missing_key": 1.0})
        result = loss.calculate({"sharpe_ratio": 2.0})
        assert result == -2.0


class TestGetLossFunction:
    def test_valid_names(self):
        for name in LOSS_FUNCTIONS:
            fn = get_loss_function(name)
            assert isinstance(fn, BaseLossFunction)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown loss function"):
            get_loss_function("nonexistent")


# --- Enums ---

class TestPrunerType:
    def test_all_values(self):
        assert PrunerType.NONE.value == "none"
        assert PrunerType.MEDIAN.value == "median"
        assert PrunerType.PERCENTILE.value == "percentile"
        assert PrunerType.HYPERBAND.value == "hyperband"
        assert PrunerType.THRESHOLD.value == "threshold"


class TestSamplerType:
    def test_nsgaii(self):
        assert SamplerType.NSGAII.value == "nsgaii"


# --- ParameterRange ---

class TestParameterRange:
    def test_float_range(self):
        pr = ParameterRange(name="x", min_value=0.0, max_value=1.0, param_type="float")
        assert pr.name == "x"
        assert pr.min_value == 0.0
        assert pr.max_value == 1.0

    def test_int_range(self):
        pr = ParameterRange(name="n", min_value=1, max_value=10, step=1, param_type="int")
        assert pr.param_type == "int"

    def test_categorical_range(self):
        pr = ParameterRange(name="method", min_value=0, max_value=0, param_type="categorical", choices=["a", "b"])
        assert pr.choices == ["a", "b"]


# --- OptimizationConfig ---

class TestOptimizationConfig:
    def test_default_config(self):
        from datetime import date
        config = OptimizationConfig(
            strategy_name="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert config.n_trials == 100
        assert config.sampler == SamplerType.TPE
        assert config.pruner == PrunerType.NONE
        assert config.storage_url is None
        assert config.objectives is None

    def test_multi_objective_config(self):
        from datetime import date
        config = OptimizationConfig(
            strategy_name="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            objectives=[OptimizationObjective.SHARPE, OptimizationObjective.MIN_DRAWDOWN],
            sampler=SamplerType.NSGAII,
        )
        assert len(config.objectives) == 2
        assert config.sampler == SamplerType.NSGAII


# --- OptimizationService ---

class TestOptimizationService:
    def test_init(self):
        svc = OptimizationService()
        assert svc._current_study is None
        assert svc._current_run_id is None

    def test_default_parameter_ranges(self):
        svc = OptimizationService()
        ranges = svc.get_default_parameter_ranges("wsb-dip-bot")
        names = [r.name for r in ranges]
        assert "position_size_pct" in names
        assert "stop_loss_pct" in names
        assert "rsi_period" in names

    def test_default_ranges_unknown_strategy(self):
        svc = OptimizationService()
        ranges = svc.get_default_parameter_ranges("unknown_strategy")
        # Should return common parameters only
        names = [r.name for r in ranges]
        assert "position_size_pct" in names
        assert len(ranges) == 4  # 4 common params

    def test_run_backtest_sync(self):
        svc = OptimizationService()
        from datetime import date
        config = OptimizationConfig(
            strategy_name="test",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        params = {"position_size_pct": 5.0, "stop_loss_pct": 5.0, "take_profit_pct": 15.0}
        metrics = svc._run_backtest_sync(config, params)
        assert "sharpe_ratio" in metrics
        assert "total_return_pct" in metrics
        assert "max_drawdown_pct" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics

    def test_create_pruner_none(self):
        svc = OptimizationService()
        assert svc._create_pruner(PrunerType.NONE) is None

    def test_create_sampler(self):
        svc = OptimizationService()
        try:
            import optuna
            sampler = svc._create_sampler(SamplerType.TPE)
            assert isinstance(sampler, optuna.samplers.TPESampler)
        except ImportError:
            pytest.skip("Optuna not installed")
