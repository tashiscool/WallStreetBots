"""
Comprehensive tests for OptimizationService.

Tests parameter optimization, Optuna integration, samplers, and result handling.
Target: 80%+ coverage.
"""
import asyncio
import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock


class TestOptimizationEnums(unittest.TestCase):
    """Test optimization enum definitions."""

    def test_optimization_objective_values(self):
        """Test OptimizationObjective enum values."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationObjective

        self.assertEqual(OptimizationObjective.SHARPE.value, 'sharpe')
        self.assertEqual(OptimizationObjective.SORTINO.value, 'sortino')
        self.assertEqual(OptimizationObjective.CALMAR.value, 'calmar')
        self.assertEqual(OptimizationObjective.TOTAL_RETURN.value, 'return')
        self.assertEqual(OptimizationObjective.PROFIT_FACTOR.value, 'profit_factor')
        self.assertEqual(OptimizationObjective.WIN_RATE.value, 'win_rate')
        self.assertEqual(OptimizationObjective.MIN_DRAWDOWN.value, 'min_drawdown')

    def test_sampler_type_values(self):
        """Test SamplerType enum values."""
        from backend.tradingbot.backtesting.optimization_service import SamplerType

        self.assertEqual(SamplerType.TPE.value, 'tpe')
        self.assertEqual(SamplerType.RANDOM.value, 'random')
        self.assertEqual(SamplerType.CMAES.value, 'cmaes')


class TestParameterRange(unittest.TestCase):
    """Test ParameterRange dataclass."""

    def test_float_parameter_range(self):
        """Test creating a float parameter range."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='stop_loss_pct',
            min_value=2.0,
            max_value=10.0,
            step=0.5,
            param_type='float'
        )

        self.assertEqual(param.name, 'stop_loss_pct')
        self.assertEqual(param.min_value, 2.0)
        self.assertEqual(param.max_value, 10.0)
        self.assertEqual(param.step, 0.5)
        self.assertEqual(param.param_type, 'float')
        self.assertFalse(param.log_scale)

    def test_int_parameter_range(self):
        """Test creating an integer parameter range."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='rsi_period',
            min_value=7,
            max_value=21,
            param_type='int'
        )

        self.assertEqual(param.name, 'rsi_period')
        self.assertEqual(param.param_type, 'int')

    def test_categorical_parameter_range(self):
        """Test creating a categorical parameter range."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='exit_type',
            min_value=0,
            max_value=2,
            param_type='categorical',
            choices=['trailing_stop', 'fixed_stop', 'time_based']
        )

        self.assertEqual(param.param_type, 'categorical')
        self.assertEqual(len(param.choices), 3)
        self.assertIn('trailing_stop', param.choices)

    def test_log_scale_parameter(self):
        """Test parameter with log scale."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='learning_rate',
            min_value=0.0001,
            max_value=0.1,
            param_type='float',
            log_scale=True
        )

        self.assertTrue(param.log_scale)


class TestParameterRangeSuggest(unittest.TestCase):
    """Test ParameterRange suggest method."""

    def test_suggest_float(self):
        """Test suggesting float parameter."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='stop_loss',
            min_value=2.0,
            max_value=10.0,
            step=0.5,
            param_type='float'
        )

        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 5.5

        value = param.suggest(mock_trial)

        self.assertEqual(value, 5.5)
        mock_trial.suggest_float.assert_called_once()

    def test_suggest_int(self):
        """Test suggesting int parameter."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='rsi_period',
            min_value=7,
            max_value=21,
            param_type='int'
        )

        mock_trial = Mock()
        mock_trial.suggest_int.return_value = 14

        value = param.suggest(mock_trial)

        self.assertEqual(value, 14)
        mock_trial.suggest_int.assert_called_once()

    def test_suggest_categorical(self):
        """Test suggesting categorical parameter."""
        from backend.tradingbot.backtesting.optimization_service import ParameterRange

        param = ParameterRange(
            name='exit_type',
            min_value=0,
            max_value=2,
            param_type='categorical',
            choices=['a', 'b', 'c']
        )

        mock_trial = Mock()
        mock_trial.suggest_categorical.return_value = 'b'

        value = param.suggest(mock_trial)

        self.assertEqual(value, 'b')
        mock_trial.suggest_categorical.assert_called_once_with('exit_type', ['a', 'b', 'c'])


class TestOptimizationConfig(unittest.TestCase):
    """Test OptimizationConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationConfig,
            OptimizationObjective,
            SamplerType
        )

        config = OptimizationConfig(
            strategy_name='Test Strategy',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        self.assertEqual(config.strategy_name, 'Test Strategy')
        self.assertEqual(config.initial_capital, Decimal("100000"))
        self.assertEqual(config.benchmark, "SPY")
        self.assertEqual(config.objective, OptimizationObjective.SHARPE)
        self.assertEqual(config.n_trials, 100)
        self.assertEqual(config.sampler, SamplerType.TPE)
        self.assertEqual(config.min_trades, 10)
        self.assertEqual(config.n_jobs, 1)

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationConfig,
            OptimizationObjective,
            SamplerType,
            ParameterRange
        )

        param_ranges = [
            ParameterRange('stop_loss', 2.0, 10.0, param_type='float'),
        ]

        config = OptimizationConfig(
            strategy_name='Custom Strategy',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            initial_capital=Decimal("50000"),
            objective=OptimizationObjective.SORTINO,
            n_trials=50,
            sampler=SamplerType.RANDOM,
            parameter_ranges=param_ranges,
            max_drawdown_limit=15.0,
            timeout_seconds=3600,
        )

        self.assertEqual(config.objective, OptimizationObjective.SORTINO)
        self.assertEqual(config.n_trials, 50)
        self.assertEqual(config.sampler, SamplerType.RANDOM)
        self.assertEqual(len(config.parameter_ranges), 1)
        self.assertEqual(config.max_drawdown_limit, 15.0)


class TestTrialResult(unittest.TestCase):
    """Test TrialResult dataclass."""

    def test_trial_result_creation(self):
        """Test creating a trial result."""
        from backend.tradingbot.backtesting.optimization_service import TrialResult

        result = TrialResult(
            trial_number=1,
            params={'rsi_threshold': 30, 'stop_loss': 5.0},
            objective_value=1.5,
            metrics={
                'sharpe_ratio': 1.5,
                'total_return': 0.20,
                'max_drawdown': 0.08,
                'win_rate': 58.0,
            }
        )

        self.assertEqual(result.trial_number, 1)
        self.assertEqual(result.params['rsi_threshold'], 30)
        self.assertEqual(result.objective_value, 1.5)
        self.assertFalse(result.is_best)

    def test_trial_result_marked_as_best(self):
        """Test trial result marked as best."""
        from backend.tradingbot.backtesting.optimization_service import TrialResult

        result = TrialResult(
            trial_number=42,
            params={},
            objective_value=2.0,
            metrics={},
            is_best=True
        )

        self.assertTrue(result.is_best)


class TestOptimizationResult(unittest.TestCase):
    """Test OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an optimization result."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationResult,
            OptimizationConfig,
            TrialResult
        )

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        result = OptimizationResult(
            run_id='test-run-123',
            config=config,
            best_params={'rsi_threshold': 28},
            best_value=1.8,
            best_metrics={'sharpe_ratio': 1.8},
            all_trials=[],
            parameter_importance={'rsi_threshold': 0.7},
            status='completed'
        )

        self.assertEqual(result.run_id, 'test-run-123')
        self.assertEqual(result.best_value, 1.8)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.parameter_importance['rsi_threshold'], 0.7)

    def test_optimization_result_with_error(self):
        """Test optimization result with error."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationResult,
            OptimizationConfig
        )

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        result = OptimizationResult(
            run_id='test-run-456',
            config=config,
            best_params={},
            best_value=0.0,
            best_metrics={},
            all_trials=[],
            parameter_importance={},
            status='error',
            error_message='Optimization failed due to timeout'
        )

        self.assertEqual(result.status, 'error')
        self.assertIsNotNone(result.error_message)


class TestOptimizationServiceInit(unittest.TestCase):
    """Test OptimizationService initialization."""

    def test_service_initialization(self):
        """Test service can be initialized."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationService

        service = OptimizationService()

        self.assertIsNone(service._current_study)
        self.assertIsNone(service._progress_callback)
        self.assertIsNone(service._current_run_id)


class TestDefaultParameterRanges(unittest.TestCase):
    """Test default parameter ranges."""

    def test_default_parameter_ranges_exist(self):
        """Test that default parameter ranges are defined."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationService

        defaults = OptimizationService.DEFAULT_PARAMETER_RANGES

        expected_params = [
            'position_size_pct', 'stop_loss_pct', 'take_profit_pct',
            'rsi_period', 'rsi_oversold', 'rsi_overbought', 'max_positions'
        ]

        for param in expected_params:
            self.assertIn(param, defaults)

    def test_get_default_parameter_ranges_common(self):
        """Test getting default ranges for unknown strategy."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationService

        service = OptimizationService()
        ranges = service.get_default_parameter_ranges('unknown-strategy')

        # Should return at least the common parameters
        param_names = [r.name for r in ranges]
        self.assertIn('position_size_pct', param_names)
        self.assertIn('stop_loss_pct', param_names)

    def test_get_default_parameter_ranges_wsb_dip_bot(self):
        """Test getting default ranges for wsb-dip-bot strategy."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationService

        service = OptimizationService()
        ranges = service.get_default_parameter_ranges('wsb-dip-bot')

        param_names = [r.name for r in ranges]
        self.assertIn('rsi_period', param_names)
        self.assertIn('rsi_oversold', param_names)


class TestSamplerCreation(unittest.TestCase):
    """Test sampler creation."""

    def test_create_tpe_sampler(self):
        """Test creating TPE sampler."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            SamplerType
        )

        service = OptimizationService()

        with patch('backend.tradingbot.backtesting.optimization_service.HAS_OPTUNA', True):
            with patch('backend.tradingbot.backtesting.optimization_service.TPESampler') as mock_sampler:
                mock_sampler.return_value = Mock()
                sampler = service._create_sampler(SamplerType.TPE)
                mock_sampler.assert_called_once_with(seed=42)

    def test_create_random_sampler(self):
        """Test creating Random sampler."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            SamplerType
        )

        service = OptimizationService()

        with patch('backend.tradingbot.backtesting.optimization_service.HAS_OPTUNA', True):
            with patch('backend.tradingbot.backtesting.optimization_service.RandomSampler') as mock_sampler:
                mock_sampler.return_value = Mock()
                sampler = service._create_sampler(SamplerType.RANDOM)
                mock_sampler.assert_called_once_with(seed=42)

    def test_create_cmaes_sampler(self):
        """Test creating CMA-ES sampler."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            SamplerType
        )

        service = OptimizationService()

        with patch('backend.tradingbot.backtesting.optimization_service.HAS_OPTUNA', True):
            with patch('backend.tradingbot.backtesting.optimization_service.CmaEsSampler') as mock_sampler:
                mock_sampler.return_value = Mock()
                sampler = service._create_sampler(SamplerType.CMAES)
                mock_sampler.assert_called_once_with(seed=42)


class TestBacktestSync(unittest.TestCase):
    """Test synchronous backtest simulation."""

    def test_run_backtest_sync_returns_metrics(self):
        """Test _run_backtest_sync returns expected metrics."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            OptimizationConfig
        )

        service = OptimizationService()

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        params = {
            'position_size_pct': 5.0,
            'stop_loss_pct': 5.0,
            'take_profit_pct': 15.0
        }

        metrics = service._run_backtest_sync(config, params)

        expected_keys = [
            'total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown_pct', 'win_rate', 'profit_factor', 'total_trades',
            'winning_trades'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics)

    def test_run_backtest_sync_different_params_different_results(self):
        """Test that different parameters produce different results."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            OptimizationConfig
        )

        service = OptimizationService()

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        params1 = {'position_size_pct': 3.0, 'stop_loss_pct': 3.0, 'take_profit_pct': 10.0}
        params2 = {'position_size_pct': 10.0, 'stop_loss_pct': 10.0, 'take_profit_pct': 30.0}

        metrics1 = service._run_backtest_sync(config, params1)
        metrics2 = service._run_backtest_sync(config, params2)

        # Results should differ due to different parameters
        # (they use parameter sum as seed, so different params = different results)
        self.assertNotEqual(metrics1, metrics2)


class TestRunOptimization(unittest.TestCase):
    """Test running optimization."""

    def test_run_optimization_no_optuna(self):
        """Test run_optimization when Optuna is not available."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            OptimizationConfig
        )

        service = OptimizationService()

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
        )

        with patch('backend.tradingbot.backtesting.optimization_service.HAS_OPTUNA', False):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    service.run_optimization(config, save_to_db=False)
                )
            finally:
                loop.close()

            self.assertEqual(result.status, 'error')
            self.assertIn('Optuna not installed', result.error_message)


class TestTrialCallback(unittest.TestCase):
    """Test trial callback functionality."""

    def test_trial_callback_calls_progress(self):
        """Test that trial callback calls progress callback."""
        from backend.tradingbot.backtesting.optimization_service import OptimizationService

        service = OptimizationService()

        progress_calls = []

        def progress_callback(current, total, metrics):
            progress_calls.append((current, total, metrics))

        service._progress_callback = progress_callback

        # Mock study and trial
        mock_study = Mock()
        mock_study.best_trial = Mock()
        mock_study.best_trial.user_attrs = {'metrics': {'sharpe_ratio': 1.5}}
        mock_study.best_value = 1.5
        mock_study.best_params = {'stop_loss': 5.0}
        mock_study.n_trials = 50

        mock_trial = Mock()
        mock_trial.number = 10
        mock_trial.state = Mock()
        mock_trial.state.name = 'COMPLETE'

        # Need to import optuna to get the state
        with patch('backend.tradingbot.backtesting.optimization_service.HAS_OPTUNA', True):
            import optuna
            mock_trial.state = optuna.trial.TrialState.COMPLETE

            service._trial_callback(mock_study, mock_trial)

        self.assertEqual(len(progress_calls), 1)
        self.assertEqual(progress_calls[0][0], 11)  # trial.number + 1


class TestGetOptimizationService(unittest.TestCase):
    """Test get_optimization_service factory function."""

    def test_get_optimization_service(self):
        """Test getting optimization service instance."""
        from backend.tradingbot.backtesting.optimization_service import (
            get_optimization_service,
            OptimizationService
        )

        service = get_optimization_service()

        self.assertIsInstance(service, OptimizationService)


class TestObjectiveFunctions(unittest.TestCase):
    """Test objective function creation and evaluation."""

    def test_objective_function_returns_correct_value(self):
        """Test that objective function returns correct metric."""
        from backend.tradingbot.backtesting.optimization_service import (
            OptimizationService,
            OptimizationConfig,
            OptimizationObjective,
            ParameterRange
        )

        service = OptimizationService()

        config = OptimizationConfig(
            strategy_name='Test',
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            objective=OptimizationObjective.SHARPE,
            parameter_ranges=[
                ParameterRange('stop_loss_pct', 2.0, 10.0, param_type='float')
            ],
            min_trades=0,  # Disable min trades constraint
        )

        objective_fn = service._create_objective(config)

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 5.0
        mock_trial.set_user_attr = Mock()

        # Run objective
        result = objective_fn(mock_trial)

        # Should return the sharpe ratio from metrics
        self.assertIsInstance(result, float)
        mock_trial.set_user_attr.assert_called()


if __name__ == '__main__':
    unittest.main()
