"""Parameter Optimization Service using Optuna.

Provides hyperparameter optimization for trading strategies with:
- Multiple optimization objectives (Sharpe, Sortino, Return, etc.)
- Various samplers (TPE, Random, CMA-ES)
- Progress tracking and persistence
- Parallel trial execution
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.utils import timezone

logger = logging.getLogger(__name__)

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not available - optimization features disabled")


class OptimizationObjective(Enum):
    """Optimization objectives."""
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    TOTAL_RETURN = "return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    MIN_DRAWDOWN = "min_drawdown"


class SamplerType(Enum):
    """Optuna sampler types."""
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"
    NSGAII = "nsgaii"


class PrunerType(Enum):
    """Optuna pruner types for early trial termination."""
    NONE = "none"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    THRESHOLD = "threshold"


@dataclass
class ParameterRange:
    """Definition of a parameter range for optimization."""
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    log_scale: bool = False
    param_type: str = "float"  # "float", "int", "categorical"
    choices: Optional[List[Any]] = None

    def suggest(self, trial: 'optuna.Trial') -> Any:
        """Suggest a value for this parameter using Optuna trial."""
        if self.param_type == "categorical" and self.choices:
            return trial.suggest_categorical(self.name, self.choices)
        elif self.param_type == "int":
            return trial.suggest_int(
                self.name,
                int(self.min_value),
                int(self.max_value),
                step=int(self.step) if self.step else 1,
                log=self.log_scale
            )
        else:
            if self.step:
                return trial.suggest_float(
                    self.name,
                    self.min_value,
                    self.max_value,
                    step=self.step,
                    log=self.log_scale
                )
            return trial.suggest_float(
                self.name,
                self.min_value,
                self.max_value,
                log=self.log_scale
            )


@dataclass
class OptimizationConfig:
    """Configuration for an optimization run."""
    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: Decimal = Decimal("100000")
    benchmark: str = "SPY"

    # Optimization settings
    objective: OptimizationObjective = OptimizationObjective.SHARPE
    objectives: Optional[List[OptimizationObjective]] = None  # Multi-objective
    n_trials: int = 100
    sampler: SamplerType = SamplerType.TPE
    pruner: PrunerType = PrunerType.NONE
    parameter_ranges: List[ParameterRange] = field(default_factory=list)

    # Loss function (alternative to objective enum - uses loss_functions.py)
    loss_function_name: Optional[str] = None  # e.g. "sharpe", "sortino", "calmar", "profit", "max_drawdown"

    # Vectorized backtesting for faster parameter sweeps
    use_vectorized: bool = False  # Use VectorizedBacktestEngine instead of simulated results

    # Constraints
    min_trades: int = 10
    max_drawdown_limit: Optional[float] = None

    # Execution
    n_jobs: int = 1
    timeout_seconds: Optional[int] = None

    # Storage for distributed optimization
    storage_url: Optional[str] = None  # e.g. "sqlite:///optuna.db" or PostgreSQL URL


@dataclass
class TrialResult:
    """Result from a single optimization trial."""
    trial_number: int
    params: Dict[str, Any]
    objective_value: float
    metrics: Dict[str, float]
    is_best: bool = False


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    run_id: str
    config: OptimizationConfig
    best_params: Dict[str, Any]
    best_value: float
    best_metrics: Dict[str, float]
    all_trials: List[TrialResult]
    parameter_importance: Dict[str, float]
    status: str = "completed"
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameter_space_size: Optional[int] = None
    overfitting_warnings: List[str] = field(default_factory=list)
    actual_trials_evaluated: int = 0  # includes pruned/failed (real DSR input)


class OptimizationService:
    """Service for running parameter optimization on strategies.

    Uses Optuna for Bayesian optimization with support for:
    - Multiple objectives (maximize Sharpe, minimize drawdown, etc.)
    - Parameter importance analysis
    - Progress callbacks for real-time updates
    - Database persistence
    """

    # Default parameter ranges for common strategy parameters
    DEFAULT_PARAMETER_RANGES = {
        'position_size_pct': ParameterRange(
            name='position_size_pct',
            min_value=1.0,
            max_value=10.0,
            step=0.5,
            param_type='float'
        ),
        'stop_loss_pct': ParameterRange(
            name='stop_loss_pct',
            min_value=2.0,
            max_value=15.0,
            step=1.0,
            param_type='float'
        ),
        'take_profit_pct': ParameterRange(
            name='take_profit_pct',
            min_value=5.0,
            max_value=30.0,
            step=2.0,
            param_type='float'
        ),
        'rsi_period': ParameterRange(
            name='rsi_period',
            min_value=7,
            max_value=21,
            step=1,
            param_type='int'
        ),
        'rsi_oversold': ParameterRange(
            name='rsi_oversold',
            min_value=20,
            max_value=40,
            step=5,
            param_type='int'
        ),
        'rsi_overbought': ParameterRange(
            name='rsi_overbought',
            min_value=60,
            max_value=80,
            step=5,
            param_type='int'
        ),
        'max_positions': ParameterRange(
            name='max_positions',
            min_value=3,
            max_value=15,
            step=1,
            param_type='int'
        ),
    }

    def __init__(self, price_data: Optional[Any] = None):
        """Initialize the optimization service.

        Args:
            price_data: Optional price data (numpy array or pandas Series/DataFrame)
                       for use with VectorizedBacktestEngine. Required if
                       config.use_vectorized is True.
        """
        self._current_study: Optional['optuna.Study'] = None
        self._progress_callback: Optional[Callable] = None
        self._current_run_id: Optional[str] = None
        self._price_data = price_data
        self._vectorized_engine = None

    async def run_optimization(
        self,
        config: OptimizationConfig,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
        save_to_db: bool = True,
        user=None,
    ) -> OptimizationResult:
        """Run parameter optimization for a strategy.

        Args:
            config: Optimization configuration
            progress_callback: Callback(current_trial, total_trials, best_metrics)
            save_to_db: Whether to persist results to database
            user: User who initiated the optimization

        Returns:
            OptimizationResult with best parameters and all trials
        """
        if not HAS_OPTUNA:
            return OptimizationResult(
                run_id=str(uuid.uuid4()),
                config=config,
                best_params={},
                best_value=0.0,
                best_metrics={},
                all_trials=[],
                parameter_importance={},
                status="error",
                error_message="Optuna not installed"
            )

        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        self._progress_callback = progress_callback
        started_at = datetime.now()

        logger.info(f"Starting optimization run {run_id} for {config.strategy_name}")

        try:
            # Create sampler and pruner
            sampler = self._create_sampler(config.sampler)
            pruner = self._create_pruner(config.pruner) if HAS_OPTUNA else None

            # Storage for distributed optimization
            storage = config.storage_url if config.storage_url else None

            # Multi-objective or single-objective
            study_name = f"opt_{config.strategy_name}_{run_id[:8]}"
            if config.objectives and len(config.objectives) > 1:
                directions = [
                    "minimize" if obj == OptimizationObjective.MIN_DRAWDOWN else "maximize"
                    for obj in config.objectives
                ]
                study = optuna.create_study(
                    directions=directions,
                    sampler=sampler,
                    pruner=pruner,
                    study_name=study_name,
                    storage=storage,
                )
            else:
                direction = "minimize" if config.objective == OptimizationObjective.MIN_DRAWDOWN else "maximize"
                study = optuna.create_study(
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner,
                    study_name=study_name,
                    storage=storage,
                )
            self._current_study = study

            # Create objective function
            objective_fn = self._create_objective(config)

            # Run optimization
            study.optimize(
                objective_fn,
                n_trials=config.n_trials,
                timeout=config.timeout_seconds,
                n_jobs=config.n_jobs,
                callbacks=[self._trial_callback],
                show_progress_bar=False,
            )

            # Count ALL evaluated trials (complete + pruned + failed)
            # This is the real num_trials for DSR, not config.n_trials
            actual_evaluated = sum(
                1 for t in study.trials
                if t.state in (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.PRUNED,
                    optuna.trial.TrialState.FAIL,
                )
            )

            # Calculate parameter importance
            try:
                importance = optuna.importance.get_param_importances(study)
            except Exception:
                importance = {}

            # Collect all trial results
            all_trials = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    all_trials.append(TrialResult(
                        trial_number=trial.number,
                        params=trial.params,
                        objective_value=trial.value,
                        metrics=trial.user_attrs.get('metrics', {}),
                        is_best=(trial.number == study.best_trial.number)
                    ))

            # Get best trial metrics
            best_metrics = study.best_trial.user_attrs.get('metrics', {})
            best_metrics[config.objective.value] = study.best_value

            # Parameter space analysis
            param_space = self._calculate_parameter_space_size(config)
            overfit_warnings = self._check_overfitting_risk(config)
            if overfit_warnings:
                logger.warning(f"Overfitting risk: {overfit_warnings}")

            result = OptimizationResult(
                run_id=run_id,
                config=config,
                best_params=study.best_params,
                best_value=study.best_value,
                best_metrics=best_metrics,
                all_trials=all_trials,
                parameter_importance=importance,
                status="completed",
                started_at=started_at,
                completed_at=datetime.now(),
                parameter_space_size=param_space,
                overfitting_warnings=overfit_warnings,
                actual_trials_evaluated=actual_evaluated,
            )

            # Save to database
            if save_to_db and user:
                await self._save_to_database(result, user)

            logger.info(f"Optimization {run_id} completed. Best {config.objective.value}: {study.best_value:.4f}")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                run_id=run_id,
                config=config,
                best_params={},
                best_value=0.0,
                best_metrics={},
                all_trials=[],
                parameter_importance={},
                status="error",
                error_message=str(e),
                started_at=started_at,
                completed_at=datetime.now(),
            )

    @staticmethod
    def _calculate_parameter_space_size(config: OptimizationConfig) -> int:
        """Calculate the total cardinality of the parameter search space."""
        if not config.parameter_ranges:
            return 0

        total = 1
        for pr in config.parameter_ranges:
            if pr.param_type == "categorical" and pr.choices:
                total *= len(pr.choices)
            elif pr.step and pr.step > 0:
                total *= int((pr.max_value - pr.min_value) / pr.step) + 1
            else:
                # Continuous — estimate as a large number
                total *= 1000
        return total

    @staticmethod
    def _check_overfitting_risk(
        config: OptimizationConfig,
        n_observations: int = 252,
        training_window_obs: Optional[int] = None,
    ) -> List[str]:
        """Generate warnings about potential overfitting.

        Parameters
        ----------
        config : OptimizationConfig
        n_observations : int
            Total number of data observations.
        training_window_obs : int, optional
            Observations in a single training window (for walk-forward).
            If provided, uses this for trial-to-sample checks instead of
            total n_observations.
        """
        warnings: List[str] = []
        space_size = OptimizationService._calculate_parameter_space_size(config)

        if space_size > 0 and config.n_trials > space_size * 0.5:
            warnings.append(
                f"Over-explored: {config.n_trials} trials explore "
                f">{50}% of {space_size} possible combinations"
            )

        if space_size > n_observations:
            warnings.append(
                f"More parameter combos ({space_size}) than data points ({n_observations})"
            )

        # Effective degrees of freedom: continuous ranges count more
        eff_dof = 0.0
        for pr in config.parameter_ranges:
            if pr.param_type == "categorical" and pr.choices:
                eff_dof += max(1, len(pr.choices) - 1)
            elif pr.step and pr.step > 0:
                n_vals = int((pr.max_value - pr.min_value) / pr.step) + 1
                eff_dof += max(1, n_vals - 1)
            else:
                # Continuous — high effective DoF
                eff_dof += 10  # penalty for unconstrained continuous range

        if eff_dof > 20:
            warnings.append(
                f"High effective degrees of freedom: {eff_dof:.0f} "
                f"(continuous ranges penalised; recommended ≤20)"
            )

        if len(config.parameter_ranges) > 7:
            warnings.append(
                f"Too many parameter dimensions: {len(config.parameter_ranges)} "
                f"(recommended ≤7)"
            )

        # Trial-to-sample ratio (use training window if available)
        sample_size = training_window_obs or n_observations
        if sample_size > 0 and config.n_trials > 0:
            ratio = config.n_trials / sample_size
            if ratio > 0.5:
                warnings.append(
                    f"Trial-to-sample ratio {ratio:.2f} > 0.5: "
                    f"{config.n_trials} trials vs {sample_size} training observations"
                )

        return warnings

    def _create_sampler(self, sampler_type: SamplerType) -> 'optuna.samplers.BaseSampler':
        """Create an Optuna sampler based on type."""
        if sampler_type == SamplerType.TPE:
            return TPESampler(seed=42)
        elif sampler_type == SamplerType.RANDOM:
            return RandomSampler(seed=42)
        elif sampler_type == SamplerType.CMAES:
            return CmaEsSampler(seed=42)
        elif sampler_type == SamplerType.NSGAII:
            return optuna.samplers.NSGAIISampler(seed=42)
        else:
            return TPESampler(seed=42)

    def _create_pruner(self, pruner_type: PrunerType) -> Optional['optuna.pruners.BasePruner']:
        """Create an Optuna pruner for early trial stopping."""
        if pruner_type == PrunerType.NONE:
            return None
        elif pruner_type == PrunerType.MEDIAN:
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        elif pruner_type == PrunerType.PERCENTILE:
            return optuna.pruners.PercentilePruner(25.0, n_startup_trials=5)
        elif pruner_type == PrunerType.HYPERBAND:
            return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100)
        elif pruner_type == PrunerType.THRESHOLD:
            return optuna.pruners.ThresholdPruner(lower=0.0)
        return None

    def _create_objective(self, config: OptimizationConfig) -> Callable:
        """Create the objective function for optimization.

        Supports two modes:
        1. loss_function_name set -> uses modular loss functions from loss_functions.py
        2. Otherwise -> uses legacy objective enum mapping
        """
        # Try to resolve a modular loss function
        loss_fn = None
        if config.loss_function_name:
            try:
                from .loss_functions import get_loss_function
                loss_fn = get_loss_function(config.loss_function_name)
            except (ImportError, ValueError) as e:
                logger.warning(f"Could not load loss function '{config.loss_function_name}': {e}")

        def objective(trial: 'optuna.Trial') -> float:
            # Suggest parameter values
            params = {}
            for param_range in config.parameter_ranges:
                params[param_range.name] = param_range.suggest(trial)

            # Run backtest with these parameters
            try:
                metrics = self._run_backtest_sync(config, params)
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()

            # Check constraints
            if config.min_trades and metrics.get('total_trades', 0) < config.min_trades:
                raise optuna.TrialPruned()

            if config.max_drawdown_limit and metrics.get('max_drawdown_pct', 100) > config.max_drawdown_limit:
                raise optuna.TrialPruned()

            # Store metrics for later retrieval
            trial.set_user_attr('metrics', metrics)

            # Use modular loss function if available (returns negative for maximization)
            if loss_fn is not None:
                return -loss_fn.calculate(metrics)  # Negate: loss is "lower is better", Optuna maximizes

            # Legacy objective enum mapping
            objective_map = {
                OptimizationObjective.SHARPE: metrics.get('sharpe_ratio', 0),
                OptimizationObjective.SORTINO: metrics.get('sortino_ratio', 0),
                OptimizationObjective.CALMAR: metrics.get('calmar_ratio', 0),
                OptimizationObjective.TOTAL_RETURN: metrics.get('total_return_pct', 0),
                OptimizationObjective.PROFIT_FACTOR: metrics.get('profit_factor', 0),
                OptimizationObjective.WIN_RATE: metrics.get('win_rate', 0),
                OptimizationObjective.MIN_DRAWDOWN: metrics.get('max_drawdown_pct', 100),
            }

            return objective_map.get(config.objective, 0)

        return objective

    def _run_backtest_sync(self, config: OptimizationConfig, params: Dict) -> Dict[str, float]:
        """Run a synchronous backtest with given parameters.

        Supports two modes:
        1. use_vectorized=True: Uses VectorizedBacktestEngine for 100x speed
        2. Default: Uses simulated results (for development/testing)

        Returns key metrics for objective evaluation.
        """
        import numpy as np

        # Vectorized backtesting mode - 100x faster for parameter sweeps
        if config.use_vectorized and self._price_data is not None:
            return self._run_vectorized_backtest(config, params)

        # Simulate backtest results based on parameters
        # In production, this would call the actual BacktestEngine
        np.random.seed(int(sum(params.values()) * 1000) % 2**31)

        # Parameter effects on metrics (simplified model)
        position_size = params.get('position_size_pct', 5.0)
        stop_loss = params.get('stop_loss_pct', 5.0)
        take_profit = params.get('take_profit_pct', 15.0)

        # Base metrics with parameter influence
        base_return = np.random.normal(10, 15)

        # Stop loss reduces drawdown but may hurt returns
        drawdown_factor = max(0.5, 1 - stop_loss / 30)
        return_factor = max(0.7, 1 - stop_loss / 50)

        # Take profit affects win rate and average win
        win_rate = 45 + np.random.normal(0, 5) + (stop_loss / take_profit) * 20
        win_rate = min(75, max(30, win_rate))

        # Position size affects volatility
        volatility = 15 * (position_size / 5)

        total_return = base_return * return_factor + np.random.normal(0, 5)
        max_drawdown = abs(np.random.normal(10, 5)) * drawdown_factor
        max_drawdown = min(50, max(2, max_drawdown))

        # Calculate Sharpe
        risk_free = 5.0
        sharpe = (total_return - risk_free) / volatility if volatility > 0 else 0
        sortino = sharpe * 1.2  # Simplified
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        total_trades = int(50 + np.random.normal(0, 20))
        total_trades = max(5, total_trades)
        winning_trades = int(total_trades * win_rate / 100)

        profit_factor = (winning_trades * take_profit) / ((total_trades - winning_trades) * stop_loss + 0.001)
        profit_factor = min(5, max(0.5, profit_factor))

        return {
            'total_return_pct': round(total_return, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'calmar_ratio': round(calmar, 3),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
        }

    def _run_vectorized_backtest(self, config: OptimizationConfig, params: Dict) -> Dict[str, float]:
        """Run backtest using VectorizedBacktestEngine for fast parameter sweeps.

        Uses the vectorized engine which operates on numpy arrays for 100x speedup
        over event-driven backtesting.
        """
        import numpy as np
        from .vectorized_engine import VectorizedBacktestEngine

        # Lazily initialize engine
        if self._vectorized_engine is None:
            self._vectorized_engine = VectorizedBacktestEngine(
                prices=np.asarray(self._price_data, dtype=np.float64),
                initial_capital=float(config.initial_capital),
            )

        # Generate signals from parameters (simple threshold-based)
        prices = np.asarray(self._price_data, dtype=np.float64)
        n = len(prices)
        signals = np.zeros(n)

        # Use params to generate signals
        rsi_period = int(params.get('rsi_period', 14))
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)

        # Simple RSI-like signal generation
        if n > rsi_period:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.convolve(gains, np.ones(rsi_period) / rsi_period, mode='valid')
            avg_loss = np.convolve(losses, np.ones(rsi_period) / rsi_period, mode='valid')

            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - 100 / (1 + rs)

            offset = rsi_period
            for i in range(len(rsi)):
                if rsi[i] < rsi_oversold:
                    signals[offset + i] = 1  # Buy
                elif rsi[i] > rsi_overbought:
                    signals[offset + i] = -1  # Sell

        # Run vectorized backtest
        result = self._vectorized_engine.run(
            signals=signals,
            commission=params.get('commission', 0.001),
        )

        return {
            'total_return_pct': round(result.total_return * 100, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 3),
            'sortino_ratio': round(result.sortino_ratio, 3),
            'calmar_ratio': round(result.calmar_ratio, 3),
            'max_drawdown_pct': round(result.max_drawdown * 100, 2),
            'win_rate': round(result.win_rate * 100, 1),
            'profit_factor': round(result.profit_factor, 2),
            'total_trades': result.total_trades,
        }

    def _trial_callback(self, study: 'optuna.Study', trial: 'optuna.trial.FrozenTrial'):
        """Callback called after each trial completes."""
        if self._progress_callback and trial.state == optuna.trial.TrialState.COMPLETE:
            best_metrics = study.best_trial.user_attrs.get('metrics', {}) if study.best_trial else {}
            self._progress_callback(
                trial.number + 1,
                study.n_trials if hasattr(study, 'n_trials') else 100,
                {
                    'best_value': study.best_value if study.best_trial else None,
                    'best_params': study.best_params if study.best_trial else {},
                    **best_metrics
                }
            )

    async def _save_to_database(self, result: OptimizationResult, user) -> None:
        """Save optimization results to database."""
        from backend.tradingbot.models.models import OptimizationRun

        OptimizationRun.objects.create(
            run_id=result.run_id,
            user=user,
            name=f"{result.config.strategy_name} Optimization",
            strategy_name=result.config.strategy_name,
            start_date=result.config.start_date,
            end_date=result.config.end_date,
            initial_capital=result.config.initial_capital,
            loss_function=result.config.objective.value,
            n_trials=result.config.n_trials,
            sampler=result.config.sampler.value,
            parameter_ranges={
                pr.name: {
                    'min': pr.min_value,
                    'max': pr.max_value,
                    'step': pr.step,
                    'type': pr.param_type,
                }
                for pr in result.config.parameter_ranges
            },
            status=result.status,
            progress=100,
            current_trial=len(result.all_trials),
            best_params=result.best_params,
            best_value=result.best_value,
            best_sharpe=result.best_metrics.get('sharpe_ratio'),
            best_return_pct=result.best_metrics.get('total_return_pct'),
            best_drawdown_pct=result.best_metrics.get('max_drawdown_pct'),
            all_trials=[
                {
                    'trial': t.trial_number,
                    'params': t.params,
                    'value': t.objective_value,
                    'metrics': t.metrics,
                }
                for t in result.all_trials
            ],
            parameter_importance=result.parameter_importance,
            completed_at=timezone.now(),
        )

    def get_default_parameter_ranges(self, strategy_name: str) -> List[ParameterRange]:
        """Get default parameter ranges for a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            List of ParameterRange objects
        """
        # Common parameters for all strategies
        common = [
            self.DEFAULT_PARAMETER_RANGES['position_size_pct'],
            self.DEFAULT_PARAMETER_RANGES['stop_loss_pct'],
            self.DEFAULT_PARAMETER_RANGES['take_profit_pct'],
            self.DEFAULT_PARAMETER_RANGES['max_positions'],
        ]

        # Strategy-specific parameters
        strategy_specific = {
            'wsb-dip-bot': [
                self.DEFAULT_PARAMETER_RANGES['rsi_period'],
                self.DEFAULT_PARAMETER_RANGES['rsi_oversold'],
            ],
            'momentum-weeklies': [
                ParameterRange('momentum_period', 5, 20, 1, param_type='int'),
                ParameterRange('volume_threshold', 1.0, 3.0, 0.5),
            ],
            'swing-trading': [
                self.DEFAULT_PARAMETER_RANGES['rsi_period'],
                self.DEFAULT_PARAMETER_RANGES['rsi_oversold'],
                self.DEFAULT_PARAMETER_RANGES['rsi_overbought'],
            ],
        }

        return common + strategy_specific.get(strategy_name, [])


def get_optimization_service() -> OptimizationService:
    """Get the optimization service instance."""
    return OptimizationService()
