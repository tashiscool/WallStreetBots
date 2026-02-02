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
    n_trials: int = 100
    sampler: SamplerType = SamplerType.TPE
    parameter_ranges: List[ParameterRange] = field(default_factory=list)

    # Constraints
    min_trades: int = 10
    max_drawdown_limit: Optional[float] = None

    # Execution
    n_jobs: int = 1
    timeout_seconds: Optional[int] = None


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

    def __init__(self):
        """Initialize the optimization service."""
        self._current_study: Optional['optuna.Study'] = None
        self._progress_callback: Optional[Callable] = None
        self._current_run_id: Optional[str] = None

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
            # Create sampler
            sampler = self._create_sampler(config.sampler)

            # Create study
            direction = "minimize" if config.objective == OptimizationObjective.MIN_DRAWDOWN else "maximize"
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                study_name=f"opt_{config.strategy_name}_{run_id[:8]}"
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

    def _create_sampler(self, sampler_type: SamplerType) -> 'optuna.samplers.BaseSampler':
        """Create an Optuna sampler based on type."""
        if sampler_type == SamplerType.TPE:
            return TPESampler(seed=42)
        elif sampler_type == SamplerType.RANDOM:
            return RandomSampler(seed=42)
        elif sampler_type == SamplerType.CMAES:
            return CmaEsSampler(seed=42)
        else:
            return TPESampler(seed=42)

    def _create_objective(self, config: OptimizationConfig) -> Callable:
        """Create the objective function for optimization."""

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

            # Return objective value
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

        This is a simplified backtest for optimization purposes.
        Returns key metrics for objective evaluation.
        """
        import numpy as np

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
