"""
Hyperopt Parallel Execution.

Ported from freqtrade's hyperopt system.
Parallel hyperparameter optimization for trading strategies.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import time
import hashlib

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods available."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    TPE = "tpe"  # Tree-structured Parzen Estimator


class ParameterType(Enum):
    """Types of hyperparameters."""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class HyperParameter:
    """Definition of a hyperparameter to optimize."""
    name: str
    param_type: ParameterType
    default: Any
    space: Optional[List[Any]] = None  # For categorical
    low: Optional[float] = None  # For numeric
    high: Optional[float] = None  # For numeric
    step: Optional[float] = None  # For numeric grid
    log_scale: bool = False  # Use log scale for sampling

    def sample(self) -> Any:
        """Sample a random value from the parameter space."""
        if self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.space or [self.default])

        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])

        elif self.param_type == ParameterType.INTEGER:
            if self.log_scale and self.low and self.high:
                import math
                log_low = math.log(max(1, self.low))
                log_high = math.log(max(1, self.high))
                return int(round(math.exp(random.uniform(log_low, log_high))))
            return random.randint(int(self.low or 0), int(self.high or 100))

        elif self.param_type == ParameterType.FLOAT:
            if self.log_scale and self.low and self.high:
                import math
                log_low = math.log(max(0.0001, self.low))
                log_high = math.log(max(0.0001, self.high))
                return math.exp(random.uniform(log_low, log_high))
            return random.uniform(self.low or 0.0, self.high or 1.0)

        return self.default

    def grid_values(self, num_points: int = 10) -> List[Any]:
        """Generate grid values for this parameter."""
        if self.param_type == ParameterType.CATEGORICAL:
            return self.space or [self.default]

        elif self.param_type == ParameterType.BOOLEAN:
            return [True, False]

        elif self.param_type == ParameterType.INTEGER:
            low = int(self.low or 0)
            high = int(self.high or 100)
            step = int(self.step or max(1, (high - low) // num_points))
            return list(range(low, high + 1, step))

        elif self.param_type == ParameterType.FLOAT:
            low = self.low or 0.0
            high = self.high or 1.0
            step = self.step or (high - low) / num_points
            values = []
            current = low
            while current <= high:
                values.append(round(current, 6))
                current += step
            return values

        return [self.default]


@dataclass
class HyperoptSpace:
    """Collection of hyperparameters to optimize."""
    parameters: List[HyperParameter] = field(default_factory=list)

    def add(
        self,
        name: str,
        param_type: Union[ParameterType, str],
        default: Any,
        **kwargs,
    ) -> "HyperoptSpace":
        """Add a parameter to the space."""
        if isinstance(param_type, str):
            param_type = ParameterType(param_type)

        self.parameters.append(HyperParameter(
            name=name,
            param_type=param_type,
            default=default,
            **kwargs,
        ))
        return self

    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
        default: Optional[int] = None,
        step: int = 1,
        log_scale: bool = False,
    ) -> "HyperoptSpace":
        """Add integer parameter."""
        return self.add(
            name, ParameterType.INTEGER, default or low,
            low=low, high=high, step=step, log_scale=log_scale
        )

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        default: Optional[float] = None,
        step: Optional[float] = None,
        log_scale: bool = False,
    ) -> "HyperoptSpace":
        """Add float parameter."""
        return self.add(
            name, ParameterType.FLOAT, default or low,
            low=low, high=high, step=step, log_scale=log_scale
        )

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        default: Optional[Any] = None,
    ) -> "HyperoptSpace":
        """Add categorical parameter."""
        return self.add(
            name, ParameterType.CATEGORICAL, default or choices[0],
            space=choices
        )

    def add_boolean(
        self,
        name: str,
        default: bool = True,
    ) -> "HyperoptSpace":
        """Add boolean parameter."""
        return self.add(name, ParameterType.BOOLEAN, default)

    def sample(self) -> Dict[str, Any]:
        """Sample random values for all parameters."""
        return {p.name: p.sample() for p in self.parameters}

    def defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {p.name: p.default for p in self.parameters}


@dataclass
class HyperoptResult:
    """Result of a single hyperopt trial."""
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float]
    objective_value: float
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "objective_value": self.objective_value,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class HyperoptSummary:
    """Summary of hyperopt run."""
    total_trials: int
    successful_trials: int
    failed_trials: int
    best_trial: Optional[HyperoptResult]
    best_params: Dict[str, Any]
    best_objective: float
    all_results: List[HyperoptResult]
    duration_seconds: float
    search_space: HyperoptSpace

    def top_n_results(self, n: int = 10) -> List[HyperoptResult]:
        """Get top N results by objective."""
        sorted_results = sorted(
            [r for r in self.all_results if r.error is None],
            key=lambda x: x.objective_value,
            reverse=True
        )
        return sorted_results[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "best_params": self.best_params,
            "best_objective": self.best_objective,
            "duration_seconds": self.duration_seconds,
            "top_results": [r.to_dict() for r in self.top_n_results(10)],
        }


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""

    @abstractmethod
    def evaluate(
        self,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate parameters and return objective value.

        Args:
            params: Parameter dictionary

        Returns:
            Tuple of (objective_value, metrics_dict)
        """
        pass


class BacktestObjective(ObjectiveFunction):
    """
    Objective function that runs backtests.

    Evaluates strategy parameters by running backtests
    and computing performance metrics.
    """

    def __init__(
        self,
        strategy_class: Type,
        data: Any,
        start_date: date,
        end_date: date,
        initial_capital: Decimal = Decimal("100000"),
        metric: str = "sharpe_ratio",
    ):
        """
        Initialize backtest objective.

        Args:
            strategy_class: Strategy class to instantiate
            data: Historical data for backtesting
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
        """
        self.strategy_class = strategy_class
        self.data = data
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.metric = metric

    def evaluate(
        self,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Run backtest and return metrics."""
        try:
            # Instantiate strategy with parameters
            strategy = self.strategy_class(**params)

            # Run backtest (placeholder - integrate with actual backtest engine)
            # In production, this would call the backtesting engine
            metrics = self._run_backtest(strategy)

            # Get objective value
            objective = metrics.get(self.metric, 0.0)

            return objective, metrics

        except Exception as e:
            logger.error(f"Backtest evaluation error: {e}")
            return float("-inf"), {"error": str(e)}

    def _run_backtest(self, strategy) -> Dict[str, float]:
        """
        Run backtest and compute metrics.

        This is a placeholder - integrate with actual backtesting engine.
        """
        # Placeholder metrics
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "trades": 0,
        }


def _worker_evaluate(args: Tuple) -> HyperoptResult:
    """Worker function for parallel evaluation."""
    trial_id, params, objective_func_pickle = args

    start_time = time.time()
    try:
        # Unpickle the objective function
        objective = pickle.loads(objective_func_pickle)

        # Evaluate
        objective_value, metrics = objective.evaluate(params)

        duration = time.time() - start_time

        return HyperoptResult(
            trial_id=trial_id,
            params=params,
            metrics=metrics,
            objective_value=objective_value,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        return HyperoptResult(
            trial_id=trial_id,
            params=params,
            metrics={},
            objective_value=float("-inf"),
            duration_seconds=duration,
            error=str(e),
        )


class HyperoptEngine:
    """
    Parallel hyperparameter optimization engine.

    Features:
    - Multiple search strategies (grid, random, Bayesian)
    - Parallel execution using multiprocessing
    - Result caching and checkpointing
    - Early stopping
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        space: HyperoptSpace,
        method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
        n_trials: int = 100,
        n_jobs: int = -1,  # -1 = all CPUs
        random_state: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize hyperopt engine.

        Args:
            objective: Objective function to optimize
            space: Hyperparameter space
            method: Optimization method
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            random_state: Random seed for reproducibility
            checkpoint_dir: Directory for checkpoints
        """
        self.objective = objective
        self.space = space
        self.method = method
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.random_state = random_state
        self.checkpoint_dir = checkpoint_dir

        if random_state is not None:
            random.seed(random_state)

        self._results: List[HyperoptResult] = []
        self._best_result: Optional[HyperoptResult] = None
        self._callbacks: List[Callable[[HyperoptResult], None]] = []

    def on_trial_complete(
        self,
        callback: Callable[[HyperoptResult], None],
    ) -> None:
        """Register callback for trial completion."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, result: HyperoptResult) -> None:
        """Notify callbacks of completed trial."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Hyperopt callback error: {e}")

    def _generate_trials_grid(self) -> List[Dict[str, Any]]:
        """Generate trial configurations using grid search."""
        from itertools import product

        param_grids = {
            p.name: p.grid_values()
            for p in self.space.parameters
        }

        # Generate all combinations
        keys = list(param_grids.keys())
        values = [param_grids[k] for k in keys]

        trials = []
        for combo in product(*values):
            params = dict(zip(keys, combo))
            trials.append(params)

        # Limit to n_trials
        if len(trials) > self.n_trials:
            random.shuffle(trials)
            trials = trials[:self.n_trials]

        return trials

    def _generate_trials_random(self) -> List[Dict[str, Any]]:
        """Generate trial configurations using random search."""
        return [self.space.sample() for _ in range(self.n_trials)]

    def _generate_trials_bayesian(self) -> List[Dict[str, Any]]:
        """
        Generate trials using Bayesian optimization.

        Uses Gaussian Process surrogate model (if scipy available).
        Falls back to random search otherwise.
        """
        try:
            from scipy.optimize import minimize
            from scipy.stats import norm
            import numpy as np

            # Start with some random trials
            initial_trials = [self.space.sample() for _ in range(min(10, self.n_trials))]
            return initial_trials  # Simplified - full implementation would use GP

        except ImportError:
            logger.warning("scipy not available, falling back to random search")
            return self._generate_trials_random()

    def _generate_trials_tpe(self) -> List[Dict[str, Any]]:
        """
        Generate trials using Tree-structured Parzen Estimator.

        Simplified implementation - full version would use proper TPE.
        """
        # Simplified: Use random with preference for good regions
        return self._generate_trials_random()

    def _generate_trials_genetic(self) -> List[Dict[str, Any]]:
        """Generate trials using genetic algorithm."""
        # Initialize population
        population = [self.space.sample() for _ in range(min(50, self.n_trials))]
        return population  # Simplified - full GA would evolve population

    def _get_param_hash(self, params: Dict[str, Any]) -> str:
        """Generate hash for parameter set (for caching)."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def _save_checkpoint(self) -> None:
        """Save current results to checkpoint."""
        if not self.checkpoint_dir:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"hyperopt_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        checkpoint = {
            "results": [r.to_dict() for r in self._results],
            "best_params": self._best_result.params if self._best_result else None,
            "best_objective": self._best_result.objective_value if self._best_result else None,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self) -> List[HyperoptResult]:
        """Load results from most recent checkpoint."""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = sorted([
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("hyperopt_checkpoint")
        ])

        if not checkpoints:
            return []

        latest = os.path.join(self.checkpoint_dir, checkpoints[-1])

        try:
            with open(latest, "r") as f:
                data = json.load(f)

            results = []
            for r in data.get("results", []):
                results.append(HyperoptResult(
                    trial_id=r["trial_id"],
                    params=r["params"],
                    metrics=r["metrics"],
                    objective_value=r["objective_value"],
                    duration_seconds=r["duration_seconds"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    error=r.get("error"),
                ))

            logger.info(f"Loaded {len(results)} results from checkpoint")
            return results

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return []

    def optimize(
        self,
        resume: bool = False,
        early_stopping_rounds: Optional[int] = None,
    ) -> HyperoptSummary:
        """
        Run hyperparameter optimization.

        Args:
            resume: Resume from checkpoint if available
            early_stopping_rounds: Stop if no improvement for N rounds

        Returns:
            HyperoptSummary with results
        """
        start_time = time.time()

        # Load checkpoint if resuming
        if resume:
            self._results = self._load_checkpoint()
            completed_hashes = {
                self._get_param_hash(r.params) for r in self._results
            }
        else:
            self._results = []
            completed_hashes = set()

        # Generate trial configurations
        if self.method == OptimizationMethod.GRID_SEARCH:
            all_trials = self._generate_trials_grid()
        elif self.method == OptimizationMethod.RANDOM_SEARCH:
            all_trials = self._generate_trials_random()
        elif self.method == OptimizationMethod.BAYESIAN:
            all_trials = self._generate_trials_bayesian()
        elif self.method == OptimizationMethod.TPE:
            all_trials = self._generate_trials_tpe()
        elif self.method == OptimizationMethod.GENETIC:
            all_trials = self._generate_trials_genetic()
        else:
            all_trials = self._generate_trials_random()

        # Filter out already completed trials
        pending_trials = [
            t for t in all_trials
            if self._get_param_hash(t) not in completed_hashes
        ]

        logger.info(
            f"Starting hyperopt: {len(pending_trials)} trials "
            f"({len(completed_hashes)} already complete)"
        )

        # Pickle the objective function for workers
        objective_pickle = pickle.dumps(self.objective)

        # Prepare work items
        trial_id_start = len(self._results)
        work_items = [
            (trial_id_start + i, params, objective_pickle)
            for i, params in enumerate(pending_trials)
        ]

        # Track best for early stopping
        rounds_without_improvement = 0
        best_so_far = max(
            (r.objective_value for r in self._results),
            default=float("-inf")
        )

        # Run parallel optimization
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(_worker_evaluate, item): item[0]
                for item in work_items
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    self._results.append(result)
                    self._notify_callbacks(result)

                    # Update best
                    if result.error is None and result.objective_value > best_so_far:
                        best_so_far = result.objective_value
                        self._best_result = result
                        rounds_without_improvement = 0
                        logger.info(
                            f"New best: {result.objective_value:.4f} "
                            f"(trial {result.trial_id})"
                        )
                    else:
                        rounds_without_improvement += 1

                    # Early stopping check
                    if early_stopping_rounds and rounds_without_improvement >= early_stopping_rounds:
                        logger.info(
                            f"Early stopping: no improvement for "
                            f"{early_stopping_rounds} rounds"
                        )
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break

                    # Periodic checkpoint
                    if len(self._results) % 10 == 0:
                        self._save_checkpoint()

                except Exception as e:
                    logger.error(f"Trial error: {e}")

        # Final checkpoint
        self._save_checkpoint()

        # Find best result
        successful_results = [r for r in self._results if r.error is None]
        if successful_results:
            self._best_result = max(successful_results, key=lambda x: x.objective_value)
        else:
            self._best_result = None

        duration = time.time() - start_time

        summary = HyperoptSummary(
            total_trials=len(self._results),
            successful_trials=len(successful_results),
            failed_trials=len(self._results) - len(successful_results),
            best_trial=self._best_result,
            best_params=self._best_result.params if self._best_result else {},
            best_objective=self._best_result.objective_value if self._best_result else float("-inf"),
            all_results=self._results,
            duration_seconds=duration,
            search_space=self.space,
        )

        logger.info(
            f"Hyperopt complete: {summary.successful_trials}/{summary.total_trials} "
            f"successful, best={summary.best_objective:.4f}, "
            f"duration={duration:.1f}s"
        )

        return summary


class HyperoptSpaceBuilder:
    """
    Builder for common hyperparameter spaces.

    Provides predefined spaces for common strategy types.
    """

    @staticmethod
    def moving_average_strategy() -> HyperoptSpace:
        """Space for moving average crossover strategies."""
        space = HyperoptSpace()
        space.add_integer("fast_period", low=5, high=50, default=10)
        space.add_integer("slow_period", low=20, high=200, default=50)
        space.add_categorical("ma_type", choices=["SMA", "EMA", "WMA"], default="EMA")
        space.add_float("entry_threshold", low=0.0, high=0.05, default=0.01)
        space.add_float("exit_threshold", low=0.0, high=0.05, default=0.01)
        return space

    @staticmethod
    def rsi_strategy() -> HyperoptSpace:
        """Space for RSI-based strategies."""
        space = HyperoptSpace()
        space.add_integer("rsi_period", low=7, high=21, default=14)
        space.add_integer("oversold_threshold", low=20, high=40, default=30)
        space.add_integer("overbought_threshold", low=60, high=80, default=70)
        space.add_boolean("use_divergence", default=False)
        return space

    @staticmethod
    def bollinger_band_strategy() -> HyperoptSpace:
        """Space for Bollinger Band strategies."""
        space = HyperoptSpace()
        space.add_integer("bb_period", low=10, high=50, default=20)
        space.add_float("bb_std", low=1.0, high=3.0, default=2.0)
        space.add_categorical("entry_type", choices=["touch", "cross", "squeeze"])
        space.add_float("squeeze_threshold", low=0.01, high=0.10, default=0.05)
        return space

    @staticmethod
    def options_strategy() -> HyperoptSpace:
        """Space for options strategy parameters."""
        space = HyperoptSpace()
        space.add_integer("min_dte", low=7, high=45, default=21)
        space.add_integer("max_dte", low=30, high=90, default=45)
        space.add_float("target_delta", low=0.10, high=0.30, default=0.16)
        space.add_integer("wing_width", low=2, high=10, default=5)
        space.add_float("profit_target_pct", low=0.30, high=0.80, default=0.50)
        space.add_float("stop_loss_multiplier", low=1.0, high=3.0, default=2.0)
        space.add_boolean("use_iv_filter", default=True)
        space.add_float("min_iv_percentile", low=0.20, high=0.50, default=0.30)
        return space

    @staticmethod
    def risk_management() -> HyperoptSpace:
        """Space for risk management parameters."""
        space = HyperoptSpace()
        space.add_float("position_size_pct", low=0.01, high=0.10, default=0.02)
        space.add_float("max_portfolio_risk", low=0.05, high=0.20, default=0.10)
        space.add_float("max_single_loss", low=0.01, high=0.05, default=0.02)
        space.add_integer("max_open_positions", low=5, high=50, default=20)
        space.add_float("correlation_threshold", low=0.50, high=0.90, default=0.70)
        return space


def create_hyperopt_engine(
    objective: ObjectiveFunction,
    space: HyperoptSpace,
    method: str = "random",
    n_trials: int = 100,
    n_jobs: int = -1,
) -> HyperoptEngine:
    """
    Factory function to create hyperopt engine.

    Args:
        objective: Objective function
        space: Parameter space
        method: Optimization method name
        n_trials: Number of trials
        n_jobs: Number of parallel jobs

    Returns:
        Configured HyperoptEngine
    """
    method_map = {
        "grid": OptimizationMethod.GRID_SEARCH,
        "random": OptimizationMethod.RANDOM_SEARCH,
        "bayesian": OptimizationMethod.BAYESIAN,
        "genetic": OptimizationMethod.GENETIC,
        "tpe": OptimizationMethod.TPE,
    }

    opt_method = method_map.get(method.lower(), OptimizationMethod.RANDOM_SEARCH)

    return HyperoptEngine(
        objective=objective,
        space=space,
        method=opt_method,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
