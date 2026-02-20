"""
Hyperopt Engine

ML-based strategy parameter optimization using Optuna.
Inspired by freqtrade's hyperopt system.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .parameters import (
    BaseParameter,
    HyperoptSpace,
    get_hyperoptable_parameters,
    apply_parameters,
)
from .hyperopt_loss import (
    IHyperoptLoss,
    BacktestResult,
    SharpeHyperoptLoss,
    SortinoHyperoptLoss,
    MaxDrawdownHyperoptLoss,
    CalmarHyperoptLoss,
    ProfitHyperoptLoss,
    MultiMetricHyperoptLoss,
)

logger = logging.getLogger(__name__)


@dataclass
class HyperoptConfig:
    """Configuration for hyperopt optimization."""
    # General settings
    epochs: int = 100
    min_trades: int = 10
    spaces: List[HyperoptSpace] = field(default_factory=lambda: [
        HyperoptSpace.BUY, HyperoptSpace.SELL
    ])

    # Loss function
    loss_function: str = 'sharpe'  # sharpe, sortino, max_drawdown, calmar, profit, multi

    # Optuna settings
    sampler: str = 'tpe'  # tpe, random, cmaes
    n_startup_trials: int = 10
    multivariate: bool = True

    # Backtest settings
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    timeframe: str = '1d'

    # Output settings
    results_dir: str = 'hyperopt_results'
    print_all_results: bool = False
    print_colorized: bool = True

    # Parallel execution
    n_jobs: int = 1  # -1 for all cores


class HyperoptEngine:
    """
    ML-based strategy parameter optimization.

    Uses Optuna for Bayesian optimization with multiple loss functions.

    Example:
        engine = HyperoptEngine(
            strategy_class=MyStrategy,
            backtest_runner=backtest_func,
            config=HyperoptConfig(epochs=100, loss_function='sharpe')
        )
        best_params, best_result = engine.optimize()
    """

    LOSS_FUNCTIONS: ClassVar[Dict[str, Type[IHyperoptLoss]]] = {
        'sharpe': SharpeHyperoptLoss,
        'sortino': SortinoHyperoptLoss,
        'max_drawdown': MaxDrawdownHyperoptLoss,
        'calmar': CalmarHyperoptLoss,
        'profit': ProfitHyperoptLoss,
        'multi': MultiMetricHyperoptLoss,
    }

    def __init__(
        self,
        strategy_class: Type,
        backtest_runner: Callable[[Type, Dict[str, Any], HyperoptConfig], BacktestResult],
        config: Optional[HyperoptConfig] = None,
    ):
        """
        Initialize HyperoptEngine.

        Args:
            strategy_class: The strategy class to optimize
            backtest_runner: Function that runs backtest and returns BacktestResult
                Signature: (strategy_class, params_dict, config) -> BacktestResult
            config: Hyperopt configuration
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperopt. Install with: pip install optuna"
            )

        self.strategy_class = strategy_class
        self.backtest_runner = backtest_runner
        self.config = config or HyperoptConfig()

        # Extract hyperoptable parameters
        self.parameters = get_hyperoptable_parameters(strategy_class)
        logger.info(f"Found {len(self.parameters)} hyperoptable parameters")

        # Filter by spaces
        self.active_parameters = {
            name: param for name, param in self.parameters.items()
            if param.space in self.config.spaces
        }
        logger.info(f"Optimizing {len(self.active_parameters)} parameters in spaces: {self.config.spaces}")

        # Get loss function
        self.loss_function = self._get_loss_function()

        # Results storage
        self.trials: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_result: Optional[BacktestResult] = None
        self.best_loss: float = float('inf')

    def _get_loss_function(self) -> Type[IHyperoptLoss]:
        """Get the loss function class."""
        loss_name = self.config.loss_function.lower()
        if loss_name not in self.LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss function: {loss_name}. "
                f"Available: {list(self.LOSS_FUNCTIONS.keys())}"
            )
        return self.LOSS_FUNCTIONS[loss_name]

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Get Optuna sampler based on config."""
        if self.config.sampler == 'tpe':
            return TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                multivariate=self.config.multivariate,
            )
        elif self.config.sampler == 'random':
            return RandomSampler()
        elif self.config.sampler == 'cmaes':
            return CmaEsSampler()
        else:
            raise ValueError(f"Unknown sampler: {self.config.sampler}")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Suggests parameters, runs backtest, returns loss.
        """
        # Generate parameter suggestions
        params = {}
        for name, param in self.active_parameters.items():
            params[name] = param.get_optuna_distribution(trial, name)

        # Run backtest with these parameters
        try:
            result = self.backtest_runner(
                self.strategy_class,
                params,
                self.config
            )
        except Exception as e:
            logger.warning(f"Backtest failed with params {params}: {e}")
            return 10.0  # High loss for failed backtests

        # Calculate loss
        loss = self.loss_function.hyperopt_loss_function(
            results=result,
            trade_count=result.total_trades,
            min_date=self.config.start_date or datetime(2020, 1, 1),
            max_date=self.config.end_date or datetime.now(),
            config={},
        )

        # Store trial results
        self.trials.append({
            'trial_number': trial.number,
            'params': params,
            'result': {
                'total_trades': result.total_trades,
                'total_profit_pct': result.total_profit_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'win_rate': result.win_rate,
            },
            'loss': loss,
        })

        # Track best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params
            self.best_result = result
            logger.info(f"New best: loss={loss:.4f}, params={params}")

        # Print progress
        if trial.number % 10 == 0:
            logger.info(
                f"Trial {trial.number}/{self.config.epochs}: "
                f"loss={loss:.4f}, best_loss={self.best_loss:.4f}"
            )

        return loss

    def optimize(self) -> tuple[Dict[str, Any], BacktestResult]:
        """
        Run hyperopt optimization.

        Returns:
            Tuple of (best_params, best_result)
        """
        logger.info(
            f"Starting hyperopt with {self.config.epochs} epochs, "
            f"loss function: {self.config.loss_function}"
        )

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=self._get_sampler(),
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.config.epochs,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )

        # Log results
        logger.info("Optimization complete!")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        if self.best_result:
            logger.info(f"Best result: {self.best_result.total_profit_pct:.2f}% profit, "
                       f"{self.best_result.total_trades} trades, "
                       f"Sharpe: {self.best_result.sharpe_ratio:.2f}")

        # Save results
        self._save_results()

        return self.best_params, self.best_result

    def _save_results(self) -> None:
        """Save optimization results to disk."""
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = self.strategy_class.__name__

        # Save all trials
        trials_file = results_dir / f"{strategy_name}_{timestamp}_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(self.trials, f, indent=2, default=str)
        logger.info(f"Saved trials to {trials_file}")

        # Save best parameters
        if self.best_params:
            params_file = results_dir / f"{strategy_name}_{timestamp}_best_params.json"
            with open(params_file, 'w') as f:
                json.dump({
                    'params': self.best_params,
                    'loss': self.best_loss,
                    'result': {
                        'total_profit_pct': self.best_result.total_profit_pct if self.best_result else None,
                        'sharpe_ratio': self.best_result.sharpe_ratio if self.best_result else None,
                        'max_drawdown_pct': self.best_result.max_drawdown_pct if self.best_result else None,
                        'total_trades': self.best_result.total_trades if self.best_result else None,
                        'win_rate': self.best_result.win_rate if self.best_result else None,
                    },
                    'config': {
                        'epochs': self.config.epochs,
                        'loss_function': self.config.loss_function,
                        'spaces': [s.value for s in self.config.spaces],
                    },
                    'timestamp': timestamp,
                }, f, indent=2, default=str)
            logger.info(f"Saved best params to {params_file}")

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        return {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'best_result': {
                'total_profit_pct': self.best_result.total_profit_pct if self.best_result else None,
                'sharpe_ratio': self.best_result.sharpe_ratio if self.best_result else None,
                'max_drawdown_pct': self.best_result.max_drawdown_pct if self.best_result else None,
                'total_trades': self.best_result.total_trades if self.best_result else None,
            },
            'total_trials': len(self.trials),
            'config': {
                'epochs': self.config.epochs,
                'loss_function': self.config.loss_function,
            }
        }


def create_backtest_runner(backtester, data_provider):
    """
    Create a backtest runner function for use with HyperoptEngine.

    Args:
        backtester: Your backtesting engine instance
        data_provider: Data provider for historical data

    Returns:
        Callable that runs backtest with given parameters
    """
    def runner(strategy_class, params: Dict[str, Any], config: HyperoptConfig) -> BacktestResult:
        # Create strategy instance with parameters
        strategy = strategy_class()
        apply_parameters(strategy, params)

        # Run backtest
        result = backtester.run(
            strategy=strategy,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
        )

        # Convert to BacktestResult
        return BacktestResult(
            total_trades=result.get('total_trades', 0),
            winning_trades=result.get('winning_trades', 0),
            losing_trades=result.get('losing_trades', 0),
            total_profit=result.get('total_profit', 0),
            total_profit_pct=result.get('total_profit_pct', 0),
            avg_profit_pct=result.get('avg_profit_pct', 0),
            max_drawdown=result.get('max_drawdown', 0),
            max_drawdown_pct=result.get('max_drawdown_pct', 0),
            sharpe_ratio=result.get('sharpe_ratio', 0),
            sortino_ratio=result.get('sortino_ratio', 0),
            calmar_ratio=result.get('calmar_ratio', 0),
            profit_factor=result.get('profit_factor', 0),
            win_rate=result.get('win_rate', 0),
            avg_trade_duration=result.get('avg_trade_duration', 0),
            trade_count_long=result.get('trade_count_long', 0),
            trade_count_short=result.get('trade_count_short', 0),
            profit_long=result.get('profit_long', 0),
            profit_short=result.get('profit_short', 0),
            equity_curve=result.get('equity_curve', []),
            drawdown_curve=result.get('drawdown_curve', []),
            start_date=config.start_date or datetime(2020, 1, 1),
            end_date=config.end_date or datetime.now(),
        )

    return runner
