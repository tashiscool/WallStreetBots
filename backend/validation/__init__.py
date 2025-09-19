"""
WallStreetBots Validation Framework

Comprehensive validation infrastructure for trading strategies including:
- Multiple hypothesis testing (Reality Check / SPA)
- Live drift detection (CUSUM, PSR)
- Factor analysis with HAC standard errors
- Regime robustness testing
- Cross-market validation
- Ensemble correlation analysis
- Capital efficiency analysis
- Alpha validation gates

Usage:
    from backend.validation import ValidationRunner
    runner = ValidationRunner()
    results = runner.run_comprehensive_validation(strategy_returns, benchmark_returns, market_data)
"""

from .statistical_rigor.reality_check import WhitesRealityCheck as RealityCheckValidator
from .drift_monitor import PerformanceDriftMonitor, CUSUMDrift, PSRDrift
from .factor_analysis import AlphaFactorAnalyzer, FactorResult
from .regime_testing import RegimeValidator
from .ensemble_evaluator import EnsembleValidator
from .capital_efficiency import CapitalEfficiencyAnalyzer, KellyResult
from .validation_runner import ValidationRunner
from .alpha_validation_gate import AlphaValidationGate, ValidationCriteria
from .parameter_registry import ParameterRegistry, FrozenParams
from .operations.clock_guard import ClockGuard, ClockGuardConfig
from .operations.state_adapter import ValidationStateAdapter, TradingState
from .broker_accounting.wash_sale import WashSaleEngine, TaxLotManager
from .reporting import ValidationReporter
from .fast_edge.overnight_close_open import OvernightCloseOpenStrategy
from .fast_edge.spy_csp_baseline import SPYCSPBaselineStrategy

__all__ = [
    'AlphaFactorAnalyzer',
    'AlphaValidationGate',
    'CUSUMDrift',
    'CapitalEfficiencyAnalyzer',
    'ClockGuard',
    'ClockGuardConfig',
    'EnsembleValidator',
    'FactorResult',
    'FrozenParams',
    'KellyResult',
    'OvernightCloseOpenStrategy',
    'PSRDrift',
    'ParameterRegistry',
    'PerformanceDriftMonitor',
    'RealityCheckValidator',
    'RegimeRobustnessTester',
    'RegimeValidator',
    'SPYCSPBaselineStrategy',
    'TaxLotManager',
    'TradingState',
    'ValidationCriteria',
    'ValidationReporter',
    'ValidationRunner',
    'ValidationStateAdapter',
    'WashSaleEngine',
]

__version__ = '1.0.0'
__author__ = 'WallStreetBots Team'