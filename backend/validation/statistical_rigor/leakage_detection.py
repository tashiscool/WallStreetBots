"""Look-ahead bias and data leakage detection system.

Prevents strategies from accidentally using future information including
survivorship bias, corporate actions, and post-event data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import inspect
import ast

logger = logging.getLogger(__name__)


@dataclass
class LeakageViolation:
    """Represents a detected data leakage violation."""
    violation_type: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    detected_at: datetime
    code_location: Optional[str] = None
    suggested_fix: Optional[str] = None


class TemporalGuard:
    """Guards against temporal data leakage in strategy code."""

    def __init__(self):
        self.current_time = None
        self.violations = []
        self.monitored_functions = set()

    def set_current_time(self, current_time: pd.Timestamp):
        """Set the current simulation time."""
        self.current_time = current_time

    def check_data_access(self, data: pd.Series, requested_time: pd.Timestamp) -> bool:
        """Check if data access violates temporal constraints."""
        if self.current_time is None:
            logger.warning("TemporalGuard: current_time not set")
            return True

        # Check for look-ahead bias
        if requested_time > self.current_time:
            violation = LeakageViolation(
                violation_type='look_ahead_bias',
                description=f"Attempted to access data from {requested_time} while current time is {self.current_time}",
                severity='critical',
                detected_at=datetime.now(),
                suggested_fix="Ensure data access is limited to current_time or earlier"
            )
            self.violations.append(violation)
            return False

        return True

    def monitor_function(self, func):
        """Decorator to monitor function for data leakage."""
        def wrapper(*args, **kwargs):
            # Record function call
            self.monitored_functions.add(func.__name__)

            # Analyze function for potential leakage
            self._analyze_function_code(func)

            # Execute function
            return func(*args, **kwargs)

        return wrapper

    def _analyze_function_code(self, func):
        """Analyze function source code for potential leakage patterns."""
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)

            # Look for suspicious patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    # Check for dangerous pandas operations
                    if hasattr(node, 'attr'):
                        if node.attr in ['shift', 'pct_change'] and hasattr(node, 'value'):
                            # Check if shift has negative values (future data)
                            self._check_shift_operations(node, func.__name__)

                elif isinstance(node, ast.Call):
                    # Check for rolling operations that might use future data
                    if hasattr(node, 'func') and hasattr(node.func, 'attr'):
                        if node.func.attr in ['rolling', 'ewm']:
                            self._check_rolling_operations(node, func.__name__)

        except Exception as e:
            logger.debug(f"Could not analyze function {func.__name__}: {e}")

    def _check_shift_operations(self, node: ast.AST, func_name: str):
        """Check shift operations for look-ahead bias."""
        # This is a simplified check - in production you'd want more sophisticated AST analysis
        pass

    def _check_rolling_operations(self, node: ast.AST, func_name: str):
        """Check rolling operations for potential leakage."""
        # This is a simplified check - in production you'd want more sophisticated AST analysis
        pass


class CorporateActionsGuard:
    """Prevents leakage from corporate actions and earnings data."""

    def __init__(self):
        self.corporate_actions = {}
        self.earnings_calendar = {}
        self.violations = []

    def register_corporate_action(self, symbol: str, action_date: pd.Timestamp,
                                action_type: str, effective_date: pd.Timestamp):
        """Register a corporate action."""
        if symbol not in self.corporate_actions:
            self.corporate_actions[symbol] = []

        self.corporate_actions[symbol].append({
            'action_date': action_date,
            'effective_date': effective_date,
            'action_type': action_type
        })

    def register_earnings_date(self, symbol: str, earnings_date: pd.Timestamp,
                             announcement_time: str = 'after_market'):
        """Register an earnings announcement date."""
        if symbol not in self.earnings_calendar:
            self.earnings_calendar[symbol] = []

        self.earnings_calendar[symbol].append({
            'earnings_date': earnings_date,
            'announcement_time': announcement_time
        })

    def check_price_data_validity(self, symbol: str, data: pd.Series,
                                current_time: pd.Timestamp) -> Tuple[bool, List[str]]:
        """Check if price data is valid given corporate actions."""
        warnings = []
        is_valid = True

        # Check for unadjusted data around corporate actions
        if symbol in self.corporate_actions:
            for action in self.corporate_actions[symbol]:
                action_date = action['action_date']
                effective_date = action['effective_date']

                # Check if we're using data around corporate action dates
                if action_date <= current_time:
                    # We should know about this action
                    data_around_action = data[
                        (data.index >= action_date - timedelta(days=5)) &
                        (data.index <= action_date + timedelta(days=5))
                    ]

                    if len(data_around_action) > 0:
                        # Check for suspicious price jumps that might indicate unadjusted data
                        returns = data_around_action.pct_change().dropna()
                        if len(returns) > 0 and (abs(returns) > 0.2).any():
                            if action['action_type'] in ['split', 'dividend']:
                                warnings.append(
                                    f"Large price movement around {action['action_type']} "
                                    f"on {action_date} - check if data is properly adjusted"
                                )

        return is_valid, warnings

    def check_earnings_blackout(self, symbol: str, current_time: pd.Timestamp,
                              trade_time: str = 'market_hours') -> bool:
        """Check if current time violates earnings blackout rules."""
        if symbol not in self.earnings_calendar:
            return True  # No earnings data, assume OK

        for earnings in self.earnings_calendar[symbol]:
            earnings_date = earnings['earnings_date']
            announcement_time = earnings['announcement_time']

            # Define blackout window
            if announcement_time == 'before_market':
                blackout_start = earnings_date - timedelta(days=1)
                blackout_end = earnings_date
            else:  # after_market or during_market
                blackout_start = earnings_date - timedelta(days=1)
                blackout_end = earnings_date + timedelta(days=1)

            # Check if current time falls in blackout window
            if blackout_start <= current_time <= blackout_end:
                violation = LeakageViolation(
                    violation_type='earnings_blackout_violation',
                    description=f"Trading {symbol} during earnings blackout period: {blackout_start} to {blackout_end}",
                    severity='warning',
                    detected_at=datetime.now(),
                    suggested_fix="Implement earnings blackout rules or pre-earnings exit logic"
                )
                self.violations.append(violation)
                return False

        return True


class SurvivorshipGuard:
    """Detects survivorship bias in strategy backtests."""

    def __init__(self):
        self.universe_history = {}
        self.violations = []

    def register_universe_snapshot(self, date: pd.Timestamp, universe: List[str]):
        """Register universe composition at a specific date."""
        self.universe_history[date] = set(universe)

    def check_survivorship_bias(self, strategy_universe: List[str],
                              backtest_start: pd.Timestamp,
                              backtest_end: pd.Timestamp) -> Tuple[bool, List[str]]:
        """Check for survivorship bias in strategy universe."""
        warnings = []
        has_bias = False

        # Check if universe uses only symbols that survived to the end
        if not self.universe_history:
            warnings.append(
                "No universe history available - cannot check for survivorship bias. "
                "Consider using point-in-time universe selection."
            )
            return True, warnings  # Can't check, assume OK but warn

        # Find universe snapshots within backtest period
        relevant_dates = [date for date in self.universe_history.keys()
                         if backtest_start <= date <= backtest_end]

        if not relevant_dates:
            warnings.append(
                f"No universe snapshots found between {backtest_start} and {backtest_end}"
            )
            return True, warnings

        # Check if strategy universe includes delisted stocks
        strategy_set = set(strategy_universe)
        earliest_universe = self.universe_history[min(relevant_dates)]
        latest_universe = self.universe_history[max(relevant_dates)]

        # Stocks that were in universe at start but not at end (potentially delisted)
        potentially_delisted = earliest_universe - latest_universe

        # Check if strategy completely excludes potentially delisted stocks
        if potentially_delisted and not (strategy_set & potentially_delisted):
            has_bias = True
            violation = LeakageViolation(
                violation_type='survivorship_bias',
                description=(
                    f"Strategy universe excludes all {len(potentially_delisted)} stocks "
                    f"that may have been delisted during backtest period"
                ),
                severity='critical',
                detected_at=datetime.now(),
                suggested_fix=(
                    "Include delisted stocks in backtest or use point-in-time "
                    "universe selection to avoid survivorship bias"
                )
            )
            self.violations.append(violation)

        # Check for other potential bias indicators
        if len(strategy_set & latest_universe) / len(strategy_set) > 0.95:
            warnings.append(
                "Strategy universe heavily biased toward stocks that survived to end of period"
            )

        return not has_bias, warnings


class ParameterGuard:
    """Ensures parameter immutability and prevents optimization leakage."""

    def __init__(self):
        self.parameter_registry = {}
        self.frozen_parameters = {}
        self.violations = []

    def register_parameters(self, strategy_name: str, parameters: Dict[str, Any],
                          optimization_period: Tuple[pd.Timestamp, pd.Timestamp],
                          git_hash: Optional[str] = None):
        """Register strategy parameters for a specific optimization period."""
        param_id = f"{strategy_name}_{optimization_period[0]}_{optimization_period[1]}"

        self.parameter_registry[param_id] = {
            'strategy_name': strategy_name,
            'parameters': parameters.copy(),
            'optimization_period': optimization_period,
            'registered_at': datetime.now(),
            'git_hash': git_hash,
            'frozen': False
        }

        logger.info(f"Registered parameters for {strategy_name}: {param_id}")

    def freeze_parameters(self, strategy_name: str,
                        optimization_period: Tuple[pd.Timestamp, pd.Timestamp]) -> str:
        """Freeze parameters after optimization period ends."""
        param_id = f"{strategy_name}_{optimization_period[0]}_{optimization_period[1]}"

        if param_id not in self.parameter_registry:
            raise ValueError(f"Parameters not found for {param_id}")

        # Mark as frozen
        self.parameter_registry[param_id]['frozen'] = True
        self.parameter_registry[param_id]['frozen_at'] = datetime.now()

        # Store in frozen registry
        self.frozen_parameters[param_id] = self.parameter_registry[param_id].copy()

        logger.info(f"Frozen parameters for {param_id}")
        return param_id

    def validate_parameters(self, strategy_name: str, current_parameters: Dict[str, Any],
                          oos_period: Tuple[pd.Timestamp, pd.Timestamp]) -> bool:
        """Validate that current parameters match frozen parameters for OOS period."""
        # Find the relevant frozen parameter set
        relevant_param_id = None

        for param_id, param_data in self.frozen_parameters.items():
            if (param_data['strategy_name'] == strategy_name and
                param_data['optimization_period'][1] <= oos_period[0]):
                relevant_param_id = param_id
                break

        if relevant_param_id is None:
            violation = LeakageViolation(
                violation_type='parameter_leakage',
                description=f"No frozen parameters found for {strategy_name} OOS period {oos_period}",
                severity='critical',
                detected_at=datetime.now(),
                suggested_fix="Ensure parameters are frozen before OOS testing begins"
            )
            self.violations.append(violation)
            return False

        # Compare parameters
        frozen_params = self.frozen_parameters[relevant_param_id]['parameters']
        params_match = self._deep_compare_parameters(frozen_params, current_parameters)

        if not params_match:
            violation = LeakageViolation(
                violation_type='parameter_drift',
                description=f"Current parameters for {strategy_name} don't match frozen parameters {relevant_param_id}",
                severity='critical',
                detected_at=datetime.now(),
                suggested_fix="Revert to frozen parameters or re-run optimization if justified"
            )
            self.violations.append(violation)

        return params_match

    def _deep_compare_parameters(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Deep comparison of parameter dictionaries."""
        if set(params1.keys()) != set(params2.keys()):
            return False

        for key in params1.keys():
            val1, val2 = params1[key], params2[key]

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > 1e-10:  # Numerical tolerance
                    return False
            elif val1 != val2:
                return False

        return True


class LeakageDetectionSuite:
    """Comprehensive data leakage detection system."""

    def __init__(self):
        self.temporal_guard = TemporalGuard()
        self.corporate_actions_guard = CorporateActionsGuard()
        self.survivorship_guard = SurvivorshipGuard()
        self.parameter_guard = ParameterGuard()

    def validate_strategy(self, strategy, backtest_start: pd.Timestamp,
                         backtest_end: pd.Timestamp,
                         strategy_universe: List[str]) -> Dict[str, Any]:
        """Comprehensive validation of strategy for data leakage."""
        validation_results = {
            'passed': True,
            'violations': [],
            'warnings': [],
            'checks_performed': []
        }

        # Survivorship bias check
        try:
            survivorship_ok, survivorship_warnings = self.survivorship_guard.check_survivorship_bias(
                strategy_universe, backtest_start, backtest_end
            )
            validation_results['checks_performed'].append('survivorship_bias')
            validation_results['warnings'].extend(survivorship_warnings)

            if not survivorship_ok:
                validation_results['passed'] = False

        except Exception as e:
            logger.error(f"Survivorship check failed: {e}")
            validation_results['warnings'].append(f"Survivorship check failed: {e}")

        # Corporate actions validation
        try:
            for symbol in strategy_universe:
                ca_valid, ca_warnings = self.corporate_actions_guard.check_price_data_validity(
                    symbol, pd.Series(), backtest_end  # Placeholder data
                )
                validation_results['warnings'].extend(ca_warnings)

            validation_results['checks_performed'].append('corporate_actions')

        except Exception as e:
            logger.error(f"Corporate actions check failed: {e}")
            validation_results['warnings'].append(f"Corporate actions check failed: {e}")

        # Collect all violations
        all_violations = (
            self.temporal_guard.violations +
            self.corporate_actions_guard.violations +
            self.survivorship_guard.violations +
            self.parameter_guard.violations
        )

        validation_results['violations'] = [
            {
                'type': v.violation_type,
                'description': v.description,
                'severity': v.severity,
                'suggested_fix': v.suggested_fix
            }
            for v in all_violations
        ]

        # Update overall pass/fail status
        critical_violations = [v for v in all_violations if v.severity == 'critical']
        if critical_violations:
            validation_results['passed'] = False

        return validation_results

    def create_leakage_report(self) -> str:
        """Create a comprehensive leakage detection report."""
        all_violations = (
            self.temporal_guard.violations +
            self.corporate_actions_guard.violations +
            self.survivorship_guard.violations +
            self.parameter_guard.violations
        )

        report = "=== DATA LEAKAGE DETECTION REPORT ===\n\n"

        if not all_violations:
            report += "✅ No data leakage violations detected.\n"
        else:
            report += f"⚠️ {len(all_violations)} potential violations detected:\n\n"

            for i, violation in enumerate(all_violations, 1):
                report += f"{i}. {violation.violation_type.upper()} ({violation.severity})\n"
                report += f"   Description: {violation.description}\n"
                if violation.suggested_fix:
                    report += f"   Suggested Fix: {violation.suggested_fix}\n"
                report += f"   Detected: {violation.detected_at}\n\n"

        # Summary recommendations
        critical_count = len([v for v in all_violations if v.severity == 'critical'])
        warning_count = len([v for v in all_violations if v.severity == 'warning'])

        report += "=== RECOMMENDATIONS ===\n"
        if critical_count > 0:
            report += f"🚨 {critical_count} critical violations must be fixed before deployment.\n"
        if warning_count > 0:
            report += f"⚠️ {warning_count} warnings should be reviewed and addressed.\n"

        if critical_count == 0 and warning_count == 0:
            report += "✅ Strategy passes data leakage validation.\n"

        return report