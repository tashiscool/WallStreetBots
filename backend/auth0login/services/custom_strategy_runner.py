"""
Custom Strategy Runner - Interprets and executes user-defined strategies.

Converts JSON strategy definitions into executable trading logic.
Supports technical indicators, condition operators, and signal generation.
"""
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CustomStrategyRunner:
    """
    Converts JSON strategy definition to executable strategy.

    Evaluates entry/exit conditions using technical indicators,
    generates trading signals, and validates strategy definitions.
    """

    # Available technical indicators with their calculation functions
    INDICATORS = {
        'rsi': {
            'name': 'Relative Strength Index',
            'description': 'Momentum oscillator measuring speed and change of price movements',
            'default_period': 14,
            'value_range': (0, 100),
            'typical_buy': 30,
            'typical_sell': 70,
        },
        'macd': {
            'name': 'MACD Line',
            'description': 'Trend-following momentum indicator',
            'default_period': None,
            'value_range': None,
        },
        'macd_signal': {
            'name': 'MACD Signal Line',
            'description': '9-period EMA of MACD line',
            'default_period': None,
            'value_range': None,
        },
        'macd_histogram': {
            'name': 'MACD Histogram',
            'description': 'Difference between MACD and Signal line',
            'default_period': None,
            'value_range': None,
        },
        'sma': {
            'name': 'Simple Moving Average',
            'description': 'Average price over specified period',
            'default_period': 20,
            'value_range': None,
        },
        'ema': {
            'name': 'Exponential Moving Average',
            'description': 'Weighted moving average with more weight on recent prices',
            'default_period': 20,
            'value_range': None,
        },
        'bb_upper': {
            'name': 'Bollinger Band Upper',
            'description': 'Upper band at 2 standard deviations above SMA',
            'default_period': 20,
            'value_range': None,
        },
        'bb_middle': {
            'name': 'Bollinger Band Middle',
            'description': 'Middle band (SMA)',
            'default_period': 20,
            'value_range': None,
        },
        'bb_lower': {
            'name': 'Bollinger Band Lower',
            'description': 'Lower band at 2 standard deviations below SMA',
            'default_period': 20,
            'value_range': None,
        },
        'bb_width': {
            'name': 'Bollinger Band Width',
            'description': 'Distance between upper and lower bands as percentage',
            'default_period': 20,
            'value_range': (0, 100),
        },
        'atr': {
            'name': 'Average True Range',
            'description': 'Volatility indicator measuring range of price movements',
            'default_period': 14,
            'value_range': None,
        },
        'adx': {
            'name': 'Average Directional Index',
            'description': 'Trend strength indicator',
            'default_period': 14,
            'value_range': (0, 100),
        },
        'stoch_k': {
            'name': 'Stochastic %K',
            'description': 'Fast stochastic oscillator',
            'default_period': 14,
            'value_range': (0, 100),
            'typical_buy': 20,
            'typical_sell': 80,
        },
        'stoch_d': {
            'name': 'Stochastic %D',
            'description': 'Slow stochastic oscillator (3-period SMA of %K)',
            'default_period': 14,
            'value_range': (0, 100),
        },
        'cci': {
            'name': 'Commodity Channel Index',
            'description': 'Momentum oscillator measuring price relative to average',
            'default_period': 20,
            'value_range': (-200, 200),
            'typical_buy': -100,
            'typical_sell': 100,
        },
        'williams_r': {
            'name': 'Williams %R',
            'description': 'Momentum indicator measuring overbought/oversold levels',
            'default_period': 14,
            'value_range': (-100, 0),
            'typical_buy': -80,
            'typical_sell': -20,
        },
        'volume_sma': {
            'name': 'Volume SMA',
            'description': 'Simple moving average of volume',
            'default_period': 20,
            'value_range': None,
        },
        'volume_ratio': {
            'name': 'Volume Ratio',
            'description': 'Current volume divided by average volume',
            'default_period': 20,
            'value_range': (0, 10),
        },
        'price_change_pct': {
            'name': 'Price Change %',
            'description': 'Percentage change in price over period',
            'default_period': 1,
            'value_range': (-50, 50),
        },
        'price_from_high': {
            'name': 'Price from 52-week High %',
            'description': 'Current price as percentage below 52-week high',
            'default_period': 252,
            'value_range': (-100, 0),
        },
        'price_from_low': {
            'name': 'Price from 52-week Low %',
            'description': 'Current price as percentage above 52-week low',
            'default_period': 252,
            'value_range': (0, 500),
        },
    }

    # Available comparison operators
    OPERATORS = {
        'less_than': {
            'name': 'Less Than',
            'symbol': '<',
            'description': 'Value is below threshold',
            'eval': lambda a, b: a < b,
        },
        'less_equal': {
            'name': 'Less Than or Equal',
            'symbol': '<=',
            'description': 'Value is at or below threshold',
            'eval': lambda a, b: a <= b,
        },
        'greater_than': {
            'name': 'Greater Than',
            'symbol': '>',
            'description': 'Value is above threshold',
            'eval': lambda a, b: a > b,
        },
        'greater_equal': {
            'name': 'Greater Than or Equal',
            'symbol': '>=',
            'description': 'Value is at or above threshold',
            'eval': lambda a, b: a >= b,
        },
        'equals': {
            'name': 'Equals',
            'symbol': '=',
            'description': 'Value equals threshold (with tolerance)',
            'eval': lambda a, b: abs(a - b) < 0.001,
        },
        'crosses_above': {
            'name': 'Crosses Above',
            'symbol': 'crosses above',
            'description': 'Value crosses from below to above threshold',
            'requires_history': True,
        },
        'crosses_below': {
            'name': 'Crosses Below',
            'symbol': 'crosses below',
            'description': 'Value crosses from above to below threshold',
            'requires_history': True,
        },
        'between': {
            'name': 'Between',
            'symbol': 'between',
            'description': 'Value is between two thresholds',
            'requires_range': True,
        },
    }

    # Exit condition types
    EXIT_TYPES = {
        'take_profit': {
            'name': 'Take Profit',
            'description': 'Exit when profit reaches target percentage',
            'unit': '%',
        },
        'stop_loss': {
            'name': 'Stop Loss',
            'description': 'Exit when loss reaches limit percentage',
            'unit': '%',
        },
        'trailing_stop': {
            'name': 'Trailing Stop',
            'description': 'Dynamic stop that follows price higher',
            'unit': '%',
        },
        'time_based': {
            'name': 'Time Limit',
            'description': 'Exit after holding for specified days',
            'unit': 'days',
        },
        'indicator': {
            'name': 'Indicator Signal',
            'description': 'Exit when indicator condition is met',
            'unit': None,
        },
    }

    def __init__(self, definition: Dict[str, Any]):
        """
        Initialize runner with strategy definition.

        Args:
            definition: Strategy definition JSON
        """
        self.definition = definition
        self._data_cache = {}

    def calculate_indicator(
        self,
        df: pd.DataFrame,
        indicator: str,
        period: int = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate technical indicator values.

        Args:
            df: DataFrame with OHLCV data
            indicator: Indicator name
            period: Optional period override

        Returns:
            Series with indicator values
        """
        if indicator not in self.INDICATORS:
            raise ValueError(f"Unknown indicator: {indicator}")

        info = self.INDICATORS[indicator]
        period = period or info.get('default_period') or 14

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        if indicator == 'rsi':
            return self._calculate_rsi(close, period)

        elif indicator in ('macd', 'macd_signal', 'macd_histogram'):
            macd_line, signal_line, histogram = self._calculate_macd(close)
            if indicator == 'macd':
                return macd_line
            elif indicator == 'macd_signal':
                return signal_line
            else:
                return histogram

        elif indicator == 'sma':
            return close.rolling(window=period).mean()

        elif indicator == 'ema':
            return close.ewm(span=period, adjust=False).mean()

        elif indicator in ('bb_upper', 'bb_middle', 'bb_lower', 'bb_width'):
            std_mult = kwargs.get('std', 2)
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()

            if indicator == 'bb_upper':
                return sma + (std * std_mult)
            elif indicator == 'bb_middle':
                return sma
            elif indicator == 'bb_lower':
                return sma - (std * std_mult)
            else:  # bb_width
                upper = sma + (std * std_mult)
                lower = sma - (std * std_mult)
                return ((upper - lower) / sma) * 100

        elif indicator == 'atr':
            return self._calculate_atr(high, low, close, period)

        elif indicator == 'adx':
            return self._calculate_adx(high, low, close, period)

        elif indicator in ('stoch_k', 'stoch_d'):
            k, d = self._calculate_stochastic(high, low, close, period)
            return k if indicator == 'stoch_k' else d

        elif indicator == 'cci':
            return self._calculate_cci(high, low, close, period)

        elif indicator == 'williams_r':
            return self._calculate_williams_r(high, low, close, period)

        elif indicator == 'volume_sma':
            return volume.rolling(window=period).mean()

        elif indicator == 'volume_ratio':
            vol_sma = volume.rolling(window=period).mean()
            return volume / vol_sma

        elif indicator == 'price_change_pct':
            return close.pct_change(periods=period) * 100

        elif indicator == 'price_from_high':
            rolling_high = high.rolling(window=period).max()
            return ((close - rolling_high) / rolling_high) * 100

        elif indicator == 'price_from_low':
            rolling_low = low.rolling(window=period).min()
            return ((close - rolling_low) / rolling_low) * 100

        else:
            raise ValueError(f"Indicator calculation not implemented: {indicator}")

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where(
            (plus_dm > minus_dm) & (plus_dm > 0), 0
        )
        minus_dm = minus_dm.where(
            (minus_dm > plus_dm) & (minus_dm > 0), 0
        )

        atr = self._calculate_atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(period).sum() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.rolling(window=period).mean()

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
        k_smooth: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic %K and %D."""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=k_smooth).mean()

        return k, d

    def _calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)

    def _calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    def evaluate_condition(
        self,
        current_value: float,
        previous_value: float,
        condition: Dict[str, Any]
    ) -> bool:
        """
        Evaluate if a single condition is met.

        Args:
            current_value: Current indicator value
            previous_value: Previous indicator value (for crossover)
            condition: Condition definition

        Returns:
            True if condition is met
        """
        operator = condition.get('operator')
        threshold = condition.get('value', 0)

        if operator not in self.OPERATORS:
            raise ValueError(f"Unknown operator: {operator}")

        op_info = self.OPERATORS[operator]

        if operator == 'crosses_above':
            return previous_value < threshold and current_value >= threshold

        elif operator == 'crosses_below':
            return previous_value > threshold and current_value <= threshold

        elif operator == 'between':
            low = condition.get('value_low', 0)
            high = condition.get('value_high', 100)
            return low <= current_value <= high

        else:
            eval_func = op_info.get('eval')
            if eval_func:
                return eval_func(current_value, threshold)

        return False

    def evaluate_nested_conditions(
        self,
        condition_group: Dict[str, Any],
        df: pd.DataFrame,
        idx: int = -1
    ) -> Tuple[bool, List[Dict]]:
        """
        Recursively evaluate nested condition groups.

        Supports nested AND/OR groups for complex condition logic like:
        (RSI < 30 AND (MACD > 0 OR Stoch < 20))

        Args:
            condition_group: Condition group with 'type', 'logic', and 'conditions'
            df: DataFrame with OHLCV data
            idx: Index to check (default -1 for latest)

        Returns:
            Tuple of (conditions_met, details)

        Example condition_group:
            {
                "type": "group",
                "logic": "AND",
                "conditions": [
                    {"type": "condition", "indicator": "rsi", "operator": "less_than", "value": 30},
                    {
                        "type": "group",
                        "logic": "OR",
                        "conditions": [
                            {"type": "condition", "indicator": "macd", "operator": "greater_than", "value": 0},
                            {"type": "condition", "indicator": "stoch_k", "operator": "less_than", "value": 20}
                        ]
                    }
                ]
            }
        """
        group_type = condition_group.get('type', 'group')
        logic = condition_group.get('logic', 'AND').upper()
        conditions = condition_group.get('conditions', [])

        if not conditions:
            return False, [{'error': 'Empty condition group'}]

        results = []
        met_values = []

        for cond in conditions:
            cond_type = cond.get('type', 'condition')

            if cond_type == 'group':
                # Recursive call for nested groups
                nested_met, nested_details = self.evaluate_nested_conditions(cond, df, idx)
                results.append({
                    'type': 'group',
                    'logic': cond.get('logic', 'AND'),
                    'met': nested_met,
                    'nested_results': nested_details,
                })
                met_values.append(nested_met)

            elif cond_type in ('condition', 'indicator'):
                # Evaluate single condition
                indicator = cond.get('indicator')
                period = cond.get('period')

                try:
                    values = self.calculate_indicator(df, indicator, period)
                    current_value = float(values.iloc[idx])
                    previous_value = float(values.iloc[idx - 1]) if idx != 0 else current_value

                    met = self.evaluate_condition(current_value, previous_value, cond)

                    results.append({
                        'type': 'condition',
                        'indicator': indicator,
                        'operator': cond.get('operator'),
                        'threshold': cond.get('value'),
                        'current_value': round(current_value, 2),
                        'met': met,
                    })
                    met_values.append(met)
                except Exception as e:
                    results.append({
                        'type': 'condition',
                        'indicator': indicator,
                        'error': str(e),
                        'met': False,
                    })
                    met_values.append(False)

            elif cond_type == 'exit_rule':
                # Exit rules are handled separately in check_exit_conditions
                pass

        # Apply logic
        if logic == 'AND':
            group_met = all(met_values) if met_values else False
        elif logic == 'OR':
            group_met = any(met_values) if met_values else False
        else:
            group_met = False

        return group_met, results

    def _is_nested_format(self, entry_conditions: Any) -> bool:
        """Check if entry conditions use the new nested format."""
        if isinstance(entry_conditions, dict):
            return entry_conditions.get('type') == 'group'
        return False

    def _convert_flat_to_nested(self, flat_conditions: List[Dict], logic: str = 'all') -> Dict:
        """Convert flat condition list to nested format for unified processing."""
        nested_logic = 'AND' if logic == 'all' else 'OR'
        return {
            'type': 'group',
            'logic': nested_logic,
            'conditions': [
                {'type': 'condition', **cond}
                for cond in flat_conditions
            ]
        }

    def check_entry_conditions(
        self,
        df: pd.DataFrame,
        idx: int = -1
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if entry conditions are met for given data.

        Supports both flat condition lists (legacy) and nested condition groups.

        Args:
            df: DataFrame with OHLCV data
            idx: Index to check (default -1 for latest)

        Returns:
            Tuple of (conditions_met, details)

        Flat format (legacy):
            {"entry_conditions": [{"indicator": "rsi", "operator": "less_than", "value": 30}]}

        Nested format (new):
            {"entry_conditions": {"type": "group", "logic": "AND", "conditions": [...]}}
        """
        entry_conditions = self.definition.get('entry_conditions', [])
        entry_logic = self.definition.get('entry_logic', 'all')

        if not entry_conditions:
            return False, [{'error': 'No entry conditions defined'}]

        # Check if using nested format
        if self._is_nested_format(entry_conditions):
            # Use new recursive nested evaluation
            return self.evaluate_nested_conditions(entry_conditions, df, idx)

        # Legacy flat format - convert to nested for unified processing
        # or handle directly for backward compatibility
        results = []

        for cond in entry_conditions:
            # Check if this is a nested group within a flat list
            if cond.get('type') == 'group':
                nested_met, nested_details = self.evaluate_nested_conditions(cond, df, idx)
                results.append({
                    'type': 'group',
                    'logic': cond.get('logic', 'AND'),
                    'met': nested_met,
                    'nested_results': nested_details,
                })
                continue

            indicator = cond.get('indicator')
            period = cond.get('period')

            try:
                values = self.calculate_indicator(df, indicator, period)
                current_value = float(values.iloc[idx])
                previous_value = float(values.iloc[idx - 1]) if idx != 0 else current_value

                met = self.evaluate_condition(current_value, previous_value, cond)

                results.append({
                    'indicator': indicator,
                    'operator': cond.get('operator'),
                    'threshold': cond.get('value'),
                    'current_value': round(current_value, 2),
                    'met': met,
                })
            except Exception as e:
                results.append({
                    'indicator': indicator,
                    'error': str(e),
                    'met': False,
                })

        # Apply logic
        met_list = [r.get('met', False) for r in results]
        if entry_logic == 'all':
            all_met = all(met_list) if met_list else False
        else:  # 'any'
            all_met = any(met_list) if met_list else False

        return all_met, results

    def check_exit_conditions(
        self,
        df: pd.DataFrame,
        entry_price: float,
        entry_date: datetime,
        idx: int = -1,
        highest_since_entry: float = None
    ) -> Tuple[bool, str, Dict]:
        """
        Check if exit conditions are met for a position.

        Args:
            df: DataFrame with OHLCV data
            entry_price: Price at which position was entered
            entry_date: Date position was entered
            idx: Index to check (default -1 for latest)
            highest_since_entry: Highest price since entry (for trailing stop)

        Returns:
            Tuple of (should_exit, exit_reason, details)
        """
        exit_conditions = self.definition.get('exit_conditions', [])
        exit_logic = self.definition.get('exit_logic', 'any')

        if not exit_conditions:
            return False, None, {}

        current_price = float(df['close'].iloc[idx])
        current_date = df.index[idx] if hasattr(df.index[idx], 'date') else datetime.now()

        # Calculate P&L percentage
        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Days held
        if isinstance(current_date, datetime):
            days_held = (current_date - entry_date).days
        else:
            days_held = 0

        results = []

        for cond in exit_conditions:
            cond_type = cond.get('type')
            met = False
            details = {}

            if cond_type == 'take_profit':
                target = cond.get('value', 10)
                met = pnl_pct >= target
                details = {
                    'type': 'take_profit',
                    'target': target,
                    'current_pnl_pct': round(pnl_pct, 2),
                    'met': met,
                }

            elif cond_type == 'stop_loss':
                limit = cond.get('value', 5)
                met = pnl_pct <= -limit
                details = {
                    'type': 'stop_loss',
                    'limit': limit,
                    'current_pnl_pct': round(pnl_pct, 2),
                    'met': met,
                }

            elif cond_type == 'trailing_stop':
                trail_pct = cond.get('value', 7)
                if highest_since_entry:
                    trail_price = highest_since_entry * (1 - trail_pct / 100)
                    met = current_price <= trail_price
                    details = {
                        'type': 'trailing_stop',
                        'trail_pct': trail_pct,
                        'highest': highest_since_entry,
                        'trail_price': round(trail_price, 2),
                        'current_price': current_price,
                        'met': met,
                    }

            elif cond_type == 'time_based':
                max_days = cond.get('days', 5)
                met = days_held >= max_days
                details = {
                    'type': 'time_based',
                    'max_days': max_days,
                    'days_held': days_held,
                    'met': met,
                }

            elif cond_type == 'indicator':
                indicator = cond.get('indicator')
                period = cond.get('period')

                try:
                    values = self.calculate_indicator(df, indicator, period)
                    current_value = float(values.iloc[idx])
                    previous_value = float(values.iloc[idx - 1])

                    met = self.evaluate_condition(current_value, previous_value, cond)
                    details = {
                        'type': 'indicator',
                        'indicator': indicator,
                        'operator': cond.get('operator'),
                        'threshold': cond.get('value'),
                        'current_value': round(current_value, 2),
                        'met': met,
                    }
                except Exception as e:
                    details = {
                        'type': 'indicator',
                        'error': str(e),
                        'met': False,
                    }

            results.append(details)

        # Apply exit logic
        met_conditions = [r for r in results if r.get('met')]

        if exit_logic == 'any':
            should_exit = len(met_conditions) > 0
        else:  # 'all'
            should_exit = len(met_conditions) == len(results)

        exit_reason = None
        if should_exit and met_conditions:
            exit_reason = met_conditions[0].get('type', 'condition_met')

        return should_exit, exit_reason, {'conditions': results}

    def _validate_nested_conditions(
        self,
        condition_group: Any,
        prefix: str = 'Entry'
    ) -> Tuple[List[str], List[str]]:
        """
        Recursively validate nested condition groups.

        Args:
            condition_group: Condition group to validate
            prefix: Prefix for error messages ('Entry' or 'Exit')

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Handle nested group format
        if isinstance(condition_group, dict) and condition_group.get('type') == 'group':
            logic = condition_group.get('logic', 'AND').upper()
            if logic not in ('AND', 'OR'):
                errors.append(f'{prefix} group: logic must be "AND" or "OR"')

            conditions = condition_group.get('conditions', [])
            if not conditions:
                errors.append(f'{prefix} group: at least one condition is required')

            for i, cond in enumerate(conditions):
                cond_type = cond.get('type', 'condition')

                if cond_type == 'group':
                    # Recursive validation for nested groups
                    nested_errors, nested_warnings = self._validate_nested_conditions(
                        cond, f'{prefix} group[{i + 1}]'
                    )
                    errors.extend(nested_errors)
                    warnings.extend(nested_warnings)

                elif cond_type in ('condition', 'indicator'):
                    cond_errors, cond_warnings = self._validate_single_condition(
                        cond, f'{prefix} condition {i + 1}'
                    )
                    errors.extend(cond_errors)
                    warnings.extend(cond_warnings)

        # Handle flat list format (legacy)
        elif isinstance(condition_group, list):
            for i, cond in enumerate(condition_group):
                if cond.get('type') == 'group':
                    nested_errors, nested_warnings = self._validate_nested_conditions(
                        cond, f'{prefix} group[{i + 1}]'
                    )
                    errors.extend(nested_errors)
                    warnings.extend(nested_warnings)
                else:
                    cond_errors, cond_warnings = self._validate_single_condition(
                        cond, f'{prefix} condition {i + 1}'
                    )
                    errors.extend(cond_errors)
                    warnings.extend(cond_warnings)

        return errors, warnings

    def _validate_single_condition(
        self,
        cond: Dict,
        prefix: str
    ) -> Tuple[List[str], List[str]]:
        """Validate a single indicator condition."""
        errors = []
        warnings = []

        indicator = cond.get('indicator')
        operator = cond.get('operator')

        if not indicator:
            errors.append(f'{prefix}: indicator is required')
        elif indicator not in self.INDICATORS:
            errors.append(f'{prefix}: unknown indicator "{indicator}"')

        if not operator:
            errors.append(f'{prefix}: operator is required')
        elif operator not in self.OPERATORS:
            errors.append(f'{prefix}: unknown operator "{operator}"')

        if 'value' not in cond and operator not in ('crosses_above', 'crosses_below'):
            if operator != 'between':
                errors.append(f'{prefix}: value is required')

        # Check value ranges
        if indicator in self.INDICATORS:
            info = self.INDICATORS[indicator]
            value_range = info.get('value_range')
            value = cond.get('value')

            if value_range and value is not None:
                if value < value_range[0] or value > value_range[1]:
                    warnings.append(
                        f'{prefix}: {indicator} value {value} '
                        f'is outside typical range {value_range}'
                    )

        return errors, warnings

    def validate_definition(self, definition: Dict = None) -> Dict[str, Any]:
        """
        Validate strategy definition for errors and warnings.

        Supports both flat condition lists and nested condition groups.

        Args:
            definition: Definition to validate (uses self.definition if not provided)

        Returns:
            Dict with 'valid', 'errors', and 'warnings'
        """
        definition = definition or self.definition
        errors = []
        warnings = []

        # Check entry conditions (supports both nested and flat formats)
        entry_conditions = definition.get('entry_conditions', [])
        if not entry_conditions:
            errors.append('At least one entry condition is required')
        else:
            entry_errors, entry_warnings = self._validate_nested_conditions(
                entry_conditions, 'Entry'
            )
            errors.extend(entry_errors)
            warnings.extend(entry_warnings)

        # Check entry logic (for flat format)
        entry_logic = definition.get('entry_logic')
        if entry_logic and entry_logic not in ('all', 'any'):
            errors.append('entry_logic must be "all" or "any"')

        # Check exit conditions
        exit_conditions = definition.get('exit_conditions', [])
        if not exit_conditions:
            warnings.append('No exit conditions defined. Consider adding stop loss.')
        else:
            has_stop_loss = False
            has_take_profit = False

            for i, cond in enumerate(exit_conditions):
                cond_type = cond.get('type')

                if not cond_type:
                    errors.append(f'Exit condition {i + 1}: type is required')
                elif cond_type not in self.EXIT_TYPES:
                    errors.append(f'Exit condition {i + 1}: unknown type "{cond_type}"')

                if cond_type == 'stop_loss':
                    has_stop_loss = True
                    value = cond.get('value')
                    if value and value > 20:
                        warnings.append(f'Stop loss of {value}% is very wide')

                if cond_type == 'take_profit':
                    has_take_profit = True

                if cond_type == 'indicator':
                    indicator = cond.get('indicator')
                    if indicator and indicator not in self.INDICATORS:
                        errors.append(f'Exit condition {i + 1}: unknown indicator "{indicator}"')

            if not has_stop_loss:
                warnings.append('No stop loss defined. This is risky.')

            if has_take_profit and not has_stop_loss:
                warnings.append('Take profit without stop loss creates asymmetric risk')

        # Check position sizing
        sizing = definition.get('position_sizing', {})
        if sizing:
            size_type = sizing.get('type')
            value = sizing.get('value', 0)

            if size_type == 'fixed_percent' and value > 25:
                warnings.append(f'Position size of {value}% is very aggressive')

            max_pos = sizing.get('max_positions', 5)
            if max_pos > 20:
                warnings.append(f'Max positions of {max_pos} may be too many')

        # Check filters
        filters = definition.get('filters', {})
        min_price = filters.get('min_price', 0)
        if min_price > 100:
            warnings.append(f'Min price of ${min_price} is very restrictive')

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        start_idx: int = 50
    ) -> List[Dict]:
        """
        Generate entry signals for historical data.

        Args:
            df: DataFrame with OHLCV data
            start_idx: Index to start scanning from (need history for indicators)

        Returns:
            List of signal dictionaries
        """
        signals = []

        for i in range(start_idx, len(df)):
            met, details = self.check_entry_conditions(df, idx=i)

            if met:
                signals.append({
                    'date': df.index[i].isoformat() if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    'price': float(df['close'].iloc[i]),
                    'conditions': details,
                })

        return signals

    @classmethod
    def get_available_indicators(cls) -> List[Dict]:
        """Get list of available indicators with metadata."""
        return [
            {
                'id': key,
                'name': info['name'],
                'description': info['description'],
                'default_period': info.get('default_period'),
                'value_range': info.get('value_range'),
                'typical_buy': info.get('typical_buy'),
                'typical_sell': info.get('typical_sell'),
            }
            for key, info in cls.INDICATORS.items()
        ]

    @classmethod
    def get_available_operators(cls) -> List[Dict]:
        """Get list of available operators with metadata."""
        return [
            {
                'id': key,
                'name': info['name'],
                'symbol': info['symbol'],
                'description': info['description'],
                'requires_history': info.get('requires_history', False),
                'requires_range': info.get('requires_range', False),
            }
            for key, info in cls.OPERATORS.items()
        ]

    @classmethod
    def get_exit_types(cls) -> List[Dict]:
        """Get list of available exit condition types."""
        return [
            {
                'id': key,
                'name': info['name'],
                'description': info['description'],
                'unit': info['unit'],
            }
            for key, info in cls.EXIT_TYPES.items()
        ]


# Strategy templates for users to start from
STRATEGY_TEMPLATES = {
    'rsi_oversold': {
        'name': 'RSI Oversold Bounce',
        'description': 'Buy when RSI indicates oversold conditions, sell when overbought or hit targets',
        'definition': {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 30, 'period': 14}
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 10},
                {'type': 'stop_loss', 'value': 5},
                {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 70}
            ],
            'exit_logic': 'any',
            'position_sizing': {'type': 'fixed_percent', 'value': 5, 'max_positions': 5},
            'filters': {'min_price': 10, 'min_volume': 500000},
        },
    },
    'macd_crossover': {
        'name': 'MACD Crossover',
        'description': 'Buy when MACD crosses above signal line, with momentum confirmation',
        'definition': {
            'entry_conditions': [
                {'indicator': 'macd_histogram', 'operator': 'crosses_above', 'value': 0},
                {'indicator': 'rsi', 'operator': 'greater_than', 'value': 40}
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 15},
                {'type': 'stop_loss', 'value': 7},
                {'type': 'trailing_stop', 'value': 10},
                {'type': 'indicator', 'indicator': 'macd_histogram', 'operator': 'crosses_below', 'value': 0}
            ],
            'exit_logic': 'any',
            'position_sizing': {'type': 'fixed_percent', 'value': 5, 'max_positions': 4},
            'filters': {'min_price': 20, 'min_volume': 1000000},
        },
    },
    'bollinger_squeeze': {
        'name': 'Bollinger Band Squeeze',
        'description': 'Buy when price touches lower band with RSI confirmation',
        'definition': {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'less_than', 'value': 35, 'period': 14},
                {'indicator': 'bb_width', 'operator': 'less_than', 'value': 15, 'period': 20}
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 12},
                {'type': 'stop_loss', 'value': 6},
                {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 65}
            ],
            'exit_logic': 'any',
            'position_sizing': {'type': 'fixed_percent', 'value': 4, 'max_positions': 5},
            'filters': {'min_price': 15, 'min_volume': 750000},
        },
    },
    'momentum_breakout': {
        'name': 'Momentum Breakout',
        'description': 'Buy strong momentum stocks breaking out with volume',
        'definition': {
            'entry_conditions': [
                {'indicator': 'rsi', 'operator': 'between', 'value_low': 50, 'value_high': 70, 'period': 14},
                {'indicator': 'volume_ratio', 'operator': 'greater_than', 'value': 1.5, 'period': 20},
                {'indicator': 'price_change_pct', 'operator': 'greater_than', 'value': 2, 'period': 1}
            ],
            'entry_logic': 'all',
            'exit_conditions': [
                {'type': 'take_profit', 'value': 20},
                {'type': 'trailing_stop', 'value': 8},
                {'type': 'time_based', 'days': 10}
            ],
            'exit_logic': 'any',
            'position_sizing': {'type': 'fixed_percent', 'value': 3, 'max_positions': 6},
            'filters': {'min_price': 25, 'min_volume': 2000000},
        },
    },
}


def get_strategy_templates() -> List[Dict]:
    """Get all available strategy templates."""
    return [
        {
            'id': key,
            'name': template['name'],
            'description': template['description'],
            'definition': template['definition'],
        }
        for key, template in STRATEGY_TEMPLATES.items()
    ]
