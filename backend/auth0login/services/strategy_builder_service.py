"""
Strategy Builder Service - Visual strategy builder API backend.

Provides validation, indicator metadata, operator metadata, and preset
strategy templates for the drag-and-drop strategy builder UI. Leverages
the CustomStrategyRunner's INDICATORS, OPERATORS, and EXIT_TYPES registries
as the single source of truth.
"""
import logging
from typing import Any, Dict, List

from .custom_strategy_runner import CustomStrategyRunner

logger = logging.getLogger(__name__)


# Indicator category mapping. Each key from CustomStrategyRunner.INDICATORS
# is placed in one logical category for the UI grouping.
INDICATOR_CATEGORIES = {
    'rsi': 'momentum',
    'macd': 'trend',
    'macd_signal': 'trend',
    'macd_histogram': 'trend',
    'sma': 'trend',
    'ema': 'trend',
    'bb_upper': 'volatility',
    'bb_middle': 'volatility',
    'bb_lower': 'volatility',
    'bb_width': 'volatility',
    'atr': 'volatility',
    'adx': 'trend',
    'stoch_k': 'momentum',
    'stoch_d': 'momentum',
    'cci': 'momentum',
    'williams_r': 'momentum',
    'volume_sma': 'volume',
    'volume_ratio': 'volume',
    'price_change_pct': 'price',
    'price_from_high': 'price',
    'price_from_low': 'price',
}

# Display-friendly names that map runner indicator keys to strategy builder
# display names. Supplements the runner's 'name' field with a parenthetical
# abbreviation where useful.
INDICATOR_DISPLAY_NAMES = {
    'rsi': 'Relative Strength Index (RSI)',
    'macd': 'MACD Line',
    'macd_signal': 'MACD Signal Line',
    'macd_histogram': 'MACD Histogram',
    'sma': 'Simple Moving Average (SMA)',
    'ema': 'Exponential Moving Average (EMA)',
    'bb_upper': 'Bollinger Band Upper',
    'bb_middle': 'Bollinger Band Middle',
    'bb_lower': 'Bollinger Band Lower',
    'bb_width': 'Bollinger Band Width',
    'atr': 'Average True Range (ATR)',
    'adx': 'Average Directional Index (ADX)',
    'stoch_k': 'Stochastic %K',
    'stoch_d': 'Stochastic %D',
    'cci': 'Commodity Channel Index (CCI)',
    'williams_r': 'Williams %R',
    'volume_sma': 'Volume SMA',
    'volume_ratio': 'Volume Ratio',
    'price_change_pct': 'Price Change %',
    'price_from_high': 'Price from 52-week High %',
    'price_from_low': 'Price from 52-week Low %',
}

# Additional builder-specific indicator aliases that resolve to runner keys.
# The strategy builder UI may use friendlier names like 'bollinger_upper';
# these map to the runner's canonical keys.
BUILDER_INDICATOR_ALIASES = {
    'bollinger_upper': 'bb_upper',
    'bollinger_lower': 'bb_lower',
    'bollinger_middle': 'bb_middle',
    'stochastic_k': 'stoch_k',
    'stochastic_d': 'stoch_d',
    'mfi': 'rsi',  # MFI approximated by RSI in the runner
    'roc': 'price_change_pct',  # Rate of change maps to price change %
    'obv': 'volume_sma',  # OBV approximated by volume SMA in the runner
    'vwap': 'ema',  # VWAP approximated by EMA in the runner
}

# Valid position sizing types
VALID_POSITION_SIZING_TYPES = {
    'fixed_percent',
    'fixed_amount',
    'equal_weight',
    'volatility_scaled',
    'kelly_criterion',
}


class StrategyBuilderService:
    """
    Service for the visual strategy builder API.

    Provides config validation, indicator/operator metadata, and
    pre-built strategy presets. Delegates heavy-duty validation and
    indicator knowledge to CustomStrategyRunner.
    """

    # The canonical set of valid operator keys.
    VALID_OPERATORS = set(CustomStrategyRunner.OPERATORS.keys())

    # The canonical set of valid exit condition types.
    VALID_EXIT_TYPES = set(CustomStrategyRunner.EXIT_TYPES.keys())

    # The canonical set of valid indicator keys (runner + builder aliases).
    VALID_INDICATORS = set(CustomStrategyRunner.INDICATORS.keys()) | set(BUILDER_INDICATOR_ALIASES.keys())

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a strategy builder configuration dict.

        Checks entry_conditions for valid indicators, operators, and
        numeric thresholds. Checks exit_conditions for valid types and
        reasonable values. Checks position_sizing for valid types.

        Args:
            config: Strategy configuration dict with entry_conditions,
                    exit_conditions, and optionally position_sizing.

        Returns:
            Dict with 'valid' (bool), 'errors' (list), and 'warnings' (list).
        """
        errors: List[str] = []
        warnings: List[str] = []

        # --- Entry Conditions ---
        entry_conditions = config.get('entry_conditions', [])
        if not entry_conditions:
            errors.append('At least one entry condition is required')
        else:
            self._validate_entry_conditions(entry_conditions, errors, warnings)

        # Validate entry_logic if present (flat format)
        entry_logic = config.get('entry_logic')
        if entry_logic and entry_logic not in ('all', 'any'):
            errors.append('entry_logic must be "all" or "any"')

        # --- Exit Conditions ---
        exit_conditions = config.get('exit_conditions', [])
        if not exit_conditions:
            warnings.append('No exit conditions defined. Consider adding stop loss.')
        else:
            self._validate_exit_conditions(exit_conditions, errors, warnings)

        # --- Position Sizing ---
        position_sizing = config.get('position_sizing', {})
        if position_sizing:
            self._validate_position_sizing(position_sizing, errors, warnings)

        # --- Filters ---
        filters = config.get('filters', {})
        if filters:
            self._validate_filters(filters, warnings)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }

    def _validate_entry_conditions(
        self,
        entry_conditions: Any,
        errors: List[str],
        warnings: List[str],
        prefix: str = 'Entry',
    ) -> None:
        """Validate entry conditions (supports flat list and nested group)."""
        if isinstance(entry_conditions, dict) and entry_conditions.get('type') == 'group':
            self._validate_condition_group(entry_conditions, errors, warnings, prefix)
        elif isinstance(entry_conditions, list):
            for i, cond in enumerate(entry_conditions):
                cond_prefix = f'{prefix} condition {i + 1}'
                if isinstance(cond, dict) and cond.get('type') == 'group':
                    self._validate_condition_group(cond, errors, warnings, cond_prefix)
                else:
                    self._validate_single_condition(cond, errors, warnings, cond_prefix)
        else:
            errors.append(f'{prefix} conditions must be a list or a condition group')

    def _validate_condition_group(
        self,
        group: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        prefix: str,
    ) -> None:
        """Recursively validate a nested condition group."""
        logic = group.get('logic', 'AND').upper()
        if logic not in ('AND', 'OR'):
            errors.append(f'{prefix}: group logic must be "AND" or "OR"')

        conditions = group.get('conditions', [])
        if not conditions:
            errors.append(f'{prefix}: condition group must have at least one condition')
            return

        for i, cond in enumerate(conditions):
            cond_prefix = f'{prefix} [{i + 1}]'
            cond_type = cond.get('type', 'condition')
            if cond_type == 'group':
                self._validate_condition_group(cond, errors, warnings, cond_prefix)
            else:
                self._validate_single_condition(cond, errors, warnings, cond_prefix)

    def _validate_single_condition(
        self,
        cond: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        prefix: str,
    ) -> None:
        """Validate a single indicator condition."""
        indicator = cond.get('indicator')
        operator = cond.get('operator')

        # Indicator check
        if not indicator:
            errors.append(f'{prefix}: indicator is required')
        elif indicator not in self.VALID_INDICATORS:
            errors.append(f'{prefix}: unknown indicator "{indicator}"')

        # Operator check
        if not operator:
            errors.append(f'{prefix}: operator is required')
        elif operator not in self.VALID_OPERATORS:
            errors.append(f'{prefix}: unknown operator "{operator}"')

        # Value/threshold check
        if operator == 'between':
            value_low = cond.get('value_low')
            value_high = cond.get('value_high')
            if value_low is None or value_high is None:
                errors.append(f'{prefix}: "between" operator requires value_low and value_high')
            else:
                if not isinstance(value_low, (int, float)):
                    errors.append(f'{prefix}: value_low must be numeric')
                if not isinstance(value_high, (int, float)):
                    errors.append(f'{prefix}: value_high must be numeric')
                if isinstance(value_low, (int, float)) and isinstance(value_high, (int, float)):
                    if value_low >= value_high:
                        errors.append(f'{prefix}: value_low must be less than value_high')
        elif operator in ('crosses_above', 'crosses_below'):
            value = cond.get('value')
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f'{prefix}: value must be numeric')
        else:
            value = cond.get('value')
            if value is None and operator is not None:
                errors.append(f'{prefix}: value is required for operator "{operator}"')
            elif value is not None and not isinstance(value, (int, float)):
                errors.append(f'{prefix}: value must be numeric')

        # Range warnings using runner's value_range metadata
        resolved = self._resolve_indicator(indicator) if indicator else None
        if resolved and resolved in CustomStrategyRunner.INDICATORS:
            info = CustomStrategyRunner.INDICATORS[resolved]
            value_range = info.get('value_range')
            value = cond.get('value')
            if value_range and value is not None and isinstance(value, (int, float)):
                if value < value_range[0] or value > value_range[1]:
                    warnings.append(
                        f'{prefix}: {indicator} value {value} is outside '
                        f'typical range {value_range}'
                    )

    def _validate_exit_conditions(
        self,
        exit_conditions: List[Dict[str, Any]],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate exit conditions."""
        has_stop_loss = False
        has_take_profit = False

        for i, cond in enumerate(exit_conditions):
            prefix = f'Exit condition {i + 1}'
            cond_type = cond.get('type')

            if not cond_type:
                errors.append(f'{prefix}: type is required')
                continue

            if cond_type not in self.VALID_EXIT_TYPES:
                errors.append(f'{prefix}: unknown exit type "{cond_type}"')
                continue

            if cond_type == 'stop_loss':
                has_stop_loss = True
                value = cond.get('value')
                if value is None:
                    errors.append(f'{prefix}: stop_loss requires a value')
                elif not isinstance(value, (int, float)):
                    errors.append(f'{prefix}: value must be numeric')
                elif value <= 0:
                    errors.append(f'{prefix}: stop_loss value must be positive')
                elif value > 50:
                    warnings.append(f'{prefix}: stop loss of {value}% is extremely wide')
                elif value > 20:
                    warnings.append(f'{prefix}: stop loss of {value}% is very wide')

            elif cond_type == 'take_profit':
                has_take_profit = True
                value = cond.get('value')
                if value is None:
                    errors.append(f'{prefix}: take_profit requires a value')
                elif not isinstance(value, (int, float)):
                    errors.append(f'{prefix}: value must be numeric')
                elif value <= 0:
                    errors.append(f'{prefix}: take_profit value must be positive')

            elif cond_type == 'trailing_stop':
                value = cond.get('value')
                if value is None:
                    errors.append(f'{prefix}: trailing_stop requires a value')
                elif not isinstance(value, (int, float)):
                    errors.append(f'{prefix}: value must be numeric')
                elif value <= 0:
                    errors.append(f'{prefix}: trailing_stop value must be positive')

            elif cond_type == 'time_based':
                days = cond.get('days')
                if days is None:
                    errors.append(f'{prefix}: time_based requires a days value')
                elif not isinstance(days, (int, float)):
                    errors.append(f'{prefix}: days must be numeric')
                elif days <= 0:
                    errors.append(f'{prefix}: days must be positive')

            elif cond_type == 'indicator':
                indicator = cond.get('indicator')
                if not indicator:
                    errors.append(f'{prefix}: indicator exit requires an indicator')
                elif indicator not in self.VALID_INDICATORS:
                    errors.append(f'{prefix}: unknown indicator "{indicator}"')

                operator = cond.get('operator')
                if not operator:
                    errors.append(f'{prefix}: indicator exit requires an operator')
                elif operator not in self.VALID_OPERATORS:
                    errors.append(f'{prefix}: unknown operator "{operator}"')

        if not has_stop_loss and exit_conditions:
            warnings.append('No stop loss defined. This is risky.')

        if has_take_profit and not has_stop_loss:
            warnings.append('Take profit without stop loss creates asymmetric risk')

    def _validate_position_sizing(
        self,
        sizing: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate position sizing configuration."""
        size_type = sizing.get('type')
        if size_type and size_type not in VALID_POSITION_SIZING_TYPES:
            errors.append(f'Unknown position sizing type "{size_type}"')

        value = sizing.get('value')
        if value is not None:
            if not isinstance(value, (int, float)):
                errors.append('Position sizing value must be numeric')
            elif value <= 0:
                errors.append('Position sizing value must be positive')
            elif size_type == 'fixed_percent' and value > 100:
                errors.append('Position sizing percent cannot exceed 100%')
            elif size_type == 'fixed_percent' and value > 25:
                warnings.append(f'Position size of {value}% is very aggressive')

        max_positions = sizing.get('max_positions')
        if max_positions is not None:
            if not isinstance(max_positions, (int, float)):
                errors.append('max_positions must be numeric')
            elif max_positions <= 0:
                errors.append('max_positions must be positive')
            elif max_positions > 20:
                warnings.append(f'Max positions of {max_positions} may be too many')

    def _validate_filters(
        self,
        filters: Dict[str, Any],
        warnings: List[str],
    ) -> None:
        """Validate optional filter settings."""
        min_price = filters.get('min_price')
        if min_price is not None and isinstance(min_price, (int, float)):
            if min_price > 100:
                warnings.append(f'Min price of ${min_price} is very restrictive')

        min_volume = filters.get('min_volume')
        if min_volume is not None and isinstance(min_volume, (int, float)):
            if min_volume > 10_000_000:
                warnings.append(
                    f'Min volume of {min_volume:,.0f} is very restrictive; '
                    f'many stocks will be excluded'
                )

    def _resolve_indicator(self, indicator: str) -> str:
        """Resolve a builder indicator alias to the runner's canonical key."""
        return BUILDER_INDICATOR_ALIASES.get(indicator, indicator)

    def get_available_indicators(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all available indicators.

        Each entry includes:
        - name: canonical key
        - display_name: human-friendly label
        - category: momentum, trend, volatility, volume, or price
        - default_period: default lookback period (or None)
        - value_range: [min, max] or None if unbounded
        - description: short explanation
        """
        indicators = []
        for key, info in CustomStrategyRunner.INDICATORS.items():
            value_range = info.get('value_range')
            indicators.append({
                'name': key,
                'display_name': INDICATOR_DISPLAY_NAMES.get(key, info['name']),
                'category': INDICATOR_CATEGORIES.get(key, 'other'),
                'default_period': info.get('default_period'),
                'value_range': list(value_range) if value_range else None,
                'description': info['description'],
            })
        return indicators

    def get_available_operators(self) -> List[Dict[str, Any]]:
        """
        Return metadata for all available comparison operators.

        Each entry includes:
        - name: canonical key (e.g. 'less_than')
        - display_name: human-friendly label
        - symbol: mathematical symbol
        - description: short explanation
        - requires_history: whether the operator needs previous-bar data
        - requires_range: whether the operator needs value_low/value_high
        """
        operators = []
        for key, info in CustomStrategyRunner.OPERATORS.items():
            operators.append({
                'name': key,
                'display_name': info['name'],
                'symbol': info['symbol'],
                'description': info['description'],
                'requires_history': info.get('requires_history', False),
                'requires_range': info.get('requires_range', False),
            })
        return operators

    def get_presets(self) -> List[Dict[str, Any]]:
        """
        Return pre-configured strategy templates.

        Each preset is a complete CustomStrategy-compatible config dict
        that can be loaded directly into the strategy builder, validated,
        and backtested.

        Returns:
            List of preset dicts, each with id, name, description, and config.
        """
        return [
            {
                'id': 'rsi_oversold_bounce',
                'name': 'RSI Oversold Bounce',
                'description': (
                    'Buy when RSI indicates oversold conditions and the '
                    'short-term SMA is above the long-term SMA (uptrend filter). '
                    'Exit on RSI overbought, take profit at 10%, or stop loss at 5%.'
                ),
                'config': {
                    'entry_conditions': [
                        {
                            'indicator': 'rsi',
                            'operator': 'less_than',
                            'value': 30,
                            'period': 14,
                        },
                        {
                            'indicator': 'sma',
                            'operator': 'greater_than',
                            'value': 0,
                            'period': 20,
                            'compare_to': {'indicator': 'sma', 'period': 50},
                        },
                    ],
                    'entry_logic': 'all',
                    'exit_conditions': [
                        {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 70},
                        {'type': 'take_profit', 'value': 10},
                        {'type': 'stop_loss', 'value': 5},
                    ],
                    'exit_logic': 'any',
                    'position_sizing': {
                        'type': 'fixed_percent',
                        'value': 5,
                        'max_positions': 5,
                    },
                    'filters': {
                        'min_price': 10,
                        'min_volume': 500000,
                    },
                },
            },
            {
                'id': 'macd_cross',
                'name': 'MACD Cross',
                'description': (
                    'Enter when the MACD histogram crosses above zero '
                    '(bullish momentum shift) while ADX confirms trend strength '
                    'above 25. Exit on histogram reversal or stop loss.'
                ),
                'config': {
                    'entry_conditions': [
                        {
                            'indicator': 'macd_histogram',
                            'operator': 'crosses_above',
                            'value': 0,
                        },
                        {
                            'indicator': 'adx',
                            'operator': 'greater_than',
                            'value': 25,
                            'period': 14,
                        },
                    ],
                    'entry_logic': 'all',
                    'exit_conditions': [
                        {'type': 'indicator', 'indicator': 'macd_histogram', 'operator': 'crosses_below', 'value': 0},
                        {'type': 'stop_loss', 'value': 5},
                    ],
                    'exit_logic': 'any',
                    'position_sizing': {
                        'type': 'fixed_percent',
                        'value': 5,
                        'max_positions': 4,
                    },
                    'filters': {
                        'min_price': 20,
                        'min_volume': 1000000,
                    },
                },
            },
            {
                'id': 'bollinger_bounce',
                'name': 'Bollinger Bounce',
                'description': (
                    'Buy when price falls below the lower Bollinger Band '
                    'and RSI confirms oversold conditions below 40. '
                    'Exit when price returns to the middle band, or on '
                    'take profit at 8% / stop loss at 4%.'
                ),
                'config': {
                    'entry_conditions': [
                        {
                            'indicator': 'bb_lower',
                            'operator': 'greater_than',
                            'value': 0,
                            'period': 20,
                        },
                        {
                            'indicator': 'rsi',
                            'operator': 'less_than',
                            'value': 40,
                            'period': 14,
                        },
                    ],
                    'entry_logic': 'all',
                    'exit_conditions': [
                        {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 55},
                        {'type': 'take_profit', 'value': 8},
                        {'type': 'stop_loss', 'value': 4},
                    ],
                    'exit_logic': 'any',
                    'position_sizing': {
                        'type': 'fixed_percent',
                        'value': 4,
                        'max_positions': 5,
                    },
                    'filters': {
                        'min_price': 15,
                        'min_volume': 750000,
                    },
                },
            },
            {
                'id': 'volume_breakout',
                'name': 'Volume Breakout',
                'description': (
                    'Enter when price breaks above the 20-day SMA with '
                    'volume at least 2x the average (volume_ratio > 2) and '
                    'ADX above 20 confirming trend. Exit on SMA breakdown or stop loss.'
                ),
                'config': {
                    'entry_conditions': [
                        {
                            'indicator': 'sma',
                            'operator': 'less_than',
                            'value': 0,
                            'period': 20,
                            'compare_to': 'close',
                        },
                        {
                            'indicator': 'volume_ratio',
                            'operator': 'greater_than',
                            'value': 2,
                            'period': 20,
                        },
                        {
                            'indicator': 'adx',
                            'operator': 'greater_than',
                            'value': 20,
                            'period': 14,
                        },
                    ],
                    'entry_logic': 'all',
                    'exit_conditions': [
                        {'type': 'indicator', 'indicator': 'adx', 'operator': 'less_than', 'value': 15},
                        {'type': 'stop_loss', 'value': 5},
                    ],
                    'exit_logic': 'any',
                    'position_sizing': {
                        'type': 'fixed_percent',
                        'value': 4,
                        'max_positions': 5,
                    },
                    'filters': {
                        'min_price': 15,
                        'min_volume': 1000000,
                    },
                },
            },
            {
                'id': 'mean_reversion',
                'name': 'Mean Reversion',
                'description': (
                    'Buy extreme oversold conditions: RSI below 25 and '
                    'price below the lower Bollinger Band. Target a quick '
                    'reversion with take profit at 5% and tight stop loss at 3%.'
                ),
                'config': {
                    'entry_conditions': [
                        {
                            'indicator': 'rsi',
                            'operator': 'less_than',
                            'value': 25,
                            'period': 14,
                        },
                        {
                            'indicator': 'bb_lower',
                            'operator': 'greater_than',
                            'value': 0,
                            'period': 20,
                        },
                    ],
                    'entry_logic': 'all',
                    'exit_conditions': [
                        {'type': 'indicator', 'indicator': 'rsi', 'operator': 'greater_than', 'value': 50},
                        {'type': 'take_profit', 'value': 5},
                        {'type': 'stop_loss', 'value': 3},
                    ],
                    'exit_logic': 'any',
                    'position_sizing': {
                        'type': 'fixed_percent',
                        'value': 3,
                        'max_positions': 6,
                    },
                    'filters': {
                        'min_price': 10,
                        'min_volume': 500000,
                    },
                },
            },
        ]
