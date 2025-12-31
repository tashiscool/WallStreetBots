"""
Multi-Timeframe Analysis

Decorator and utilities for combining signals from multiple timeframes.
Inspired by freqtrade's @informative decorator.
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# Timeframe conversion helpers
TIMEFRAME_MINUTES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '1d': 1440,
    '1w': 10080,
}


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    return TIMEFRAME_MINUTES.get(timeframe, 60)


def resample_dataframe(
    df: pd.DataFrame,
    source_timeframe: str,
    target_timeframe: str,
) -> pd.DataFrame:
    """
    Resample OHLCV data from one timeframe to another.

    Args:
        df: DataFrame with OHLCV columns and datetime index
        source_timeframe: Source timeframe (e.g., '5m')
        target_timeframe: Target timeframe (e.g., '1h')

    Returns:
        Resampled DataFrame
    """
    source_mins = timeframe_to_minutes(source_timeframe)
    target_mins = timeframe_to_minutes(target_timeframe)

    if target_mins <= source_mins:
        logger.warning(f"Cannot resample to smaller timeframe: {source_timeframe} -> {target_timeframe}")
        return df

    # Resample rules
    resample_rule = f'{target_mins}T'  # e.g., '60T' for 1 hour

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }

    resampled = df.resample(resample_rule).agg(ohlc_dict)
    resampled = resampled.dropna()

    return resampled


def informative(
    timeframe: str,
    asset: Optional[str] = None,
    fmt: Optional[str] = None,
):
    """
    Decorator to add informative data from another timeframe.

    Usage:
        class MyStrategy(BaseStrategy):

            @informative('1h')
            def populate_indicators_1h(self, dataframe, metadata):
                dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=14)
                return dataframe

            @informative('1d', 'BTC/USDT')  # Different asset
            def populate_indicators_btc_1d(self, dataframe, metadata):
                dataframe['btc_trend'] = dataframe['close'] > dataframe['close'].shift(1)
                return dataframe

    Args:
        timeframe: The informative timeframe (e.g., '1h', '4h', '1d')
        asset: Optional different asset (default: same as main)
        fmt: Column name format (default: '{column}_{timeframe}')
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, dataframe: pd.DataFrame, metadata: dict):
            # Store informative metadata
            func._informative_timeframe = timeframe
            func._informative_asset = asset

            # Call the original function
            result = func(self, dataframe, metadata)

            return result

        wrapper._informative_timeframe = timeframe
        wrapper._informative_asset = asset
        wrapper._fmt = fmt or '{column}_{timeframe}'

        return wrapper

    return decorator


def merge_informative(
    main_df: pd.DataFrame,
    informative_df: pd.DataFrame,
    suffix: str = '_inf',
    ffill: bool = True,
) -> pd.DataFrame:
    """
    Merge informative dataframe into main dataframe.

    Args:
        main_df: Main dataframe (higher frequency)
        informative_df: Informative dataframe (lower frequency)
        suffix: Column suffix for informative columns
        ffill: Forward fill informative values

    Returns:
        Merged DataFrame
    """
    # Add suffix to informative columns
    inf_columns = {
        col: f'{col}{suffix}'
        for col in informative_df.columns
    }
    informative_df = informative_df.rename(columns=inf_columns)

    # Merge on index (datetime)
    merged = main_df.merge(
        informative_df,
        how='left',
        left_index=True,
        right_index=True,
    )

    # Forward fill informative values
    if ffill:
        inf_cols = list(inf_columns.values())
        merged[inf_cols] = merged[inf_cols].ffill()

    return merged


class MultiTimeframeAnalyzer:
    """
    Analyze data across multiple timeframes.

    Example:
        analyzer = MultiTimeframeAnalyzer(data_client)
        analyzer.add_timeframe('1h', ['rsi', 'ema_20'])
        analyzer.add_timeframe('1d', ['ema_200'])

        signals = analyzer.analyze('AAPL')
        # Returns: {'rsi_1h': 45, 'ema_20_1h': 185.5, 'ema_200_1d': 180.2}
    """

    def __init__(self, data_client):
        """
        Initialize MultiTimeframeAnalyzer.

        Args:
            data_client: Data client for fetching historical data
        """
        self.data_client = data_client
        self._timeframes: Dict[str, List[str]] = {}  # timeframe -> indicators
        self._indicator_funcs: Dict[str, Callable] = {}

    def add_timeframe(
        self,
        timeframe: str,
        indicators: List[str],
    ) -> None:
        """
        Add a timeframe with indicators to analyze.

        Args:
            timeframe: Timeframe string (e.g., '1h', '4h')
            indicators: List of indicator names to calculate
        """
        self._timeframes[timeframe] = indicators

    def register_indicator(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> None:
        """
        Register a custom indicator function.

        Args:
            name: Indicator name
            func: Function taking DataFrame, returning Series
        """
        self._indicator_funcs[name] = func

    def analyze(
        self,
        symbol: str,
        base_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Analyze symbol across all configured timeframes.

        Args:
            symbol: Ticker symbol
            base_data: Optional base dataframe (if not provided, fetched from client)

        Returns:
            Dictionary of indicator values keyed by '{indicator}_{timeframe}'
        """
        results = {}

        for timeframe, indicators in self._timeframes.items():
            # Fetch data for this timeframe
            tf_data = self._get_timeframe_data(symbol, timeframe, base_data)

            if tf_data is None or tf_data.empty:
                logger.warning(f"No data for {symbol} at {timeframe}")
                continue

            # Calculate each indicator
            for indicator in indicators:
                try:
                    value = self._calculate_indicator(tf_data, indicator)
                    key = f'{indicator}_{timeframe}'
                    results[key] = value
                except Exception as e:
                    logger.error(f"Error calculating {indicator} for {symbol}: {e}")

        return results

    def _get_timeframe_data(
        self,
        symbol: str,
        timeframe: str,
        base_data: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Fetch or resample data for timeframe."""
        if base_data is not None:
            # Resample from base data
            base_tf = self._detect_timeframe(base_data)
            return resample_dataframe(base_data, base_tf, timeframe)

        # Fetch from client
        try:
            return self.data_client.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=200,
            )
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None

    def _detect_timeframe(self, df: pd.DataFrame) -> str:
        """Detect timeframe from DataFrame index."""
        if len(df) < 2:
            return '1d'

        diff = (df.index[1] - df.index[0]).total_seconds() / 60

        for tf, minutes in TIMEFRAME_MINUTES.items():
            if abs(diff - minutes) < 1:
                return tf

        return '1d'  # Default

    def _calculate_indicator(
        self,
        df: pd.DataFrame,
        indicator: str,
    ) -> Any:
        """Calculate a single indicator."""
        # Check for registered custom indicator
        if indicator in self._indicator_funcs:
            return self._indicator_funcs[indicator](df).iloc[-1]

        # Built-in indicators
        if indicator == 'rsi':
            return self._calculate_rsi(df)
        elif indicator.startswith('ema_'):
            period = int(indicator.split('_')[1])
            return self._calculate_ema(df, period)
        elif indicator.startswith('sma_'):
            period = int(indicator.split('_')[1])
            return self._calculate_sma(df, period)
        elif indicator == 'atr':
            return self._calculate_atr(df)
        elif indicator == 'macd':
            return self._calculate_macd(df)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI."""
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calculate EMA."""
        return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

    def _calculate_sma(self, df: pd.DataFrame, period: int) -> float:
        """Calculate SMA."""
        return df['close'].rolling(window=period).mean().iloc[-1]

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1]

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD."""
        close = df['close']
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
        }


def create_multi_timeframe_dataframe(
    data_client,
    symbol: str,
    base_timeframe: str,
    informative_timeframes: List[str],
    lookback_periods: int = 200,
) -> pd.DataFrame:
    """
    Create a single DataFrame with all timeframe data merged.

    Args:
        data_client: Data client
        symbol: Ticker symbol
        base_timeframe: Base timeframe for main data
        informative_timeframes: List of informative timeframes
        lookback_periods: Number of periods to fetch

    Returns:
        Merged DataFrame with columns suffixed by timeframe
    """
    # Fetch base data
    base_df = data_client.get_ohlcv(
        symbol=symbol,
        timeframe=base_timeframe,
        limit=lookback_periods,
    )

    if base_df is None or base_df.empty:
        return pd.DataFrame()

    # Merge each informative timeframe
    for inf_tf in informative_timeframes:
        inf_df = data_client.get_ohlcv(
            symbol=symbol,
            timeframe=inf_tf,
            limit=lookback_periods,
        )

        if inf_df is not None and not inf_df.empty:
            base_df = merge_informative(
                base_df,
                inf_df,
                suffix=f'_{inf_tf}',
            )

    return base_df
