"""
FindatapyDataProvider - Unified data provider using findatapy library.

Provides multi-source data fetching with automatic fallback chain:
Yahoo Finance → Quandl → FRED → Alpha Vantage

Features:
- Unified API across 15+ data sources
- Built-in caching (SpeedCache/Redis)
- Automatic fallback on source failure
- Configurable data sources per asset class
"""

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add vendored findatapy to path
VENDOR_PATH = Path(__file__).parent.parent.parent.parent / "vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))

try:
    from findatapy.market import Market, MarketDataRequest, MarketDataGenerator
    FINDATAPY_AVAILABLE = True
except ImportError:
    FINDATAPY_AVAILABLE = False
    Market = None
    MarketDataRequest = None
    MarketDataGenerator = None


logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for a data source with fallback chain."""

    primary: str = "yahoo"
    fallbacks: List[str] = field(default_factory=lambda: ["quandl", "fred", "alphavantage"])
    cache_algo: str = "internet_load_return"  # Cache after fetching

    # Source-specific settings
    yahoo_fields: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    fred_fields: List[str] = field(default_factory=lambda: ["close"])

    # API keys (set via environment or direct assignment)
    quandl_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None


@dataclass
class FetchResult:
    """Result of a data fetch operation."""

    data: Optional[pd.DataFrame]
    source: str
    success: bool
    error: Optional[str] = None
    fetch_time: float = 0.0
    from_cache: bool = False


class FindatapyDataProvider:
    """
    Unified data provider using findatapy for multi-source data fetching.

    Provides automatic fallback chain and intelligent caching for:
    - Historical OHLCV data
    - Real-time quotes
    - Economic indicators
    - FX rates

    Example:
        provider = FindatapyDataProvider()
        df = provider.get_historical_data("AAPL", days=365)
        df = provider.get_historical_data(["AAPL", "MSFT", "GOOGL"], days=252)
    """

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the findatapy data provider.

        Args:
            config: Data source configuration with fallback chain
            use_cache: Enable/disable caching (default: True)
        """
        if not FINDATAPY_AVAILABLE:
            raise ImportError(
                "findatapy is not available. Install it or ensure ../findatapy is accessible."
            )

        self.config = config or DataSourceConfig()
        self.use_cache = use_cache

        # Initialize findatapy Market
        self._market = Market(market_data_generator=MarketDataGenerator())

        # Track source health for intelligent fallback
        self._source_health: Dict[str, Dict[str, Any]] = {}

        # Standard column mapping
        self._column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }

        logger.info(f"FindatapyDataProvider initialized with primary source: {self.config.primary}")

    def get_historical_data(
        self,
        tickers: str | List[str],
        days: int = 365,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        fields: Optional[List[str]] = None,
        freq: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic fallback.

        Args:
            tickers: Single ticker or list of tickers
            days: Number of days of history (used if start_date not provided)
            start_date: Start date for data
            end_date: End date for data (default: today)
            fields: Fields to fetch (default: OHLCV)
            freq: Data frequency ("daily", "hourly", "minute")

        Returns:
            DataFrame with OHLCV data, columns lowercase
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Calculate date range
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=days))

        fields = fields or self.config.yahoo_fields

        # Build fallback chain
        sources = [self.config.primary] + self.config.fallbacks

        last_error = None
        for source in sources:
            try:
                result = self._fetch_from_source(
                    tickers=tickers,
                    source=source,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    freq=freq,
                )

                if result.success and result.data is not None and not result.data.empty:
                    self._update_source_health(source, success=True)
                    logger.info(
                        f"Successfully fetched {len(tickers)} ticker(s) from {source} "
                        f"({len(result.data)} rows, cache={result.from_cache})"
                    )
                    return result.data

            except Exception as e:
                last_error = str(e)
                self._update_source_health(source, success=False, error=last_error)
                logger.warning(f"Source {source} failed: {e}, trying next...")
                continue

        # All sources failed
        logger.error(f"All data sources failed for {tickers}. Last error: {last_error}")
        return pd.DataFrame()

    def _fetch_from_source(
        self,
        tickers: List[str],
        source: str,
        start_date: datetime,
        end_date: datetime,
        fields: List[str],
        freq: str,
    ) -> FetchResult:
        """
        Fetch data from a specific source using findatapy.

        Args:
            tickers: List of tickers
            source: Data source name
            start_date: Start date
            end_date: End date
            fields: Fields to fetch
            freq: Frequency

        Returns:
            FetchResult with data or error
        """
        import time
        start_time = time.time()

        try:
            # Build MarketDataRequest
            md_request = MarketDataRequest(
                start_date=start_date.strftime("%d %b %Y"),
                finish_date=end_date.strftime("%d %b %Y"),
                data_source=source,
                tickers=tickers,
                vendor_tickers=self._get_vendor_tickers(tickers, source),
                fields=fields,
                vendor_fields=self._get_vendor_fields(fields, source),
                freq=freq,
                cache_algo=self.config.cache_algo if self.use_cache else "internet_load",
            )

            # Fetch data
            df = self._market.fetch_market(md_request)

            fetch_time = time.time() - start_time

            if df is None or df.empty:
                return FetchResult(
                    data=None,
                    source=source,
                    success=False,
                    error="Empty result",
                    fetch_time=fetch_time,
                )

            # Standardize column names
            df = self._standardize_columns(df)

            return FetchResult(
                data=df,
                source=source,
                success=True,
                fetch_time=fetch_time,
                from_cache=self.config.cache_algo == "cache_algo_return",
            )

        except Exception as e:
            return FetchResult(
                data=None,
                source=source,
                success=False,
                error=str(e),
                fetch_time=time.time() - start_time,
            )

    def _get_vendor_tickers(self, tickers: List[str], source: str) -> List[str]:
        """
        Convert standard tickers to vendor-specific format.

        For most sources, tickers pass through unchanged.
        Override this for custom ticker mappings.
        """
        # Yahoo uses lowercase for some tickers
        if source == "yahoo":
            return [t.lower() if len(t) <= 5 else t for t in tickers]
        return tickers

    def _get_vendor_fields(self, fields: List[str], source: str) -> List[str]:
        """
        Convert standard fields to vendor-specific format.
        """
        vendor_field_map = {
            "yahoo": {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            "fred": {
                "close": "close",
            },
            "quandl": {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
        }

        field_map = vendor_field_map.get(source, {})
        return [field_map.get(f, f) for f in fields]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame columns to lowercase.
        """
        # Handle multi-level columns (ticker, field)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten to single level with ticker prefix
            new_cols = []
            for col in df.columns:
                if len(col) == 2:
                    ticker, field = col
                    field_lower = self._column_map.get(field, field.lower())
                    new_cols.append(f"{ticker}_{field_lower}")
                else:
                    new_cols.append("_".join(str(c) for c in col).lower())
            df.columns = new_cols
        else:
            # Simple column rename
            df.columns = [self._column_map.get(c, c.lower()) for c in df.columns]

        return df

    def _update_source_health(
        self,
        source: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Track source health for intelligent fallback ordering."""
        if source not in self._source_health:
            self._source_health[source] = {
                "success_count": 0,
                "failure_count": 0,
                "last_success": None,
                "last_failure": None,
                "last_error": None,
            }

        health = self._source_health[source]
        if success:
            health["success_count"] += 1
            health["last_success"] = datetime.now()
        else:
            health["failure_count"] += 1
            health["last_failure"] = datetime.now()
            health["last_error"] = error

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health statistics for all sources."""
        return self._source_health.copy()

    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current price for a ticker.

        Uses 1-day history and returns the last close price.
        """
        df = self.get_historical_data(ticker, days=5)
        if df.empty:
            return None

        # Find the close column
        close_col = None
        for col in df.columns:
            if "close" in col.lower():
                close_col = col
                break

        if close_col is None:
            return None

        return float(df[close_col].iloc[-1])

    def get_multiple_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple tickers."""
        df = self.get_historical_data(tickers, days=5)

        prices = {}
        for ticker in tickers:
            # Look for ticker-specific close column
            close_col = f"{ticker}_close"
            if close_col in df.columns:
                prices[ticker] = float(df[close_col].iloc[-1])
            else:
                prices[ticker] = None

        return prices


class MLDataFetcher:
    """
    ML-optimized data fetcher using findatapy.

    Designed for training ML models with:
    - Technical indicator calculation
    - Data normalization
    - Train/test splitting
    - Sequence preparation for LSTM/Transformer
    """

    def __init__(
        self,
        provider: Optional[FindatapyDataProvider] = None,
        config: Optional[DataSourceConfig] = None,
    ):
        """
        Initialize ML data fetcher.

        Args:
            provider: Optional FindatapyDataProvider instance
            config: Data source configuration
        """
        self.provider = provider or FindatapyDataProvider(config=config)

    def fetch_training_data(
        self,
        tickers: str | List[str],
        days: int = 1000,
        add_indicators: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch data suitable for ML training.

        Args:
            tickers: Ticker(s) to fetch
            days: Days of history
            add_indicators: Add technical indicators

        Returns:
            DataFrame with OHLCV and optional indicators
        """
        df = self.provider.get_historical_data(tickers, days=days)

        if df.empty:
            logger.warning(f"No data fetched for {tickers}")
            return df

        if add_indicators:
            df = self._add_technical_indicators(df)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.

        Adds: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, returns, volatility
        """
        # Find the close column
        close_col = None
        high_col = None
        low_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "close" in col_lower and close_col is None:
                close_col = col
            elif "high" in col_lower and high_col is None:
                high_col = col
            elif "low" in col_lower and low_col is None:
                low_col = col

        if close_col is None:
            logger.warning("No close column found, skipping indicators")
            return df

        close = df[close_col]

        # Simple Moving Averages
        df["sma_10"] = close.rolling(window=10).mean()
        df["sma_20"] = close.rolling(window=20).mean()
        df["sma_50"] = close.rolling(window=50).mean()

        # Exponential Moving Averages
        df["ema_10"] = close.ewm(span=10, adjust=False).mean()
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.inf)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20

        # Returns and volatility
        df["returns"] = close.pct_change()
        df["volatility_20"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

        # ATR (if high/low available)
        if high_col and low_col:
            high = df[high_col]
            low = df[low_col]

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr_14"] = tr.rolling(window=14).mean()

        return df

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        seq_length: int = 60,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM/Transformer training.

        Args:
            df: DataFrame with features
            seq_length: Sequence length
            target_col: Target column (default: first close column)
            feature_cols: Feature columns (default: all numeric)

        Returns:
            Tuple of (X sequences, y targets)
        """
        # Find target column
        if target_col is None:
            for col in df.columns:
                if "close" in col.lower():
                    target_col = col
                    break

        if target_col is None:
            raise ValueError("No target column found")

        # Select feature columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Drop NaN rows
        df_clean = df[feature_cols].dropna()

        if len(df_clean) < seq_length + 1:
            raise ValueError(f"Not enough data: {len(df_clean)} rows, need {seq_length + 1}")

        # Create sequences
        X, y = [], []
        values = df_clean.values
        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0

        for i in range(len(values) - seq_length):
            X.append(values[i:i + seq_length])
            y.append(values[i + seq_length, target_idx])

        return np.array(X), np.array(y)


# Factory function for easy instantiation
def create_findatapy_provider(
    primary_source: str = "yahoo",
    fallbacks: Optional[List[str]] = None,
    use_cache: bool = True,
) -> FindatapyDataProvider:
    """
    Factory function to create a FindatapyDataProvider.

    Args:
        primary_source: Primary data source
        fallbacks: Fallback sources
        use_cache: Enable caching

    Returns:
        Configured FindatapyDataProvider
    """
    config = DataSourceConfig(
        primary=primary_source,
        fallbacks=fallbacks or ["quandl", "fred", "alphavantage"],
    )
    return FindatapyDataProvider(config=config, use_cache=use_cache)


# Check availability
def is_findatapy_available() -> bool:
    """Check if findatapy is available."""
    return FINDATAPY_AVAILABLE
