#!/usr/bin/env python3
"""
Robust error handling utilities for WallStreetBots trading strategies.
Handles yfinance API failures, market hours, network issues, and data validation.
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional, Any, Callable, TypeVar
from functools import wraps
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class MarketDataError(Exception):
    """Custom exception for market data related errors."""
    pass

class MarketClosedError(MarketDataError):
    """Raised when trying to access market data outside trading hours."""
    pass

class DataValidationError(MarketDataError):
    """Raised when market data fails validation checks."""
    pass

def is_market_hours() -> bool:
    """
    Check if US stock market is currently open.
    Returns True if market is open, False otherwise.
    """
    now = datetime.now(timezone.utc)
    
    # Convert to EST/EDT (market timezone)
    import pytz
    market_tz = pytz.timezone('US/Eastern')
    market_time = now.astimezone(market_tz)
    
    # Market hours: 9:30 AM - 4:00 PM EST/EDT, Monday-Friday
    if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_open = market_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = market_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= market_time <= market_close

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    if jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)  # Add ±50% jitter
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=2.0)
def safe_yfinance_history(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    timeout: int = 30
) -> pd.DataFrame:
    """
    Safely fetch historical data from yfinance with retry logic.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame with historical data
        
    Raises:
        MarketDataError: If data cannot be fetched or is invalid
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Set timeout for the request
        stock.session.timeout = timeout
        
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            raise MarketDataError(f"No data returned for {ticker}")
        
        # Validate data quality
        if len(data) < 2:
            raise DataValidationError(f"Insufficient data points for {ticker}: {len(data)}")
        
        # Check for all NaN values
        if data['Close'].isna().all():
            raise DataValidationError(f"All close prices are NaN for {ticker}")
        
        # Check for recent data (within last 7 days for intraday data)
        if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
            latest_date = data.index[-1]
            days_old = (datetime.now() - latest_date.replace(tzinfo=None)).days
            if days_old > 7:
                logger.warning(f"Data for {ticker} is {days_old} days old")
        
        logger.debug(f"Successfully fetched {len(data)} data points for {ticker}")
        return data
        
    except Exception as e:
        if "No data found" in str(e):
            raise MarketDataError(f"No data found for ticker {ticker}. It may be delisted or invalid.")
        elif "timeout" in str(e).lower():
            raise MarketDataError(f"Timeout fetching data for {ticker}")
        else:
            raise MarketDataError(f"Error fetching data for {ticker}: {e}")

@retry_with_backoff(max_retries=2, base_delay=1.0)
def safe_yfinance_options_chain(
    ticker: str,
    expiry: str,
    timeout: int = 30
) -> Optional[Any]:
    """
    Safely fetch options chain from yfinance with retry logic.
    
    Args:
        ticker: Stock ticker symbol
        expiry: Options expiry date (YYYY-MM-DD)
        timeout: Request timeout in seconds
    
    Returns:
        Options chain object or None if not available
        
    Raises:
        MarketDataError: If options chain cannot be fetched
    """
    try:
        stock = yf.Ticker(ticker)
        stock.session.timeout = timeout
        
        chain = stock.option_chain(expiry)
        
        if chain.calls.empty and chain.puts.empty:
            logger.warning(f"No options data available for {ticker} {expiry}")
            return None
        
        # Validate options data quality
        if not chain.calls.empty:
            calls = chain.calls
            # Check for reasonable bid/ask spreads
            valid_calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0) & (calls['ask'] > calls['bid'])]
            if len(valid_calls) == 0:
                logger.warning(f"No valid call options with positive bid/ask for {ticker} {expiry}")
        
        logger.debug(f"Successfully fetched options chain for {ticker} {expiry}")
        return chain
        
    except Exception as e:
        if "No options data" in str(e):
            logger.warning(f"No options data available for {ticker} {expiry}")
            return None
        elif "timeout" in str(e).lower():
            raise MarketDataError(f"Timeout fetching options for {ticker} {expiry}")
        else:
            raise MarketDataError(f"Error fetching options for {ticker} {expiry}: {e}")

def validate_market_data(data: pd.DataFrame, ticker: str, min_points: int = 10) -> bool:
    """
    Validate market data quality.
    
    Args:
        data: DataFrame with market data
        ticker: Stock ticker for logging
        min_points: Minimum number of data points required
    
    Returns:
        True if data is valid, False otherwise
    """
    if data.empty:
        logger.error(f"Empty data for {ticker}")
        return False
    
    if len(data) < min_points:
        logger.error(f"Insufficient data points for {ticker}: {len(data)} < {min_points}")
        return False
    
    # Check for required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Missing columns for {ticker}: {missing_columns}")
        return False
    
    # Check for all NaN values in critical columns
    for col in ['Close', 'Volume']:
        if data[col].isna().all():
            logger.error(f"All {col} values are NaN for {ticker}")
            return False
    
    # Check for negative prices
    if (data['Close'] <= 0).any():
        logger.error(f"Non-positive close prices found for {ticker}")
        return False
    
    # Check for negative volume
    if (data['Volume'] < 0).any():
        logger.error(f"Negative volume found for {ticker}")
        return False
    
    logger.debug(f"Data validation passed for {ticker}")
    return True

def get_safe_current_price(ticker: str) -> Optional[float]:
    """
    Safely get current price for a ticker with fallback mechanisms.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Current price or None if unavailable
    """
    try:
        # Try intraday data first
        data = safe_yfinance_history(ticker, period="1d", interval="1m")
        if validate_market_data(data, ticker, min_points=1):
            return float(data['Close'].iloc[-1])
    except MarketDataError:
        pass
    
    try:
        # Fallback to daily data
        data = safe_yfinance_history(ticker, period="5d", interval="1d")
        if validate_market_data(data, ticker, min_points=1):
            return float(data['Close'].iloc[-1])
    except MarketDataError:
        pass
    
    logger.error(f"Could not fetch current price for {ticker}")
    return None

def log_trading_hours_warning():
    """Log a warning if trying to trade outside market hours."""
    if not is_market_hours():
        logger.warning("⚠️  Market is currently CLOSED. Live trading data may be stale.")
        logger.info("Market hours: 9:30 AM - 4:00 PM EST/EDT, Monday-Friday")
