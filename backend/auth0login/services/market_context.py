"""
Market Context Service - Comprehensive market overview for the dashboard.

Provides:
- Major indices (SPY, QQQ, IWM, DIA) with prices and changes
- VIX monitoring with level classification
- Sector performance heat map data
- Holdings events (earnings, dividends)
- Economic calendar
- Market status (open/closed/pre-market/after-hours)

Uses aggressive caching to minimize API calls:
- Prices: 1-5 minute TTL during market hours
- Events: 1 hour TTL
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market trading session status."""
    PRE_MARKET = "pre_market"       # 4:00 AM - 9:30 AM ET
    OPEN = "open"                   # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"     # 4:00 PM - 8:00 PM ET
    CLOSED = "closed"               # 8:00 PM - 4:00 AM ET


class VIXLevel(Enum):
    """VIX volatility classification."""
    LOW = "low"           # VIX < 15
    NORMAL = "normal"     # VIX 15-20
    ELEVATED = "elevated" # VIX 20-25
    HIGH = "high"         # VIX 25-35
    EXTREME = "extreme"   # VIX > 35


@dataclass
class IndexData:
    """Data for a market index."""
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    trend: str  # "up", "down", "flat"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': round(self.price, 2),
            'change': round(self.change, 2),
            'change_pct': round(self.change_pct, 2),
            'trend': self.trend,
        }


@dataclass
class SectorData:
    """Data for a sector ETF."""
    symbol: str
    name: str
    change_pct: float
    trend: str

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'change_pct': round(self.change_pct, 2),
            'trend': self.trend,
        }


@dataclass
class HoldingEvent:
    """Upcoming event for a held position."""
    symbol: str
    event_type: str  # "earnings", "ex-dividend", "split"
    date: date
    days_until: int
    details: str = ""

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'event': self.event_type,
            'date': self.date.isoformat(),
            'days_until': self.days_until,
            'details': self.details,
        }


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    event: str
    date: date
    time: Optional[str]
    importance: str  # "high", "medium", "low"
    forecast: Optional[str] = None
    previous: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'event': self.event,
            'date': self.date.isoformat(),
            'time': self.time,
            'importance': self.importance,
            'forecast': self.forecast,
            'previous': self.previous,
        }


class MarketContextCache:
    """Simple TTL cache for market data."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str, ttl: int = 60) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache value with current timestamp."""
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


# Sector ETF mappings
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
    'XLB': 'Materials',
}

# Major index symbols
INDEX_SYMBOLS = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'DIA': 'Dow Jones',
}


class MarketContextService:
    """
    Comprehensive market context service for dashboard display.

    Features:
    - Major indices with price and change data
    - VIX monitoring with level classification
    - Sector performance heat map
    - Holdings-specific events (earnings, dividends)
    - Economic calendar
    - Market status tracking
    """

    CACHE_TTL_PRICES = 60       # 1 minute for prices during market hours
    CACHE_TTL_PRICES_CLOSED = 300  # 5 minutes when market closed
    CACHE_TTL_EVENTS = 3600     # 1 hour for events
    CACHE_TTL_VIX = 60          # 1 minute for VIX

    def __init__(self):
        self.cache = MarketContextCache()
        self._et_tz = ZoneInfo("America/New_York")

    def _get_et_now(self) -> datetime:
        """Get current time in Eastern timezone."""
        return datetime.now(self._et_tz)

    def get_market_status(self) -> dict:
        """
        Get current market status and next event.

        Returns:
            Dict with status, next_event, and time info
        """
        now = self._get_et_now()
        day = now.weekday()
        hour = now.hour
        minute = now.minute
        total_minutes = hour * 60 + minute

        # Market hours in minutes from midnight
        pre_market_start = 4 * 60       # 4:00 AM
        market_open = 9 * 60 + 30       # 9:30 AM
        market_close = 16 * 60          # 4:00 PM
        after_hours_end = 20 * 60       # 8:00 PM

        # Weekend check
        if day >= 5:  # Saturday or Sunday
            status = MarketStatus.CLOSED
            days_until_monday = 7 - day
            next_event = "Market opens Monday at 9:30 AM ET"
        # Pre-market
        elif pre_market_start <= total_minutes < market_open:
            status = MarketStatus.PRE_MARKET
            minutes_until = market_open - total_minutes
            hours = minutes_until // 60
            mins = minutes_until % 60
            next_event = f"Market opens in {hours}h {mins}m"
        # Regular hours
        elif market_open <= total_minutes < market_close:
            status = MarketStatus.OPEN
            minutes_until = market_close - total_minutes
            hours = minutes_until // 60
            mins = minutes_until % 60
            next_event = f"Market closes in {hours}h {mins}m"
        # After hours
        elif market_close <= total_minutes < after_hours_end:
            status = MarketStatus.AFTER_HOURS
            minutes_until = after_hours_end - total_minutes
            hours = minutes_until // 60
            mins = minutes_until % 60
            next_event = f"After-hours ends in {hours}h {mins}m"
        # Night (before pre-market)
        elif total_minutes < pre_market_start:
            status = MarketStatus.CLOSED
            minutes_until = pre_market_start - total_minutes
            hours = minutes_until // 60
            mins = minutes_until % 60
            next_event = f"Pre-market opens in {hours}h {mins}m"
        else:
            status = MarketStatus.CLOSED
            next_event = "Market opens at 9:30 AM ET"

        return {
            'status': status.value,
            'is_open': status == MarketStatus.OPEN,
            'is_trading': status in (MarketStatus.PRE_MARKET, MarketStatus.OPEN, MarketStatus.AFTER_HOURS),
            'next_event': next_event,
            'current_time': now.strftime('%I:%M %p ET'),
        }

    def get_market_overview(self, force_refresh: bool = False) -> dict:
        """
        Get current market snapshot with major indices and VIX.

        Returns:
            Dict with indices, VIX, market status
        """
        cache_key = "market_overview"
        market_status = self.get_market_status()
        ttl = self.CACHE_TTL_PRICES if market_status['is_trading'] else self.CACHE_TTL_PRICES_CLOSED

        if not force_refresh:
            cached = self.cache.get(cache_key, ttl=ttl)
            if cached:
                cached['market_status'] = market_status
                return cached

        indices = {}
        vix_data = {}

        try:
            import yfinance as yf

            # Fetch indices
            symbols = [*list(INDEX_SYMBOLS.keys()), '^VIX']
            tickers = yf.Tickers(' '.join(symbols))

            for symbol in INDEX_SYMBOLS.keys():
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period='2d')
                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change = current - prev
                        change_pct = (change / prev) * 100

                        indices[symbol] = IndexData(
                            symbol=symbol,
                            name=INDEX_SYMBOLS[symbol],
                            price=current,
                            change=change,
                            change_pct=change_pct,
                            trend='up' if change > 0 else ('down' if change < 0 else 'flat'),
                        ).to_dict()
                    elif len(hist) >= 1:
                        current = hist['Close'].iloc[-1]
                        indices[symbol] = IndexData(
                            symbol=symbol,
                            name=INDEX_SYMBOLS[symbol],
                            price=current,
                            change=0,
                            change_pct=0,
                            trend='flat',
                        ).to_dict()
                except Exception as e:
                    logger.warning(f"Failed to fetch {symbol}: {e}")

            # Fetch VIX
            try:
                vix_ticker = yf.Ticker('^VIX')
                vix_hist = vix_ticker.history(period='2d')
                if len(vix_hist) >= 1:
                    vix_value = vix_hist['Close'].iloc[-1]
                    vix_change = 0
                    if len(vix_hist) >= 2:
                        vix_change = vix_value - vix_hist['Close'].iloc[-2]

                    # Classify VIX level
                    if vix_value < 15:
                        level = VIXLevel.LOW
                    elif vix_value < 20:
                        level = VIXLevel.NORMAL
                    elif vix_value < 25:
                        level = VIXLevel.ELEVATED
                    elif vix_value < 35:
                        level = VIXLevel.HIGH
                    else:
                        level = VIXLevel.EXTREME

                    vix_data = {
                        'value': round(vix_value, 2),
                        'change': round(vix_change, 2),
                        'level': level.value,
                    }
            except Exception as e:
                logger.warning(f"Failed to fetch VIX: {e}")

        except ImportError:
            logger.warning("yfinance not installed")
        except Exception as e:
            logger.error(f"Market overview fetch error: {e}")

        result = {
            'indices': indices,
            'vix': vix_data,
            'market_status': market_status,
            'last_updated': datetime.now().isoformat(),
        }

        self.cache.set(cache_key, result)
        return result

    def get_sector_performance(self, force_refresh: bool = False) -> list[dict]:
        """
        Get sector ETF performance for heat map display.

        Returns:
            List of sector data dicts sorted by performance
        """
        cache_key = "sector_performance"
        market_status = self.get_market_status()
        ttl = self.CACHE_TTL_PRICES if market_status['is_trading'] else self.CACHE_TTL_PRICES_CLOSED

        if not force_refresh:
            cached = self.cache.get(cache_key, ttl=ttl)
            if cached:
                return cached

        sectors = []

        try:
            import yfinance as yf

            symbols = list(SECTOR_ETFS.keys())
            tickers = yf.Tickers(' '.join(symbols))

            for symbol, name in SECTOR_ETFS.items():
                try:
                    ticker = tickers.tickers[symbol]
                    hist = ticker.history(period='2d')

                    if len(hist) >= 2:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2]
                        change_pct = ((current - prev) / prev) * 100

                        sectors.append(SectorData(
                            symbol=symbol,
                            name=name,
                            change_pct=change_pct,
                            trend='up' if change_pct > 0 else ('down' if change_pct < 0 else 'flat'),
                        ).to_dict())
                except Exception as e:
                    logger.warning(f"Failed to fetch sector {symbol}: {e}")

        except ImportError:
            logger.warning("yfinance not installed")
        except Exception as e:
            logger.error(f"Sector performance fetch error: {e}")

        # Sort by performance (best to worst)
        sectors.sort(key=lambda x: x['change_pct'], reverse=True)

        self.cache.set(cache_key, sectors)
        return sectors

    def get_holdings_events(self, symbols: list[str], days_ahead: int = 14) -> list[dict]:
        """
        Get upcoming events for held positions.

        Args:
            symbols: List of stock symbols to check
            days_ahead: Number of days to look ahead

        Returns:
            List of event dicts sorted by date
        """
        if not symbols:
            return []

        cache_key = f"holdings_events_{'-'.join(sorted(symbols))}"

        cached = self.cache.get(cache_key, ttl=self.CACHE_TTL_EVENTS)
        if cached:
            return cached

        events = []
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        try:
            import yfinance as yf

            for symbol in symbols[:20]:  # Limit to 20 symbols
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info or {}

                    # Check for earnings date
                    earnings_dates = info.get('earningsTimestamp')
                    if earnings_dates:
                        try:
                            earnings_date = datetime.fromtimestamp(earnings_dates).date()
                            if today <= earnings_date <= cutoff:
                                days_until = (earnings_date - today).days
                                events.append(HoldingEvent(
                                    symbol=symbol,
                                    event_type='earnings',
                                    date=earnings_date,
                                    days_until=days_until,
                                    details='Quarterly Earnings',
                                ).to_dict())
                        except (ValueError, TypeError):
                            pass

                    # Check for ex-dividend date
                    ex_div_date = info.get('exDividendDate')
                    if ex_div_date:
                        try:
                            ex_date = datetime.fromtimestamp(ex_div_date).date()
                            if today <= ex_date <= cutoff:
                                days_until = (ex_date - today).days
                                div_yield = info.get('dividendYield', 0)
                                events.append(HoldingEvent(
                                    symbol=symbol,
                                    event_type='ex-dividend',
                                    date=ex_date,
                                    days_until=days_until,
                                    details=f'Yield: {div_yield * 100:.2f}%' if div_yield else '',
                                ).to_dict())
                        except (ValueError, TypeError):
                            pass

                except Exception as e:
                    logger.warning(f"Failed to fetch events for {symbol}: {e}")

        except ImportError:
            logger.warning("yfinance not installed")
        except Exception as e:
            logger.error(f"Holdings events fetch error: {e}")

        # Sort by days until event
        events.sort(key=lambda x: x['days_until'])

        self.cache.set(cache_key, events)
        return events

    def get_economic_calendar(self, days_ahead: int = 7) -> list[dict]:
        """
        Get upcoming economic events.

        Note: This provides a static list of major recurring events.
        For a real implementation, integrate with an economic calendar API.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of economic event dicts
        """
        cache_key = "economic_calendar"

        cached = self.cache.get(cache_key, ttl=self.CACHE_TTL_EVENTS)
        if cached:
            return cached

        # Known major recurring events (simplified)
        # In production, use an API like TradingEconomics or Investing.com
        events = []
        today = date.today()

        # Add some placeholder major events
        # The first Friday of each month is Jobs Report
        # FOMC meetings are 8 times per year (roughly every 6 weeks)
        # CPI is around the 12th of each month

        # Jobs Report - First Friday
        first_day = today.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        jobs_date = first_day + timedelta(days=days_until_friday)
        if jobs_date < today:
            # Move to next month
            if today.month == 12:
                first_day = today.replace(year=today.year + 1, month=1, day=1)
            else:
                first_day = today.replace(month=today.month + 1, day=1)
            days_until_friday = (4 - first_day.weekday()) % 7
            jobs_date = first_day + timedelta(days=days_until_friday)

        if jobs_date <= today + timedelta(days=days_ahead):
            events.append(EconomicEvent(
                event='Jobs Report (NFP)',
                date=jobs_date,
                time='8:30 AM ET',
                importance='high',
            ).to_dict())

        # CPI - Around 12th of month
        cpi_date = today.replace(day=12)
        if cpi_date < today:
            if today.month == 12:
                cpi_date = today.replace(year=today.year + 1, month=1, day=12)
            else:
                cpi_date = today.replace(month=today.month + 1, day=12)

        if cpi_date <= today + timedelta(days=days_ahead):
            events.append(EconomicEvent(
                event='CPI (Inflation)',
                date=cpi_date,
                time='8:30 AM ET',
                importance='high',
            ).to_dict())

        # Sort by date
        events.sort(key=lambda x: x['date'])

        self.cache.set(cache_key, events)
        return events

    def get_full_context(self, holding_symbols: list[str] | None = None) -> dict:
        """
        Get complete market context for dashboard.

        Args:
            holding_symbols: List of user's current holding symbols

        Returns:
            Complete market context dict
        """
        return {
            'overview': self.get_market_overview(),
            'sectors': self.get_sector_performance(),
            'holdings_events': self.get_holdings_events(holding_symbols or []),
            'economic_calendar': self.get_economic_calendar(),
            'last_updated': datetime.now().isoformat(),
        }


# Global singleton
_market_context_service: Optional[MarketContextService] = None


def get_market_context_service() -> MarketContextService:
    """Get or create the global MarketContextService instance."""
    global _market_context_service
    if _market_context_service is None:
        _market_context_service = MarketContextService()
    return _market_context_service
