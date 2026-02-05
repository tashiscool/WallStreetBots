"""Comprehensive tests for MarketContextService."""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from backend.auth0login.services.market_context import (
    MarketContextService,
    MarketStatus,
    VIXLevel,
    MarketContextCache,
    get_market_context_service
)


@pytest.fixture
def service():
    return MarketContextService()


@pytest.fixture
def cache():
    return MarketContextCache()


class TestMarketContextCache:
    """Test cache functionality."""

    def test_cache_get_miss(self, cache):
        result = cache.get('missing_key')
        assert result is None

    def test_cache_set_get(self, cache):
        cache.set('test_key', {'data': 'value'})
        result = cache.get('test_key', ttl=60)
        assert result == {'data': 'value'}

    def test_cache_expiry(self, cache):
        cache.set('test_key', 'value')
        result = cache.get('test_key', ttl=0)
        assert result is None

    def test_cache_clear(self, cache):
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        assert cache.get('key1') is None


class TestMarketStatus:
    """Test market status determination."""

    def test_get_market_status_open(self, service):
        with patch.object(service, '_get_et_now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 15, 14, 0, 0, tzinfo=service._et_tz)
            result = service.get_market_status()
            assert result['status'] == MarketStatus.OPEN.value
            assert result['is_open'] is True

    def test_get_market_status_pre_market(self, service):
        with patch.object(service, '_get_et_now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 15, 8, 0, 0, tzinfo=service._et_tz)
            result = service.get_market_status()
            assert result['status'] == MarketStatus.PRE_MARKET.value

    def test_get_market_status_weekend(self, service):
        with patch.object(service, '_get_et_now') as mock_now:
            mock_now.return_value = datetime(2024, 1, 13, 14, 0, 0, tzinfo=service._et_tz)
            result = service.get_market_status()
            assert result['status'] == MarketStatus.CLOSED.value


class TestMarketOverview:
    """Test market overview."""

    @patch.dict('sys.modules', {'yfinance': Mock()})
    def test_get_market_overview(self, service):
        import sys
        mock_yf = sys.modules['yfinance']
        mock_ticker = Mock()
        mock_ticker.history.return_value = Mock(Close=Mock(iloc=[-150.0, -155.0]))
        mock_ticker.history.return_value.__len__ = Mock(return_value=2)

        mock_tickers = Mock()
        mock_tickers.tickers = {'SPY': mock_ticker}
        mock_yf.Tickers.return_value = mock_tickers

        result = service.get_market_overview(force_refresh=True)
        assert 'indices' in result


class TestHoldingsEvents:
    """Test holdings events."""

    def test_get_holdings_events_empty(self, service):
        result = service.get_holdings_events([])
        assert result == []


class TestFullContext:
    """Test full context."""

    @patch.object(MarketContextService, 'get_market_overview')
    @patch.object(MarketContextService, 'get_sector_performance')
    @patch.object(MarketContextService, 'get_holdings_events')
    @patch.object(MarketContextService, 'get_economic_calendar')
    def test_get_full_context(self, mock_cal, mock_events, mock_sectors, mock_overview, service):
        mock_overview.return_value = {'indices': {}}
        mock_sectors.return_value = []
        mock_events.return_value = []
        mock_cal.return_value = []

        result = service.get_full_context()
        assert 'overview' in result
