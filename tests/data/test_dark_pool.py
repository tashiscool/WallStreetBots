"""Tests for DarkPoolDataSource — FINRA ATS data."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.data.sources.dark_pool import (
    DarkPoolData,
    DarkPoolDataSource,
)


def _mock_ats_response(weeks, base_volume=1000000):
    """Build a mock FINRA ATS response."""
    items = []
    base_date = datetime.utcnow()
    for i in range(weeks):
        d = base_date - timedelta(weeks=i)
        # Increase volume for first (most recent) week to test accumulation
        vol = base_volume * (2 if i == 0 else 1)
        items.append({
            "weekStartDate": d.strftime("%Y-%m-%d"),
            "totalWeeklyShareQuantity": vol,
            "totalWeeklyTradeCount": vol // 100,
            "issueSymbolIdentifier": "AAPL",
        })
    return items


@pytest.fixture
def source():
    src = DarkPoolDataSource()
    src._session = MagicMock()
    return src


class TestGetAtsData:
    def test_parses_response(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_ats_response(4)
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        data = source.get_ats_data("AAPL", weeks_back=4)
        assert len(data) == 4
        assert all(isinstance(d, DarkPoolData) for d in data)
        assert data[0].ticker == "AAPL"
        assert data[0].total_shares > 0

    def test_empty_response(self, source):
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        data = source.get_ats_data("XYZ")
        assert data == []

    def test_api_error(self, source):
        source._session.post.side_effect = Exception("timeout")
        data = source.get_ats_data("AAPL")
        assert data == []

    def test_no_session(self):
        src = DarkPoolDataSource()
        src._session = None
        assert src.get_ats_data("AAPL") == []


class TestDarkPoolRatio:
    def test_calculates_ratio(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_ats_response(1, base_volume=1000000)
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        ratio = source.get_dark_pool_ratio("AAPL", exchange_volume=3000000)
        # DP volume = 2000000 (doubled for i==0), total = 5000000
        assert ratio is not None
        assert 0.0 < ratio < 1.0

    def test_zero_exchange_volume(self, source):
        ratio = source.get_dark_pool_ratio("AAPL", exchange_volume=0)
        assert ratio is None

    def test_no_data_returns_none(self, source):
        resp = MagicMock()
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        ratio = source.get_dark_pool_ratio("XYZ", exchange_volume=1000000)
        assert ratio is None


class TestAccumulationDetection:
    def test_detects_accumulation(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_ats_response(8, base_volume=1000000)
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        result = source.detect_accumulation("AAPL", weeks_back=8, threshold_pct=20.0)
        assert result["detected"] is True
        assert result["change_pct"] > 20.0
        assert result["current_volume"] > result["avg_volume"]

    def test_no_accumulation_flat(self, source):
        # All weeks same volume
        items = []
        base_date = datetime.utcnow()
        for i in range(8):
            items.append({
                "weekStartDate": (base_date - timedelta(weeks=i)).strftime("%Y-%m-%d"),
                "totalWeeklyShareQuantity": 1000000,
                "totalWeeklyTradeCount": 10000,
            })

        resp = MagicMock()
        resp.json.return_value = items
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        result = source.detect_accumulation("AAPL", threshold_pct=20.0)
        assert result["detected"] is False
        assert result["change_pct"] == 0.0

    def test_insufficient_data(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_ats_response(1)
        resp.raise_for_status = MagicMock()
        source._session.post.return_value = resp

        result = source.detect_accumulation("AAPL")
        assert result["detected"] is False
        assert result["current_volume"] == 0


class TestDarkPoolData:
    def test_dataclass(self):
        d = DarkPoolData(
            ticker="AAPL",
            week_ending=datetime(2025, 6, 1),
            total_shares=5000000,
            total_trades=50000,
            ats_name="UBSS",
        )
        assert d.ticker == "AAPL"
        assert d.total_shares == 5000000
        assert d.ats_name == "UBSS"
