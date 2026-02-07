"""Tests for FREDDataSource."""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.data.sources.fred import (
    FREDDataSource,
    FREDObservation,
    FRED_SERIES,
)


def _mock_observations(values):
    """Build a mock FRED API response."""
    obs = []
    for i, val in enumerate(values):
        obs.append({
            "date": f"2025-01-{i + 1:02d}",
            "value": str(val) if val != "." else ".",
        })
    return {"observations": obs}


@pytest.fixture
def source():
    src = FREDDataSource(api_key="test-key")
    src._session = MagicMock()
    return src


class TestGetSeries:
    def test_parses_observations(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_observations([4.5, 4.3, 4.1])
        resp.raise_for_status = MagicMock()
        source._session.get.return_value = resp

        obs = source.get_series("DGS10")
        assert len(obs) == 3
        assert obs[0].series_id == "DGS10"
        assert obs[0].value == 4.5
        assert isinstance(obs[0].date, datetime)

    def test_skips_missing_values(self, source):
        resp = MagicMock()
        resp.json.return_value = _mock_observations([4.5, ".", 4.1])
        resp.raise_for_status = MagicMock()
        source._session.get.return_value = resp

        obs = source.get_series("DGS10")
        assert len(obs) == 2  # Skipped the "."

    def test_empty_response(self, source):
        resp = MagicMock()
        resp.json.return_value = {"observations": []}
        resp.raise_for_status = MagicMock()
        source._session.get.return_value = resp

        obs = source.get_series("DGS10")
        assert obs == []

    def test_api_error_returns_empty(self, source):
        source._session.get.side_effect = Exception("timeout")
        obs = source.get_series("DGS10")
        assert obs == []

    def test_no_session_returns_empty(self):
        src = FREDDataSource(api_key="")
        src._session = None
        assert src.get_series("DGS10") == []


class TestYieldCurve:
    def test_normal_yield_curve(self, source):
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:  # DGS2
                resp.json.return_value = _mock_observations([3.5])
            else:  # DGS10
                resp.json.return_value = _mock_observations([4.5])
            return resp

        source._session.get.side_effect = mock_get
        curve = source.get_yield_curve()
        assert curve["inverted"] is False
        assert curve["spread"] == 1.0

    def test_inverted_yield_curve(self, source):
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:  # DGS2
                resp.json.return_value = _mock_observations([5.0])
            else:  # DGS10
                resp.json.return_value = _mock_observations([4.0])
            return resp

        source._session.get.side_effect = mock_get
        curve = source.get_yield_curve()
        assert curve["inverted"] is True
        assert curve["spread"] == -1.0

    def test_is_yield_curve_inverted(self, source):
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:
                resp.json.return_value = _mock_observations([5.0])
            else:
                resp.json.return_value = _mock_observations([4.0])
            return resp

        source._session.get.side_effect = mock_get
        assert source.is_yield_curve_inverted() is True


class TestMacroSnapshot:
    def test_returns_all_indicators(self, source):
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            values = {
                1: [3.6],   # unemployment
                2: [5.25],  # fed_funds
                3: [18.5],  # vix
                4: [308.0], # cpi
                5: [3.5],   # DGS2 for yield spread
                6: [4.5],   # DGS10 for yield spread
            }
            resp.json.return_value = _mock_observations(values.get(call_count, [0.0]))
            return resp

        source._session.get.side_effect = mock_get
        snapshot = source.get_macro_snapshot()
        assert "unemployment" in snapshot
        assert "fed_funds" in snapshot
        assert "vix" in snapshot
        assert "cpi" in snapshot
        assert "yield_spread" in snapshot


class TestFREDObservation:
    def test_defaults(self):
        obs = FREDObservation(
            series_id="DGS10",
            date=datetime(2025, 1, 1),
            value=4.5,
        )
        assert obs.metadata == {}


class TestFREDSeries:
    def test_known_series_ids(self):
        assert "DGS10" in FRED_SERIES
        assert "DGS2" in FRED_SERIES
        assert "UNRATE" in FRED_SERIES
        assert "FEDFUNDS" in FRED_SERIES
        assert "VIXCLS" in FRED_SERIES
