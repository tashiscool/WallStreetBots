"""Tests for InsiderTransactionSource — SEC EDGAR Form 4 parsing."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.sentiment.sources.insider_source import (
    InsiderTransaction,
    InsiderTransactionSource,
)


@pytest.fixture
def source():
    return InsiderTransactionSource()


def _mock_submissions(forms, dates, descriptions=None):
    """Build a mock EDGAR submissions JSON response."""
    n = len(forms)
    return {
        "name": "John Doe",
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": dates,
                "accessionNumber": [f"0001-{i:04d}" for i in range(n)],
                "primaryDocDescription": descriptions or [""] * n,
            }
        },
    }


def _mock_cik_response():
    return {"0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple Inc"}}


class TestInsiderTransaction:
    def test_value_property(self):
        txn = InsiderTransaction(
            ticker="AAPL",
            insider_name="Tim Cook",
            title="CEO",
            transaction_type="sell",
            shares=10000,
            price=150.0,
            date=datetime.utcnow(),
        )
        assert txn.value == 1500000.0


class TestParseTransactions:
    def test_parse_form4_filings(self, source):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        data = _mock_submissions(
            forms=["4", "10-Q", "4"],
            dates=[today, today, today],
            descriptions=["purchase", "quarterly report", "sale"],
        )
        txns = source._parse_transactions(data, "AAPL", days_back=30, limit=50)
        assert len(txns) == 2
        assert txns[0].transaction_type == "buy"
        assert txns[1].transaction_type == "sell"

    def test_filters_old_filings(self, source):
        old = (datetime.utcnow() - timedelta(days=200)).strftime("%Y-%m-%d")
        data = _mock_submissions(forms=["4"], dates=[old])
        txns = source._parse_transactions(data, "AAPL", days_back=30, limit=50)
        assert len(txns) == 0

    def test_respects_limit(self, source):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        data = _mock_submissions(
            forms=["4"] * 10,
            dates=[today] * 10,
        )
        txns = source._parse_transactions(data, "AAPL", days_back=30, limit=3)
        assert len(txns) == 3

    def test_empty_filings(self, source):
        data = {"name": "X", "filings": {"recent": {"form": [], "filingDate": [], "accessionNumber": [], "primaryDocDescription": []}}}
        txns = source._parse_transactions(data, "AAPL", days_back=30, limit=50)
        assert txns == []


class TestGetInsiderTransactions:
    @patch("backend.tradingbot.sentiment.sources.insider_source.requests")
    def test_calls_edgar_api(self, mock_requests, source):
        mock_session = MagicMock()
        source._session = mock_session

        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Mock CIK resolution
        cik_resp = MagicMock()
        cik_resp.json.return_value = _mock_cik_response()
        cik_resp.raise_for_status = MagicMock()

        # Mock submissions
        sub_resp = MagicMock()
        sub_resp.json.return_value = _mock_submissions(
            forms=["4"], dates=[today], descriptions=["purchase"]
        )
        sub_resp.raise_for_status = MagicMock()

        mock_session.get.side_effect = [cik_resp, sub_resp]

        txns = source.get_insider_transactions("AAPL", days_back=30)
        assert len(txns) == 1
        assert txns[0].ticker == "AAPL"

    def test_no_session_returns_empty(self):
        source = InsiderTransactionSource()
        source._session = None
        assert source.get_insider_transactions("AAPL") == []


class TestClusterBuys:
    @patch.object(InsiderTransactionSource, "get_insider_transactions")
    def test_detects_cluster(self, mock_get):
        mock_get.return_value = [
            InsiderTransaction("AAPL", "Alice", "CEO", "buy", 1000, 150.0, datetime.utcnow()),
            InsiderTransaction("AAPL", "Bob", "CFO", "buy", 500, 150.0, datetime.utcnow()),
            InsiderTransaction("AAPL", "Carol", "Director", "buy", 200, 150.0, datetime.utcnow()),
        ]
        source = InsiderTransactionSource()
        buys = source.get_cluster_buys("AAPL", min_insiders=3)
        assert len(buys) == 3

    @patch.object(InsiderTransactionSource, "get_insider_transactions")
    def test_no_cluster_below_threshold(self, mock_get):
        mock_get.return_value = [
            InsiderTransaction("AAPL", "Alice", "CEO", "buy", 1000, 150.0, datetime.utcnow()),
        ]
        source = InsiderTransactionSource()
        buys = source.get_cluster_buys("AAPL", min_insiders=3)
        assert buys == []

    @patch.object(InsiderTransactionSource, "get_insider_transactions")
    def test_ignores_sells(self, mock_get):
        mock_get.return_value = [
            InsiderTransaction("AAPL", "Alice", "CEO", "sell", 10000, 150.0, datetime.utcnow()),
            InsiderTransaction("AAPL", "Bob", "CFO", "sell", 5000, 150.0, datetime.utcnow()),
            InsiderTransaction("AAPL", "Carol", "Director", "sell", 2000, 150.0, datetime.utcnow()),
        ]
        source = InsiderTransactionSource()
        buys = source.get_cluster_buys("AAPL", min_insiders=3)
        assert buys == []
