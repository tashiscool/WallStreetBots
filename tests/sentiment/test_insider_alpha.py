"""Tests for InsiderAlphaModel."""

import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.tradingbot.framework.alpha_models.insider_alpha import (
    InsiderAlphaModel,
    TITLE_WEIGHTS,
)
from backend.tradingbot.framework.insight import InsightDirection


@dataclass
class MockTransaction:
    """Mimics InsiderTransaction."""
    ticker: str = "AAPL"
    insider_name: str = "John Doe"
    title: str = "CEO"
    transaction_type: str = "buy"
    shares: int = 1000
    price: float = 150.0
    date: datetime = None

    def __post_init__(self):
        if self.date is None:
            self.date = datetime.utcnow()


@pytest.fixture
def model():
    return InsiderAlphaModel(
        cluster_buy_threshold=3,
        large_sale_shares=50000,
        min_confidence=0.3,
    )


class TestClusterBuy:
    def test_cluster_buy_generates_up(self, model):
        """3+ distinct insiders buying → strong bullish."""
        txns = [
            MockTransaction(insider_name="Alice", title="CEO", transaction_type="buy"),
            MockTransaction(insider_name="Bob", title="CFO", transaction_type="buy"),
            MockTransaction(insider_name="Carol", title="Director", transaction_type="buy"),
        ]
        data = {"insider": {"AAPL": txns}}
        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.UP
        assert insights[0].confidence >= 0.7

    def test_two_buyers_not_cluster(self, model):
        """Only 2 buyers → not a cluster, still may produce signal from weight."""
        txns = [
            MockTransaction(insider_name="Alice", title="CEO", transaction_type="buy"),
            MockTransaction(insider_name="Bob", title="CFO", transaction_type="buy"),
        ]
        data = {"insider": {"AAPL": txns}}
        insights = model.generate_insights(data, ["AAPL"])
        # Should still generate UP from buy pressure, just lower confidence
        if insights:
            assert insights[0].direction == InsightDirection.UP
            assert insights[0].confidence < 0.8  # Not cluster-level confidence


class TestLargeSale:
    def test_large_ceo_sale_bearish(self, model):
        """Large CEO sale → bearish."""
        txns = [
            MockTransaction(
                insider_name="CEO", title="CEO",
                transaction_type="sell", shares=100000
            ),
        ]
        data = {"insider": {"AAPL": txns}}
        insights = model.generate_insights(data, ["AAPL"])
        assert len(insights) == 1
        assert insights[0].direction == InsightDirection.DOWN

    def test_small_sale_weaker_signal(self, model):
        """Small director sale → weaker signal."""
        txns = [
            MockTransaction(
                insider_name="Dir", title="Director",
                transaction_type="sell", shares=1000
            ),
        ]
        data = {"insider": {"AAPL": txns}}
        insights = model.generate_insights(data, ["AAPL"])
        if insights:
            assert insights[0].direction == InsightDirection.DOWN
            assert insights[0].confidence < 0.7


class TestTitleWeighting:
    def test_ceo_weight(self):
        assert TITLE_WEIGHTS["ceo"] == 1.0

    def test_cfo_weight(self):
        assert TITLE_WEIGHTS["cfo"] == 0.9

    def test_director_weight(self):
        assert TITLE_WEIGHTS["director"] == 0.5

    def test_weight_lookup(self, model):
        txn = MockTransaction(title="CEO")
        weight = model._get_title_weight(txn)
        assert weight == 1.0

    def test_unknown_title_default(self, model):
        txn = MockTransaction(title="Janitor")
        weight = model._get_title_weight(txn)
        assert weight == 0.3  # Default


class TestEdgeCases:
    def test_no_insider_data(self, model):
        insights = model.generate_insights({}, ["AAPL"])
        assert insights == []

    def test_empty_transactions(self, model):
        data = {"insider": {"AAPL": []}}
        insights = model.generate_insights(data, ["AAPL"])
        assert insights == []

    def test_symbol_not_in_data(self, model):
        data = {"insider": {"AAPL": [MockTransaction()]}}
        insights = model.generate_insights(data, ["MSFT"])
        assert insights == []

    def test_multiple_symbols(self, model):
        data = {
            "insider": {
                "AAPL": [
                    MockTransaction(insider_name="A", title="CEO", transaction_type="buy"),
                    MockTransaction(insider_name="B", title="CFO", transaction_type="buy"),
                    MockTransaction(insider_name="C", title="COO", transaction_type="buy"),
                ],
                "TSLA": [
                    MockTransaction(ticker="TSLA", insider_name="X", title="CEO",
                                    transaction_type="sell", shares=100000),
                ],
            }
        }
        insights = model.generate_insights(data, ["AAPL", "TSLA"])
        directions = {i.symbol: i.direction for i in insights}
        assert directions.get("AAPL") == InsightDirection.UP
        assert directions.get("TSLA") == InsightDirection.DOWN
