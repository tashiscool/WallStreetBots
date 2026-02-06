"""Tests for L2 Order Book module."""

import pytest
from datetime import datetime

from backend.tradingbot.execution.order_book import (
    OrderBook,
    OrderBookFeatures,
    OrderBookLevel,
    OrderBookManager,
    OrderBookSnapshot,
)


class TestOrderBookLevel:
    def test_creation(self):
        level = OrderBookLevel(price=150.0, size=100.0, num_orders=3)
        assert level.price == 150.0
        assert level.size == 100.0
        assert level.num_orders == 3

    def test_frozen(self):
        level = OrderBookLevel(price=150.0, size=100.0)
        with pytest.raises(AttributeError):
            level.price = 151.0


class TestOrderBookSnapshot:
    def _make_snapshot(self):
        bids = [OrderBookLevel(150.0, 100, 3), OrderBookLevel(149.99, 200, 5)]
        asks = [OrderBookLevel(150.02, 80, 2), OrderBookLevel(150.03, 150, 4)]
        return OrderBookSnapshot(
            symbol="AAPL", timestamp=datetime.now(), bids=bids, asks=asks
        )

    def test_best_bid_ask(self):
        snap = self._make_snapshot()
        assert snap.best_bid == 150.0
        assert snap.best_ask == 150.02

    def test_mid_price(self):
        snap = self._make_snapshot()
        assert snap.mid_price == pytest.approx(150.01, abs=0.001)

    def test_spread(self):
        snap = self._make_snapshot()
        assert snap.spread == pytest.approx(0.02, abs=0.001)

    def test_spread_bps(self):
        snap = self._make_snapshot()
        expected_bps = (0.02 / 150.01) * 10000
        assert snap.spread_bps == pytest.approx(expected_bps, rel=0.01)

    def test_empty_snapshot(self):
        snap = OrderBookSnapshot(symbol="X", timestamp=datetime.now(), bids=[], asks=[])
        assert snap.best_bid is None
        assert snap.best_ask is None
        assert snap.mid_price is None
        assert snap.spread is None


class TestOrderBook:
    def test_update_from_quote_creates_levels(self):
        book = OrderBook("AAPL", max_depth=5)
        book.update_from_quote(bid=150.0, ask=150.02, bid_size=100, ask_size=80)

        assert not book.is_empty
        assert book.is_simulated
        snap = book.get_snapshot()
        assert snap is not None
        assert len(snap.bids) == 5
        assert len(snap.asks) == 5
        assert snap.best_bid == 150.0
        assert snap.best_ask == 150.02

    def test_update_book_with_real_data(self):
        book = OrderBook("AAPL", max_depth=3)
        bids = [(150.0, 100, 3), (149.99, 200, 5), (149.98, 300, 8)]
        asks = [(150.02, 80, 2), (150.03, 150, 4), (150.04, 250, 6)]
        book.update_book(bids=bids, asks=asks)

        assert not book.is_simulated
        snap = book.get_snapshot()
        assert len(snap.bids) == 3
        assert len(snap.asks) == 3
        assert snap.best_bid == 150.0
        assert snap.best_ask == 150.02

    def test_get_features(self):
        book = OrderBook("AAPL", max_depth=5)
        book.update_from_quote(bid=150.0, ask=150.02, bid_size=200, ask_size=100)

        features = book.get_features()
        assert features is not None
        assert features.mid_price == pytest.approx(150.01, abs=0.01)
        assert features.spread_bps > 0
        # More bids than asks at top -> positive imbalance
        assert features.imbalance > 0
        assert features.depth_bid_5 > 0
        assert features.depth_ask_5 > 0
        assert 0.0 <= features.depth_ratio <= 1.0
        assert 0.0 <= features.resilience_score <= 1.0

    def test_microprice(self):
        book = OrderBook("AAPL", max_depth=5)
        # Large ask size pulls microprice toward bid (opposite side weight)
        book.update_from_quote(bid=100.0, ask=100.10, bid_size=100, ask_size=900)
        features = book.get_features()
        # Microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
        expected = (100.0 * 900 + 100.10 * 100) / 1000
        assert features.microprice == pytest.approx(expected, abs=0.01)

    def test_estimate_market_impact_buy(self):
        book = OrderBook("AAPL", max_depth=5)
        bids = [(150.0, 100, 3), (149.99, 200, 5)]
        asks = [(150.02, 50, 2), (150.03, 100, 4), (150.05, 200, 6)]
        book.update_book(bids=bids, asks=asks)

        # Buy 50 shares: fits in first level
        impact = book.estimate_market_impact('buy', 50)
        assert impact['levels_consumed'] == 1
        assert impact['avg_fill_price'] == 150.02
        assert impact['remaining_quantity'] == 0

    def test_estimate_market_impact_large_order(self):
        book = OrderBook("AAPL", max_depth=3)
        asks = [(150.02, 50, 2), (150.03, 100, 4), (150.05, 200, 6)]
        bids = [(150.0, 100, 3)]
        book.update_book(bids=bids, asks=asks)

        # Buy 200 shares: consumes first two levels + part of third
        impact = book.estimate_market_impact('buy', 200)
        assert impact['levels_consumed'] == 3
        assert impact['impact_bps'] > 0
        assert impact['filled_quantity'] == 200

    def test_get_liquidity_at_bps(self):
        book = OrderBook("AAPL", max_depth=5)
        bids = [(150.0, 100, 3), (149.95, 200, 5)]
        asks = [(150.02, 80, 2), (150.07, 150, 4)]
        book.update_book(bids=bids, asks=asks)

        liq = book.get_liquidity_at_bps(50)  # 50 bps from mid
        assert liq['bid_liquidity'] >= 100
        assert liq['ask_liquidity'] >= 80
        assert liq['total_liquidity'] > 0

    def test_empty_book(self):
        book = OrderBook("X")
        assert book.is_empty
        assert book.get_snapshot() is None
        assert book.get_features() is None

    def test_to_dict(self):
        book = OrderBook("AAPL")
        book.update_from_quote(bid=150.0, ask=150.02)
        d = book.to_dict()
        assert d['symbol'] == 'AAPL'
        assert d['is_simulated'] is True
        assert d['best_bid'] == 150.0
        assert d['best_ask'] == 150.02


class TestOrderBookManager:
    def test_get_or_create(self):
        mgr = OrderBookManager()
        book = mgr.get_or_create("AAPL")
        assert book.symbol == "AAPL"
        assert mgr.get_or_create("AAPL") is book

    def test_update_from_quote(self):
        mgr = OrderBookManager()
        mgr.update_from_quote("AAPL", bid=150.0, ask=150.02)
        features = mgr.get_features("AAPL")
        assert features is not None
        assert features.mid_price == pytest.approx(150.01, abs=0.01)

    def test_get_all_features(self):
        mgr = OrderBookManager()
        mgr.update_from_quote("AAPL", bid=150.0, ask=150.02)
        mgr.update_from_quote("GOOG", bid=140.0, ask=140.05)
        all_features = mgr.get_all_features()
        assert "AAPL" in all_features
        assert "GOOG" in all_features

    def test_estimate_impact(self):
        mgr = OrderBookManager()
        mgr.update_book(
            "AAPL",
            bids=[(150.0, 100, 3)],
            asks=[(150.02, 50, 2), (150.03, 100, 4)],
        )
        impact = mgr.estimate_impact("AAPL", "buy", 50)
        assert impact['levels_consumed'] == 1

    def test_unknown_symbol(self):
        mgr = OrderBookManager()
        assert mgr.get_features("UNKNOWN") is None

    def test_remove(self):
        mgr = OrderBookManager()
        mgr.update_from_quote("AAPL", bid=150.0, ask=150.02)
        assert "AAPL" in mgr.symbols
        mgr.remove("AAPL")
        assert "AAPL" not in mgr.symbols
