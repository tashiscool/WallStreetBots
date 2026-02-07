"""Tests for Pre-Trade Compliance Service."""

import pytest

from backend.tradingbot.execution.interfaces import OrderRequest
from backend.tradingbot.execution.pre_trade_compliance import (
    ComplianceLimits,
    ComplianceResult,
    PreTradeComplianceService,
)


@pytest.fixture
def limits():
    return ComplianceLimits(
        max_position_pct=0.20,
        max_position_shares=10_000,
        max_order_notional=100_000,
        max_daily_notional=500_000,
        max_single_order_pct=0.10,
        max_orders_per_minute=5,
        max_orders_per_symbol_per_minute=3,
        allow_short_selling=False,
        allow_market_orders=True,
    )


@pytest.fixture
def service(limits):
    return PreTradeComplianceService(limits=limits, portfolio_value=100_000)


def _order(symbol="AAPL", qty=10, side="buy", order_type="limit", limit_price=150.0):
    return OrderRequest(
        client_order_id="test_001",
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        limit_price=limit_price,
    )


class TestBasicApproval:
    def test_simple_order_approved(self, service):
        result = service.check(_order(), current_price=150.0)
        assert result.approved is True
        assert len(result.violations) == 0

    def test_result_has_checked_by(self, service):
        result = service.check(_order(), current_price=150.0)
        assert result.checked_by == "PreTradeCompliance"


class TestRestrictedSymbols:
    def test_restricted_symbol_denied(self, service):
        service.add_restricted_symbol("AAPL")
        result = service.check(_order(symbol="AAPL"), current_price=150.0)
        assert result.approved is False
        assert any("RESTRICTED" in v for v in result.violations)

    def test_unrestricted_symbol_passes(self, service):
        service.add_restricted_symbol("TSLA")
        result = service.check(_order(symbol="AAPL"), current_price=150.0)
        assert result.approved is True

    def test_remove_restricted_symbol(self, service):
        service.add_restricted_symbol("AAPL")
        service.remove_restricted_symbol("AAPL")
        result = service.check(_order(symbol="AAPL"), current_price=150.0)
        assert result.approved is True


class TestPositionLimits:
    def test_exceeds_position_concentration(self, service):
        # 200 shares * $150 = $30K = 30% of $100K portfolio > 20% limit
        result = service.check(_order(qty=200), current_price=150.0)
        assert result.approved is False
        assert any("POSITION_LIMIT" in v for v in result.violations)

    def test_within_position_concentration(self, service):
        # 10 shares * $150 = $1.5K = 1.5% of $100K < 20% limit
        result = service.check(_order(qty=10), current_price=150.0)
        assert result.approved is True

    def test_exceeds_max_shares(self, service):
        result = service.check(_order(qty=20_000), current_price=1.0)
        assert result.approved is False
        assert any("SHARE_LIMIT" in v for v in result.violations)


class TestNotionalLimits:
    def test_exceeds_order_notional(self, service):
        # 1000 shares * $150 = $150K > $100K max
        result = service.check(_order(qty=1000), current_price=150.0)
        assert result.approved is False
        assert any("ORDER_NOTIONAL" in v for v in result.violations)

    def test_exceeds_single_order_pct(self, service):
        # 100 shares * $150 = $15K = 15% > 10% max
        result = service.check(_order(qty=100), current_price=150.0)
        assert result.approved is False
        assert any("ORDER_SIZE" in v for v in result.violations)

    def test_exceeds_daily_notional(self):
        limits = ComplianceLimits(
            max_position_pct=1.0,
            max_position_shares=1_000_000,
            max_order_notional=1_000_000,
            max_daily_notional=50_000,
            max_single_order_pct=1.0,
            max_orders_per_minute=1000,
            max_orders_per_symbol_per_minute=1000,
        )
        svc = PreTradeComplianceService(limits=limits, portfolio_value=1_000_000)
        # Submit 5 orders of $10K each = $50K = max daily
        for i in range(5):
            svc.check(_order(qty=100, symbol=f"SYM{i}"), current_price=100.0)
        # Next order should exceed daily notional
        result = svc.check(_order(qty=100, symbol="QQQ"), current_price=100.0)
        assert result.approved is False
        assert any("DAILY_NOTIONAL" in v for v in result.violations)


class TestShortSelling:
    def test_short_sell_blocked(self, service):
        result = service.check(_order(side="sell", qty=10), current_price=150.0)
        assert result.approved is False
        assert any("SHORT_SELL" in v for v in result.violations)

    def test_sell_within_position_ok(self, service):
        service.update_position("AAPL", 100)
        result = service.check(_order(side="sell", qty=50), current_price=150.0)
        # Should pass short sell check (have 100, selling 50)
        assert not any("SHORT_SELL" in v for v in result.violations)

    def test_short_sell_allowed_when_configured(self, limits):
        limits.allow_short_selling = True
        svc = PreTradeComplianceService(limits=limits, portfolio_value=100_000)
        result = svc.check(_order(side="sell", qty=10), current_price=150.0)
        assert not any("SHORT_SELL" in v for v in result.violations)


class TestRateLimits:
    def test_exceeds_orders_per_minute(self, service):
        for i in range(5):
            service.check(
                OrderRequest(
                    client_order_id=f"test_{i}",
                    symbol=f"SYM{i}",
                    qty=1,
                    side="buy",
                    type="limit",
                    limit_price=10.0,
                ),
                current_price=10.0,
            )
        # 6th should fail
        result = service.check(_order(symbol="NEWONE"), current_price=10.0)
        assert result.approved is False
        assert any("RATE_LIMIT" in v for v in result.violations)

    def test_exceeds_symbol_rate(self, service):
        for i in range(3):
            service.check(
                OrderRequest(
                    client_order_id=f"test_{i}",
                    symbol="AAPL",
                    qty=1,
                    side="buy",
                    type="limit",
                    limit_price=10.0,
                ),
                current_price=10.0,
            )
        result = service.check(_order(symbol="AAPL", qty=1), current_price=10.0)
        assert result.approved is False
        assert any("SYMBOL_RATE" in v for v in result.violations)


class TestMarketOrders:
    def test_market_orders_blocked_when_configured(self, limits):
        limits.allow_market_orders = False
        svc = PreTradeComplianceService(limits=limits, portfolio_value=100_000)
        result = svc.check(_order(order_type="market"), current_price=150.0)
        assert result.approved is False
        assert any("MARKET_ORDER" in v for v in result.violations)


class TestStateManagement:
    def test_update_portfolio_value(self, service):
        service.update_portfolio_value(1_000_000)
        # 100 shares * $150 = $15K = 1.5% of $1M — now fine
        result = service.check(_order(qty=100), current_price=150.0)
        assert result.approved is True

    def test_audit_log(self, service):
        service.check(_order(), current_price=150.0)
        log = service.get_audit_log()
        assert len(log) == 1
        assert log[0].approved is True

    def test_daily_stats(self, service):
        service.check(_order(), current_price=150.0)
        stats = service.get_daily_stats()
        assert stats["total_checks"] == 1
        assert stats["approved"] == 1
        assert stats["denied"] == 0
