"""Tests for Order Management System (OMS)."""

import pytest

from backend.tradingbot.execution.interfaces import (
    ExecutionClient,
    OrderAck,
    OrderFill,
    OrderRequest,
)
from backend.tradingbot.execution.order_state_machine import OrderStatus
from backend.tradingbot.execution.pre_trade_compliance import (
    ComplianceLimits,
    PreTradeComplianceService,
)
from backend.tradingbot.execution.maker_checker import (
    ApprovalStatus,
    ApprovalThresholds,
    MakerCheckerService,
)
from backend.tradingbot.execution.oms import ManagedOrder, OrderManagementSystem


class RejectingClient(ExecutionClient):
    """Client that rejects all orders."""

    def place_order(self, req: OrderRequest) -> OrderAck:
        return OrderAck(
            client_order_id=req.client_order_id,
            broker_order_id=None,
            accepted=False,
            reason="Broker offline",
        )


class ErrorClient(ExecutionClient):
    """Client that raises exceptions."""

    def place_order(self, req: OrderRequest) -> OrderAck:
        raise ConnectionError("Network failure")


@pytest.fixture
def compliance():
    limits = ComplianceLimits(
        max_position_pct=0.50,
        max_position_shares=100_000,
        max_order_notional=1_000_000,
        max_daily_notional=5_000_000,
        max_single_order_pct=0.50,
        max_orders_per_minute=100,
        max_orders_per_symbol_per_minute=50,
    )
    return PreTradeComplianceService(limits=limits, portfolio_value=1_000_000)


@pytest.fixture
def oms(compliance):
    return OrderManagementSystem(
        execution_client=ExecutionClient(),
        compliance=compliance,
    )


class TestBasicOrderFlow:
    def test_submit_order_approved_and_routed(self, oms):
        managed = oms.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        assert managed.status == OrderStatus.SUBMITTED
        assert managed.broker_ack is not None
        assert managed.broker_ack.accepted is True
        assert managed.compliance_result.approved is True

    def test_order_gets_unique_id(self, oms):
        m1 = oms.submit_order(symbol="AAPL", qty=10, side="buy", limit_price=150.0, current_price=150.0)
        m2 = oms.submit_order(symbol="GOOGL", qty=5, side="buy", limit_price=100.0, current_price=100.0)
        assert m1.order_id != m2.order_id

    def test_order_id_starts_with_oms(self, oms):
        managed = oms.submit_order(symbol="AAPL", qty=10, side="buy", limit_price=150.0, current_price=150.0)
        assert managed.order_id.startswith("OMS_")


class TestComplianceDenial:
    def test_restricted_symbol_denied(self, oms, compliance):
        compliance.add_restricted_symbol("TSLA")
        managed = oms.submit_order(
            symbol="TSLA", qty=10, side="buy",
            limit_price=200.0, current_price=200.0,
        )
        assert managed.status == OrderStatus.DENIED
        assert "RESTRICTED" in managed.error
        assert managed.broker_ack is None  # Never reached broker

    def test_reject_callback_fires(self, compliance):
        rejections = []
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
        )
        oms_obj.set_on_reject(lambda m: rejections.append(m))
        compliance.add_restricted_symbol("TSLA")
        oms_obj.submit_order(symbol="TSLA", qty=10, side="buy", limit_price=200.0, current_price=200.0)
        assert len(rejections) == 1


class TestMakerCheckerIntegration:
    def test_large_order_held_for_approval(self, compliance):
        thresholds = ApprovalThresholds(
            auto_approve_notional=1_000,
            auto_approve_shares=10,
            require_approval_for_new_symbols=False,
        )
        mc = MakerCheckerService(thresholds=thresholds)
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
            maker_checker=mc,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=100, side="buy",
            limit_price=150.0, current_price=150.0,
            submitted_by="trader_a",
        )
        assert managed.status == OrderStatus.INITIALIZED
        assert managed.approval.status == ApprovalStatus.PENDING
        assert managed.broker_ack is None

    def test_small_order_auto_approved(self, compliance):
        thresholds = ApprovalThresholds(
            auto_approve_notional=100_000,
            auto_approve_shares=1_000,
            require_approval_for_new_symbols=False,
        )
        mc = MakerCheckerService(thresholds=thresholds)
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
            maker_checker=mc,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        assert managed.status == OrderStatus.SUBMITTED
        assert managed.approval.status == ApprovalStatus.AUTO_APPROVED

    def test_on_approval_routes_to_broker(self, compliance):
        thresholds = ApprovalThresholds(
            auto_approve_notional=1_000,
            auto_approve_shares=10,
            require_approval_for_new_symbols=False,
        )
        mc = MakerCheckerService(thresholds=thresholds)
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
            maker_checker=mc,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=100, side="buy",
            limit_price=150.0, current_price=150.0,
            submitted_by="trader_a",
        )
        # Approve via maker-checker
        mc.approve(managed.approval.request_id, checker_id="manager_b")
        # OMS should route it now
        result = oms_obj.on_approval(managed.approval.request_id)
        assert result is not None
        assert result.status == OrderStatus.SUBMITTED


class TestBrokerFailures:
    def test_broker_rejects_order(self, compliance):
        oms_obj = OrderManagementSystem(
            execution_client=RejectingClient(),
            compliance=compliance,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        assert managed.status == OrderStatus.REJECTED
        assert "Broker" in managed.error

    def test_broker_error(self, compliance):
        oms_obj = OrderManagementSystem(
            execution_client=ErrorClient(),
            compliance=compliance,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        assert managed.status == OrderStatus.ERROR
        assert "Network failure" in managed.error


class TestOrderCancellation:
    def test_cancel_submitted_order(self, oms):
        managed = oms.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        success = oms.cancel_order(managed.order_id)
        assert success is True
        assert oms.get_order(managed.order_id).status == OrderStatus.CANCELED

    def test_cancel_unknown_order(self, oms):
        assert oms.cancel_order("nonexistent") is False

    def test_cancel_pending_order(self, compliance):
        thresholds = ApprovalThresholds(
            auto_approve_notional=1_000,
            auto_approve_shares=10,
            require_approval_for_new_symbols=False,
        )
        mc = MakerCheckerService(thresholds=thresholds)
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
            maker_checker=mc,
        )
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=100, side="buy",
            limit_price=150.0, current_price=150.0,
            submitted_by="trader_a",
        )
        # Cancel before broker submission
        success = oms_obj.cancel_order(managed.order_id)
        assert success is True


class TestFillProcessing:
    def test_process_fill(self, oms):
        managed = oms.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        fill = OrderFill(
            broker_order_id=managed.broker_ack.broker_order_id,
            symbol="AAPL",
            avg_price=149.50,
            filled_qty=10,
            status="filled",
        )
        result = oms.process_fill(managed.broker_ack.broker_order_id, fill)
        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.fill.avg_price == 149.50

    def test_partial_fill(self, oms):
        managed = oms.submit_order(
            symbol="AAPL", qty=100, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        fill = OrderFill(
            broker_order_id=managed.broker_ack.broker_order_id,
            symbol="AAPL",
            avg_price=149.50,
            filled_qty=50,
            status="partially_filled",
        )
        result = oms.process_fill(managed.broker_ack.broker_order_id, fill)
        assert result.status == OrderStatus.PARTIALLY_FILLED

    def test_fill_callback(self, compliance):
        fills = []
        oms_obj = OrderManagementSystem(
            execution_client=ExecutionClient(),
            compliance=compliance,
        )
        oms_obj.set_on_fill(lambda m: fills.append(m))
        managed = oms_obj.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        fill = OrderFill(
            broker_order_id=managed.broker_ack.broker_order_id,
            symbol="AAPL",
            avg_price=149.50,
            filled_qty=10,
            status="filled",
        )
        oms_obj.process_fill(managed.broker_ack.broker_order_id, fill)
        assert len(fills) == 1

    def test_fill_updates_position(self, oms, compliance):
        managed = oms.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        fill = OrderFill(
            broker_order_id=managed.broker_ack.broker_order_id,
            symbol="AAPL",
            avg_price=149.50,
            filled_qty=10,
            status="filled",
        )
        oms.process_fill(managed.broker_ack.broker_order_id, fill)
        assert compliance._positions.get("AAPL", 0) == 10

    def test_unknown_fill_returns_none(self, oms):
        fill = OrderFill(
            broker_order_id="unknown_broker_id",
            symbol="AAPL",
            avg_price=149.50,
            filled_qty=10,
            status="filled",
        )
        result = oms.process_fill("unknown_broker_id", fill)
        assert result is None


class TestQueries:
    def test_get_order(self, oms):
        managed = oms.submit_order(
            symbol="AAPL", qty=10, side="buy",
            limit_price=150.0, current_price=150.0,
        )
        found = oms.get_order(managed.order_id)
        assert found is not None
        assert found.order.symbol == "AAPL"

    def test_get_open_orders(self, oms):
        oms.submit_order(symbol="AAPL", qty=10, side="buy", limit_price=150.0, current_price=150.0)
        oms.submit_order(symbol="GOOGL", qty=5, side="buy", limit_price=100.0, current_price=100.0)
        open_orders = oms.get_open_orders()
        assert len(open_orders) == 2

    def test_get_all_orders(self, oms, compliance):
        oms.submit_order(symbol="AAPL", qty=10, side="buy", limit_price=150.0, current_price=150.0)
        compliance.add_restricted_symbol("TSLA")
        oms.submit_order(symbol="TSLA", qty=10, side="buy", limit_price=200.0, current_price=200.0)
        all_orders = oms.get_all_orders()
        assert len(all_orders) == 2

    def test_get_order_count(self, oms, compliance):
        oms.submit_order(symbol="AAPL", qty=10, side="buy", limit_price=150.0, current_price=150.0)
        compliance.add_restricted_symbol("TSLA")
        oms.submit_order(symbol="TSLA", qty=10, side="buy", limit_price=200.0, current_price=200.0)
        counts = oms.get_order_count()
        assert counts.get("submitted", 0) == 1
        assert counts.get("denied", 0) == 1
