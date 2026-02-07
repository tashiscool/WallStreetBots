"""Tests for Maker-Checker (Four-Eyes) Approval Workflow."""

import pytest
from datetime import datetime, timedelta

from backend.tradingbot.execution.interfaces import OrderRequest
from backend.tradingbot.execution.maker_checker import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalThresholds,
    MakerCheckerService,
)


@pytest.fixture
def thresholds():
    return ApprovalThresholds(
        auto_approve_notional=10_000,
        auto_approve_shares=1_000,
        require_approval_notional=10_000,
        require_approval_shares=1_000,
        approval_timeout_minutes=30,
        max_pending_approvals=5,
        require_approval_for_new_symbols=False,
    )


@pytest.fixture
def service(thresholds):
    return MakerCheckerService(thresholds=thresholds)


def _order(symbol="AAPL", qty=10, side="buy"):
    return OrderRequest(
        client_order_id="test_001",
        symbol=symbol,
        qty=qty,
        side=side,
        type="limit",
        limit_price=150.0,
    )


class TestAutoApproval:
    def test_small_order_auto_approved(self, service):
        req = service.submit(_order(qty=10), maker_id="trader_a", current_price=100.0)
        assert req.status == ApprovalStatus.AUTO_APPROVED
        assert req.checker_id == "system"

    def test_large_notional_requires_approval(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        assert req.status == ApprovalStatus.PENDING

    def test_large_qty_requires_approval(self, service):
        req = service.submit(_order(qty=2000), maker_id="trader_a", current_price=1.0)
        assert req.status == ApprovalStatus.PENDING


class TestNewSymbolApproval:
    def test_first_order_new_symbol_requires_approval(self):
        thresholds = ApprovalThresholds(
            auto_approve_notional=10_000,
            auto_approve_shares=1_000,
            require_approval_for_new_symbols=True,
        )
        svc = MakerCheckerService(thresholds=thresholds)
        req = svc.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        assert req.status == ApprovalStatus.PENDING

    def test_second_order_same_symbol_auto_approved(self):
        thresholds = ApprovalThresholds(
            auto_approve_notional=10_000,
            auto_approve_shares=1_000,
            require_approval_for_new_symbols=True,
        )
        svc = MakerCheckerService(thresholds=thresholds)
        # First: requires approval (new symbol)
        svc.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        # Second: auto-approved (symbol now known)
        req2 = svc.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        assert req2.status == ApprovalStatus.AUTO_APPROVED


class TestApprovalWorkflow:
    def test_approve_pending_order(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        assert req.status == ApprovalStatus.PENDING

        approved = service.approve(req.request_id, checker_id="manager_b", comment="ok")
        assert approved is not None
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.checker_id == "manager_b"

    def test_reject_pending_order(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        rejected = service.reject(req.request_id, checker_id="manager_b", comment="too risky")
        assert rejected is not None
        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.checker_comment == "too risky"

    def test_self_approval_blocked(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        result = service.approve(req.request_id, checker_id="trader_a")
        assert result is not None
        assert result.status == ApprovalStatus.REJECTED
        assert "cannot approve own" in result.checker_comment.lower()

    def test_approve_unknown_request_returns_none(self, service):
        result = service.approve("nonexistent_id", checker_id="manager_b")
        assert result is None

    def test_reject_unknown_request_returns_none(self, service):
        result = service.reject("nonexistent_id", checker_id="manager_b")
        assert result is None


class TestCallbacks:
    def test_on_approved_callback(self, thresholds):
        approved_orders = []
        svc = MakerCheckerService(
            thresholds=thresholds,
            on_approved=lambda r: approved_orders.append(r),
        )
        req = svc.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        svc.approve(req.request_id, checker_id="manager_b")
        assert len(approved_orders) == 1

    def test_on_rejected_callback(self, thresholds):
        rejected_orders = []
        svc = MakerCheckerService(
            thresholds=thresholds,
            on_rejected=lambda r: rejected_orders.append(r),
        )
        req = svc.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        svc.reject(req.request_id, checker_id="manager_b", comment="no")
        assert len(rejected_orders) == 1

    def test_auto_approve_triggers_callback(self, thresholds):
        approved_orders = []
        svc = MakerCheckerService(
            thresholds=thresholds,
            on_approved=lambda r: approved_orders.append(r),
        )
        svc.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        assert len(approved_orders) == 1


class TestExpiration:
    def test_stale_approvals_expired(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        # Manually backdate the created_at to force expiry
        service._pending[req.request_id].created_at = datetime.utcnow() - timedelta(hours=1)

        # Expire on next action
        pending = service.get_pending()
        assert len(pending) == 0

        # Should show up in history as expired
        history = service.get_history()
        assert any(r.status == ApprovalStatus.EXPIRED for r in history)


class TestMaxPending:
    def test_rejects_when_too_many_pending(self, service):
        # Fill up pending queue
        for i in range(5):
            service.submit(
                OrderRequest(
                    client_order_id=f"test_{i}",
                    symbol="AAPL",
                    qty=100,
                    side="buy",
                    type="limit",
                    limit_price=200.0,
                ),
                maker_id="trader_a",
                current_price=200.0,
            )
        # 6th should be rejected
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        assert req.status == ApprovalStatus.REJECTED
        assert "too many" in req.checker_comment.lower()


class TestQueries:
    def test_get_pending(self, service):
        service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        pending = service.get_pending()
        assert len(pending) == 1

    def test_get_request_pending(self, service):
        req = service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        found = service.get_request(req.request_id)
        assert found is not None
        assert found.request_id == req.request_id

    def test_get_request_historical(self, service):
        req = service.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        # Auto-approved → in history
        found = service.get_request(req.request_id)
        assert found is not None
        assert found.status == ApprovalStatus.AUTO_APPROVED

    def test_stats(self, service):
        service.submit(_order(qty=1), maker_id="trader_a", current_price=10.0)
        service.submit(_order(qty=100), maker_id="trader_a", current_price=200.0)
        stats = service.get_stats()
        assert stats["pending"] == 1
        assert stats["total_processed"] == 1
