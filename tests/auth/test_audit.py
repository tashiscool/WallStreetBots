"""Tests for Platform Audit Trail.

Tests the AuditLog model, log_event() helper, AuditMiddleware,
and audit API endpoints.
"""

import json
import pytest
from datetime import timedelta
from django.contrib.auth.models import User
from django.test import RequestFactory
from django.utils import timezone

from backend.auth0login.audit import (
    AuditEventType,
    AuditLog,
    AuditMiddleware,
    AuditSeverity,
    get_audit_log,
    log_event,
)
from backend.auth0login.permissions import (
    PlatformRoles,
    assign_role,
    setup_platform_permissions,
)


@pytest.fixture
def rbac_setup(db):
    """Set up platform groups and permissions."""
    setup_platform_permissions()


@pytest.fixture
def factory():
    return RequestFactory()


@pytest.fixture
def admin_user(rbac_setup):
    user = User.objects.create_user(username="audit_admin", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.ADMIN)
    return user


@pytest.fixture
def trader_user(rbac_setup):
    user = User.objects.create_user(username="audit_trader", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.TRADER)
    return user


@pytest.fixture
def risk_manager_user(rbac_setup):
    user = User.objects.create_user(username="audit_riskmgr", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.RISK_MANAGER)
    return user


@pytest.fixture
def viewer_user(rbac_setup):
    user = User.objects.create_user(username="audit_viewer", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.VIEWER)
    return user


# ---------------------------------------------------------------------------
# Model basics
# ---------------------------------------------------------------------------

class TestAuditLogModel:
    def test_create_minimal(self, db):
        entry = AuditLog.objects.create(event_type="test.event")
        assert entry.id is not None
        assert entry.event_type == "test.event"
        assert entry.severity == "info"
        assert entry.timestamp is not None

    def test_to_dict(self, trader_user):
        entry = AuditLog.objects.create(
            event_type="auth.login",
            severity="info",
            user=trader_user,
            username=trader_user.username,
            description="User logged in",
            ip_address="192.168.1.1",
        )
        d = entry.to_dict()
        assert d["event_type"] == "auth.login"
        assert d["user"] == "audit_trader"
        assert d["ip_address"] == "192.168.1.1"
        assert d["id"] == entry.id

    def test_str_repr(self, db):
        entry = AuditLog.objects.create(
            event_type="rbac.role_assigned",
            username="testuser",
        )
        s = str(entry)
        assert "rbac.role_assigned" in s
        assert "testuser" in s

    def test_ordering_newest_first(self, db):
        old = AuditLog.objects.create(
            event_type="test.old",
            timestamp=timezone.now() - timedelta(hours=1),
        )
        new = AuditLog.objects.create(event_type="test.new")
        entries = list(AuditLog.objects.all())
        assert entries[0].id == new.id
        assert entries[1].id == old.id


# ---------------------------------------------------------------------------
# log_event() helper
# ---------------------------------------------------------------------------

class TestLogEvent:
    def test_basic_event(self, trader_user):
        entry = log_event(
            AuditEventType.LOGIN,
            user=trader_user,
            description="Logged in",
        )
        assert entry.event_type == "auth.login"
        assert entry.username == "audit_trader"
        assert entry.severity == "info"
        assert entry.user == trader_user

    def test_event_with_request_context(self, factory, trader_user):
        request = factory.post("/api/orders/", HTTP_USER_AGENT="TestBrowser/1.0")
        request.user = trader_user
        request.correlation_id = "test-corr-123"

        entry = log_event(
            AuditEventType.ORDER_PLACED,
            request=request,
            description="Placed order for AAPL",
            detail={"symbol": "AAPL", "qty": 10},
        )

        assert entry.user == trader_user
        assert entry.request_method == "POST"
        assert entry.request_path == "/api/orders/"
        assert entry.user_agent == "TestBrowser/1.0"
        assert entry.correlation_id == "test-corr-123"
        assert entry.detail == {"symbol": "AAPL", "qty": 10}

    def test_auto_severity_for_critical_events(self, db):
        entry = log_event(
            AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
            description="VIX breaker fired",
        )
        assert entry.severity == "critical"

    def test_auto_severity_for_warning_events(self, db):
        entry = log_event(
            AuditEventType.PERMISSION_DENIED,
            description="Access denied",
        )
        assert entry.severity == "warning"

    def test_override_severity(self, db):
        entry = log_event(
            AuditEventType.LOGIN,
            severity=AuditSeverity.CRITICAL,
            description="Suspicious login",
        )
        assert entry.severity == "critical"

    def test_target_user(self, admin_user, trader_user):
        entry = log_event(
            AuditEventType.ROLE_ASSIGNED,
            user=admin_user,
            target_user=trader_user,
            description='Assigned role "admin" to trader',
            detail={"role": "admin"},
        )
        assert entry.user == admin_user
        assert entry.target_user == trader_user

    def test_target_object(self, trader_user):
        entry = log_event(
            AuditEventType.ML_MODEL_CREATED,
            user=trader_user,
            target_type="ml_model",
            target_id="42",
            description="Created LSTM model",
        )
        assert entry.target_type == "ml_model"
        assert entry.target_id == "42"

    def test_string_event_type(self, db):
        entry = log_event(
            "custom.event",
            description="Custom event",
        )
        assert entry.event_type == "custom.event"

    def test_user_from_request_fallback(self, factory, trader_user):
        """If user not passed explicitly, extract from request."""
        request = factory.get("/test")
        request.user = trader_user

        entry = log_event(
            AuditEventType.LOGIN,
            request=request,
            description="Login",
        )
        assert entry.user == trader_user
        assert entry.username == "audit_trader"

    def test_ip_from_x_forwarded_for(self, factory, trader_user):
        request = factory.get(
            "/test",
            HTTP_X_FORWARDED_FOR="10.0.0.1, 192.168.1.1",
        )
        request.user = trader_user

        entry = log_event(
            AuditEventType.LOGIN,
            request=request,
        )
        assert entry.ip_address == "10.0.0.1"

    def test_no_user_no_request(self, db):
        """log_event works with no user and no request."""
        entry = log_event(
            AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
            description="System-triggered event",
        )
        assert entry.user is None
        assert entry.username == ""


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

class TestGetAuditLog:
    @pytest.fixture(autouse=True)
    def _seed(self, trader_user, admin_user):
        self.trader = trader_user
        self.admin = admin_user
        # Create a spread of events
        log_event(AuditEventType.LOGIN, user=trader_user, description="Login 1")
        log_event(AuditEventType.ORDER_PLACED, user=trader_user, description="Order 1")
        log_event(AuditEventType.ROLE_ASSIGNED, user=admin_user, target_user=trader_user, description="Role assign")
        log_event(AuditEventType.CIRCUIT_BREAKER_TRIGGERED, description="CB triggered")
        log_event(AuditEventType.PERMISSION_DENIED, user=trader_user, description="Denied")

    def test_all_events(self):
        entries, total = get_audit_log()
        assert total == 5

    def test_filter_by_user(self):
        entries, total = get_audit_log(user=self.trader)
        assert total == 3  # login, order, denied

    def test_filter_by_event_type_exact(self):
        entries, total = get_audit_log(event_type="auth.login")
        assert total == 1

    def test_filter_by_event_category(self):
        entries, total = get_audit_log(event_type="auth")
        assert total == 1  # just the login

    def test_filter_by_severity(self):
        entries, total = get_audit_log(severity="critical")
        assert total == 1  # circuit breaker

    def test_pagination(self):
        entries, total = get_audit_log(limit=2, offset=0)
        assert total == 5
        assert len(list(entries)) == 2

    def test_pagination_offset(self):
        entries, total = get_audit_log(limit=2, offset=3)
        assert total == 5
        assert len(list(entries)) == 2

    def test_filter_by_target_user(self):
        entries, total = get_audit_log(target_user=self.trader)
        assert total == 1  # role_assigned


# ---------------------------------------------------------------------------
# AuditMiddleware
# ---------------------------------------------------------------------------

class TestAuditMiddleware:
    def test_passes_through_normal_response(self, factory, trader_user):
        from django.http import JsonResponse

        def view(request):
            return JsonResponse({"ok": True})

        middleware = AuditMiddleware(view)
        request = factory.get("/test")
        request.user = trader_user

        response = middleware(request)
        assert response.status_code == 200
        # No audit entry for normal requests
        assert AuditLog.objects.count() == 0

    def test_records_permission_denied(self, factory, trader_user):
        from django.http import JsonResponse

        def view(request):
            return JsonResponse(
                {
                    "error_code": "INSUFFICIENT_PERMISSIONS",
                    "message": "Role required: admin",
                    "required_roles": ["admin"],
                    "user_roles": ["trader"],
                },
                status=403,
            )

        middleware = AuditMiddleware(view)
        request = factory.get("/admin/action")
        request.user = trader_user

        response = middleware(request)
        assert response.status_code == 403

        entries = AuditLog.objects.filter(event_type="rbac.permission_denied")
        assert entries.count() == 1
        entry = entries.first()
        assert entry.username == "audit_trader"
        assert "admin" in entry.detail.get("required_roles", [])

    def test_ignores_non_json_403(self, factory, trader_user):
        from django.http import HttpResponse

        def view(request):
            return HttpResponse("Forbidden", status=403, content_type="text/html")

        middleware = AuditMiddleware(view)
        request = factory.get("/test")
        request.user = trader_user

        middleware(request)
        assert AuditLog.objects.count() == 0

    def test_ignores_anonymous_403(self, factory, db):
        from django.contrib.auth.models import AnonymousUser
        from django.http import JsonResponse

        def view(request):
            return JsonResponse({"error": "forbidden"}, status=403)

        middleware = AuditMiddleware(view)
        request = factory.get("/test")
        request.user = AnonymousUser()

        middleware(request)
        assert AuditLog.objects.count() == 0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestAuditAPI:
    @pytest.fixture(autouse=True)
    def _seed(self, admin_user, trader_user, risk_manager_user, viewer_user):
        self.admin = admin_user
        self.trader = trader_user
        self.risk_mgr = risk_manager_user
        self.viewer = viewer_user
        # Seed some events
        log_event(AuditEventType.LOGIN, user=trader_user, description="Login")
        log_event(AuditEventType.ORDER_PLACED, user=trader_user, description="Order")
        log_event(AuditEventType.ROLE_ASSIGNED, user=admin_user, description="Role assigned")

    def test_admin_can_view_audit(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 3
        assert len(data["entries"]) == 3

    def test_risk_manager_can_view_audit(self, client):
        client.force_login(self.risk_mgr)
        response = client.get("/api/audit/")
        assert response.status_code == 200

    def test_trader_cannot_view_audit(self, client):
        client.force_login(self.trader)
        response = client.get("/api/audit/")
        assert response.status_code == 403

    def test_viewer_cannot_view_audit(self, client):
        client.force_login(self.viewer)
        response = client.get("/api/audit/")
        assert response.status_code == 403

    def test_unauthenticated_redirects(self, client):
        response = client.get("/api/audit/")
        assert response.status_code == 302

    def test_filter_by_event_type(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/?event_type=auth.login")
        data = response.json()
        assert data["total"] == 1

    def test_filter_by_category(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/?event_type=auth")
        data = response.json()
        assert data["total"] == 1

    def test_filter_by_severity(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/?severity=info")
        data = response.json()
        assert data["total"] == 3

    def test_pagination(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/?limit=2&offset=0")
        data = response.json()
        assert data["total"] == 3
        assert len(data["entries"]) == 2

    def test_summary_endpoint(self, client):
        client.force_login(self.admin)
        response = client.get("/api/audit/summary/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_events"] == 3
        assert len(data["by_type"]) > 0
        assert len(data["by_severity"]) > 0

    def test_summary_blocked_for_trader(self, client):
        client.force_login(self.trader)
        response = client.get("/api/audit/summary/")
        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Integration: audit events from role assignment API
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestAuditIntegration:
    def test_role_assign_creates_audit_entry(self, client, admin_user, viewer_user):
        # Ensure setup_platform_permissions has been called via fixture
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": viewer_user.username, "role": "trader", "action": "add"}),
            content_type="application/json",
        )
        assert response.status_code == 200

        entries = AuditLog.objects.filter(event_type="rbac.role_assigned")
        assert entries.count() == 1
        entry = entries.first()
        assert entry.username == admin_user.username
        assert entry.target_user == viewer_user
        assert entry.detail["role"] == "trader"

    def test_role_remove_creates_audit_entry(self, client, admin_user, trader_user):
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": trader_user.username, "role": "trader", "action": "remove"}),
            content_type="application/json",
        )
        assert response.status_code == 200

        entries = AuditLog.objects.filter(event_type="rbac.role_removed")
        assert entries.count() == 1

    @pytest.fixture
    def admin_user(self, rbac_setup):
        user = User.objects.create_user(username="int_admin", password="testpass123")
        user.groups.clear()
        assign_role(user, PlatformRoles.ADMIN)
        return user

    @pytest.fixture
    def trader_user(self, rbac_setup):
        user = User.objects.create_user(username="int_trader", password="testpass123")
        user.groups.clear()
        assign_role(user, PlatformRoles.TRADER)
        return user

    @pytest.fixture
    def viewer_user(self, rbac_setup):
        user = User.objects.create_user(username="int_viewer", password="testpass123")
        user.groups.clear()
        assign_role(user, PlatformRoles.VIEWER)
        return user

    @pytest.fixture
    def rbac_setup(self, db):
        setup_platform_permissions()


# ---------------------------------------------------------------------------
# Event type coverage
# ---------------------------------------------------------------------------

class TestEventTypes:
    def test_all_event_types_are_loggable(self, db):
        """Every defined event type can be logged without error."""
        for et in AuditEventType:
            entry = log_event(et, description=f"Test: {et.value}")
            assert entry.id is not None
            assert entry.event_type == et.value

    def test_severity_mapping(self, db):
        """Critical/warning event types get auto-severity."""
        critical = [
            AuditEventType.LIVE_TRADING_REVOKED,
            AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
        ]
        warning = [
            AuditEventType.LOGIN_FAILED,
            AuditEventType.PERMISSION_DENIED,
            AuditEventType.CIRCUIT_BREAKER_RESET,
            AuditEventType.ML_MODEL_DELETED,
            AuditEventType.CREDENTIALS_ROTATED,
            AuditEventType.ADMIN_ACTION,
        ]
        for et in critical:
            entry = log_event(et, description="test")
            assert entry.severity == "critical", f"{et} should be critical"
        for et in warning:
            entry = log_event(et, description="test")
            assert entry.severity == "warning", f"{et} should be warning"
