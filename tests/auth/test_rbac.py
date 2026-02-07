"""Tests for Platform RBAC (Role-Based Access Control).

Tests the Django Group/Permission-based RBAC system including:
- Role assignment and removal
- Permission checking
- Decorators (@role_required, @permission_required_json)
- Auto-assignment signal for new users
- API endpoints for role management
"""

import json
import pytest
from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.test import RequestFactory

from backend.auth0login.permissions import (
    DEFAULT_ROLE,
    PLATFORM_PERMISSIONS,
    ROLE_PERMISSIONS,
    PlatformRoles,
    RoleRequiredMixin,
    assign_role,
    get_user_permissions,
    get_user_roles,
    has_any_role,
    has_platform_permission,
    has_role,
    permission_required_json,
    remove_role,
    role_required,
    set_roles,
    setup_platform_permissions,
)


@pytest.fixture
def rbac_setup(db):
    """Set up platform groups and permissions (replaces migration in tests)."""
    setup_platform_permissions()


@pytest.fixture
def factory():
    return RequestFactory()


@pytest.fixture
def viewer_user(rbac_setup):
    user = User.objects.create_user(username="viewer_user", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.VIEWER)
    return user


@pytest.fixture
def trader_user(rbac_setup):
    user = User.objects.create_user(username="trader_user", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.TRADER)
    return user


@pytest.fixture
def risk_manager_user(rbac_setup):
    user = User.objects.create_user(username="risk_mgr", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.RISK_MANAGER)
    return user


@pytest.fixture
def admin_user(rbac_setup):
    user = User.objects.create_user(username="admin_user", password="testpass123")
    user.groups.clear()
    assign_role(user, PlatformRoles.ADMIN)
    return user


@pytest.fixture
def superuser(rbac_setup):
    return User.objects.create_superuser(username="super", password="testpass123")


@pytest.fixture
def anon_user():
    """Unauthenticated user stub."""
    from django.contrib.auth.models import AnonymousUser
    return AnonymousUser()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

class TestSetup:
    def test_setup_creates_groups(self, rbac_setup):
        for role in PlatformRoles.ALL:
            assert Group.objects.filter(name=role).exists()

    def test_setup_creates_permissions(self, rbac_setup):
        ct = ContentType.objects.get(app_label="auth0login", model="credential")
        for codename, _ in PLATFORM_PERMISSIONS:
            assert Permission.objects.filter(codename=codename, content_type=ct).exists()

    def test_setup_is_idempotent(self, rbac_setup):
        # Run again — should not fail or duplicate
        g, p = setup_platform_permissions()
        assert g == 0  # No new groups
        assert p == 0  # No new permissions

    def test_viewer_group_has_correct_permissions(self, rbac_setup):
        viewer = Group.objects.get(name=PlatformRoles.VIEWER)
        perm_codenames = set(viewer.permissions.values_list("codename", flat=True))
        expected = set(ROLE_PERMISSIONS[PlatformRoles.VIEWER])
        assert perm_codenames == expected

    def test_trader_group_has_correct_permissions(self, rbac_setup):
        trader = Group.objects.get(name=PlatformRoles.TRADER)
        perm_codenames = set(trader.permissions.values_list("codename", flat=True))
        expected = set(ROLE_PERMISSIONS[PlatformRoles.TRADER])
        assert perm_codenames == expected

    def test_admin_group_has_all_permissions(self, rbac_setup):
        admin_group = Group.objects.get(name=PlatformRoles.ADMIN)
        perm_codenames = set(admin_group.permissions.values_list("codename", flat=True))
        all_perms = {p[0] for p in PLATFORM_PERMISSIONS}
        assert perm_codenames == all_perms


# ---------------------------------------------------------------------------
# Role helpers
# ---------------------------------------------------------------------------

class TestHasRole:
    def test_viewer_has_viewer_role(self, viewer_user):
        assert has_role(viewer_user, PlatformRoles.VIEWER) is True

    def test_viewer_does_not_have_trader_role(self, viewer_user):
        assert has_role(viewer_user, PlatformRoles.TRADER) is False

    def test_superuser_has_any_role(self, superuser):
        assert has_role(superuser, PlatformRoles.ADMIN) is True
        assert has_role(superuser, PlatformRoles.TRADER) is True
        assert has_role(superuser, PlatformRoles.VIEWER) is True

    def test_anon_has_no_role(self, anon_user):
        assert has_role(anon_user, PlatformRoles.VIEWER) is False


class TestHasAnyRole:
    def test_trader_matches_trader_or_admin(self, trader_user):
        assert has_any_role(trader_user, [PlatformRoles.TRADER, PlatformRoles.ADMIN]) is True

    def test_viewer_does_not_match_trader_or_admin(self, viewer_user):
        assert has_any_role(viewer_user, [PlatformRoles.TRADER, PlatformRoles.ADMIN]) is False


class TestGetUserRoles:
    def test_single_role(self, trader_user):
        roles = get_user_roles(trader_user)
        assert roles == [PlatformRoles.TRADER]

    def test_multiple_roles(self, rbac_setup):
        user = User.objects.create_user(username="multi_role", password="test")
        user.groups.clear()
        assign_role(user, PlatformRoles.TRADER)
        assign_role(user, PlatformRoles.RISK_MANAGER)
        roles = get_user_roles(user)
        assert PlatformRoles.TRADER in roles
        assert PlatformRoles.RISK_MANAGER in roles

    def test_superuser_includes_admin(self, superuser):
        roles = get_user_roles(superuser)
        assert PlatformRoles.ADMIN in roles

    def test_anon_empty(self, anon_user):
        assert get_user_roles(anon_user) == []


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------

class TestPermissions:
    def test_trader_can_place_orders(self, trader_user):
        assert has_platform_permission(trader_user, "can_place_orders") is True

    def test_viewer_cannot_place_orders(self, viewer_user):
        assert has_platform_permission(viewer_user, "can_place_orders") is False

    def test_viewer_can_view_analytics(self, viewer_user):
        assert has_platform_permission(viewer_user, "can_view_analytics") is True

    def test_risk_manager_can_manage_circuit_breakers(self, risk_manager_user):
        assert has_platform_permission(risk_manager_user, "can_manage_circuit_breakers") is True

    def test_trader_cannot_manage_circuit_breakers(self, trader_user):
        assert has_platform_permission(trader_user, "can_manage_circuit_breakers") is False

    def test_admin_has_all_permissions(self, admin_user):
        for codename, _ in PLATFORM_PERMISSIONS:
            assert has_platform_permission(admin_user, codename) is True

    def test_superuser_has_all_permissions(self, superuser):
        for codename, _ in PLATFORM_PERMISSIONS:
            assert has_platform_permission(superuser, codename) is True

    def test_get_user_permissions_trader(self, trader_user):
        perms = get_user_permissions(trader_user)
        assert "can_place_orders" in perms
        assert "can_manage_strategies" in perms
        assert "can_manage_circuit_breakers" not in perms


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

class TestAssignRole:
    def test_assign_valid_role(self, rbac_setup):
        user = User.objects.create_user(username="assign_test", password="test")
        user.groups.clear()
        result = assign_role(user, PlatformRoles.RISK_MANAGER)
        assert result is True
        assert has_role(user, PlatformRoles.RISK_MANAGER) is True

    def test_assign_invalid_role(self, rbac_setup):
        user = User.objects.create_user(username="bad_role", password="test")
        result = assign_role(user, "nonexistent_role")
        assert result is False

    def test_remove_role(self, trader_user):
        assert has_role(trader_user, PlatformRoles.TRADER) is True
        result = remove_role(trader_user, PlatformRoles.TRADER)
        assert result is True
        assert has_role(trader_user, PlatformRoles.TRADER) is False

    def test_remove_nonexistent_role(self, trader_user):
        result = remove_role(trader_user, "nonexistent")
        assert result is False

    def test_set_roles_replaces(self, rbac_setup):
        user = User.objects.create_user(username="set_roles", password="test")
        assign_role(user, PlatformRoles.TRADER)
        set_roles(user, [PlatformRoles.VIEWER, PlatformRoles.RISK_MANAGER])
        roles = get_user_roles(user)
        assert PlatformRoles.TRADER not in roles
        assert PlatformRoles.VIEWER in roles
        assert PlatformRoles.RISK_MANAGER in roles


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

class TestRoleRequiredDecorator:
    def test_allows_correct_role(self, factory, trader_user):
        @role_required(PlatformRoles.TRADER)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = trader_user
        response = my_view(request)
        assert response.status_code == 200

    def test_blocks_wrong_role(self, factory, viewer_user):
        @role_required(PlatformRoles.TRADER)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = viewer_user
        response = my_view(request)
        assert response.status_code == 403
        data = json.loads(response.content)
        assert data["error_code"] == "INSUFFICIENT_PERMISSIONS"

    def test_blocks_anonymous(self, factory, anon_user):
        @role_required(PlatformRoles.TRADER)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = anon_user
        response = my_view(request)
        assert response.status_code == 401

    def test_allows_superuser(self, factory, superuser):
        @role_required(PlatformRoles.RISK_MANAGER)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = superuser
        response = my_view(request)
        assert response.status_code == 200

    def test_allows_any_of_multiple_roles(self, factory, risk_manager_user):
        @role_required(PlatformRoles.TRADER, PlatformRoles.RISK_MANAGER)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = risk_manager_user
        response = my_view(request)
        assert response.status_code == 200

    def test_response_includes_user_roles(self, factory, viewer_user):
        @role_required(PlatformRoles.ADMIN)
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = viewer_user
        response = my_view(request)
        data = json.loads(response.content)
        assert "user_roles" in data
        assert PlatformRoles.VIEWER in data["user_roles"]


class TestPermissionRequiredDecorator:
    def test_allows_with_permission(self, factory, trader_user):
        @permission_required_json("can_place_orders")
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = trader_user
        response = my_view(request)
        assert response.status_code == 200

    def test_blocks_without_permission(self, factory, viewer_user):
        @permission_required_json("can_place_orders")
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = viewer_user
        response = my_view(request)
        assert response.status_code == 403

    def test_checks_multiple_permissions(self, factory, trader_user):
        @permission_required_json("can_place_orders", "can_manage_circuit_breakers")
        def my_view(request):
            from django.http import JsonResponse
            return JsonResponse({"ok": True})

        request = factory.get("/test")
        request.user = trader_user
        response = my_view(request)
        assert response.status_code == 403
        data = json.loads(response.content)
        assert "can_manage_circuit_breakers" in data["message"]


# ---------------------------------------------------------------------------
# Signal: auto-assignment
# ---------------------------------------------------------------------------

class TestAutoAssignSignal:
    def test_new_user_gets_default_role(self, rbac_setup):
        user = User.objects.create_user(username="new_user_signal", password="test")
        roles = get_user_roles(user)
        assert DEFAULT_ROLE in roles

    def test_new_superuser_gets_admin_role(self, rbac_setup):
        user = User.objects.create_superuser(username="new_super_signal", password="test")
        roles = get_user_roles(user)
        assert PlatformRoles.ADMIN in roles

    def test_existing_user_save_does_not_reassign(self, trader_user):
        # Remove role and save — should NOT get reassigned
        remove_role(trader_user, PlatformRoles.TRADER)
        trader_user.first_name = "Updated"
        trader_user.save()
        roles = get_user_roles(trader_user)
        assert PlatformRoles.TRADER not in roles


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@pytest.mark.django_db
class TestRolesAPI:
    def test_get_roles_authenticated(self, client, trader_user):
        client.force_login(trader_user)
        response = client.get("/api/roles/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert PlatformRoles.TRADER in data["roles"]
        assert "can_place_orders" in data["permissions"]
        assert "available_roles" in data

    def test_get_roles_unauthenticated(self, client):
        response = client.get("/api/roles/")
        # @login_required redirects to login page
        assert response.status_code == 302

    def test_assign_role_as_admin(self, client, admin_user, viewer_user):
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": viewer_user.username, "role": "trader", "action": "add"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        viewer_user.refresh_from_db()
        assert has_role(viewer_user, PlatformRoles.TRADER) is True

    def test_assign_role_as_non_admin_blocked(self, client, trader_user, viewer_user):
        client.force_login(trader_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": viewer_user.username, "role": "admin", "action": "add"}),
            content_type="application/json",
        )
        assert response.status_code == 403

    def test_remove_role_as_admin(self, client, admin_user, trader_user):
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": trader_user.username, "role": "trader", "action": "remove"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        trader_user.refresh_from_db()
        assert has_role(trader_user, PlatformRoles.TRADER) is False

    def test_assign_invalid_role(self, client, admin_user, viewer_user):
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": viewer_user.username, "role": "supreme_overlord"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_assign_nonexistent_user(self, client, admin_user):
        client.force_login(admin_user)
        response = client.post(
            "/api/roles/assign/",
            json.dumps({"username": "nobody", "role": "trader"}),
            content_type="application/json",
        )
        assert response.status_code == 404
