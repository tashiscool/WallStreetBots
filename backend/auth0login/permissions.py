"""
Platform-level RBAC using Django's built-in Group and Permission system.

Defines four platform roles mapped to Django Groups:
    - viewer:       Read-only access to dashboards and analytics
    - trader:       Can place orders, manage strategies, run backtests
    - risk_manager: Can manage circuit breakers, approve live trading, view risk
    - admin:        Full platform access (superset of all roles)

Usage:
    from backend.auth0login.permissions import role_required, has_role

    @login_required
    @role_required('trader')
    def submit_order(request):
        ...

    if has_role(request.user, 'risk_manager'):
        ...
"""

import functools
import logging
from typing import List, Optional, Tuple

from django.contrib.auth.models import Group, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.http import JsonResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

class PlatformRoles:
    """Platform role constants — each maps to a Django Group."""

    VIEWER = "viewer"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    ADMIN = "admin"

    ALL = [VIEWER, TRADER, RISK_MANAGER, ADMIN]

    # Human-readable descriptions
    DESCRIPTIONS = {
        VIEWER: "Read-only access to dashboards, positions, and analytics",
        TRADER: "Can place orders, manage strategies, and run backtests",
        RISK_MANAGER: "Can manage circuit breakers, approve live trading, and view risk reports",
        ADMIN: "Full platform access including user management",
    }


# Custom permissions defined on the auth0login Credential model
# (we use Credential as the anchor since it's the core auth0login model)
PLATFORM_PERMISSIONS = [
    # Trading permissions
    ("can_place_orders", "Can place and cancel orders"),
    ("can_view_positions", "Can view positions and P&L"),
    ("can_manage_strategies", "Can create, edit, and activate strategies"),
    ("can_run_backtests", "Can run strategy backtests"),
    # Analytics & reporting
    ("can_view_analytics", "Can view analytics dashboards"),
    ("can_export_reports", "Can export PDF and CSV reports"),
    # Risk management
    ("can_manage_circuit_breakers", "Can trip and reset circuit breakers"),
    ("can_approve_live_trading", "Can approve users for live trading"),
    ("can_view_risk_dashboard", "Can view risk management dashboard"),
    # ML & training
    ("can_manage_ml_models", "Can train, promote, and retire ML models"),
    # System administration
    ("can_manage_alerts", "Can configure alert rules and thresholds"),
    ("can_manage_users", "Can view and modify user roles and settings"),
]

# Which permissions each role gets
ROLE_PERMISSIONS = {
    PlatformRoles.VIEWER: [
        "can_view_positions",
        "can_view_analytics",
    ],
    PlatformRoles.TRADER: [
        "can_view_positions",
        "can_view_analytics",
        "can_place_orders",
        "can_manage_strategies",
        "can_run_backtests",
        "can_export_reports",
    ],
    PlatformRoles.RISK_MANAGER: [
        "can_view_positions",
        "can_view_analytics",
        "can_view_risk_dashboard",
        "can_manage_circuit_breakers",
        "can_approve_live_trading",
        "can_export_reports",
    ],
    PlatformRoles.ADMIN: [
        # All permissions
        perm[0] for perm in PLATFORM_PERMISSIONS
    ],
}

# Default role for new users
DEFAULT_ROLE = PlatformRoles.TRADER


# ---------------------------------------------------------------------------
# Role helpers
# ---------------------------------------------------------------------------

def has_role(user: User, role: str) -> bool:
    """Check if a user belongs to a platform role (Django Group).

    Admin users (is_superuser) implicitly have all roles.
    """
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    return user.groups.filter(name=role).exists()


def has_any_role(user: User, roles: List[str]) -> bool:
    """Check if user belongs to any of the specified roles."""
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    return user.groups.filter(name__in=roles).exists()


def get_user_roles(user: User) -> List[str]:
    """Get all platform roles for a user."""
    if not user.is_authenticated:
        return []
    roles = list(user.groups.filter(name__in=PlatformRoles.ALL).values_list("name", flat=True))
    if user.is_superuser and PlatformRoles.ADMIN not in roles:
        roles.append(PlatformRoles.ADMIN)
    return sorted(roles)


def get_user_permissions(user: User) -> List[str]:
    """Get all platform permissions for a user (from roles + direct)."""
    if not user.is_authenticated:
        return []
    if user.is_superuser:
        return [p[0] for p in PLATFORM_PERMISSIONS]
    perms = set()
    for perm in user.get_all_permissions():
        # Django permissions are 'app_label.codename'
        codename = perm.split(".")[-1]
        if any(codename == p[0] for p in PLATFORM_PERMISSIONS):
            perms.add(codename)
    return sorted(perms)


def has_platform_permission(user: User, permission: str) -> bool:
    """Check if user has a specific platform permission.

    Args:
        user: Django User object.
        permission: Permission codename (e.g., 'can_place_orders').
    """
    if not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    return user.has_perm(f"auth0login.{permission}")


def assign_role(user: User, role: str) -> bool:
    """Assign a platform role to a user.

    Returns True if the role was assigned, False if it doesn't exist.
    """
    if role not in PlatformRoles.ALL:
        logger.warning("Unknown role: %s", role)
        return False
    group, _ = Group.objects.get_or_create(name=role)
    user.groups.add(group)
    logger.info("Assigned role '%s' to user %s", role, user.username)
    return True


def remove_role(user: User, role: str) -> bool:
    """Remove a platform role from a user."""
    try:
        group = Group.objects.get(name=role)
        user.groups.remove(group)
        logger.info("Removed role '%s' from user %s", role, user.username)
        return True
    except Group.DoesNotExist:
        return False


def set_roles(user: User, roles: List[str]) -> None:
    """Replace all platform roles for a user."""
    valid_roles = [r for r in roles if r in PlatformRoles.ALL]
    groups = Group.objects.filter(name__in=valid_roles)
    # Remove old platform roles, keep non-platform groups
    user.groups.remove(*Group.objects.filter(name__in=PlatformRoles.ALL))
    user.groups.add(*groups)
    logger.info("Set roles for user %s: %s", user.username, valid_roles)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def role_required(*roles: str):
    """Decorator that requires the user to have at least one of the specified roles.

    Usage::

        @login_required
        @role_required('trader', 'admin')
        def submit_order(request):
            ...

    Returns 403 JSON if the user lacks the required role.
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({
                    "status": "error",
                    "error_code": "AUTHENTICATION_REQUIRED",
                    "message": "Authentication required.",
                }, status=401)

            if not has_any_role(request.user, list(roles)):
                logger.warning(
                    "Access denied for user %s: requires role %s",
                    request.user.username, roles,
                )
                return JsonResponse({
                    "status": "error",
                    "error_code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"This action requires one of these roles: {', '.join(roles)}",
                    "user_roles": get_user_roles(request.user),
                }, status=403)

            return view_func(request, *args, **kwargs)
        return _wrapped
    return decorator


def permission_required_json(*permissions: str):
    """Decorator that requires specific platform permissions.

    Usage::

        @login_required
        @permission_required_json('can_place_orders')
        def submit_order(request):
            ...

    Returns 403 JSON if the user lacks the permission.
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return JsonResponse({
                    "status": "error",
                    "error_code": "AUTHENTICATION_REQUIRED",
                    "message": "Authentication required.",
                }, status=401)

            missing = [
                p for p in permissions
                if not has_platform_permission(request.user, p)
            ]
            if missing:
                logger.warning(
                    "Permission denied for user %s: missing %s",
                    request.user.username, missing,
                )
                return JsonResponse({
                    "status": "error",
                    "error_code": "INSUFFICIENT_PERMISSIONS",
                    "message": f"Missing permissions: {', '.join(missing)}",
                    "user_roles": get_user_roles(request.user),
                }, status=403)

            return view_func(request, *args, **kwargs)
        return _wrapped
    return decorator


class RoleRequiredMixin:
    """Mixin for class-based views that requires a platform role.

    Usage::

        class OrderCreateView(RoleRequiredMixin, CreateView):
            required_roles = ['trader']
            ...
    """

    required_roles: List[str] = []

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({
                "status": "error",
                "error_code": "AUTHENTICATION_REQUIRED",
                "message": "Authentication required.",
            }, status=401)

        if self.required_roles and not has_any_role(request.user, self.required_roles):
            return JsonResponse({
                "status": "error",
                "error_code": "INSUFFICIENT_PERMISSIONS",
                "message": f"Requires role: {', '.join(self.required_roles)}",
            }, status=403)

        return super().dispatch(request, *args, **kwargs)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_platform_permissions() -> Tuple[int, int]:
    """Create platform permissions and groups with correct assignments.

    Safe to run multiple times — idempotent.

    Returns:
        (groups_created, permissions_created) counts.
    """
    from .models import Credential

    ct = ContentType.objects.get_for_model(Credential)
    perms_created = 0
    groups_created = 0

    # Create permissions
    for codename, description in PLATFORM_PERMISSIONS:
        _, created = Permission.objects.get_or_create(
            codename=codename,
            content_type=ct,
            defaults={"name": description},
        )
        if created:
            perms_created += 1

    # Create groups and assign permissions
    for role_name in PlatformRoles.ALL:
        group, created = Group.objects.get_or_create(name=role_name)
        if created:
            groups_created += 1

        perm_codenames = ROLE_PERMISSIONS.get(role_name, [])
        perms = Permission.objects.filter(codename__in=perm_codenames, content_type=ct)
        group.permissions.set(perms)

    return groups_created, perms_created
