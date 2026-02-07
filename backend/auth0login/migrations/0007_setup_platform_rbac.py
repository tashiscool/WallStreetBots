"""
Data migration: Set up platform RBAC groups and permissions.

Creates four platform roles (viewer, trader, risk_manager, admin) as Django
Groups with appropriate permissions assigned to each.

Also assigns the default 'trader' role to all existing users who don't
already have a platform role.
"""

from django.db import migrations


PLATFORM_PERMISSIONS = [
    ("can_place_orders", "Can place and cancel orders"),
    ("can_view_positions", "Can view positions and P&L"),
    ("can_manage_strategies", "Can create, edit, and activate strategies"),
    ("can_run_backtests", "Can run strategy backtests"),
    ("can_view_analytics", "Can view analytics dashboards"),
    ("can_export_reports", "Can export PDF and CSV reports"),
    ("can_manage_circuit_breakers", "Can trip and reset circuit breakers"),
    ("can_approve_live_trading", "Can approve users for live trading"),
    ("can_view_risk_dashboard", "Can view risk management dashboard"),
    ("can_manage_ml_models", "Can train, promote, and retire ML models"),
    ("can_manage_alerts", "Can configure alert rules and thresholds"),
    ("can_manage_users", "Can view and modify user roles and settings"),
]

ROLE_PERMISSIONS = {
    "viewer": [
        "can_view_positions",
        "can_view_analytics",
    ],
    "trader": [
        "can_view_positions",
        "can_view_analytics",
        "can_place_orders",
        "can_manage_strategies",
        "can_run_backtests",
        "can_export_reports",
    ],
    "risk_manager": [
        "can_view_positions",
        "can_view_analytics",
        "can_view_risk_dashboard",
        "can_manage_circuit_breakers",
        "can_approve_live_trading",
        "can_export_reports",
    ],
    "admin": [p[0] for p in PLATFORM_PERMISSIONS],
}

DEFAULT_ROLE = "trader"


def setup_rbac(apps, schema_editor):
    """Create groups, permissions, and assign defaults."""
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")
    ContentType = apps.get_model("contenttypes", "ContentType")
    User = apps.get_model("auth", "User")

    # Get content type for Credential model (our permission anchor)
    ct, _ = ContentType.objects.get_or_create(
        app_label="auth0login",
        model="credential",
    )

    # Create permissions
    for codename, description in PLATFORM_PERMISSIONS:
        Permission.objects.get_or_create(
            codename=codename,
            content_type=ct,
            defaults={"name": description},
        )

    # Create groups and assign permissions
    for role_name, perm_codenames in ROLE_PERMISSIONS.items():
        group, _ = Group.objects.get_or_create(name=role_name)
        perms = Permission.objects.filter(codename__in=perm_codenames, content_type=ct)
        group.permissions.set(perms)

    # Assign default role to existing users without any platform role
    platform_groups = Group.objects.filter(name__in=list(ROLE_PERMISSIONS.keys()))
    default_group = Group.objects.get(name=DEFAULT_ROLE)

    users_without_roles = User.objects.exclude(groups__in=platform_groups).distinct()
    for user in users_without_roles:
        if not user.is_superuser:
            user.groups.add(default_group)

    # Superusers get admin role
    for user in User.objects.filter(is_superuser=True):
        admin_group = Group.objects.get(name="admin")
        user.groups.add(admin_group)


def reverse_rbac(apps, schema_editor):
    """Remove platform RBAC groups and permissions."""
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")
    ContentType = apps.get_model("contenttypes", "ContentType")

    Group.objects.filter(name__in=list(ROLE_PERMISSIONS.keys())).delete()

    try:
        ct = ContentType.objects.get(app_label="auth0login", model="credential")
        Permission.objects.filter(
            codename__in=[p[0] for p in PLATFORM_PERMISSIONS],
            content_type=ct,
        ).delete()
    except ContentType.DoesNotExist:
        pass


class Migration(migrations.Migration):

    dependencies = [
        ("auth0login", "0006_credential_created_at_credential_is_valid_and_more"),
        ("auth", "0012_alter_user_first_name_max_length"),
        ("contenttypes", "0002_remove_content_type_name"),
    ]

    operations = [
        migrations.RunPython(setup_rbac, reverse_rbac),
    ]
