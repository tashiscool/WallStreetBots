"""Add AuditLog model for platform audit trail."""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("auth0login", "0007_setup_platform_rbac"),
    ]

    operations = [
        migrations.CreateModel(
            name="AuditLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "username",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Username snapshot (preserved even if user is deleted)",
                        max_length=150,
                    ),
                ),
                (
                    "event_type",
                    models.CharField(
                        db_index=True,
                        help_text="Dot-separated event category (e.g. rbac.role_assigned)",
                        max_length=60,
                    ),
                ),
                (
                    "severity",
                    models.CharField(
                        db_index=True,
                        default="info",
                        help_text="info / warning / critical",
                        max_length=10,
                    ),
                ),
                (
                    "description",
                    models.TextField(
                        blank=True,
                        default="",
                        help_text="Human-readable description of what happened",
                    ),
                ),
                (
                    "detail",
                    models.JSONField(
                        blank=True,
                        default=dict,
                        help_text="Structured payload with event-specific data",
                    ),
                ),
                (
                    "target_type",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Type of target object",
                        max_length=60,
                    ),
                ),
                (
                    "target_id",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="ID of target object",
                        max_length=255,
                    ),
                ),
                (
                    "ip_address",
                    models.GenericIPAddressField(
                        blank=True,
                        help_text="Client IP address",
                        null=True,
                    ),
                ),
                (
                    "user_agent",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Client user-agent string",
                        max_length=500,
                    ),
                ),
                (
                    "correlation_id",
                    models.CharField(
                        blank=True,
                        db_index=True,
                        default="",
                        help_text="Request correlation ID for tracing",
                        max_length=50,
                    ),
                ),
                (
                    "request_method",
                    models.CharField(blank=True, default="", max_length=10),
                ),
                (
                    "request_path",
                    models.CharField(blank=True, default="", max_length=500),
                ),
                (
                    "timestamp",
                    models.DateTimeField(
                        db_index=True,
                        default=django.utils.timezone.now,
                        help_text="When the event occurred",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        help_text="User who performed the action",
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="audit_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "target_user",
                    models.ForeignKey(
                        blank=True,
                        help_text="User being acted upon",
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="audit_target_events",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Audit Log",
                "verbose_name_plural": "Audit Logs",
                "ordering": ["-timestamp"],
                "indexes": [
                    models.Index(
                        fields=["user", "-timestamp"],
                        name="auth0login_audit_user_ts",
                    ),
                    models.Index(
                        fields=["event_type", "-timestamp"],
                        name="auth0login_audit_type_ts",
                    ),
                    models.Index(
                        fields=["severity", "-timestamp"],
                        name="auth0login_audit_sev_ts",
                    ),
                    models.Index(
                        fields=["target_user", "-timestamp"],
                        name="auth0login_audit_target_ts",
                    ),
                ],
            },
        ),
    ]
