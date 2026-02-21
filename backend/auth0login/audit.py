"""Platform Audit Trail.

Provides persistent, queryable audit logging for security-critical actions.
Every role change, circuit breaker operation, ML model mutation, config change,
and authentication event is captured with full context (who, what, when, where,
correlation ID).

Usage:
    from backend.auth0login.audit import log_event, AuditEventType

    log_event(
        event_type=AuditEventType.ROLE_ASSIGNED,
        user=request.user,
        request=request,
        target_user=target_user,
        detail={"role": "trader"},
    )
"""

import json
import logging
from enum import Enum
from typing import ClassVar

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.http import HttpRequest
from django.utils import timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class AuditEventType(str, Enum):
    """Categorised event types for the audit trail."""

    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"

    # RBAC
    ROLE_ASSIGNED = "rbac.role_assigned"
    ROLE_REMOVED = "rbac.role_removed"
    ROLES_SET = "rbac.roles_set"
    PERMISSION_DENIED = "rbac.permission_denied"

    # Trading
    ORDER_PLACED = "trading.order_placed"
    ORDER_CANCELLED = "trading.order_cancelled"
    LIVE_TRADING_APPROVED = "trading.live_trading_approved"
    LIVE_TRADING_DENIED = "trading.live_trading_denied"
    LIVE_TRADING_REVOKED = "trading.live_trading_revoked"

    # Circuit breakers
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker.triggered"
    CIRCUIT_BREAKER_RESET = "circuit_breaker.reset"
    CIRCUIT_BREAKER_RECOVERY = "circuit_breaker.recovery"

    # ML models
    ML_MODEL_CREATED = "ml.model_created"
    ML_MODEL_DELETED = "ml.model_deleted"
    ML_MODEL_TRAINED = "ml.model_trained"
    ML_MODEL_STATUS_CHANGED = "ml.model_status_changed"
    RL_AGENT_TRAINED = "ml.rl_agent_trained"

    # Configuration
    CONFIG_CHANGED = "config.changed"
    STRATEGY_ACTIVATED = "config.strategy_activated"
    STRATEGY_DEACTIVATED = "config.strategy_deactivated"
    ALLOCATION_CHANGED = "config.allocation_changed"
    RISK_PROFILE_CHANGED = "config.risk_profile_changed"

    # Credentials
    CREDENTIALS_UPDATED = "credentials.updated"
    CREDENTIALS_ROTATED = "credentials.rotated"

    # Admin
    ADMIN_ACTION = "admin.action"


class AuditSeverity(str, Enum):
    """Severity level of audit events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Map event types to default severities
_EVENT_SEVERITIES = {
    AuditEventType.LOGIN_FAILED: AuditSeverity.WARNING,
    AuditEventType.PERMISSION_DENIED: AuditSeverity.WARNING,
    AuditEventType.LIVE_TRADING_REVOKED: AuditSeverity.CRITICAL,
    AuditEventType.CIRCUIT_BREAKER_TRIGGERED: AuditSeverity.CRITICAL,
    AuditEventType.CIRCUIT_BREAKER_RESET: AuditSeverity.WARNING,
    AuditEventType.ML_MODEL_DELETED: AuditSeverity.WARNING,
    AuditEventType.CREDENTIALS_ROTATED: AuditSeverity.WARNING,
    AuditEventType.ADMIN_ACTION: AuditSeverity.WARNING,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AuditLog(models.Model):
    """Immutable audit record for security-critical platform actions."""

    # Who
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_events",
        help_text="User who performed the action",
    )
    username = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Username snapshot (preserved even if user is deleted)",
    )

    # What
    event_type = models.CharField(
        max_length=60,
        db_index=True,
        help_text="Dot-separated event category (e.g. rbac.role_assigned)",
    )
    severity = models.CharField(
        max_length=10,
        default="info",
        db_index=True,
        help_text="info / warning / critical",
    )
    description = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable description of what happened",
    )
    detail = models.JSONField(
        default=dict,
        blank=True,
        help_text="Structured payload with event-specific data",
    )

    # Target (optional - the object being acted upon)
    target_user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="audit_target_events",
        help_text="User being acted upon (e.g. role assigned TO this user)",
    )
    target_type = models.CharField(
        max_length=60,
        blank=True,
        default="",
        help_text="Type of target object (e.g. 'ml_model', 'circuit_breaker')",
    )
    target_id = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="ID of target object",
    )

    # Where
    ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Client IP address",
    )
    user_agent = models.CharField(
        max_length=500,
        blank=True,
        default="",
        help_text="Client user-agent string",
    )
    correlation_id = models.CharField(
        max_length=50,
        blank=True,
        default="",
        db_index=True,
        help_text="Request correlation ID for tracing",
    )
    request_method = models.CharField(
        max_length=10,
        blank=True,
        default="",
    )
    request_path = models.CharField(
        max_length=500,
        blank=True,
        default="",
    )

    # When
    timestamp = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When the event occurred",
    )

    class Meta:
        app_label = "auth0login"
        ordering: ClassVar[list] = ["-timestamp"]
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"
        indexes: ClassVar[list] = [
            models.Index(fields=["user", "-timestamp"]),
            models.Index(fields=["event_type", "-timestamp"]),
            models.Index(fields=["severity", "-timestamp"]),
            models.Index(fields=["target_user", "-timestamp"]),
        ]

    def __str__(self):
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.event_type} by {self.username}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "severity": self.severity,
            "description": self.description,
            "detail": self.detail,
            "user": self.username,
            "target_user": self.target_user.username if self.target_user else None,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "ip_address": self.ip_address,
            "correlation_id": self.correlation_id,
            "request_method": self.request_method,
            "request_path": self.request_path,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Helper - the main API for creating audit entries
# ---------------------------------------------------------------------------

def log_event(
    event_type: AuditEventType | str,
    *,
    user: User | None = None,
    request: HttpRequest | None = None,
    description: str = "",
    detail: dict | None = None,
    target_user: User | None = None,
    target_type: str = "",
    target_id: str = "",
    severity: AuditSeverity | str | None = None,
) -> AuditLog:
    """Create an audit log entry.

    This is the primary interface for recording audit events.  All parameters
    except ``event_type`` are optional; supply as much context as available.
    """
    if isinstance(event_type, AuditEventType):
        event_str = event_type.value
    else:
        event_str = str(event_type)

    # Resolve severity
    if severity is None:
        if isinstance(event_type, AuditEventType):
            severity = _EVENT_SEVERITIES.get(event_type, AuditSeverity.INFO).value
        else:
            severity = AuditSeverity.INFO.value
    elif isinstance(severity, AuditSeverity):
        severity = severity.value

    # Extract request context
    ip_address = None
    user_agent = ""
    correlation_id = ""
    method = ""
    path = ""

    if request is not None:
        ip_address = _get_client_ip(request)
        headers = getattr(request, "headers", {}) or {}
        if not hasattr(headers, "get"):
            headers = {}
        user_agent = str(headers.get("User-Agent", ""))[:500]
        correlation_id = getattr(request, "correlation_id", "")
        method = getattr(request, "method", "") or ""
        path = getattr(request, "path", "") or ""
        if user is None and hasattr(request, "user") and request.user.is_authenticated:
            user = request.user

    username = ""
    if user is not None and hasattr(user, "username"):
        username = user.username

    user_for_fk = user if isinstance(user, User) and getattr(user, "pk", None) else None

    try:
        entry = AuditLog.objects.create(
            user=user_for_fk,
            username=username,
            event_type=event_str,
            severity=severity,
            description=description,
            detail=detail or {},
            target_user=target_user,
            target_type=target_type,
            target_id=str(target_id) if target_id else "",
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            request_method=method,
            request_path=path,
            timestamp=timezone.now(),
        )
    except Exception as exc:
        logger.warning("Audit DB write skipped for %s: %s", event_str, exc)
        entry = None

    # Also log to standard Python logger for log aggregation
    logger.info(
        "AUDIT: %s by %s — %s",
        event_str,
        username or "anonymous",
        description or "(no description)",
        extra={
            "audit_event": event_str,
            "audit_user": username,
            "correlation_id": correlation_id,
        },
    )

    return entry


# ---------------------------------------------------------------------------
# Middleware - captures auth events automatically
# ---------------------------------------------------------------------------

class AuditMiddleware:
    """Middleware that records login/logout events and permission denials.

    Must be placed AFTER ``AuthenticationMiddleware`` in the middleware stack.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Record permission denials (403 from our RBAC decorators)
        if (
            response.status_code == 403
            and hasattr(request, "user")
            and request.user.is_authenticated
        ):
            try:
                content_type = response.get("Content-Type", "")
                if "application/json" in content_type:
                    body = json.loads(response.content)
                    if body.get("error_code") == "INSUFFICIENT_PERMISSIONS":
                        log_event(
                            AuditEventType.PERMISSION_DENIED,
                            user=request.user,
                            request=request,
                            description=body.get("message", "Permission denied"),
                            detail={
                                "required_roles": body.get("required_roles", []),
                                "user_roles": body.get("user_roles", []),
                            },
                        )
            except (json.JSONDecodeError, AttributeError):
                pass

        return response


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_audit_log(
    *,
    user: User | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    since: str | None = None,
    until: str | None = None,
    target_user: User | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Query audit log with filters. Returns (queryset, total_count)."""
    qs = AuditLog.objects.all()

    if user:
        qs = qs.filter(user=user)
    if event_type:
        if "." in event_type:
            qs = qs.filter(event_type=event_type)
        else:
            # Category filter: e.g. "rbac" matches "rbac.*"
            qs = qs.filter(event_type__startswith=event_type + ".")
    if severity:
        qs = qs.filter(severity=severity)
    if since:
        qs = qs.filter(timestamp__gte=since)
    if until:
        qs = qs.filter(timestamp__lte=until)
    if target_user:
        qs = qs.filter(target_user=target_user)

    total = qs.count()
    entries = qs[offset:offset + limit]
    return entries, total


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_client_ip(request: HttpRequest) -> str | None:
    """Extract client IP, respecting X-Forwarded-For."""
    meta = getattr(request, "META", {}) or {}
    if not hasattr(meta, "get"):
        return None

    xff = meta.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    addr = meta.get("REMOTE_ADDR")
    return addr if addr else None
