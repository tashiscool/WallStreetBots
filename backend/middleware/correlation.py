"""Correlation ID middleware for distributed tracing."""

import logging
import threading
import uuid
from typing import Callable, Optional

from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)

# Thread-local storage for correlation ID
_correlation_id = threading.local()

# Header names for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID for logging/tracing."""
    return getattr(_correlation_id, "value", None)


def set_correlation_id(value: str) -> None:
    """Set the correlation ID for the current thread."""
    _correlation_id.value = value


def clear_correlation_id() -> None:
    """Clear the correlation ID."""
    if hasattr(_correlation_id, "value"):
        del _correlation_id.value


class CorrelationIdMiddleware:
    """Add correlation IDs to requests for distributed tracing.

    Features:
    - Generates UUID for each request if not provided
    - Accepts existing correlation ID from upstream services
    - Adds correlation ID to response headers
    - Stores in thread-local for use in logging
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Get existing correlation ID from headers or generate new one
        correlation_id = (
            request.headers.get(CORRELATION_ID_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
            or str(uuid.uuid4())
        )

        # Store in thread-local for logging access
        set_correlation_id(correlation_id)

        # Store on request for view access
        request.correlation_id = correlation_id

        # Log request with correlation ID
        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.path,
                "user_agent": request.headers.get("User-Agent", ""),
                "remote_addr": self._get_client_ip(request),
            },
        )

        try:
            response = self.get_response(request)

            # Add correlation ID to response
            response[CORRELATION_ID_HEADER] = correlation_id
            response[REQUEST_ID_HEADER] = correlation_id

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "status_code": response.status_code,
                    "path": request.path,
                },
            )

            return response

        except Exception as e:
            logger.exception(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "path": request.path,
                    "error": str(e),
                },
            )
            raise

        finally:
            clear_correlation_id()

    @staticmethod
    def _get_client_ip(request: HttpRequest) -> str:
        """Get client IP address handling proxies."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "")


class CorrelationIdFilter(logging.Filter):
    """Logging filter to add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True
