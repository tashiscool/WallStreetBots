"""Security headers middleware for production hardening."""

import logging
from typing import Callable

from django.http import HttpRequest, HttpResponse

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware:
    """Add security headers to all responses.

    Implements OWASP recommended security headers:
    - Content-Security-Policy: Restrict resource loading
    - X-Content-Type-Options: Prevent MIME sniffing
    - X-Frame-Options: Prevent clickjacking
    - X-XSS-Protection: Enable XSS filter (legacy browsers)
    - Referrer-Policy: Control referrer information
    - Permissions-Policy: Restrict browser features
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)

        # Content Security Policy - restrictive default
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' https://api.alpaca.markets wss://stream.data.alpaca.markets",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
        ]
        response["Content-Security-Policy"] = "; ".join(csp_directives)

        # Prevent MIME type sniffing
        response["X-Content-Type-Options"] = "nosniff"

        # Clickjacking protection
        response["X-Frame-Options"] = "DENY"

        # XSS protection for legacy browsers
        response["X-XSS-Protection"] = "1; mode=block"

        # Control referrer information
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Restrict browser features
        permissions = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()",
        ]
        response["Permissions-Policy"] = ", ".join(permissions)

        # Remove server header that reveals tech stack
        if "Server" in response:
            del response["Server"]

        return response
