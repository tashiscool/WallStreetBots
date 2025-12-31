"""Rate limiting middleware using token bucket algorithm."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, Optional

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Implements the token bucket algorithm:
    - Bucket starts full with 'capacity' tokens
    - Tokens are consumed on each request
    - Tokens refill at 'refill_rate' per second
    - Requests are rejected when no tokens available
    """

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self.lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_retry_after(self) -> float:
        """Calculate seconds until a token will be available."""
        with self.lock:
            if self.tokens >= 1:
                return 0.0
            needed = 1 - self.tokens
            return needed / self.refill_rate


class RateLimitMiddleware:
    """Rate limiting middleware with per-IP and per-user limits.

    Features:
    - Per-IP rate limiting for anonymous users
    - Per-user rate limiting for authenticated users
    - Token bucket algorithm for burst handling
    - Configurable limits via settings
    - Rate limit headers in responses
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]):
        self.get_response = get_response

        # Get config from settings
        self.requests_per_minute = getattr(settings, "RATE_LIMIT_PER_MINUTE", 60)
        self.burst_size = getattr(settings, "RATE_LIMIT_BURST", 10)

        # Calculate refill rate (tokens per second)
        self.refill_rate = self.requests_per_minute / 60.0

        # Token buckets per client (IP or user ID)
        self._buckets: Dict[str, TokenBucket] = defaultdict(self._create_bucket)
        self._cleanup_lock = Lock()
        self._last_cleanup = time.time()

        # Paths to exclude from rate limiting
        self.excluded_paths = {
            "/health/",
            "/health/live/",
            "/health/ready/",
            "/metrics/",
        }

    def _create_bucket(self) -> TokenBucket:
        """Factory for creating new token buckets."""
        return TokenBucket(
            capacity=self.burst_size,
            refill_rate=self.refill_rate,
        )

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Skip rate limiting for health checks and metrics
        if request.path in self.excluded_paths:
            return self.get_response(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Try to consume a token
        bucket = self._buckets[client_id]

        if not bucket.consume():
            retry_after = bucket.get_retry_after()
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_id": client_id,
                    "path": request.path,
                    "retry_after": retry_after,
                },
            )
            response = JsonResponse(
                {
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please slow down.",
                    "retry_after": int(retry_after) + 1,
                },
                status=429,
            )
            response["Retry-After"] = str(int(retry_after) + 1)
            response["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response["X-RateLimit-Remaining"] = "0"
            return response

        # Periodic cleanup of old buckets
        self._maybe_cleanup()

        # Process request
        response = self.get_response(request)

        # Add rate limit headers
        response["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        response["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response

    def _get_client_id(self, request: HttpRequest) -> str:
        """Get unique client identifier for rate limiting."""
        # Use user ID for authenticated users
        if hasattr(request, "user") and request.user.is_authenticated:
            return f"user:{request.user.id}"

        # Fall back to IP address
        return f"ip:{self._get_client_ip(request)}"

    @staticmethod
    def _get_client_ip(request: HttpRequest) -> str:
        """Get client IP address handling proxies."""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "unknown")

    def _maybe_cleanup(self) -> None:
        """Periodically clean up old buckets to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < 300:  # Every 5 minutes
            return

        with self._cleanup_lock:
            if now - self._last_cleanup < 300:
                return

            # Remove buckets that haven't been used in 10 minutes
            cutoff = now - 600
            to_remove = [
                key for key, bucket in self._buckets.items()
                if bucket.last_refill < cutoff
            ]

            for key in to_remove:
                del self._buckets[key]

            self._last_cleanup = now

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} stale rate limit buckets")
