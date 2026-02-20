"""Health check views for production monitoring.

Provides comprehensive health checks including:
- Database connectivity
- Redis connectivity (if configured)
- Memory and CPU usage
- Trading system components
"""
import logging
import time
from typing import Any, Dict

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
from django.conf import settings
from django.db import connection
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None
    PROMETHEUS_AVAILABLE = False

try:
    from backend.tradingbot.metrics.collectors import trading_registry
except ModuleNotFoundError:
    trading_registry = None
from backend.tradingbot.monitoring.health_check import health_checker

log = logging.getLogger(__name__)


def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    start = time.time()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        latency_ms = (time.time() - start) * 1000
        return {
            "status": "healthy",
            "latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def check_redis() -> Dict[str, Any]:
    """Check Redis connectivity if configured."""
    try:
        from django.core.cache import cache

        # Check if Redis is the cache backend
        cache_backend = settings.CACHES.get("default", {}).get("BACKEND", "")
        if "redis" not in cache_backend.lower():
            return {"status": "not_configured"}

        start = time.time()
        cache.set("health_check", "ok", timeout=5)
        result = cache.get("health_check")
        latency_ms = (time.time() - start) * 1000

        if result == "ok":
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
            }
        return {"status": "unhealthy", "error": "Cache read failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage."""
    if not PSUTIL_AVAILABLE:
        return {
            "status": "not_configured",
            "error": "psutil is not installed",
        }

    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        status = "healthy"
        warnings = []

        if memory.percent > 90:
            status = "degraded"
            warnings.append(f"High memory usage: {memory.percent}%")
        if cpu_percent > 90:
            status = "degraded"
            warnings.append(f"High CPU usage: {cpu_percent}%")

        return {
            "status": status,
            "memory_percent": memory.percent,
            "memory_available_mb": round(memory.available / (1024 * 1024), 2),
            "cpu_percent": cpu_percent,
            "warnings": warnings if warnings else None,
        }
    except Exception as e:
        return {"status": "unknown", "error": str(e)}


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """Comprehensive health check endpoint for load balancers and monitoring.

    Returns detailed health status for all system components:
    - Database: Connectivity and latency
    - Redis: Connectivity (if configured)
    - System: Memory and CPU usage
    - Trading: Strategy execution components
    """
    try:
        # Collect health from all subsystems
        db_health = check_database()
        redis_health = check_redis()
        system_health = check_system_resources()

        # Get trading system health
        try:
            trading_health = health_checker.check_health()
        except Exception as e:
            log.warning(f"Trading health check failed: {e}")
            trading_health = {"status": "unknown", "error": str(e)}

        # Determine overall status
        statuses = [
            db_health.get("status"),
            trading_health.get("status"),
        ]

        # Include system status only when configured
        if system_health.get("status") != "not_configured":
            statuses.append(system_health.get("status"))
        # Don't include Redis if not configured
        if redis_health.get("status") != "not_configured":
            statuses.append(redis_health.get("status"))

        if "unhealthy" in statuses or "unknown" in statuses:
            overall_status = "unhealthy"
            status_code = 503
        elif "degraded" in statuses:
            overall_status = "degraded"
            status_code = 200  # Still serve traffic
        else:
            overall_status = "healthy"
            status_code = 200

        health_response = {
            "status": overall_status,
            "timestamp": time.time(),
            "components": {
                "database": db_health,
                "redis": redis_health,
                "system": system_health,
                "trading": trading_health,
            },
        }

        return JsonResponse(health_response, status=status_code)
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return JsonResponse(
            {"status": "unhealthy", "error": str(e), "timestamp": time.time()},
            status=503
        )


@csrf_exempt
@require_http_methods(["GET"])
def readiness_check(request):
    """Readiness check for Kubernetes."""
    try:
        if health_checker.is_ready():
            return JsonResponse({"status": "ready"}, status=200)
        else:
            return JsonResponse({"status": "not_ready"}, status=503)
    except Exception as e:
        log.error(f"Readiness check failed: {e}")
        return JsonResponse({"status": "not_ready", "error": str(e)}, status=503)


@csrf_exempt
@require_http_methods(["GET"])
def liveness_check(request):
    """Liveness check for Kubernetes."""
    try:
        if health_checker.is_live():
            return JsonResponse({"status": "alive"}, status=200)
        else:
            return JsonResponse({"status": "dead"}, status=503)
    except Exception as e:
        log.error(f"Liveness check failed: {e}")
        return JsonResponse({"status": "dead", "error": str(e)}, status=503)


@csrf_exempt
@require_http_methods(["GET"])
def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    try:
        if generate_latest is None:
            return HttpResponse("Prometheus client not installed", status=503)

        metrics_data = generate_latest(trading_registry)
        return HttpResponse(
            metrics_data,
            content_type=CONTENT_TYPE_LATEST,
            status=200
        )
    except Exception as e:
        log.error(f"Metrics endpoint failed: {e}")
        return HttpResponse("Metrics collection failed", status=500)