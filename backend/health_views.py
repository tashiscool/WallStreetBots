"""Health check views for production monitoring."""
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from django.http import HttpResponse
import logging

from backend.tradingbot.monitoring.health_check import health_checker
from backend.tradingbot.metrics.collectors import trading_registry

log = logging.getLogger(__name__)


@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for load balancers and monitoring."""
    try:
        health_status = health_checker.check_health()

        # Return appropriate HTTP status
        if health_status["status"] == "healthy":
            status_code = 200
        elif health_status["status"] == "degraded":
            status_code = 200  # Still serve traffic but log warning
        else:  # unhealthy
            status_code = 503

        return JsonResponse(health_status, status=status_code)
    except Exception as e:
        log.error(f"Health check failed: {e}")
        return JsonResponse(
            {"status": "unhealthy", "error": str(e)},
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
        metrics_data = generate_latest(trading_registry)
        return HttpResponse(
            metrics_data,
            content_type=CONTENT_TYPE_LATEST,
            status=200
        )
    except Exception as e:
        log.error(f"Metrics endpoint failed: {e}")
        return HttpResponse("Metrics collection failed", status=500)