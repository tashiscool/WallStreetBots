"""URL configuration for WallStreetBots backend."""

from django.contrib import admin
from django.urls import include, path

from backend.api_docs.views import api_docs_view, openapi_schema_view, redoc_view
from backend.health_views import (
    health_check,
    liveness_check,
    metrics_endpoint,
    readiness_check,
)

urlpatterns = [
    # Admin
    path("admin/", admin.site.urls),

    # Health checks (no auth required, excluded from rate limiting)
    path("health/", health_check, name="health-check"),
    path("health/live/", liveness_check, name="liveness-check"),
    path("health/ready/", readiness_check, name="readiness-check"),
    path("metrics/", metrics_endpoint, name="metrics"),

    # API Documentation
    path("api/docs/", api_docs_view, name="api-docs"),
    path("api/docs/openapi.json", openapi_schema_view, name="openapi-schema"),
    path("api/docs/redoc/", redoc_view, name="redoc"),

    # Application routes
    path("", include("backend.home.urls")),
    path("", include("backend.auth0login.urls")),
    path("trading/", include("backend.tradingbot.urls")),
]
