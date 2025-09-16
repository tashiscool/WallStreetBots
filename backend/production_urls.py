"""Production URL configuration for monitoring endpoints."""
from django.urls import path
from . import health_views

urlpatterns = [
    path('health/', health_views.health_check, name='health_check'),
    path('health/ready/', health_views.readiness_check, name='readiness_check'),
    path('health/live/', health_views.liveness_check, name='liveness_check'),
    path('metrics/', health_views.metrics_endpoint, name='metrics'),
]