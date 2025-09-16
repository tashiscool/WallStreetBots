"""Health check endpoint for production monitoring."""
from __future__ import annotations
from typing import Dict, Any
import time
import logging

from ..risk.circuit_breaker import CircuitBreaker
from ..data.quality import DataQualityMonitor
from ..infra.build_info import build_id, version_info

log = logging.getLogger("wsb.health")


class HealthChecker:
    """Production health check system."""

    def __init__(self):
        """Initialize health checker."""
        self.components = {}

    def register_component(self, name: str, component: Any) -> None:
        """Register a component for health checking.

        Args:
            name: Component name
            component: Component instance with status() method
        """
        self.components[name] = component

    def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check.

        Returns:
            Health status dictionary
        """
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "build_info": version_info(),
            "components": {},
            "issues": []
        }

        # Check all registered components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'status'):
                    component_status = component.status()
                    health_status["components"][name] = component_status

                    # Check for critical issues
                    if isinstance(component, CircuitBreaker):
                        if component_status.get("tripped", False):
                            health_status["issues"].append(f"Circuit breaker is open: {component_status.get('reason')}")
                            health_status["status"] = "degraded"

                    elif isinstance(component, DataQualityMonitor):
                        if not component_status.get("is_fresh", True):
                            health_status["issues"].append("Data quality issues detected")
                            health_status["status"] = "degraded"
                else:
                    health_status["components"][name] = {"status": "no_status_method"}

            except Exception as e:
                health_status["components"][name] = {"error": str(e)}
                health_status["issues"].append(f"Component {name} health check failed: {e}")
                health_status["status"] = "unhealthy"

        # Overall health determination
        if health_status["issues"]:
            if health_status["status"] != "unhealthy":
                health_status["status"] = "degraded"

        log.info(f"Health check completed: {health_status['status']}")
        return health_status

    def is_ready(self) -> bool:
        """Check if system is ready to serve requests.

        Returns:
            True if system is ready
        """
        health = self.check_health()
        return health["status"] in ("healthy", "degraded")

    def is_live(self) -> bool:
        """Liveness check for container orchestration.

        Returns:
            True if system is live
        """
        # Basic liveness - can we execute code?
        try:
            _ = build_id()
            return True
        except Exception:
            return False


# Global health checker instance
health_checker = HealthChecker()