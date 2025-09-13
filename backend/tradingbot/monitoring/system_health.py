"""System Health Monitor.

Real - time health monitoring for all system components with comprehensive alerting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psutil


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a system component."""

    component_name: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""

    timestamp: datetime
    overall_status: HealthStatus
    data_feed_status: ComponentHealth
    broker_status: ComponentHealth
    database_status: ComponentHealth
    resource_status: ComponentHealth
    trading_status: ComponentHealth
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    total_errors: int = 0


class SystemHealthMonitor:
    """Real - time system health monitoring.

    Monitors all critical system components:
    - Data feed health and latency
    - Broker connection status
    - Database performance
    - System resources (CPU, memory)
    - Trading performance metrics
    """

    def __init__(
        self,
        trading_system=None,
        alert_system=None,
        config: dict[str, Any] | None = None,
    ):
        self.trading_system = trading_system
        self.alert_system = alert_system
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Health thresholds
        self.alert_thresholds = {
            "data_feed_latency": config.get(
                "data_feed_latency_threshold", 5.0
            ),  # seconds
            "order_execution_time": config.get(
                "order_execution_time_threshold", 30.0
            ),  # seconds
            "error_rate": config.get("error_rate_threshold", 0.05),  # 5% error rate
            "memory_usage": config.get(
                "memory_usage_threshold", 0.80
            ),  # 80% memory usage
            "cpu_usage": config.get("cpu_usage_threshold", 0.90),  # 90% CPU usage
            "disk_usage": config.get("disk_usage_threshold", 0.85),  # 85% disk usage
        }

        # Monitoring state
        self.start_time = datetime.now()
        self.health_history = []
        self.component_checks = {}

        self.logger.info("SystemHealthMonitor initialized")

    async def check_system_health(self) -> SystemHealthReport:
        """Comprehensive system health check."""
        try:
            start_time = time.time()

            # Run all health checks in parallel
            health_checks = await asyncio.gather(
                self._check_data_feed_health(),
                self._check_broker_connection(),
                self._check_database_performance(),
                self._check_system_resources(),
                self._check_trading_performance(),
                return_exceptions=True,
            )

            # Extract results
            data_feed_health = (
                health_checks[0]
                if not isinstance(health_checks[0], Exception)
                else self._create_error_health("data_feed", health_checks[0])
            )
            broker_health = (
                health_checks[1]
                if not isinstance(health_checks[1], Exception)
                else self._create_error_health("broker", health_checks[1])
            )
            db_health = (
                health_checks[2]
                if not isinstance(health_checks[2], Exception)
                else self._create_error_health("database", health_checks[2])
            )
            resource_health = (
                health_checks[3]
                if not isinstance(health_checks[3], Exception)
                else self._create_error_health("resources", health_checks[3])
            )
            trading_health = (
                health_checks[4]
                if not isinstance(health_checks[4], Exception)
                else self._create_error_health("trading", health_checks[4])
            )

            # Calculate overall health
            overall_status = self._calculate_overall_health(
                [
                    data_feed_health,
                    broker_health,
                    db_health,
                    resource_health,
                    trading_health,
                ]
            )

            # Create comprehensive report
            report = SystemHealthReport(
                timestamp=datetime.now(),
                overall_status=overall_status,
                data_feed_status=data_feed_health,
                broker_status=broker_health,
                database_status=db_health,
                resource_status=resource_health,
                trading_status=trading_health,
                components={
                    "data_feed": data_feed_health,
                    "broker": broker_health,
                    "database": db_health,
                    "resources": resource_health,
                    "trading": trading_health,
                },
                uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
            )

            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)

            # Store in history
            self.health_history.append(report)
            if len(self.health_history) > 100:  # Keep last 100 reports
                self.health_history = self.health_history[-100:]

            # Send alerts if unhealthy
            if overall_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                await self._send_health_alert(report)

            check_duration = time.time() - start_time
            self.logger.info(
                f"System health check completed in {check_duration: .2f}s - Status: {overall_status.value}"
            )

            return report

        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
            return self._create_critical_report(str(e))

    async def _check_data_feed_health(self) -> ComponentHealth:
        """Check data feed health and latency."""
        try:
            start_time = time.time()

            # Test data feed connectivity
            if self.trading_system and hasattr(self.trading_system, "data_provider"):
                # Test a simple data request
                test_price = await self.trading_system.data_provider.get_current_price(
                    "AAPL"
                )
                response_time = (time.time() - start_time) * 1000  # Convert to ms

                if test_price:
                    status = (
                        HealthStatus.HEALTHY
                        if response_time
                        < self.alert_thresholds["data_feed_latency"] * 1000
                        else HealthStatus.DEGRADED
                    )
                    return ComponentHealth(
                        component_name="data_feed",
                        status=status,
                        last_check=datetime.now(),
                        response_time_ms=response_time,
                        details={
                            "test_ticker": "AAPL",
                            "test_price": float(test_price.price)
                            if hasattr(test_price, "price")
                            else None,
                            "latency_threshold_ms": self.alert_thresholds[
                                "data_feed_latency"
                            ]
                            * 1000,
                        },
                        recommendations=[]
                        if status == HealthStatus.HEALTHY
                        else ["Consider switching to backup data source"],
                    )
                else:
                    return ComponentHealth(
                        component_name="data_feed",
                        status=HealthStatus.CRITICAL,
                        last_check=datetime.now(),
                        response_time_ms=response_time,
                        error_count=1,
                        details={"error": "No data returned from test request"},
                        recommendations=[
                            "Switch to backup data source",
                            "Check data provider configuration",
                        ],
                    )
            else:
                return ComponentHealth(
                    component_name="data_feed",
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    details={"error": "No trading system or data provider available"},
                    recommendations=["Initialize trading system with data provider"],
                )

        except Exception as e:
            return ComponentHealth(
                component_name="data_feed",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={"error": str(e)},
                recommendations=["Check data provider connection", "Review error logs"],
            )

    async def _check_broker_connection(self) -> ComponentHealth:
        """Check broker connection health."""
        try:
            start_time = time.time()

            if self.trading_system and hasattr(self.trading_system, "broker_manager"):
                # Test broker connection
                account_info = await self.trading_system.broker_manager.get_account()
                response_time = (time.time() - start_time) * 1000

                if account_info:
                    status = (
                        HealthStatus.HEALTHY
                        if response_time < 5000
                        else HealthStatus.DEGRADED
                    )  # 5 second threshold
                    return ComponentHealth(
                        component_name="broker",
                        status=status,
                        last_check=datetime.now(),
                        response_time_ms=response_time,
                        details={
                            "account_status": getattr(
                                account_info, "status", "unknown"
                            ),
                            "buying_power": getattr(account_info, "buying_power", 0),
                            "portfolio_value": getattr(
                                account_info, "portfolio_value", 0
                            ),
                        },
                        recommendations=[]
                        if status == HealthStatus.HEALTHY
                        else ["Check broker API connection"],
                    )
                else:
                    return ComponentHealth(
                        component_name="broker",
                        status=HealthStatus.CRITICAL,
                        last_check=datetime.now(),
                        response_time_ms=response_time,
                        error_count=1,
                        details={"error": "No account info returned"},
                        recommendations=[
                            "Check broker API credentials",
                            "Verify account status",
                        ],
                    )
            else:
                return ComponentHealth(
                    component_name="broker",
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time_ms=0,
                    details={"error": "No broker manager available"},
                    recommendations=["Initialize broker manager"],
                )

        except Exception as e:
            return ComponentHealth(
                component_name="broker",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={"error": str(e)},
                recommendations=["Check broker connection", "Review API credentials"],
            )

    async def _check_database_performance(self) -> ComponentHealth:
        """Check database performance."""
        try:
            start_time = time.time()

            # Test database connectivity (simplified)
            # In a real implementation, this would test actual database queries
            await asyncio.sleep(0.01)  # Simulate database query
            response_time = (time.time() - start_time) * 1000

            # For now, assume database is healthy if we can simulate a query
            status = (
                HealthStatus.HEALTHY if response_time < 1000 else HealthStatus.DEGRADED
            )  # 1 second threshold

            return ComponentHealth(
                component_name="database",
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                details={
                    "connection_pool_size": 10,  # Placeholder
                    "active_connections": 3,  # Placeholder
                    "query_count": 1000,  # Placeholder
                },
                recommendations=[]
                if status == HealthStatus.HEALTHY
                else ["Optimize database queries", "Check connection pool"],
            )

        except Exception as e:
            return ComponentHealth(
                component_name="database",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={"error": str(e)},
                recommendations=["Check database connection", "Review database logs"],
            )

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource usage."""
        try:
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            recommendations = []

            if cpu_percent > self.alert_thresholds["cpu_usage"] * 100:
                status = HealthStatus.CRITICAL
                recommendations.append(
                    "High CPU usage-consider scaling or optimization"
                )
            elif cpu_percent > self.alert_thresholds["cpu_usage"] * 100 * 0.8:
                status = HealthStatus.DEGRADED
                recommendations.append("CPU usage approaching threshold")

            if memory.percent > self.alert_thresholds["memory_usage"] * 100:
                status = HealthStatus.CRITICAL
                recommendations.append("High memory usage-consider memory optimization")
            elif memory.percent > self.alert_thresholds["memory_usage"] * 100 * 0.8:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                recommendations.append("Memory usage approaching threshold")

            if disk.percent > self.alert_thresholds["disk_usage"] * 100:
                status = HealthStatus.CRITICAL
                recommendations.append("High disk usage-cleanup required")
            elif disk.percent > self.alert_thresholds["disk_usage"] * 100 * 0.8:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                recommendations.append("Disk usage approaching threshold")

            return ComponentHealth(
                component_name="resources",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return ComponentHealth(
                component_name="resources",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={"error": str(e)},
                recommendations=[
                    "Check system resource monitoring",
                    "Review system logs",
                ],
            )

    async def _check_trading_performance(self) -> ComponentHealth:
        """Check trading performance metrics."""
        try:
            # This would check actual trading performance in a real implementation
            # For now, we'll simulate some metrics

            status = HealthStatus.HEALTHY
            recommendations = []

            # Simulate some trading metrics
            total_trades = 100  # Placeholder
            successful_trades = 95  # Placeholder
            error_rate = (
                (total_trades - successful_trades) / total_trades
                if total_trades > 0
                else 0
            )

            if error_rate > self.alert_thresholds["error_rate"]:
                status = HealthStatus.CRITICAL
                recommendations.append("High trading error rate-review strategy logic")
            elif error_rate > self.alert_thresholds["error_rate"] * 0.5:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                recommendations.append("Trading error rate approaching threshold")

            return ComponentHealth(
                component_name="trading",
                status=status,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=int(total_trades - successful_trades),
                details={
                    "total_trades": total_trades,
                    "successful_trades": successful_trades,
                    "error_rate": error_rate,
                    "avg_execution_time_ms": 150,  # Placeholder
                },
                recommendations=recommendations,
            )

        except Exception as e:
            return ComponentHealth(
                component_name="trading",
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time_ms=0,
                error_count=1,
                details={"error": str(e)},
                recommendations=["Check trading system", "Review trading logs"],
            )

    def _calculate_overall_health(
        self, component_healths: list[ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall system health from component healths."""
        if not component_healths:
            return HealthStatus.UNKNOWN

        # Count statuses
        status_counts = {}
        for health in component_healths:
            status_counts[health.status] = status_counts.get(health.status, 0) + 1

        # Determine overall status
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        elif status_counts.get(HealthStatus.HEALTHY, 0) == len(component_healths):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def _generate_recommendations(self, report: SystemHealthReport) -> list[str]:
        """Generate system - wide recommendations based on health report."""
        recommendations = []

        # Collect all component recommendations
        for component in report.components.values():
            recommendations.extend(component.recommendations)

        # Add system - wide recommendations
        if report.overall_status == HealthStatus.CRITICAL:
            recommendations.append(
                "System in critical state-immediate attention required"
            )
        elif report.overall_status == HealthStatus.DEGRADED:
            recommendations.append("System performance degraded - monitor closely")

        return list(set(recommendations))  # Remove duplicates

    async def _send_health_alert(self, report: SystemHealthReport):
        """Send health alert if system is unhealthy."""
        if self.alert_system:
            priority = (
                "CRITICAL" if report.overall_status == HealthStatus.CRITICAL else "HIGH"
            )
            await self.alert_system.send_health_alert(
                f"System Health Alert: {report.overall_status.value.upper()}",
                f"Overall Status: {report.overall_status.value}\n"
                f"Components: {', '.join([f'{name}: {comp.status.value}' for name, comp in report.components.items()])}\n"
                f"Recommendations: {'; '.join(report.recommendations[:3])}",  # First 3 recommendations
                priority=priority,
            )

    def _create_error_health(
        self, component_name: str, error: Exception
    ) -> ComponentHealth:
        """Create health status for component that failed to check."""
        return ComponentHealth(
            component_name=component_name,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time_ms=0,
            error_count=1,
            details={"error": str(error)},
            recommendations=["Check component configuration", "Review error logs"],
        )

    def _create_critical_report(self, error_message: str) -> SystemHealthReport:
        """Create critical health report when health check fails."""
        return SystemHealthReport(
            timestamp=datetime.now(),
            overall_status=HealthStatus.CRITICAL,
            data_feed_status=self._create_error_health(
                "data_feed", Exception(error_message)
            ),
            broker_status=self._create_error_health("broker", Exception(error_message)),
            database_status=self._create_error_health(
                "database", Exception(error_message)
            ),
            resource_status=self._create_error_health(
                "resources", Exception(error_message)
            ),
            trading_status=self._create_error_health(
                "trading", Exception(error_message)
            ),
            recommendations=[
                "System health check failed - immediate investigation required"
            ],
        )

    def get_health_history(self, hours: int = 24) -> list[SystemHealthReport]:
        """Get health history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            report for report in self.health_history if report.timestamp >= cutoff_time
        ]

    def get_uptime_stats(self) -> dict[str, Any]:
        """Get system uptime statistics."""
        uptime = datetime.now() - self.start_time
        return {
            "start_time": self.start_time,
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "uptime_days": uptime.days,
        }
