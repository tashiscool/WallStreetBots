"""
Phase 4: Advanced Monitoring and Alerting System
Production - grade monitoring with real - time alerts and metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
from collections import defaultdict, deque

from ..core.production_logging import ProductionLogger
from ..core.production_config import ConfigManager


class AlertLevel(Enum): 
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum): 
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class AlertRule: 
    """Alert rule configuration"""
    name: str
    condition: str  # Python expression to evaluate
    threshold: float
    comparison: str  # "gt", "lt", "eq", "gte", "lte"
    level: AlertLevel
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class Metric: 
    """Metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType=MetricType.GAUGE


@dataclass
class Alert: 
    """Alert data structure"""
    id: str
    rule_name: str
    level: AlertLevel
    message: str
    timestamp: datetime
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector: 
    """Metrics collection and storage"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.metrics=defaultdict(lambda: deque(maxlen=10000))  # Store last 10k metrics per name
        self.counters=defaultdict(float)
        self.gauges=defaultdict(float)
        self.histograms=defaultdict(list)
        self.summaries=defaultdict(list)
        
        self.logger.info("MetricsCollector initialized")
    
    def record_metric(self, metric: Metric):
        """Record a metric"""
        try: 
            self.metrics[metric.name].append(metric)
            
            # Update specific metric type storage
            if metric.metric_type ==  MetricType.COUNTER: 
                self.counters[metric.name] += metric.value
            elif metric.metric_type  ==  MetricType.GAUGE: 
                self.gauges[metric.name] = metric.value
            elif metric.metric_type  ==  MetricType.HISTOGRAM: 
                self.histograms[metric.name].append(metric.value)
            elif metric.metric_type ==  MetricType.SUMMARY: 
                self.summaries[metric.name].append(metric.value)
            
            self.logger.debug(f"Recorded metric: {metric.name} = {metric.value}")
            
        except Exception as e: 
            self.logger.error(f"Error recording metric: {e}")
    
    def get_metric_value(self, name: str, metric_type: MetricType=MetricType.GAUGE)->float:
        """Get current metric value"""
        try: 
            if metric_type ==  MetricType.COUNTER: 
                return self.counters.get(name, 0.0)
            elif metric_type ==  MetricType.GAUGE: 
                return self.gauges.get(name, 0.0)
            elif metric_type ==  MetricType.HISTOGRAM: 
                values = self.histograms.get(name, [])
                return sum(values) / len(values) if values else 0.0
            elif metric_type ==  MetricType.SUMMARY: 
                values = self.summaries.get(name, [])
                return sum(values) / len(values) if values else 0.0
            
            return 0.0
            
        except Exception as e: 
            self.logger.error(f"Error getting metric value: {e}")
            return 0.0
    
    def get_metric_history(self, name: str, minutes: int=60)->List[Metric]:
        """Get metric history for specified time period"""
        try: 
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            metrics = self.metrics.get(name, deque())
            
            return [m for m in metrics if m.timestamp  >=  cutoff_time]
            
        except Exception as e: 
            self.logger.error(f"Error getting metric history: {e}")
            return []
    
    def get_metric_stats(self, name: str, minutes: int=60)->Dict[str, float]: 
        """Get metric statistics for specified time period"""
        try: 
            history = self.get_metric_history(name, minutes)
            
            if not history: 
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "sum": 0}
            
            values = [m.value for m in history]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "sum": sum(values)
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting metric stats: {e}")
            return {"count": 0, "min": 0, "max": 0, "avg": 0, "sum": 0}


class AlertManager: 
    """Alert management and evaluation"""
    
    def __init__(self, metrics_collector: MetricsCollector, logger: ProductionLogger):
        self.metrics_collector = metrics_collector
        self.logger = logger
        self.alert_rules={}
        self.active_alerts={}
        self.alert_handlers={}
        
        self.logger.info("AlertManager initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        try: 
            self.alert_rules[rule.name] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
            
        except Exception as e: 
            self.logger.error(f"Error adding alert rule: {e}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        try: 
            if rule_name in self.alert_rules: 
                del self.alert_rules[rule_name]
                self.logger.info(f"Removed alert rule: {rule_name}")
            
        except Exception as e: 
            self.logger.error(f"Error removing alert rule: {e}")
    
    def register_alert_handler(self, level: AlertLevel, handler: Callable[[Alert], None]): 
        """Register an alert handler for a specific level"""
        try: 
            if level not in self.alert_handlers: 
                self.alert_handlers[level] = []
            
            self.alert_handlers[level].append(handler)
            self.logger.info(f"Registered alert handler for {level.value}")
            
        except Exception as e: 
            self.logger.error(f"Error registering alert handler: {e}")
    
    async def evaluate_alerts(self): 
        """Evaluate all alert rules"""
        try: 
            for rule_name, rule in self.alert_rules.items(): 
                if not rule.enabled: 
                    continue
                
                # Check cooldown
                if rule.last_triggered: 
                    time_since_triggered = datetime.now() - rule.last_triggered
                    if time_since_triggered.total_seconds()  <  rule.cooldown_minutes * 60: 
                        continue
                
                # Evaluate rule condition
                if await self._evaluate_rule(rule): 
                    await self._trigger_alert(rule)
                else: 
                    await self._resolve_alert(rule_name)
            
        except Exception as e: 
            self.logger.error(f"Error evaluating alerts: {e}")
    
    async def _evaluate_rule(self, rule: AlertRule)->bool:
        """Evaluate a single alert rule"""
        try: 
            # Get metric value
            metric_value = self.metrics_collector.get_metric_value(rule.condition)
            
            # Apply comparison
            if rule.comparison ==  "gt": return metric_value  >  rule.threshold
            elif rule.comparison  ==  "lt": return metric_value  <  rule.threshold
            elif rule.comparison  ==  "eq": return metric_value  ==  rule.threshold
            elif rule.comparison  ==  "gte": return metric_value  >=  rule.threshold
            elif rule.comparison  ==  "lte": return metric_value  <=  rule.threshold
            
            return False
            
        except Exception as e: 
            self.logger.error(f"Error evaluating rule {rule.name}: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert"""
        try: 
            alert_id = f"{rule.name}_{int(time.time())}"
            metric_value = self.metrics_collector.get_metric_value(rule.condition)
            
            alert = Alert(
                id=alert_id,
                rule_name = rule.name,
                level = rule.level,
                message = f"Alert triggered: {rule.condition} {rule.comparison} {rule.threshold} (current: {metric_value:.2f})",
                timestamp = datetime.now(),
                value=metric_value,
                threshold = rule.threshold
            )
            
            self.active_alerts[rule.name] = alert
            rule.last_triggered=datetime.now()
            
            # Send to handlers
            await self._send_alert(alert)
            
            self.logger.warning(f"Alert triggered: {rule.name} - {alert.message}")
            
        except Exception as e: 
            self.logger.error(f"Error triggering alert: {e}")
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an alert"""
        try: 
            if rule_name in self.active_alerts: 
                alert = self.active_alerts[rule_name]
                alert.resolved = True
                alert.resolved_at=datetime.now()
                
                del self.active_alerts[rule_name]
                
                self.logger.info(f"Alert resolved: {rule_name}")
                
        except Exception as e: 
            self.logger.error(f"Error resolving alert: {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert to registered handlers"""
        try: 
            handlers = self.alert_handlers.get(alert.level, [])
            
            for handler in handlers: 
                try: 
                    await handler(alert)
                except Exception as e: 
                    self.logger.error(f"Error in alert handler: {e}")
            
        except Exception as e: 
            self.logger.error(f"Error sending alert: {e}")


class SystemMonitor: 
    """System monitoring and health checks"""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 alert_manager: AlertManager,
                 config: ConfigManager,
                 logger: ProductionLogger):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.config = config
        self.logger = logger
        
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("SystemMonitor initialized")
    
    def start_monitoring(self): 
        """Start system monitoring"""
        try: 
            if self.monitoring_active: 
                self.logger.warning("Monitoring already active")
                return
            
            self.monitoring_active = True
            self.monitoring_thread=threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("System monitoring started")
            
        except Exception as e: 
            self.logger.error(f"Error starting monitoring: {e}")
    
    def stop_monitoring(self): 
        """Stop system monitoring"""
        try: 
            self.monitoring_active = False
            
            if self.monitoring_thread: 
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("System monitoring stopped")
            
        except Exception as e: 
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def _monitoring_loop(self): 
        """Main monitoring loop"""
        try: 
            while self.monitoring_active: 
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect trading metrics
                self._collect_trading_metrics()
                
                # Evaluate alerts
                asyncio.run(self.alert_manager.evaluate_alerts())
                
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
        except Exception as e: 
            self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self): 
        """Collect system - level metrics"""
        try: 
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.record_metric(Metric(
                name = "system.cpu.usage",
                value=cpu_percent,
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(Metric(
                name = "system.memory.usage",
                value = memory.percent,
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics_collector.record_metric(Metric(
                name = "system.disk.usage",
                value=disk_percent,
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
        except ImportError: 
            # psutil not available, use mock metrics
            self.metrics_collector.record_metric(Metric(
                name = "system.cpu.usage",
                value = 25.0,  # Mock CPU usage
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
            self.metrics_collector.record_metric(Metric(
                name = "system.memory.usage",
                value = 60.0,  # Mock memory usage
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
        except Exception as e: 
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_trading_metrics(self): 
        """Collect trading - specific metrics"""
        try: 
            # Mock trading metrics - in production, these would come from actual trading data
            self.metrics_collector.record_metric(Metric(
                name = "trading.portfolio.value",
                value = 100000.0,  # Mock portfolio value
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
            self.metrics_collector.record_metric(Metric(
                name = "trading.positions.count",
                value = 5,  # Mock position count
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
            self.metrics_collector.record_metric(Metric(
                name = "trading.trades.today",
                value = 12,  # Mock daily trade count
                timestamp = datetime.now(),
                metric_type = MetricType.COUNTER
            ))
            
            self.metrics_collector.record_metric(Metric(
                name = "trading.pnl.daily",
                value = 1500.0,  # Mock daily P & L
                timestamp = datetime.now(),
                metric_type = MetricType.GAUGE
            ))
            
        except Exception as e: 
            self.logger.error(f"Error collecting trading metrics: {e}")


class AlertHandlers: 
    """Built - in alert handlers"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
    
    async def slack_alert_handler(self, alert: Alert):
        """Send alert to Slack"""
        try: 
            # Mock Slack integration - in production, use actual Slack webhook
            self.logger.info(f"SLACK ALERT [{alert.level.value.upper()}]: {alert.message}")
            
        except Exception as e: 
            self.logger.error(f"Error sending Slack alert: {e}")
    
    async def email_alert_handler(self, alert: Alert):
        """Send alert via email"""
        try: 
            # Mock email integration - in production, use actual SMTP
            self.logger.info(f"EMAIL ALERT [{alert.level.value.upper()}]: {alert.message}")
            
        except Exception as e: 
            self.logger.error(f"Error sending email alert: {e}")
    
    async def webhook_alert_handler(self, alert: Alert):
        """Send alert to webhook"""
        try: 
            # Mock webhook integration - in production, use actual HTTP request
            self.logger.info(f"WEBHOOK ALERT [{alert.level.value.upper()}]: {alert.message}")
            
        except Exception as e: 
            self.logger.error(f"Error sending webhook alert: {e}")


class MonitoringDashboard: 
    """Monitoring dashboard and reporting"""
    
    def __init__(self, 
                 metrics_collector: MetricsCollector,
                 alert_manager: AlertManager,
                 logger: ProductionLogger):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.logger = logger
        
        self.logger.info("MonitoringDashboard initialized")
    
    def get_dashboard_data(self)->Dict[str, Any]: 
        """Get dashboard data"""
        try: 
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "system_health": self._get_system_health(),
                "trading_metrics": self._get_trading_metrics(),
                "active_alerts": self._get_active_alerts(),
                "recent_alerts": self._get_recent_alerts(),
                "performance_metrics": self._get_performance_metrics()
            }
            
            return dashboard_data
            
        except Exception as e: 
            self.logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def _get_system_health(self)->Dict[str, Any]: 
        """Get system health metrics"""
        try: 
            cpu_usage = self.metrics_collector.get_metric_value("system.cpu.usage")
            memory_usage = self.metrics_collector.get_metric_value("system.memory.usage")
            disk_usage = self.metrics_collector.get_metric_value("system.disk.usage")
            
            health_status = "healthy"
            if cpu_usage  >  80 or memory_usage  >  80 or disk_usage  >  90: 
                health_status = "warning"
            if cpu_usage  >  95 or memory_usage  >  95 or disk_usage  >  95: 
                health_status = "critical"
            
            return {
                "status": health_status,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting system health: {e}")
            return {"status": "unknown", "error": str(e)}
    
    def _get_trading_metrics(self)->Dict[str, Any]: 
        """Get trading metrics"""
        try: 
            portfolio_value = self.metrics_collector.get_metric_value("trading.portfolio.value")
            position_count = self.metrics_collector.get_metric_value("trading.positions.count")
            daily_trades = self.metrics_collector.get_metric_value("trading.trades.today")
            daily_pnl = self.metrics_collector.get_metric_value("trading.pnl.daily")
            
            return {
                "portfolio_value": portfolio_value,
                "position_count": position_count,
                "daily_trades": daily_trades,
                "daily_pnl": daily_pnl
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting trading metrics: {e}")
            return {"error": str(e)}
    
    def _get_active_alerts(self)->List[Dict[str, Any]]: 
        """Get active alerts"""
        try: 
            active_alerts = []
            
            for alert in self.alert_manager.active_alerts.values(): 
                active_alerts.append({
                    "id": alert.id,
                    "rule_name": alert.rule_name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "value": alert.value,
                    "threshold": alert.threshold
                })
            
            return active_alerts
            
        except Exception as e: 
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    def _get_recent_alerts(self, limit: int=10)->List[Dict[str, Any]]: 
        """Get recent alerts"""
        try: 
            # Mock recent alerts - in production, this would come from a database
            recent_alerts = [
                {
                    "id": "alert_1",
                    "rule_name": "high_cpu_usage",
                    "level": "warning",
                    "message": "CPU usage above 80%",
                    "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                    "resolved": True
                },
                {
                    "id": "alert_2",
                    "rule_name": "low_memory",
                    "level": "error",
                    "message": "Memory usage above 90%",
                    "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                    "resolved": False
                }
            ]
            
            return recent_alerts[: limit]
            
        except Exception as e: 
            self.logger.error(f"Error getting recent alerts: {e}")
            return []
    
    def _get_performance_metrics(self)->Dict[str, Any]: 
        """Get performance metrics"""
        try: 
            # Get metrics for last hour
            cpu_stats = self.metrics_collector.get_metric_stats("system.cpu.usage", 60)
            memory_stats = self.metrics_collector.get_metric_stats("system.memory.usage", 60)
            portfolio_stats = self.metrics_collector.get_metric_stats("trading.portfolio.value", 60)
            
            return {
                "cpu": cpu_stats,
                "memory": memory_stats,
                "portfolio": portfolio_stats
            }
            
        except Exception as e: 
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}


class Phase4Monitoring: 
    """Main Phase 4 monitoring orchestrator"""
    
    def __init__(self, config: ConfigManager, logger: ProductionLogger):
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.metrics_collector=MetricsCollector(logger)
        self.alert_manager=AlertManager(self.metrics_collector, logger)
        self.system_monitor = SystemMonitor(
            self.metrics_collector, 
            self.alert_manager, 
            config, 
            logger
        )
        self.alert_handlers=AlertHandlers(logger)
        self.dashboard = MonitoringDashboard(
            self.metrics_collector, 
            self.alert_manager, 
            logger
        )
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Register alert handlers
        self._register_alert_handlers()
        
        self.logger.info("Phase4Monitoring initialized")
    
    def _setup_default_alert_rules(self): 
        """Setup default alert rules"""
        try: 
            # System alerts
            self.alert_manager.add_alert_rule(AlertRule(
                name = "high_cpu_usage",
                condition = "system.cpu.usage",
                threshold = 80.0,
                comparison = "gt",
                level = AlertLevel.WARNING,
                cooldown_minutes = 5
            ))
            
            self.alert_manager.add_alert_rule(AlertRule(
                name = "high_memory_usage",
                condition = "system.memory.usage",
                threshold = 85.0,
                comparison = "gt",
                level = AlertLevel.ERROR,
                cooldown_minutes = 10
            ))
            
            self.alert_manager.add_alert_rule(AlertRule(
                name = "high_disk_usage",
                condition = "system.disk.usage",
                threshold = 90.0,
                comparison = "gt",
                level = AlertLevel.CRITICAL,
                cooldown_minutes = 15
            ))
            
            # Trading alerts
            self.alert_manager.add_alert_rule(AlertRule(
                name = "low_portfolio_value",
                condition = "trading.portfolio.value",
                threshold = 50000.0,
                comparison = "lt",
                level = AlertLevel.WARNING,
                cooldown_minutes = 30
            ))
            
            self.alert_manager.add_alert_rule(AlertRule(
                name = "high_daily_loss",
                condition = "trading.pnl.daily",
                threshold = -5000.0,
                comparison = "lt",
                level = AlertLevel.ERROR,
                cooldown_minutes = 60
            ))
            
            self.logger.info("Default alert rules configured")
            
        except Exception as e: 
            self.logger.error(f"Error setting up default alert rules: {e}")
    
    def _register_alert_handlers(self): 
        """Register alert handlers"""
        try: 
            # Register handlers for each alert level
            for level in AlertLevel: 
                self.alert_manager.register_alert_handler(level, self.alert_handlers.slack_alert_handler)
                self.alert_manager.register_alert_handler(level, self.alert_handlers.email_alert_handler)
                self.alert_manager.register_alert_handler(level, self.alert_handlers.webhook_alert_handler)
            
            self.logger.info("Alert handlers registered")
            
        except Exception as e: 
            self.logger.error(f"Error registering alert handlers: {e}")
    
    def start_monitoring(self): 
        """Start the monitoring system"""
        try: 
            self.system_monitor.start_monitoring()
            self.logger.info("Phase 4 monitoring system started")
            
        except Exception as e: 
            self.logger.error(f"Error starting monitoring: {e}")
    
    def stop_monitoring(self): 
        """Stop the monitoring system"""
        try: 
            self.system_monitor.stop_monitoring()
            self.logger.info("Phase 4 monitoring system stopped")
            
        except Exception as e: 
            self.logger.error(f"Error stopping monitoring: {e}")
    
    def get_dashboard_data(self)->Dict[str, Any]: 
        """Get monitoring dashboard data"""
        return self.dashboard.get_dashboard_data()
    
    def add_custom_metric(self, name: str, value: float, metric_type: MetricType=MetricType.GAUGE, 
                         labels: Dict[str, str] = None): 
        """Add a custom metric"""
        try: 
            metric = Metric(
                name=name,
                value=value,
                timestamp = datetime.now(),
                labels = labels or {},
                metric_type = metric_type)
            
            self.metrics_collector.record_metric(metric)
            
        except Exception as e: 
            self.logger.error(f"Error adding custom metric: {e}")
    
    def add_custom_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_manager.add_alert_rule(rule)
