"""
Production Logging and Error Handling
Robust error handling with retry mechanisms and structured logging
"""

import asyncio
import logging
import structlog
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)
import json
import os


class ProductionLogger:
    """Production-grade logging system"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.logger = structlog.get_logger(name)
        self.setup_logging(log_level)
    
    def setup_logging(self, log_level: str):
        """Setup structured logging"""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup file handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(f"{log_dir}/{self.name}.log")
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.logger.critical(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(message, **kwargs)


class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}
        self.error_thresholds: Dict[str, int] = {
            'api_error': 10,
            'network_error': 5,
            'validation_error': 3,
            'broker_error': 5
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle error with context and tracking"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Increment error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Log error with context
        self.logger.error(
            f"Error occurred: {error_type}",
            error_type=error_type,
            error_message=error_message,
            error_count=self.error_counts[error_type],
            context=context or {},
            traceback=traceback.format_exc()
        )
        
        # Check if error threshold exceeded
        threshold = self.error_thresholds.get(error_type, 10)
        if self.error_counts[error_type] >= threshold:
            self.logger.critical(
                f"Error threshold exceeded for {error_type}",
                error_type=error_type,
                count=self.error_counts[error_type],
                threshold=threshold
            )
        
        return {
            'error_type': error_type,
            'error_message': error_message,
            'error_count': self.error_counts[error_type],
            'threshold_exceeded': self.error_counts[error_type] >= threshold
        }
    
    def reset_error_count(self, error_type: str):
        """Reset error count for specific error type"""
        self.error_counts[error_type] = 0
        self.logger.info(f"Reset error count for {error_type}")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logging.getLogger(func.__module__), logging.WARNING)
        )
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=base_delay, min=base_delay, max=max_delay),
            retry=retry_if_exception_type(exceptions),
            before_sleep=before_sleep_log(logging.getLogger(func.__module__), logging.WARNING)
        )
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = ProductionLogger("circuit_breaker")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.critical(
                f"Circuit breaker opened after {self.failure_count} failures",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_time: Optional[datetime] = None
        self.health_status: Dict[str, Any] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        self.last_check_time = datetime.now()
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'result': result,
                    'timestamp': self.last_check_time.isoformat()
                }
                
                self.logger.info(f"Health check {name}: {'healthy' if result else 'unhealthy'}")
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': self.last_check_time.isoformat()
                }
                
                self.logger.error(f"Health check {name} failed: {e}")
        
        self.health_status = results
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health status"""
        if not self.health_status:
            return 'unknown'
        
        unhealthy_count = sum(
            1 for check in self.health_status.values() 
            if check['status'] in ['unhealthy', 'error']
        )
        
        if unhealthy_count == 0:
            return 'healthy'
        elif unhealthy_count < len(self.health_status) / 2:
            return 'degraded'
        else:
            return 'unhealthy'


class MetricsCollector:
    """Collect and store system metrics"""
    
    def __init__(self, logger: ProductionLogger):
        self.logger = logger
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.max_metrics_per_type = 1000
    
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        metric_data = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'tags': tags or {}
        }
        
        self.metrics[metric_name].append(metric_data)
        
        # Keep only recent metrics
        if len(self.metrics[metric_name]) > self.max_metrics_per_type:
            self.metrics[metric_name] = self.metrics[metric_name][-self.max_metrics_per_type:]
        
        self.logger.debug(f"Recorded metric {metric_name}: {value}", metric_name=metric_name, value=value, tags=tags)
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary for specified time window"""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now().timestamp() - (window_minutes * 60)
        
        recent_metrics = [
            m for m in self.metrics[metric_name]
            if datetime.fromisoformat(m['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m['value'] for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else None
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            self.logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Factory functions for easy initialization
def create_production_logger(name: str, log_level: str = "INFO") -> ProductionLogger:
    """Create production logger"""
    return ProductionLogger(name, log_level)


def create_error_handler(logger: ProductionLogger) -> ErrorHandler:
    """Create error handler"""
    return ErrorHandler(logger)


def create_circuit_breaker(failure_threshold: int = 5, timeout: float = 60.0) -> CircuitBreaker:
    """Create circuit breaker"""
    return CircuitBreaker(failure_threshold, timeout)


def create_health_checker(logger: ProductionLogger) -> HealthChecker:
    """Create health checker"""
    return HealthChecker(logger)


def create_metrics_collector(logger: ProductionLogger) -> MetricsCollector:
    """Create metrics collector"""
    return MetricsCollector(logger)
