# backend/tradingbot/infra/obs.py
import logging
import json
import time
from typing import Any, Dict

# Structured logging
LOG = logging.getLogger("wsb")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
LOG.addHandler(handler)

def jlog(event: str, **kwargs: Any) -> None:
    """Structured JSON logging"""
    log_data = {
        "ts": time.time(),
        "event": event,
        **kwargs
    }
    print(json.dumps(log_data, separators=(",", ":")))

# Metrics (simplified without prometheus for now)
class SimpleMetrics:
    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}

    def counter_inc(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value
        jlog("metric", type="counter", name=name, value=self._counters[name])

    def gauge_set(self, name: str, value: float) -> None:
        self._gauges[name] = value
        jlog("metric", type="gauge", name=name, value=value)

    def histogram_observe(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
        jlog("metric", type="histogram", name=name, value=value)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "histograms": {k: {"count": len(v), "sum": sum(v)} for k, v in self._histograms.items()}
        }

# Global metrics instance
metrics = SimpleMetrics()

# Common trading metrics
def track_order_placed(symbol: str, side: str, qty: float) -> None:
    metrics.counter_inc("orders_placed_total")
    jlog("order_placed", symbol=symbol, side=side, qty=qty)

def track_order_rejected(symbol: str, reason: str) -> None:
    metrics.counter_inc("orders_rejected_total")
    jlog("order_rejected", symbol=symbol, reason=reason)

def track_data_staleness(seconds: float) -> None:
    metrics.gauge_set("data_staleness_seconds", seconds)

def track_operation_latency(operation: str, duration: float) -> None:
    metrics.histogram_observe(f"{operation}_latency_seconds", duration)

def track_position_update(symbol: str, qty: float, value: float) -> None:
    jlog("position_update", symbol=symbol, qty=qty, value=value)

def track_risk_event(event_type: str, details: Dict[str, Any]) -> None:
    metrics.counter_inc(f"risk_events_{event_type}_total")
    jlog("risk_event", type=event_type, details=details)