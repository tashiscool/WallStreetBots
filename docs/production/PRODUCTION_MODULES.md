# Production Modules for WallStreetBots

This document describes the production-grade modules implemented to close critical gaps in the trading system.

## 🚨 High-Impact Modules Implemented

### 1. Circuit Breaker (`backend/tradingbot/risk/circuit_breaker.py`)
**Purpose**: Trip on realized/MTM loss, error spikes, or stale data; require manual reset.

**Features**:
- Daily drawdown protection (default 6%)
- Error rate monitoring (default 5 errors/minute)
- Data staleness detection (default 30 seconds)
- Configurable cooldown period (default 30 minutes)
- Manual reset capability

**Usage**:
```python
from backend.tradingbot.risk.circuit_breaker import CircuitBreaker, BreakerLimits

breaker = CircuitBreaker(start_equity=1000000.0)
breaker.require_ok()  # Call before every order
breaker.mark_error()  # Call on errors
breaker.mark_data_fresh()  # Call on every tick
```

### 2. Replay Guard (`backend/tradingbot/execution/replay_guard.py`)
**Purpose**: Exactly-once order replay guard - never re-send acknowledged client_order_id.

**Features**:
- Persistent order state tracking
- JSON-based storage
- Automatic cleanup capabilities
- Restart-safe operation

**Usage**:
```python
from backend.tradingbot.execution.replay_guard import ReplayGuard

guard = ReplayGuard()
if guard.seen("order_123"):
    raise RuntimeError("Order already processed")
guard.record("order_123", "acknowledged")
```

### 3. EOD Reconciliation (`backend/tradingbot/ops/eod_recon.py`)
**Purpose**: Broker statement/exec reconciliation, P&L attribution, position affirmations.

**Features**:
- Local orders vs broker fills reconciliation
- Break detection and categorization
- Next-day disable logic
- Comprehensive logging

**Usage**:
```python
from backend.tradingbot.ops.eod_recon import EODReconciler, LocalOrder, BrokerFill

reconciler = EODReconciler()
breaks = reconciler.run_daily_reconciliation(local_orders, broker_fills)
if reconciler.check_pending_breaks():
    disable_next_day()
```

### 4. Data Quality Monitor (`backend/tradingbot/data/quality.py`)
**Purpose**: Hard stops for staleness, outlier jumps, missing bars, corrupted caches.

**Features**:
- Real-time staleness monitoring
- Outlier detection using Z-scores
- OHLC relationship validation
- Missing data detection

**Usage**:
```python
from backend.tradingbot.data.quality import DataQualityMonitor

monitor = DataQualityMonitor(max_staleness_sec=20, max_return_z=6.0)
monitor.assert_fresh()  # Call before strategy decisions
monitor.check_all(bars, symbol)  # Comprehensive check
```

### 5. Greek Exposure Limits (`backend/tradingbot/risk/greek_exposure_limits.py`)
**Purpose**: Cap dollar, β-adjusted, and Greeks (Δ/Γ/Θ/ν) at portfolio and per-name levels.

**Features**:
- Portfolio-level Greek limits
- Per-name exposure limits
- Beta-adjusted exposure tracking
- Pre-trade validation

**Usage**:
```python
from backend.tradingbot.risk.greek_exposure_limits import GreekExposureLimiter, PositionGreeks

limiter = GreekExposureLimiter()
limiter.add_position(PositionGreeks("AAPL", delta=1000, gamma=50, theta=-10, vega=200, beta=1.2, notional=100000))
limiter.require_ok()  # Call before every order
```

### 6. Shadow Execution Client (`backend/tradingbot/execution/shadow_client.py`)
**Purpose**: Shadow trade (decide but don't place) + per-strategy canary allocation.

**Features**:
- Logs orders without submitting
- Canary allocation limits
- Daily order limits
- Performance monitoring

**Usage**:
```python
from backend.tradingbot.execution.shadow_client import ShadowExecutionClient, CanaryExecutionClient

shadow = ShadowExecutionClient(real_client)
canary = CanaryExecutionClient(real_client, canary_allocation_pct=0.1, max_daily_orders=10)
```

### 7. Build Info (`backend/tradingbot/infra/build_info.py`)
**Purpose**: Build/version stamp in every order/log for reproducibility.

**Features**:
- Git SHA extraction
- Build timestamp tracking
- Order and log stamping
- Context manager support

**Usage**:
```python
from backend.tradingbot.infra.build_info import build_id, BuildStamper

build_id_str = build_id()
with BuildStamper("trading") as stamper:
    stamped_data = stamper.stamp(order_data)
```

## 🧪 Testing

### Property-Based Tests (`tests/property/test_sizing_properties.py`)
**Purpose**: Property-based sizing test (rounding/limits/NANs).

**Features**:
- Hypothesis-based testing
- Edge case validation
- Monotonicity checks
- Error handling tests

**Usage**:
```bash
python -m pytest tests/property/test_sizing_properties.py -v
```

### Integration Tests (`tests/test_production_modules.py`)
**Purpose**: Comprehensive testing of all production modules.

**Features**:
- Unit tests for each module
- Integration scenarios
- Error condition testing
- Performance validation

## 🚨 Monitoring & Alerts

### Prometheus Alert Rules (`ops/alerts.yml`)
**Purpose**: JSON logs with trace_id, Prometheus SLOs, alert rules.

**Key Alerts**:
- Circuit breaker open
- High reject rate
- Stale data feed
- Data quality failures
- Risk limit approaching
- Reconciliation breaks

### CI Security Scanning (`.github/workflows/security.yml`)
**Purpose**: Pinned deps w/ hashes, SBOM, image scan (Trivy), secret scan (gitleaks).

**Features**:
- SBOM generation
- Filesystem security scanning
- Container image scanning
- Secret detection
- Dependency security checks
- License compliance

## 🔧 Minimal Wiring Order

### 1. Gate Every Order
```python
# Before any order:
data_quality.assert_fresh()
circuit_breaker.require_ok()
greek_limiter.require_ok()
replay_guard.seen(client_order_id)
```

### 2. On Every Tick
```python
# Update freshness and metrics:
data_quality.mark_tick()
circuit_breaker.poll()
update_prometheus_metrics()
```

### 3. At EOD
```python
# Run reconciliation:
breaks = eod_reconciler.run_daily_reconciliation(local_orders, broker_fills)
if breaks:
    disable_next_day()
```

### 4. Deploy
```bash
# With pinned deps + SBOM:
pip install -r requirements.txt
python -m pytest tests/
# Enable alerts and keep journals immutable
```

## 📊 Integration Example

See `backend/tradingbot/production/integration_example.py` for a complete example showing how to wire all modules together in a production trading system.

## 🎯 Key Benefits

1. **Risk Management**: Circuit breaker prevents catastrophic losses
2. **Data Integrity**: Quality monitoring ensures fresh, sane data
3. **Order Safety**: Replay guard prevents duplicate orders
4. **Compliance**: EOD reconciliation ensures regulatory compliance
5. **Observability**: Comprehensive monitoring and alerting
6. **Security**: Automated security scanning and SBOM generation
7. **Testing**: Property-based tests ensure robust edge case handling

## 🚀 Next Steps

1. **Wire into existing strategies**: Integrate modules into current trading strategies
2. **Configure alerts**: Set up Prometheus and alerting infrastructure
3. **Deploy security scanning**: Enable CI/CD security workflows
4. **Monitor in production**: Use shadow execution for canary testing
5. **Regular reconciliation**: Implement daily EOD reconciliation processes

All modules are production-ready and tested. They provide the critical safety mechanisms needed for institutional-grade trading systems.

