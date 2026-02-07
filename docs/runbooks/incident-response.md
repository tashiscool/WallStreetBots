# Incident Response Procedure

## Severity Levels

| Level | Definition | Response Time | Examples |
|-------|-----------|---------------|----------|
| **SEV-1** | Trading halted or significant financial impact | Immediate | Circuit breaker open, reconciliation breaks > $1K, data corruption |
| **SEV-2** | Degraded trading or elevated risk | 15 minutes | High reject rate, stale data, VIX spike |
| **SEV-3** | Non-critical issue affecting monitoring | 1 hour | High latency, disk space warnings, missing journal entries |
| **SEV-4** | Informational, no trading impact | Next business day | Low order volume, minor data outliers |

## Incident Lifecycle

### 1. Detection
- Prometheus alert fires → Alertmanager routes to email/Slack
- Manual observation of unexpected behavior
- Reconciliation report shows discrepancies

### 2. Triage (first 5 minutes)
- [ ] Acknowledge the alert
- [ ] Determine severity level
- [ ] Check: Is trading currently active?
- [ ] Check: Are there open positions at risk?
- [ ] If SEV-1: immediately halt trading via circuit breaker

### 3. Diagnosis (5-30 minutes)
- [ ] Identify the root cause using relevant runbook
- [ ] Check related systems (broker API, data feeds, database)
- [ ] Review recent deployments or configuration changes
- [ ] Gather logs and metrics for the incident timeline

### 4. Mitigation
- [ ] Apply fix or workaround
- [ ] Verify fix is effective (check metrics/alerts clear)
- [ ] If trading was halted: reconcile positions before resuming
- [ ] Resume trading in stages (restricted → cautious → normal)

### 5. Resolution
- [ ] Confirm all alerts have cleared
- [ ] Verify reconciliation passes clean
- [ ] Update monitoring if new failure mode discovered
- [ ] Notify stakeholders of resolution

### 6. Post-Incident (within 48 hours)
- [ ] Write postmortem (use template: `docs/runbooks/postmortem-template.md`)
- [ ] Identify action items to prevent recurrence
- [ ] Update runbooks if procedures were missing or wrong
- [ ] Review and tune alert thresholds if needed

## Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| Primary operator | (configure in .env) | Alertmanager email |
| Broker support | Alpaca: support@alpaca.markets | For account/API issues |
| Data provider | Per provider status page | For data feed outages |

## Emergency Commands

```bash
# Halt all trading immediately
python manage.py shell -c "
from tradingbot.risk.circuit_breaker import get_circuit_breaker
get_circuit_breaker().trip('manual_halt', 'Emergency halt by operator')
"

# Cancel all open orders
python manage.py shell -c "
from tradingbot.core.trading_interface import TradingInterface
ti = TradingInterface()
ti.cancel_all_orders()
"

# Check system health
python manage.py shell -c "
from tradingbot.monitoring.system_health import SystemHealthMonitor
m = SystemHealthMonitor()
print(m.full_health_check())
"

# Run position reconciliation
python manage.py shell -c "
from tradingbot.reconciliation.position_reconciler import PositionReconciler
print(PositionReconciler().reconcile())
"
```
