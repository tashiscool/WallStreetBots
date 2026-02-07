# Runbook: Circuit Breaker Open

**Alert:** `CircuitBreakerOpen`
**Severity:** Critical
**Component:** circuit_breaker
**Prometheus expr:** `wsb_circuit_open == 1`

## Impact

All trading is halted. No new orders will be submitted to the broker. Existing open orders remain at the broker.

## Diagnosis

1. Check the breaker reason in logs:
   ```bash
   grep "circuit.*trip" logs/trading.log | tail -20
   ```

2. Identify the trigger:
   | Trigger | Log Pattern | Root Cause |
   |---------|-------------|------------|
   | Daily loss | `daily_loss_limit` | P&L exceeded threshold (default 6%) |
   | Error rate | `error_rate_limit` | 5+ errors/minute |
   | VIX spike | `vix_threshold` | VIX > 45 (critical) or > 35 (elevated) |
   | Manual | `manual_halt` | Operator triggered |
   | Drawdown | `drawdown_limit` | Peak-to-trough exceeded limit |

3. Check the database state:
   ```bash
   python manage.py shell -c "
   from auth0login.services.circuit_breaker_persistence import CircuitBreakerPersistence
   p = CircuitBreakerPersistence()
   print(p.get_current_state())
   "
   ```

## Resolution

### If triggered by daily loss limit:
1. Review positions: `python manage.py shell -c "from tradingbot.reconciliation.position_reconciler import PositionReconciler; print(PositionReconciler().get_positions())"`
2. Verify P&L calculation is correct (not a data error)
3. If legitimate losses — breaker auto-resets at next trading day
4. If data error — fix data, then manually reset

### If triggered by error rate:
1. Check error logs: `grep ERROR logs/trading.log | tail -50`
2. Common causes: broker API down, network issues, bad data feed
3. Fix underlying issue, then reset

### If triggered by VIX:
1. VIX > 45: Wait for VIX to normalize. Recovery manager will step through stages (paused → restricted → cautious → normal)
2. VIX 35-45: Position sizing auto-reduced to 50%. Monitor.

### Manual reset:
```bash
python manage.py shell -c "
from tradingbot.risk.circuit_breaker import get_circuit_breaker
cb = get_circuit_breaker()
cb.reset()
print('Circuit breaker reset')
"
```

## Post-Incident

- [ ] Document the trigger in the incident log
- [ ] Verify all positions are correctly reconciled
- [ ] Check if parameter tuning is needed (thresholds too tight/loose)
- [ ] File postmortem if losses exceeded 2% of portfolio
