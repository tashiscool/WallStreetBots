# Runbook: EOD Reconciliation Breaks

**Alert:** `ReconciliationBreaks`
**Severity:** Critical
**Component:** compliance
**Prometheus expr:** `wsb_eod_reconciliation_breaks_total > 0`

## Impact

End-of-day reconciliation detected discrepancies between internal position tracking and broker-reported positions. This may indicate missed fills, phantom positions, or accounting errors.

## Diagnosis

1. Check the reconciliation report:
   ```bash
   python manage.py shell -c "
   from tradingbot.reconciliation.position_reconciler import PositionReconciler
   r = PositionReconciler()
   report = r.reconcile()
   for d in report.get('discrepancies', []):
       print(f'{d[\"type\"]}: {d[\"symbol\"]} — {d[\"detail\"]}')
   "
   ```

2. Break types and severity:

   | Type | Description | Auto-Halt? |
   |------|-------------|------------|
   | `MISSING_AT_BROKER` | Internal DB has position, broker doesn't | Yes |
   | `MISSING_IN_DB` | Broker has position, DB doesn't | Yes |
   | `QUANTITY_MISMATCH` | Quantities disagree | If > 10% |
   | `PRICE_MISMATCH` | Average prices disagree significantly | No |
   | `STALE_POSITION` | Position untouched for 24+ hours | No |

3. Check the EOD recon history:
   ```bash
   cat .state/eod_recon_breaks.json | python -m json.tool
   ```

## Resolution

### Missing at broker (DB has position, broker doesn't):
1. Check if a fill was missed (order filled but callback lost)
2. Query broker for recent fills: `GET /v2/orders?status=filled&after=<date>`
3. If position was closed by broker (margin call, corporate action):
   - Update DB to reflect closure
   - Record the P&L impact
4. If position never existed (phantom):
   - Remove from DB
   - Investigate how it was created

### Missing in DB (broker has position, DB doesn't):
1. Check if an order was placed outside the system (manual trade)
2. Query broker for the order that created the position
3. Add position to DB with correct entry price
4. Flag for review

### Quantity mismatch:
1. Check partial fills that may not have been recorded
2. Verify split/reverse-split adjustments
3. Correct the DB quantity to match broker
4. Re-run P&L calculations

### Resolution steps:
```bash
# After fixing discrepancies, re-run reconciliation
python manage.py shell -c "
from tradingbot.reconciliation.position_reconciler import PositionReconciler
r = PositionReconciler()
report = r.reconcile()
print(f'Breaks remaining: {report.get(\"break_count\", 0)}')
"
```

## Escalation

- If financial impact > $1,000: halt trading until resolved
- If 3+ positions missing: halt trading, full manual reconciliation
- Document all corrections in the audit log
