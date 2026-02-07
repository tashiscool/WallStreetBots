# Disaster Recovery Plan

## Recovery Objectives

| Metric | Target | Rationale |
|--------|--------|-----------|
| **RTO** (Recovery Time Objective) | 30 minutes | Must recover before next market session |
| **RPO** (Recovery Point Objective) | 1 hour | Max acceptable data loss |
| **MTTR** (Mean Time to Recovery) | 15 minutes | Target for practiced scenarios |

## Backup Strategy

### Database Backups
- **Frequency:** Every 6 hours + pre-market daily
- **Retention:** 30 days rolling
- **Location:** `backups/` directory (configure offsite in production)
- **Script:** `scripts/backup.sh`

### State Files
- **ReplayGuard state:** `.state/replay_guard.json`
- **Circuit breaker state:** Persisted in database
- **EOD recon breaks:** `.state/eod_recon_breaks.json`

### Configuration
- **Environment:** `.env` file (encrypted backup recommended)
- **Strategy configs:** Version-controlled in git

## Disaster Scenarios

### Scenario 1: Database Corruption
**Impact:** Loss of position tracking, trade history, audit trail

**Recovery:**
1. Halt trading: `python manage.py shell -c "from tradingbot.risk.circuit_breaker import get_circuit_breaker; get_circuit_breaker().trip('dr', 'DB corruption')"`
2. Restore from backup: `bash scripts/restore.sh backups/latest.sql.gz`
3. Run migrations: `python manage.py migrate`
4. Reconcile positions against broker: `python manage.py shell -c "from tradingbot.reconciliation.position_reconciler import PositionReconciler; print(PositionReconciler().reconcile())"`
5. Verify data integrity
6. Resume trading

### Scenario 2: Application Server Failure
**Impact:** Trading halted, no monitoring

**Recovery:**
1. Open orders at broker continue to work (stops, limits)
2. Restart application: `bash scripts/run.sh`
3. Circuit breaker state auto-restores from database
4. ReplayGuard prevents duplicate orders on restart
5. Run reconciliation to catch any missed fills
6. Resume trading

### Scenario 3: Broker API Outage
**Impact:** Cannot place/cancel orders, data feed may be stale

**Recovery:**
1. Circuit breaker auto-triggers on stale data / error rate
2. Open orders at broker remain active
3. Monitor broker status page
4. When API recovers:
   - Run reconciliation to sync fills
   - Clear circuit breaker
   - Resume in restricted mode first

### Scenario 4: Network Failure
**Impact:** Complete loss of connectivity

**Recovery:**
1. Circuit breaker auto-triggers on data staleness
2. Existing broker-side orders (stops, limits) remain active
3. Restore network connectivity
4. Verify all connections (broker, data feeds, database)
5. Reconcile and resume

## DR Test Procedure

Run quarterly DR tests using `scripts/dr_test.py`:

```bash
# Full DR test (non-destructive, uses test database)
python scripts/dr_test.py --full

# Individual scenario tests
python scripts/dr_test.py --scenario backup_restore
python scripts/dr_test.py --scenario circuit_breaker_recovery
python scripts/dr_test.py --scenario state_persistence
```

### DR Test Checklist
- [ ] Backup creates valid database dump
- [ ] Restore from backup produces working database
- [ ] Migrations run cleanly on restored database
- [ ] Circuit breaker state persists across restart
- [ ] ReplayGuard prevents duplicate orders after restart
- [ ] Position reconciliation detects and reports discrepancies
- [ ] Recovery manager steps through stages correctly
- [ ] All health checks pass after recovery

## DR Test Log

| Date | Scenario | Result | Notes | Tester |
|------|----------|--------|-------|--------|
| *(Record results here after each test)* | | | | |
