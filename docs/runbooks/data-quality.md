# Runbook: Data Quality Failure

**Alert:** `DataQualityFailure`
**Severity:** Critical
**Component:** data
**Prometheus expr:** `increase(wsb_data_quality_failures_total[5m]) > 0`

## Impact

Data quality validation checks are failing. Prices, volumes, or other market data may be corrupted, which can cause incorrect trading decisions.

## Diagnosis

1. Check which quality checks failed:
   ```bash
   grep "quality.*fail\|validation.*fail\|outlier" logs/trading.log | tail -20
   ```

2. Common quality failures:

   | Check | Meaning | Severity |
   |-------|---------|----------|
   | `price_outlier` | Price moved >10 std devs in one bar | High — likely bad tick |
   | `volume_zero` | Zero volume during market hours | Medium — data gap |
   | `missing_bar` | Expected bar not received | Medium — data gap |
   | `negative_price` | Negative price received | Critical — data corruption |
   | `stale_timestamp` | Timestamp is hours/days old | High — feed replay |
   | `cross_source_mismatch` | Sources disagree by >2% | Medium — investigate |

3. Check specific symbol data:
   ```bash
   python manage.py shell -c "
   from tradingbot.data.quality import DataQualityMonitor
   m = DataQualityMonitor()
   print(m.get_recent_failures(limit=20))
   "
   ```

## Resolution

### Bad tick / price outlier:
1. Verify against alternative data source
2. If confirmed bad: the quality filter should have already discarded it
3. Check if any orders were placed on the bad data
4. If orders placed — review and cancel if still open

### Missing data / gaps:
1. Check if data source is experiencing issues
2. Try fetching the missing data manually
3. If gap is small (<5 bars), strategies should handle gracefully
4. If gap is large, consider halting the affected strategy

### Data corruption:
1. Stop trading immediately for affected symbols
2. Clear cached data for the symbol
3. Re-fetch from a clean source
4. Verify data integrity before resuming

## Prevention

- Enable cross-source validation (compare prices across 2+ sources)
- Set tight outlier thresholds for illiquid symbols
- Monitor data source API status pages
