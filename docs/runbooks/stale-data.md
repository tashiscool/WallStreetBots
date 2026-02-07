# Runbook: Stale Data Feed

**Alert:** `StaleDataFeed`
**Severity:** Critical
**Component:** data
**Prometheus expr:** `wsb_data_staleness_seconds > 20`

## Impact

Market data is more than 20 seconds old. Strategies are making decisions on stale prices, which can lead to incorrect entry/exit signals and adverse fills.

## Diagnosis

1. Check data feed health:
   ```bash
   python manage.py shell -c "
   from tradingbot.monitoring.system_health import SystemHealthMonitor
   m = SystemHealthMonitor()
   print(m.check_data_feed_health())
   "
   ```

2. Identify which data source is stale:
   ```bash
   grep "stale\|staleness\|timeout" logs/trading.log | tail -20
   ```

3. Check data source connectivity:
   | Source | Health Check |
   |--------|-------------|
   | Alpaca | `curl -s https://paper-api.alpaca.markets/v2/clock -H "APCA-API-KEY-ID: $KEY"` |
   | Yahoo | `curl -s "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1m&range=1d"` |
   | Polygon | `curl -s "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=$KEY"` |

4. Check system resources:
   ```bash
   # High CPU or memory can cause data processing delays
   top -l 1 | head -10
   ```

## Resolution

### Single source stale:
1. Check if the source API is experiencing outages
2. Try reconnecting: restart the data provider connection
3. If source is down, failover to backup source (DataSourceManager handles this)

### All sources stale:
1. Check network connectivity: `ping 8.8.8.8`
2. Check DNS resolution: `nslookup api.alpaca.markets`
3. Check if firewall rules changed
4. Restart the application if network is confirmed OK

### During market hours:
1. If data is stale > 60 seconds, circuit breaker should auto-trigger
2. Do NOT manually override the staleness check
3. Wait for data to recover, then verify with live quote

### Outside market hours:
1. Staleness alerts are expected when markets are closed
2. Verify this is not a false positive by checking market calendar
3. Suppress alert if confirmed outside hours

## Prevention

- Configure DataSourceManager with multiple fallback sources
- Set up health checks for each data provider
- Monitor API rate limit consumption to avoid throttling
