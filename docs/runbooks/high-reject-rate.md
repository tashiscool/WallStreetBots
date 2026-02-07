# Runbook: High Order Reject Rate

**Alert:** `HighRejectRate`
**Severity:** Critical
**Component:** execution
**Prometheus expr:** `rate(wsb_orders_rejected_total[5m]) > 3`

## Impact

Orders are being rejected by the broker at an elevated rate (>3/min). Strategies may be unable to enter/exit positions.

## Diagnosis

1. Check recent rejections:
   ```bash
   grep "rejected\|REJECTED" logs/trading.log | tail -30
   ```

2. Common rejection reasons:

   | Broker Response | Cause | Fix |
   |-----------------|-------|-----|
   | `insufficient_buying_power` | Not enough cash/margin | Reduce position sizes or add funds |
   | `invalid_qty` | Quantity too small/large or fractional shares | Check order sizing logic |
   | `market_closed` | Trading outside market hours | Check scheduling configuration |
   | `symbol_not_found` | Delisted or invalid ticker | Update universe, remove symbol |
   | `duplicate_order` | Replay guard failure | Check ReplayGuard state file |
   | `rate_limit` | Too many API calls | Throttle order submission rate |
   | `account_restricted` | Account PDT/frozen | Contact broker |

3. Check broker API status:
   ```bash
   python manage.py shell -c "
   from tradingbot.core.trading_interface import TradingInterface
   ti = TradingInterface()
   print(ti.check_connection())
   "
   ```

4. Check if rejections are symbol-specific or global:
   ```bash
   grep "rejected" logs/trading.log | awk '{print $NF}' | sort | uniq -c | sort -rn | head -10
   ```

## Resolution

### Buying power issues:
1. Check account balance via broker API
2. Review open orders consuming buying power
3. Cancel unnecessary open orders
4. Reduce position sizes in strategy config

### Market hours issues:
1. Verify `TradingScheduler` configuration
2. Check if market holiday calendar is up to date
3. Confirm timezone settings

### Rate limiting:
1. Check order submission rate in metrics
2. Increase delay between orders in strategy config
3. Batch orders where possible

### Broker API issues:
1. Check broker status page (e.g., status.alpaca.markets)
2. If API is down, circuit breaker should auto-trigger
3. Wait for recovery, then verify with test order

## Escalation

If reject rate persists after 15 minutes:
1. Trigger manual circuit breaker halt
2. Cancel all open orders
3. Reconcile positions
4. Contact broker support if needed
