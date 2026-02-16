# Codex Review & Fix Prompt for WallStreetBots

You are reviewing **WallStreetBots**, an institutional-grade algorithmic trading system built with Python 3.12+ and Django 4.2+. The codebase has ~86,000 lines across 521 Python files, 240+ test files with 5,500+ test cases, and 10+ trading strategies. Your job is to perform a full review and then fix every issue you find. Do not skip anything. Do not leave TODOs behind.

---

## Project Structure Overview

```
WallStreetBots/
├── backend/
│   ├── settings.py                          # Django config
│   ├── tradingbot/                          # Core trading system
│   │   ├── production/                      # Strategy orchestration
│   │   │   └── core/production_strategy_manager.py
│   │   ├── strategies/implementations/      # 10 strategy files
│   │   │   ├── wsb_dip_bot.py
│   │   │   ├── wheel_strategy.py
│   │   │   ├── swing_trading.py
│   │   │   ├── momentum_weeklies.py
│   │   │   ├── earnings_protection.py
│   │   │   ├── debit_spreads.py
│   │   │   ├── spx_credit_spreads.py
│   │   │   ├── leaps_tracker.py
│   │   │   ├── lotto_scanner.py
│   │   │   └── __init__.py
│   │   ├── risk/                            # Risk management
│   │   │   ├── engines/advanced_var_engine.py
│   │   │   ├── managers/risk_integrated_production_manager.py
│   │   │   ├── circuit_breaker.py
│   │   │   └── compliance/
│   │   ├── validation/                      # Signal validation
│   │   │   ├── signal_strength_validator.py
│   │   │   └── strategy_signal_integration.py
│   │   ├── data/sources/                    # Market data providers
│   │   │   ├── alpaca.py, polygon.py, yahoo.py, fred.py, dark_pool.py
│   │   ├── execution/oms.py                 # Order management
│   │   ├── sentiment/sentiment_analyzer.py  # NLP sentiment
│   │   ├── alert_system.py                  # Discord/Slack/Email alerts
│   │   ├── models.py                        # Legacy Django models
│   │   ├── models/models.py                 # Extended Django models
│   │   ├── indicators/                      # Technical indicators
│   │   │   ├── pivot_points.py
│   │   │   └── candlestick.py
│   │   ├── client/api_client.py             # Alpaca broker client
│   │   ├── rpc/                             # Discord & Telegram bots
│   │   │   ├── notifications.py
│   │   │   └── telegram_bot.py
│   │   └── error_handling/                  # Custom exceptions & recovery
│   ├── auth0login/                          # Auth0 authentication
│   └── home/                                # Web dashboard
├── ml/tradingbots/                          # ML/RL models
│   ├── components/rl_agents.py              # PPO, DQN, SAC, TD3, DDPG, A2C
│   ├── training/rl_training.py
│   └── models/                              # Model registry
├── tests/                                   # 33 test categories, 240+ files
├── docs/                                    # Documentation
├── scripts/                                 # Setup & deployment scripts
├── docker/                                  # Container config
├── config/                                  # Config files
├── requirements.txt                         # 56 dependencies
├── pyproject.toml                           # Build config with ruff/mypy/pytest
├── .env.example                             # 175 environment variables
└── INTEGRATION_GAPS_ANALYSIS.md             # Known gap tracker
```

---

## What To Review and Fix

### 1. Known TODO/FIXME Items (Fix All of These)

These are confirmed TODOs in the codebase. Resolve every one:

- **`backend/tradingbot/models.py` line ~95**: `StockTrade` model is marked as "overly simplistic" — it needs `bought_price`, `sold_price`, and a `transaction_type` field (BUY/SELL). Refactor this model to represent individual transactions rather than trying to represent an entire buy/sell operation in one record. Create the corresponding Django migration.

- **`backend/tradingbot/models.py` line ~218**: Orphaned `# TODO` on the HMM Sharpe Ratio Monte Carlo strategy choice — either implement the missing logic or clean up the dangling comment.

- **`backend/tradingbot/strategies/implementations/wheel_strategy.py` line ~569**: `TODO: State transition handling between wheel phases` — implement the state machine for transitioning between cash-secured-put → assignment → covered-call → expiration/call-away phases.

- **`backend/tradingbot/tests.py` line ~9-14**: Tests are stale because models changed — rewrite these tests to match the current model schema. If the file is entirely obsolete, delete it and ensure coverage exists elsewhere.

- **`backend/tradingbot/urls.py` line ~5**: `TODO: add route for creating, patching company` — add the missing DRF API endpoints (POST for create, PATCH for update) for the Company model.

- **`backend/auth0login/forms.py` line ~106**: Orphaned `# TODO` — determine what form field or validation was planned and implement it, or remove the dead code.

- **`backend/validation/validation_runner.py` line ~151**: `TODO: Load actual strategy data` — replace the placeholder with real strategy data loading from the database or data providers.

### 2. Environment & Dependency Issues

- **`pyproject.toml` specifies `requires-python = ">=3.12"`** but the system is running Python 3.11. Verify all code is compatible with 3.12+ and that there are no 3.12-specific syntax issues (like `type` statements) that would break on 3.11. Either lower the requirement to `>=3.11` or confirm 3.12 is the true minimum.

- **`requirements.txt` vs `pyproject.toml` dependency drift**: `requirements.txt` lists `alpaca-trade-api>=3.0.0` while `pyproject.toml` lists `alpaca-py>=0.42.0`. These are different packages. Reconcile them — `alpaca-py` is the current SDK; remove the deprecated `alpaca-trade-api` reference. Audit all other version mismatches between the two files and make them consistent.

- **Missing dependencies in `requirements.txt`**: The codebase imports `plotly`, `weasyprint`, `hmmlearn`, `gymnasium`, `stable-baselines3`, `torch`, `optuna`, `httpx`, `aiohttp`, `cryptography`, `ruff`, `mypy` — but `requirements.txt` only has some of these. Ensure `requirements.txt` and `pyproject.toml` optional dependency groups are complete and consistent.

### 3. Integration & Architecture Gaps

According to `INTEGRATION_GAPS_ANALYSIS.md`, 6 integration gaps were marked COMPLETED in February 2026. **Verify** that the actual code matches these claims:

- **Gap 1**: Does `production_strategy_manager.py` actually call signal validation and use the results to filter/weight signals? Trace the code path.
- **Gap 2**: Does the risk manager (`risk_integrated_production_manager.py`) actually adjust position sizes based on signal strength scores? Trace the code path.
- **Gap 3**: Does signal validation actually check data quality from data providers before scoring signals?
- **Gap 4**: Is there actually monitoring/alerting when signal validation accuracy degrades? Check that `SIGNAL_VALIDATION_DEGRADATION` alerts fire.
- **Gap 5**: Is `SignalValidationHistory` actually being persisted to the database with real trade outcomes?
- **Gap 6**: Is there a real feedback loop that adjusts validation thresholds based on historical trade performance?

For any gap that is NOT actually implemented (just stubbed or partially done), complete the implementation.

### 4. Async/Await Consistency

The production strategy manager uses `async/await` patterns but many components are synchronous. Audit for:

- Blocking calls inside `async` functions (e.g., synchronous `requests.get()` inside an `async def`)
- Missing `await` on coroutines
- Potential deadlocks from mixing sync and async code
- Any `asyncio.run()` called from within an already-running event loop

Fix all async issues found.

### 5. Error Handling & Edge Cases

- Review the custom exception hierarchy in `error_handling/` — ensure all exceptions are actually raised somewhere and caught somewhere. Remove dead exception classes.
- Check that the recovery manager's retry logic actually works with the broker client — trace a failed order through the retry path.
- Verify circuit breaker thresholds trigger correctly and that the trading halt actually stops all strategies.
- Check for bare `except:` or `except Exception:` clauses that silently swallow errors that should propagate.

### 6. Security Review

- Scan for hardcoded API keys, secrets, or credentials in any `.py` file (not `.env.example`).
- Check that `.env` is in `.gitignore`.
- Verify SQL queries use parameterized queries (no string formatting with user input).
- Check that the Django `SECRET_KEY` is loaded from environment variables, not hardcoded in `settings.py`.
- Verify `DEBUG = False` is the production default.
- Check for command injection in any `subprocess` calls.
- Verify the crypto wallet encryption in the DEX integration is using proper key derivation.

### 7. Test Coverage & Quality

- Run `ruff check .` from project root and fix every error/warning it reports.
- Run `pytest tests/ -x --tb=short` and fix every failing test.
- Identify any strategy that has zero test coverage and add basic unit tests for it.
- Fix the stale test files (`backend/tradingbot/tests.py`) that reference old model schemas.
- Check that all test fixtures in `conftest.py` are actually used — remove unused fixtures.
- Ensure `pytest.ini` / `pyproject.toml [tool.pytest.ini_options]` markers match all markers actually used in test files.

### 8. Django Model & Migration Issues

- Check that all models in `models.py` and `models/models.py` have proper migrations.
- Look for model fields that should be `DecimalField` for financial precision but are using `FloatField` instead.
- Verify all ForeignKey relationships have appropriate `on_delete` behavior (not blindly using `CASCADE` where `PROTECT` or `SET_NULL` would be safer for financial records).
- Check for missing database indexes on fields that are frequently queried (symbol fields, date fields, status fields).
- Make sure the legacy `models.py` and the newer `models/models.py` don't have conflicting model names or table names.

### 9. Trading Logic Correctness

- **Wheel strategy**: Verify the transition logic between phases (CSP → assignment → CC → expiration). Check that the strategy doesn't accidentally enter both a put and call position simultaneously.
- **Risk management**: Verify VaR calculations handle edge cases (zero positions, single-asset portfolio, perfectly correlated assets, extreme tail events).
- **Order execution**: Check for race conditions in the order management system — can two strategies submit conflicting orders for the same symbol simultaneously?
- **Signal validation**: Verify the 0-100 scoring produces sensible results at boundary conditions (no volume, single data point, missing fields).
- **Circuit breaker**: Trace what happens when a circuit breaker trips mid-order. Does it cancel pending orders? Does it close existing positions or just halt new ones?

### 10. Code Quality Cleanup

- Remove any dead code (functions/classes/imports that are never called).
- Remove any duplicate logic between files (especially between legacy and new model files).
- Fix any type annotation issues that `mypy --strict` would flag.
- Ensure all public functions have return type annotations.
- Remove any `print()` statements that should be `logger.info()` or `logger.debug()`.
- Replace any `time.sleep()` in async code with `asyncio.sleep()`.

---

## Rules

1. **Do not add unnecessary abstractions.** Fix what's broken; don't refactor what works.
2. **Every Django model change must have a migration.** Run `python manage.py makemigrations` after model edits.
3. **Run `ruff check .` and `pytest tests/` after all changes.** Everything must pass.
4. **Do not remove or weaken any existing risk controls.** Trading safety is paramount.
5. **Do not change strategy parameters or thresholds** unless there is a clear bug (e.g., wrong sign, division by zero).
6. **Commit atomically.** One logical change per commit with a descriptive message.
7. **If a fix is complex and risky, add a test for it first** (test-driven fix).
