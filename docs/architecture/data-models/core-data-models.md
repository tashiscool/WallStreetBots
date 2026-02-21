# Data Models: Wallstreetbots
## Date: 2026-02-20
## Entities
- trades (docs/PRODUCTION_ROADMAP.md)
- positions (docs/PRODUCTION_ROADMAP.md)
- strategies (docs/PRODUCTION_ROADMAP.md)
- risk_limits (docs/PRODUCTION_ROADMAP.md)
- positions (scripts/dr_test.py)
- IF (db/taxlots.sql)
- Ai (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- Oi (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- Mn (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- En (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- Yn (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- Kn (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- Vo (backend/auth0login/static/assets/js/plugins/chartjs.min.js)
- IF (backend/tradingbot/core/production_database.py)
- IF (backend/tradingbot/prediction_markets/trade_storage.py)
- IF (backend/tradingbot/risk/database_schema.py)
- StockOrderRequest (backend/tradingbot/validation/request_validators.py)
- OptionOrderRequest (backend/tradingbot/validation/request_validators.py)
- PositionCloseRequest (backend/tradingbot/validation/request_validators.py)
- WatchlistRequest (backend/tradingbot/validation/request_validators.py)
- StrategyConfigRequest (backend/tradingbot/validation/request_validators.py)
- AlertConfigRequest (backend/tradingbot/validation/request_validators.py)
- BacktestRequest (backend/tradingbot/validation/request_validators.py)
- IF (backend/tradingbot/risk/compliance/regulatory_compliance_manager.py)

## Data Flow Notes
- Validate upstream input before persistence.
- Keep schema-affecting changes behind migration plans.
