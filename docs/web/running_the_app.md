# Running WallStreetBots

## Quick Start

```bash
# One-command setup
bash scripts/setup.sh

# Start the development server
bash scripts/run.sh
```

## Access Points

- **Home**: http://127.0.0.1:8000/
- **Admin Panel**: http://127.0.0.1:8000/admin/
- **API Docs**: http://127.0.0.1:8000/api/docs/
- **Health Check**: http://127.0.0.1:8000/health/

## API Endpoints

### Strategy Builder
- `GET /api/strategy-builder/indicators/` - List available indicators
- `GET /api/strategy-builder/presets/` - List strategy presets
- `POST /api/strategy-builder/validate/` - Validate strategy config
- `POST /api/strategy-builder/backtest/` - Run backtest

### Copy Trading
- `GET /api/signals/providers/` - List signal providers
- `POST /api/signals/subscribe/` - Subscribe to a provider
- `GET /api/signals/subscriptions/` - List your subscriptions

### DEX Trading
- `GET /api/dex/quote/` - Get swap quote
- `POST /api/dex/swap/` - Execute swap
- `GET /api/dex/wallet/balance/` - Check wallet balance

### Reports
- `POST /api/reports/generate/` - Generate PDF report
- `GET /api/reports/download/<id>/` - Download report

## Database Options

### SQLite (Default)
No configuration needed. Database file is created at `db.sqlite3`.

### PostgreSQL
```bash
bash scripts/setup_postgres.sh
```

## Environment Variables

Key settings in `.env`:
- `DJANGO_DEBUG` - Enable debug mode (default: True)
- `DJANGO_SECRET_KEY` - Secret key for Django
- `ALPACA_API_KEY` - Alpaca API key for trading
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `PAPER_TRADING` - Enable paper trading mode (default: True)
- `DATABASE_URL` - PostgreSQL connection string (optional)

## Creating a Superuser

```bash
python manage.py createsuperuser
```

This gives you access to the Django admin panel at `/admin/`.
