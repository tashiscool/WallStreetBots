# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- Run all backend tests: `python -Wa ./manage.py test backend`
- The `-Wa` flag enables all warnings including deprecation warnings
- Tests require a PostgreSQL database user with `createdb` permissions

### Linting
- Run flake8 linting: `flake8 .`
- Configuration in `.flake8`: max line length 160, max complexity 10, excludes migrations
- CI also runs syntax/undefined name checks: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`

### Database
- Apply migrations: `python manage.py migrate`
- Database connection configured via `.env` file (copy from `backend/.env.example`)
- Format: `postgres://<username>:<password>@<host>:<port>/<db_name>`

### Development Setup
- Bootstrap script for macOS/Linux: `./scripts/bootstrap.sh`
- Manual setup: Python 3.9.6, PostgreSQL 14, install requirements with `pip install -r requirements.txt`
- Also install ML dependencies: `pip install -r ml/requirements.txt`

### Docker
- Run with Docker: `docker-compose up`
- Database service name in Docker: `web-db`
- Web service runs on port 8000

## Architecture

### Backend (Django)
- Django REST API backend in `backend/` directory
- Main apps: `home`, `tradingbot`, `auth0login`
- Settings in `backend/settings.py` with environment-based configuration
- URL routing: API endpoints under `/api/` prefix
- Uses PostgreSQL database with Auth0 authentication

### ML Components
- Machine learning code in `ml/` directory
- Trading bot pipelines in `ml/tradingbots/pipelines/`
- Includes Monte Carlo, Hidden Markov Model implementations
- Data collection utilities in `ml/data_collection/`
- Models saved to disk and loaded by web application

### Key Libraries
- Django 3.2.8 + DRF for web framework
- Alpaca Trade API for stock trading
- Pandas, NumPy for data processing
- HMM, Plotly for ML and visualization
- Celery for background tasks
- Auth0 for authentication

### Environment Configuration
- Development settings via `.env` file in `backend/`
- Required: `SECRET_KEY`, `DEBUG`, database URL
- CORS configured for localhost:3000 (frontend)
- Allowed hosts include `wallstreetbots.org`