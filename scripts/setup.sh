#!/usr/bin/env bash
# WallStreetBots Setup Script
# Run from project root: bash scripts/setup.sh
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " WallStreetBots Setup"
echo "=========================================="

# ── 1. Check Python ──────────────────────────
echo ""
echo "[1/6] Checking Python..."
PYTHON=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3 not found. Install Python 3.11+ and retry."
    exit 1
fi
echo "  Using: $($PYTHON --version) at $(which $PYTHON)"

# ── 2. Install dependencies ──────────────────
echo ""
echo "[2/6] Installing Python dependencies..."
$PYTHON -m pip install -q --upgrade pip
$PYTHON -m pip install -q -r requirements.txt
echo "  Core dependencies installed."

# Install optional deps (non-fatal)
echo "  Installing optional dependencies..."
$PYTHON -m pip install -q vaderSentiment 2>/dev/null && echo "    vaderSentiment: OK" || echo "    vaderSentiment: skipped (optional)"
$PYTHON -m pip install -q plotly 2>/dev/null && echo "    plotly: OK" || echo "    plotly: skipped (optional)"
$PYTHON -m pip install -q kaleido 2>/dev/null && echo "    kaleido: OK" || echo "    kaleido: skipped (optional)"

# ── 3. Create .env if missing ────────────────
echo ""
echo "[3/6] Checking .env file..."
if [ ! -f .env ]; then
    SECRET=$(openssl rand -hex 32 2>/dev/null || echo "dev-fallback-secret-key-change-me")
    cat > .env << ENVEOF
# WallStreetBots Local Development Settings
DJANGO_SETTINGS_MODULE=backend.settings
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=$SECRET

# Database: SQLite by default (no setup required)
# Uncomment below for PostgreSQL:
# DATABASE_URL=postgresql://wsb_user:wsb_password@localhost:5432/wallstreetbots

# Trading APIs (fill in your keys for live data)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Paper trading mode (safe default)
PAPER_TRADING=True
ENVEOF
    echo "  Created .env with SQLite defaults and DEBUG=True."
    echo "  Edit .env to add API keys or switch to PostgreSQL."
else
    echo "  .env already exists, skipping."
fi

# Source .env
set -a
source .env 2>/dev/null || true
set +a

# ── 4. Database migrations ───────────────────
echo ""
echo "[4/6] Setting up database..."

# Generate migrations for any new models (with auto defaults for auto_now_add)
$PYTHON -c "
import os, sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django; django.setup()
from django.db.migrations.questioner import NonInteractiveMigrationQuestioner
from django.utils import timezone
original = NonInteractiveMigrationQuestioner.ask_auto_now_add_addition
NonInteractiveMigrationQuestioner.ask_auto_now_add_addition = lambda self, *a: timezone.now()
from django.core.management import call_command
call_command('makemigrations', '--noinput', verbosity=0)
" 2>&1 || echo "  (migrations already up to date)"

# Apply all migrations
$PYTHON manage.py migrate --noinput 2>&1 | tail -3
echo "  Database ready."

# ── 5. Collect static files ──────────────────
echo ""
echo "[5/6] Collecting static files..."
$PYTHON manage.py collectstatic --noinput --verbosity 0 2>&1
echo "  Static files collected."

# ── 6. Verify ────────────────────────────────
echo ""
echo "[6/6] Running system checks..."
CHECK_OUTPUT=$($PYTHON manage.py check 2>&1)
if echo "$CHECK_OUTPUT" | grep -q "no issues"; then
    echo "  System check passed - no issues."
else
    echo "$CHECK_OUTPUT"
fi

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo " Start the dev server:"
echo "   DJANGO_DEBUG=True python manage.py runserver"
echo ""
echo " Run tests:"
echo "   python -m pytest tests/ -x -q"
echo ""
echo " Create a superuser (for /admin access):"
echo "   python manage.py createsuperuser"
echo ""
echo " Access the app at:"
echo "   http://127.0.0.1:8000/        - Home"
echo "   http://127.0.0.1:8000/admin/   - Admin"
echo "   http://127.0.0.1:8000/health/  - Health Check"
echo "   http://127.0.0.1:8000/api/docs/ - API Documentation"
echo ""
