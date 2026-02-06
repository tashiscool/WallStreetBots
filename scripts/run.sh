#!/usr/bin/env bash
# WallStreetBots Run Script
# Starts the development server with correct settings
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Load environment
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Ensure DEBUG is on for dev
export DJANGO_DEBUG=${DJANGO_DEBUG:-True}
export DJANGO_SETTINGS_MODULE=${DJANGO_SETTINGS_MODULE:-backend.settings}

PORT=${1:-8000}

echo "Starting WallStreetBots on http://127.0.0.1:$PORT"
echo "  DEBUG=$DJANGO_DEBUG"
echo "  Press Ctrl+C to stop"
echo ""

python manage.py runserver "127.0.0.1:$PORT"
