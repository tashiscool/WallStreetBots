#!/bin/bash
# Production entrypoint script for WallStreetBots
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print startup banner
echo "============================================"
echo "  WallStreetBots Trading Platform"
echo "  Build: ${WSB_GIT_SHA:-unknown}"
echo "  Django Settings: ${DJANGO_SETTINGS_MODULE:-backend.settings}"
echo "============================================"

# Wait for database to be ready
wait_for_db() {
    log_info "Waiting for database..."
    local max_retries=30
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if python -c "
import sys
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()
from django.db import connection
connection.ensure_connection()
" 2>/dev/null; then
            log_info "Database is ready!"
            return 0
        fi

        retry=$((retry + 1))
        log_warn "Database not ready, retrying ($retry/$max_retries)..."
        sleep 2
    done

    log_error "Database connection failed after $max_retries attempts"
    return 1
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    python manage.py migrate --noinput

    if [ $? -eq 0 ]; then
        log_info "Migrations completed successfully"
    else
        log_error "Migration failed!"
        exit 1
    fi
}

# Collect static files (if not done at build time)
collect_static() {
    if [ "${COLLECT_STATIC:-true}" = "true" ]; then
        log_info "Collecting static files..."
        python manage.py collectstatic --noinput --clear
    fi
}

# Create superuser if credentials provided
create_superuser() {
    if [ -n "${DJANGO_SUPERUSER_USERNAME}" ] && [ -n "${DJANGO_SUPERUSER_PASSWORD}" ] && [ -n "${DJANGO_SUPERUSER_EMAIL}" ]; then
        log_info "Creating superuser if not exists..."
        python manage.py createsuperuser --noinput 2>/dev/null || log_info "Superuser already exists"
    fi
}

# Perform startup health check
startup_health_check() {
    log_info "Performing startup health check..."
    python -c "
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()
from backend.tradingbot.monitoring.health_check import health_checker
result = health_checker.check_health()
print(f'Health status: {result.get(\"status\", \"unknown\")}')
" || log_warn "Health check returned warnings"
}

# Main execution
main() {
    # Skip database operations for certain commands
    case "$1" in
        bash|sh|python|pip)
            exec "$@"
            ;;
    esac

    # Production startup sequence
    if [ "${SKIP_DB_WAIT:-false}" != "true" ]; then
        wait_for_db
    fi

    if [ "${SKIP_MIGRATIONS:-false}" != "true" ]; then
        run_migrations
    fi

    if [ "${SKIP_STATIC:-false}" != "true" ]; then
        collect_static
    fi

    create_superuser
    startup_health_check

    log_info "Starting application..."
    exec "$@"
}

main "$@"
