#!/usr/bin/env bash
# Database restore script for WallStreetBots
# Usage: bash scripts/restore.sh <backup_file>
#
# Supports both SQLite (.sqlite3.gz) and PostgreSQL (.sql.gz) backups.
# Creates a pre-restore backup before overwriting.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/restore.sh <backup_file>"
    echo ""
    echo "Examples:"
    echo "  bash scripts/restore.sh backups/db_backup_20250615_120000.sqlite3.gz"
    echo "  bash scripts/restore.sh backups/pg_backup_20250615_120000.sql.gz"
    echo "  bash scripts/restore.sh backups/latest"
    exit 1
fi

BACKUP_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Resolve symlinks
if [ -L "$BACKUP_FILE" ]; then
    BACKUP_DIR="$(dirname "$BACKUP_FILE")"
    BACKUP_FILE="$BACKUP_DIR/$(readlink "$BACKUP_FILE")"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "=== WallStreetBots Database Restore ==="
echo "Backup file: $BACKUP_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Confirm
read -p "This will overwrite the current database. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled."
    exit 0
fi

if echo "$BACKUP_FILE" | grep -q "\.sql\.gz$"; then
    echo "Restoring PostgreSQL backup..."

    DB_NAME=$(grep -i "DATABASE_NAME\|DB_NAME" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "wallstreetbots")
    DB_USER=$(grep -i "DATABASE_USER\|DB_USER" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "postgres")
    DB_HOST=$(grep -i "DATABASE_HOST\|DB_HOST" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "localhost")
    DB_PORT=$(grep -i "DATABASE_PORT\|DB_PORT" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "5432")

    # Pre-restore backup
    echo "Creating pre-restore backup..."
    PRE_RESTORE="backups/pre_restore_${TIMESTAMP}.sql.gz"
    mkdir -p backups
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" 2>/dev/null | gzip > "$PRE_RESTORE" || true

    # Restore
    gunzip -c "$BACKUP_FILE" | psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
    echo "PostgreSQL restore complete."

elif echo "$BACKUP_FILE" | grep -q "\.sqlite3\.gz$"; then
    echo "Restoring SQLite backup..."

    DB_FILE="$PROJECT_DIR/db.sqlite3"
    if [ ! -f "$DB_FILE" ]; then
        DB_FILE="$PROJECT_DIR/backend/db.sqlite3"
    fi

    # Pre-restore backup
    if [ -f "$DB_FILE" ]; then
        echo "Creating pre-restore backup..."
        PRE_RESTORE="backups/pre_restore_${TIMESTAMP}.sqlite3.gz"
        mkdir -p backups
        gzip -c "$DB_FILE" > "$PRE_RESTORE"
    fi

    # Restore
    gunzip -c "$BACKUP_FILE" > "$DB_FILE"
    echo "SQLite restore complete: $DB_FILE"
else
    echo "ERROR: Unrecognized backup format. Expected .sql.gz or .sqlite3.gz"
    exit 1
fi

# Run migrations to ensure schema is current
echo ""
echo "Running migrations..."
cd "$PROJECT_DIR"
python manage.py migrate --no-input 2>/dev/null || echo "Warning: migrations may need manual review"

echo ""
echo "=== Restore complete ==="
echo "Pre-restore backup: ${PRE_RESTORE:-none}"
echo ""
echo "Next steps:"
echo "  1. Verify data integrity"
echo "  2. Run position reconciliation"
echo "  3. Check circuit breaker state"
echo "  4. Resume trading when confident"
