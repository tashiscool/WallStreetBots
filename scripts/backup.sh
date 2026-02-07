#!/usr/bin/env bash
# Database backup script for WallStreetBots
# Usage: bash scripts/backup.sh [backup_dir]
#
# Supports both SQLite and PostgreSQL databases.
# Backups are compressed with gzip and timestamped.

set -euo pipefail

BACKUP_DIR="${1:-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$BACKUP_DIR"

echo "=== WallStreetBots Database Backup ==="
echo "Timestamp: $TIMESTAMP"
echo "Backup dir: $BACKUP_DIR"

# Detect database type from .env or settings
if [ -f "$PROJECT_DIR/.env" ]; then
    DB_ENGINE=$(grep -i "DATABASE_ENGINE\|DB_ENGINE" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "")
fi
DB_ENGINE="${DB_ENGINE:-sqlite3}"

if echo "$DB_ENGINE" | grep -qi "postgres"; then
    echo "Database: PostgreSQL"

    DB_NAME=$(grep -i "DATABASE_NAME\|DB_NAME" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "wallstreetbots")
    DB_USER=$(grep -i "DATABASE_USER\|DB_USER" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "postgres")
    DB_HOST=$(grep -i "DATABASE_HOST\|DB_HOST" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "localhost")
    DB_PORT=$(grep -i "DATABASE_PORT\|DB_PORT" "$PROJECT_DIR/.env" 2>/dev/null | head -1 | cut -d= -f2 || echo "5432")

    BACKUP_FILE="$BACKUP_DIR/pg_backup_${TIMESTAMP}.sql.gz"
    pg_dump -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_FILE"
    echo "PostgreSQL backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
else
    echo "Database: SQLite"

    # Find SQLite database files
    for db_file in "$PROJECT_DIR"/db.sqlite3 "$PROJECT_DIR"/backend/db.sqlite3; do
        if [ -f "$db_file" ]; then
            DB_BASENAME=$(basename "$db_file" .sqlite3)
            BACKUP_FILE="$BACKUP_DIR/${DB_BASENAME}_backup_${TIMESTAMP}.sqlite3.gz"
            # Use SQLite .backup for consistency (avoids partial reads)
            sqlite3 "$db_file" ".backup '/tmp/wsb_backup_temp.sqlite3'"
            gzip -c "/tmp/wsb_backup_temp.sqlite3" > "$BACKUP_FILE"
            rm -f "/tmp/wsb_backup_temp.sqlite3"
            echo "SQLite backup: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
        fi
    done
fi

# Backup state files
STATE_DIR="$PROJECT_DIR/.state"
if [ -d "$STATE_DIR" ]; then
    STATE_BACKUP="$BACKUP_DIR/state_backup_${TIMESTAMP}.tar.gz"
    tar -czf "$STATE_BACKUP" -C "$PROJECT_DIR" .state/ 2>/dev/null || true
    echo "State backup: $STATE_BACKUP"
fi

# Create latest symlink
LATEST_LINK="$BACKUP_DIR/latest"
rm -f "$LATEST_LINK"
ln -sf "$(basename "$BACKUP_FILE")" "$LATEST_LINK"

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "*_backup_*" -mtime +30 -delete 2>/dev/null || true

echo ""
echo "=== Backup complete ==="
echo "Files in $BACKUP_DIR:"
ls -lh "$BACKUP_DIR"/ | tail -10
