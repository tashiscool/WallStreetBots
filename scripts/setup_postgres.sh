#!/usr/bin/env bash
# WallStreetBots PostgreSQL Setup Script
# Run this if you want to use PostgreSQL instead of SQLite
# Prerequisites: PostgreSQL installed and running
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " WallStreetBots PostgreSQL Setup"
echo "=========================================="

# ── 1. Find PostgreSQL ───────────────────────
echo ""
echo "[1/4] Finding PostgreSQL..."
PSQL=""
for candidate in psql /usr/local/bin/psql /opt/homebrew/bin/psql /opt/homebrew/opt/postgresql@16/bin/psql /opt/homebrew/opt/postgresql@15/bin/psql /opt/homebrew/opt/postgresql@14/bin/psql; do
    if command -v "$candidate" &>/dev/null || [ -x "$candidate" ]; then
        PSQL="$candidate"
        break
    fi
done

# Also check Postgres.app
if [ -z "$PSQL" ]; then
    for ver in 16 15 14; do
        if [ -x "/Applications/Postgres.app/Contents/Versions/$ver/bin/psql" ]; then
            PSQL="/Applications/Postgres.app/Contents/Versions/$ver/bin/psql"
            break
        fi
    done
fi

if [ -z "$PSQL" ]; then
    echo "ERROR: psql not found. Make sure PostgreSQL is installed and on your PATH."
    echo ""
    echo "On macOS with Homebrew:"
    echo "  brew install postgresql@16"
    echo "  brew services start postgresql@16"
    echo "  echo 'export PATH=\"/opt/homebrew/opt/postgresql@16/bin:\$PATH\"' >> ~/.zshrc"
    echo ""
    echo "Or with Postgres.app:"
    echo "  Download from https://postgresapp.com/"
    exit 1
fi

PSQL_DIR=$(dirname "$PSQL")
CREATEDB="$PSQL_DIR/createdb"
CREATEUSER="$PSQL_DIR/createuser"

echo "  Found: $PSQL"
echo "  Version: $($PSQL --version)"

# ── 2. Create database and user ──────────────
echo ""
echo "[2/4] Creating database..."

DB_NAME="${DB_NAME:-wallstreetbots}"
DB_USER="${DB_USER:-wsb_user}"
DB_PASSWORD="${DB_PASSWORD:-wsb_password}"

# Create user (ignore error if exists)
"$CREATEUSER" -s "$DB_USER" 2>/dev/null && echo "  Created user: $DB_USER" || echo "  User $DB_USER already exists"

# Set password
"$PSQL" -c "ALTER USER $DB_USER WITH PASSWORD '$DB_PASSWORD';" postgres 2>/dev/null || true

# Create database (ignore error if exists)
"$CREATEDB" -O "$DB_USER" "$DB_NAME" 2>/dev/null && echo "  Created database: $DB_NAME" || echo "  Database $DB_NAME already exists"

# ── 3. Update .env ───────────────────────────
echo ""
echo "[3/4] Updating .env..."

DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"

if [ -f .env ]; then
    # Comment out SQLite line and add/update DATABASE_URL
    if grep -q "^DATABASE_URL=" .env; then
        sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$DATABASE_URL|" .env
        echo "  Updated DATABASE_URL in .env"
    elif grep -q "^# DATABASE_URL=" .env; then
        sed -i '' "s|^# DATABASE_URL=.*|DATABASE_URL=$DATABASE_URL|" .env
        echo "  Uncommented and set DATABASE_URL in .env"
    else
        echo "DATABASE_URL=$DATABASE_URL" >> .env
        echo "  Added DATABASE_URL to .env"
    fi
else
    echo "  WARNING: No .env file found. Run scripts/setup.sh first."
    exit 1
fi

# ── 4. Run migrations ────────────────────────
echo ""
echo "[4/4] Running migrations on PostgreSQL..."

# Source updated .env
set -a; source .env; set +a

python manage.py migrate --noinput 2>&1 | tail -5
echo "  PostgreSQL database ready."

echo ""
echo "=========================================="
echo " PostgreSQL Setup Complete!"
echo "=========================================="
echo ""
echo " Connection: $DATABASE_URL"
echo ""
echo " Start the server:"
echo "   bash scripts/run.sh"
echo ""
