#!/bin/bash
#
# build.sh - PostgreSQL Database Setup Script
#
# Description:
#   Automates PostgreSQL database setup including schema initialization,
#   data import, and read-only user creation.
#
# Usage:
#   ./build.sh


set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database connection settings
readonly DB_HOST="${DB_HOST:-localhost}"
readonly DB_PORT="${DB_PORT:-5432}"
readonly DB_USER="${DB_USER:-postgres}"
readonly DB_PASSWORD="${DB_PASSWORD:-}"
readonly TARGET_DB="${TARGET_DB:-medmcpcalc_database}"

# Read-only user settings
readonly READONLY_USER="${READONLY_USER:-medmcpcalc_readonly_user}"
readonly READONLY_PASSWORD="${READONLY_PASSWORD:-PASSWORD}"

# File paths
readonly SCHEMA_SQL="${SCHEMA_SQL:-./schema.sql}"
readonly INSTALL_SQL="${INSTALL_SQL:-./install.sql}"
readonly DATA_DIR="${DATA_DIR:-./data}"

# Export PGPASSWORD for psql authentication
export PGPASSWORD="$DB_PASSWORD"

# =============================================================================
# MAIN SCRIPT
# =============================================================================

echo "[INFO] Step 1: Resetting database: ${TARGET_DB} ..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS ${TARGET_DB};"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE ${TARGET_DB};"

echo "[INFO] Step 2: Initializing schema..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$TARGET_DB" -f "$SCHEMA_SQL"

echo "[INFO] Step 3: Importing data..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$TARGET_DB" -v data_dir="$DATA_DIR" -f "$INSTALL_SQL"

echo "[INFO] Step 4: Creating read-only user..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$TARGET_DB" <<EOF
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '${READONLY_USER}') THEN
        CREATE USER ${READONLY_USER} WITH PASSWORD '${READONLY_PASSWORD}';
    END IF;
END \$\$;

GRANT CONNECT ON DATABASE "${TARGET_DB}" TO ${READONLY_USER};
GRANT USAGE ON SCHEMA public TO ${READONLY_USER};
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${READONLY_USER};
EOF

# Cleanup
unset PGPASSWORD

echo "[INFO] All operations completed successfully!"