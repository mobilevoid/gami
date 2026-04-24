#!/bin/bash
# GAMI Backup Script
# Backs up PostgreSQL, Redis, and cold storage
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAY_OF_WEEK=$(date +%u)

# Configuration via environment variables
BACKUP_DIR="${GAMI_BACKUP_DIR:-/var/backups/gami}"
GAMI_DB_HOST="${GAMI_DB_HOST:-127.0.0.1}"
GAMI_DB_PORT="${GAMI_DB_PORT:-5432}"
GAMI_DB_NAME="${GAMI_DB_NAME:-gami}"
GAMI_DB_USER="${GAMI_DB_USER:-gami}"
GAMI_DB_PASSWORD="${GAMI_DB_PASSWORD:?ERROR: GAMI_DB_PASSWORD not set}"
GAMI_REDIS_PORT="${GAMI_REDIS_PORT:-6379}"
RETENTION_DAILY="${RETENTION_DAILY:-7}"
RETENTION_WEEKLY="${RETENTION_WEEKLY:-4}"

mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly"

echo "[$(date)] Starting GAMI backup..."

# 1. PostgreSQL dump
echo "  Dumping PostgreSQL..."
PGPASSWORD="$GAMI_DB_PASSWORD" pg_dump \
    -h "$GAMI_DB_HOST" \
    -p "$GAMI_DB_PORT" \
    -U "$GAMI_DB_USER" \
    "$GAMI_DB_NAME" \
    --exclude-schema=gami_graph \
    | gzip > "$BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz"

# 2. Redis RDB snapshot
echo "  Redis snapshot..."
redis-cli -p "$GAMI_REDIS_PORT" BGSAVE > /dev/null 2>&1 || true
sleep 3
REDIS_DIR="${REDIS_DATA_DIR:-/var/lib/redis}"
if [ -f "$REDIS_DIR/dump.rdb" ]; then
    cp "$REDIS_DIR/dump.rdb" "$BACKUP_DIR/daily/gami-redis-${TIMESTAMP}.rdb"
fi

# 3. Cold storage index (if exists)
if [ -n "${GAMI_COLD_STORE:-}" ] && [ -f "$GAMI_COLD_STORE/index/cold_index.sqlite" ]; then
    echo "  Cold index..."
    cp "$GAMI_COLD_STORE/index/cold_index.sqlite" "$BACKUP_DIR/daily/gami-cold-index-${TIMESTAMP}.sqlite"
fi

# 4. Weekly copy on Sundays
if [ "$DAY_OF_WEEK" -eq 7 ]; then
    echo "  Weekly backup..."
    cp "$BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$BACKUP_DIR/weekly/"
fi

# 5. Optional: Copy to remote/NAS
if [ -n "${GAMI_REMOTE_BACKUP_DIR:-}" ] && [ -d "$GAMI_REMOTE_BACKUP_DIR" ]; then
    echo "  Copying to remote storage..."
    mkdir -p "$GAMI_REMOTE_BACKUP_DIR/daily" "$GAMI_REMOTE_BACKUP_DIR/weekly"
    cp "$BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$GAMI_REMOTE_BACKUP_DIR/daily/"
    if [ "$DAY_OF_WEEK" -eq 7 ]; then
        cp "$BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$GAMI_REMOTE_BACKUP_DIR/weekly/"
    fi
fi

# 6. Retention cleanup
echo "  Cleaning old backups..."
find "$BACKUP_DIR/daily" -name "gami-*" -mtime +${RETENTION_DAILY} -delete 2>/dev/null || true
find "$BACKUP_DIR/weekly" -name "gami-*" -mtime +$((RETENTION_WEEKLY * 7)) -delete 2>/dev/null || true

echo "[$(date)] GAMI backup complete."
