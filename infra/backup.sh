#!/bin/bash
# GAMI Backup Script — backs up to local /mnt/16tb AND black-jesus NAS
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAY_OF_WEEK=$(date +%u)
LOCAL_BACKUP_DIR="/mnt/16tb/gami/backups"
NAS_BACKUP_DIR="/mnt/nas-media-torrents/../gami-backups"  # Will fix path below
GAMI_DB_HOST="127.0.0.1"
GAMI_DB_PORT="5433"
GAMI_DB_NAME="gami"
GAMI_DB_USER="gami"
RETENTION_DAILY=7
RETENTION_WEEKLY=4

mkdir -p "$LOCAL_BACKUP_DIR/daily" "$LOCAL_BACKUP_DIR/weekly"

echo "[$(date)] Starting GAMI backup..."

# 1. PostgreSQL dump
echo "  Dumping PostgreSQL..."
PGPASSWORD="GamiProd2026" pg_dump -h "$GAMI_DB_HOST" -p "$GAMI_DB_PORT" -U "$GAMI_DB_USER" "$GAMI_DB_NAME" --exclude-schema=gami_graph | gzip > "$LOCAL_BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz"

# 2. Redis RDB snapshot
echo "  Redis snapshot..."
redis-cli -p 6380 BGSAVE > /dev/null 2>&1
sleep 3
cp /var/lib/redis-gami/dump-gami.rdb "$LOCAL_BACKUP_DIR/daily/gami-redis-${TIMESTAMP}.rdb" 2>/dev/null || true

# 3. Cold storage SQLite index
echo "  Cold index..."
cp /mnt/16tb/gami/index/cold_index.sqlite "$LOCAL_BACKUP_DIR/daily/gami-cold-index-${TIMESTAMP}.sqlite" 2>/dev/null || true

# 4. Weekly copy on Sundays
if [ "$DAY_OF_WEEK" -eq 7 ]; then
    echo "  Weekly backup..."
    cp "$LOCAL_BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$LOCAL_BACKUP_DIR/weekly/"
fi

# 5. Copy to NAS (black-jesus)
NAS_DIR="/mnt/nas-gami-backup"
if mountpoint -q "$NAS_DIR" 2>/dev/null || mount | grep -q "nas-gami-backup"; then
    echo "  Copying to NAS..."
    mkdir -p "$NAS_DIR/daily" "$NAS_DIR/weekly"
    cp "$LOCAL_BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$NAS_DIR/daily/"
    if [ "$DAY_OF_WEEK" -eq 7 ]; then
        cp "$LOCAL_BACKUP_DIR/daily/gami-pg-${TIMESTAMP}.sql.gz" "$NAS_DIR/weekly/"
    fi
else
    echo "  NAS not mounted — skipping NAS backup"
fi

# 6. Retention cleanup
echo "  Cleaning old backups..."
find "$LOCAL_BACKUP_DIR/daily" -name "gami-*" -mtime +${RETENTION_DAILY} -delete 2>/dev/null || true
find "$LOCAL_BACKUP_DIR/weekly" -name "gami-*" -mtime +$((RETENTION_WEEKLY * 7)) -delete 2>/dev/null || true
if mountpoint -q "$NAS_DIR" 2>/dev/null; then
    find "$NAS_DIR/daily" -name "gami-*" -mtime +${RETENTION_DAILY} -delete 2>/dev/null || true
    find "$NAS_DIR/weekly" -name "gami-*" -mtime +$((RETENTION_WEEKLY * 7)) -delete 2>/dev/null || true
fi

echo "[$(date)] GAMI backup complete."
