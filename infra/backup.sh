#!/usr/bin/env bash
# GAMI Backup Script
# pg_dump + Redis RDB snapshot, compressed and timestamped.
# Retention: 7 daily, 4 weekly.
# Intended for cron: 0 2 * * * /opt/gami/infra/backup.sh
set -euo pipefail

# --- Configuration ---
BACKUP_DIR="/mnt/16tb/gami/backups"
PG_HOST="127.0.0.1"
PG_PORT="5433"
PG_USER="gami"
PG_DB="gami"
REDIS_HOST="127.0.0.1"
REDIS_PORT="6380"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday
LOG_FILE="/var/log/gami-backup.log"

# Retention
DAILY_KEEP=7
WEEKLY_KEEP=4

# --- Helpers ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

die() {
    log "ERROR: $*"
    exit 1
}

# --- Setup ---
mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly"
log "=== GAMI backup starting (timestamp: $TIMESTAMP) ==="

# --- PostgreSQL Dump ---
PG_DUMP_FILE="$BACKUP_DIR/daily/gami_pg_${TIMESTAMP}.sql.gz"
log "Dumping PostgreSQL database '$PG_DB'..."

export PGPASSWORD='GAMI_2026_Pr0d!Secure'
if pg_dump -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" "$PG_DB" \
    --no-owner --no-privileges --format=plain \
    | gzip -9 > "$PG_DUMP_FILE"; then
    PG_SIZE=$(stat -c%s "$PG_DUMP_FILE" 2>/dev/null || echo 0)
    log "PostgreSQL dump complete: $(basename "$PG_DUMP_FILE") ($(numfmt --to=iec "$PG_SIZE"))"
else
    die "PostgreSQL dump failed"
fi
unset PGPASSWORD

# --- Redis Snapshot ---
REDIS_DUMP_FILE="$BACKUP_DIR/daily/gami_redis_${TIMESTAMP}.rdb.gz"
log "Taking Redis RDB snapshot..."

# Trigger BGSAVE and wait for it
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE 2>/dev/null; then
    # Wait for background save to complete (max 60s)
    for i in $(seq 1 60); do
        STATUS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE 2>/dev/null)
        sleep 1
        NEW_STATUS=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE 2>/dev/null)
        if [ "$NEW_STATUS" != "$STATUS" ] || [ "$i" -eq 1 ]; then
            break
        fi
    done

    # Find and copy the RDB file
    REDIS_RDB_DIR=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dir 2>/dev/null | tail -1)
    REDIS_RDB_FILE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" CONFIG GET dbfilename 2>/dev/null | tail -1)

    if [ -n "$REDIS_RDB_DIR" ] && [ -n "$REDIS_RDB_FILE" ] && [ -f "$REDIS_RDB_DIR/$REDIS_RDB_FILE" ]; then
        gzip -c "$REDIS_RDB_DIR/$REDIS_RDB_FILE" > "$REDIS_DUMP_FILE"
        REDIS_SIZE=$(stat -c%s "$REDIS_DUMP_FILE" 2>/dev/null || echo 0)
        log "Redis snapshot complete: $(basename "$REDIS_DUMP_FILE") ($(numfmt --to=iec "$REDIS_SIZE"))"
    else
        log "WARNING: Redis RDB file not found at $REDIS_RDB_DIR/$REDIS_RDB_FILE"
    fi
else
    log "WARNING: Redis BGSAVE failed (Redis may not be running)"
fi

# --- Cold Index Backup ---
COLD_INDEX="/mnt/16tb/gami/index/cold_index.sqlite"
if [ -f "$COLD_INDEX" ]; then
    COLD_BACKUP="$BACKUP_DIR/daily/gami_cold_index_${TIMESTAMP}.sqlite.gz"
    gzip -c "$COLD_INDEX" > "$COLD_BACKUP"
    log "Cold index backup: $(basename "$COLD_BACKUP")"
fi

# --- Weekly Copy (on Sundays) ---
if [ "$DAY_OF_WEEK" -eq 7 ]; then
    log "Creating weekly backup copy..."
    for f in "$BACKUP_DIR/daily/gami_"*"_${TIMESTAMP}"*; do
        if [ -f "$f" ]; then
            WEEKLY_NAME=$(basename "$f" | sed "s/${TIMESTAMP}/weekly_${TIMESTAMP}/")
            cp "$f" "$BACKUP_DIR/weekly/$WEEKLY_NAME"
        fi
    done
    log "Weekly backup created."
fi

# --- Retention Cleanup ---
log "Cleaning up old backups..."

# Daily: keep last N days
DAILY_COUNT=$(find "$BACKUP_DIR/daily" -name "gami_pg_*.sql.gz" -type f | wc -l)
if [ "$DAILY_COUNT" -gt "$DAILY_KEEP" ]; then
    REMOVE_COUNT=$((DAILY_COUNT - DAILY_KEEP))
    find "$BACKUP_DIR/daily" -name "gami_*" -type f -printf '%T@ %p\n' \
        | sort -n | head -n "$((REMOVE_COUNT * 3))" | cut -d' ' -f2- \
        | while read -r OLD_FILE; do
        log "  Removing old daily: $(basename "$OLD_FILE")"
        rm -f "$OLD_FILE"
    done
fi

# Weekly: keep last N weeks
WEEKLY_COUNT=$(find "$BACKUP_DIR/weekly" -name "gami_pg_*" -type f | wc -l)
if [ "$WEEKLY_COUNT" -gt "$WEEKLY_KEEP" ]; then
    REMOVE_COUNT=$((WEEKLY_COUNT - WEEKLY_KEEP))
    find "$BACKUP_DIR/weekly" -name "gami_*" -type f -printf '%T@ %p\n' \
        | sort -n | head -n "$((REMOVE_COUNT * 3))" | cut -d' ' -f2- \
        | while read -r OLD_FILE; do
        log "  Removing old weekly: $(basename "$OLD_FILE")"
        rm -f "$OLD_FILE"
    done
fi

# --- Summary ---
TOTAL_BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | awk '{print $1}')
log "=== GAMI backup complete. Total backup storage: $TOTAL_BACKUP_SIZE ==="
