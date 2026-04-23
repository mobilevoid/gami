-- Migration: Add Phase 3 (compression) and Phase 5 (bi-temporal) columns to segments
-- Run as postgres user: psql -p 5433 -U postgres -d gami -f storage/sql/019_phase3_phase5_columns.sql

BEGIN;

-- Phase 3: Lossless Compression columns
ALTER TABLE segments ADD COLUMN IF NOT EXISTS compression_status TEXT DEFAULT 'raw';
ALTER TABLE segments ADD COLUMN IF NOT EXISTS compression_delta_id TEXT;

-- Phase 5: Bi-temporal query columns
ALTER TABLE segments ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS event_time_confidence FLOAT DEFAULT 0.0;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS event_time_source TEXT DEFAULT 'unknown';

-- Index for event_time queries
CREATE INDEX IF NOT EXISTS idx_segments_event_time
    ON segments (event_time DESC) WHERE event_time IS NOT NULL;

-- Grant permissions to gami user
GRANT SELECT, UPDATE ON segments TO gami;

COMMIT;

-- Verify
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'segments'
AND column_name IN ('compression_status', 'compression_delta_id', 'event_time', 'event_time_confidence', 'event_time_source');
