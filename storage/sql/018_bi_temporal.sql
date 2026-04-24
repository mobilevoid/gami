-- GAMI Enhancement: Bi-Temporal Query Support
-- Tracks event_time separately (since segments is owned by postgres)
-- Run: psql -h 127.0.0.1 -p 5433 -U gami -d gami -f 018_bi_temporal.sql

BEGIN;

-- Event time tracking for segments
-- Allows querying "when did this happen" vs "when was this ingested"
CREATE TABLE IF NOT EXISTS segment_event_times (
    segment_id TEXT PRIMARY KEY,

    -- Event time (when the event actually happened)
    event_time TIMESTAMPTZ,
    event_time_confidence FLOAT DEFAULT 0.0,  -- 0-1 confidence in extraction
    event_time_source TEXT DEFAULT 'unknown', -- explicit, extracted, inferred, metadata

    -- Extracted temporal references (all dates found in text)
    temporal_refs JSONB DEFAULT '[]',  -- [{timestamp, description, confidence}]

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for bi-temporal queries
CREATE INDEX IF NOT EXISTS idx_event_times_time
    ON segment_event_times(event_time DESC) WHERE event_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_event_times_confidence
    ON segment_event_times(event_time_confidence DESC);

COMMIT;
