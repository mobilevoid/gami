-- Rollback Migration 003: Innovation Extension Tables
--
-- WARNING: This will drop all innovation tables and their data!
-- Run only if you need to revert the migration.
--
-- To rollback: psql -p 5433 -U gami -d gami -f rollback_003.sql

BEGIN;

-- Drop new tables (in reverse dependency order)
DROP TABLE IF EXISTS subconscious_events CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS memory_clusters CASCADE;
DROP TABLE IF EXISTS causal_relations CASCADE;
DROP TABLE IF EXISTS prompt_templates CASCADE;
DROP TABLE IF EXISTS agent_trust_history CASCADE;
DROP TABLE IF EXISTS agent_configs CASCADE;
DROP TABLE IF EXISTS retrieval_logs CASCADE;

-- Remove added columns from existing tables
-- Note: These ALTER TABLE statements will fail silently if columns don't exist

ALTER TABLE segments DROP COLUMN IF EXISTS created_by_agent_id;
ALTER TABLE segments DROP COLUMN IF EXISTS created_by_user_id;
ALTER TABLE segments DROP COLUMN IF EXISTS derived_from;
ALTER TABLE segments DROP COLUMN IF EXISTS derivation_type;
ALTER TABLE segments DROP COLUMN IF EXISTS stability_score;
ALTER TABLE segments DROP COLUMN IF EXISTS decay_score;
ALTER TABLE segments DROP COLUMN IF EXISTS cluster_id;
ALTER TABLE segments DROP COLUMN IF EXISTS last_reinforced_at;

ALTER TABLE entities DROP COLUMN IF EXISTS created_by_agent_id;
ALTER TABLE entities DROP COLUMN IF EXISTS created_by_user_id;

ALTER TABLE claims DROP COLUMN IF EXISTS created_by_agent_id;
ALTER TABLE claims DROP COLUMN IF EXISTS created_by_user_id;
ALTER TABLE claims DROP COLUMN IF EXISTS derived_from;
ALTER TABLE claims DROP COLUMN IF EXISTS derivation_type;

ALTER TABLE relations DROP COLUMN IF EXISTS created_by_agent_id;

ALTER TABLE assistant_memories DROP COLUMN IF EXISTS created_by_agent_id;
ALTER TABLE assistant_memories DROP COLUMN IF EXISTS created_by_user_id;
ALTER TABLE assistant_memories DROP COLUMN IF EXISTS derived_from;
ALTER TABLE assistant_memories DROP COLUMN IF EXISTS cluster_id;

COMMIT;
