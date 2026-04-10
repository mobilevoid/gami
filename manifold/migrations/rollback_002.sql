-- Rollback for Migration 002: Multi-Manifold Memory System Tables
--
-- Removes all manifold-related tables.
-- WARNING: This will delete all manifold embeddings and scores.
--
-- To rollback: psql -p 5433 -U gami -d gami -f rollback_002.sql

BEGIN;

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS shadow_comparisons;
DROP TABLE IF EXISTS manifold_config;
DROP TABLE IF EXISTS query_logs;
DROP TABLE IF EXISTS promotion_scores;
DROP TABLE IF EXISTS temporal_features;
DROP TABLE IF EXISTS canonical_procedures;
DROP TABLE IF EXISTS canonical_claims;
DROP TABLE IF EXISTS manifold_embeddings;

COMMIT;

-- Note: This rollback does NOT affect existing GAMI tables.
-- All original segments, entities, claims, memories remain intact.
