-- Product Manifold Coordinates Migration
-- TRUE manifold embeddings: H^32 × S^16 × E^64 (112 dimensions total)
--
-- This replaces the fake "manifold" system (which was just 768d flat Euclidean)
-- with real geometric spaces:
-- - Hyperbolic (Poincaré ball): For hierarchical structure
-- - Spherical: For categorical/type information
-- - Euclidean: For general semantic similarity (pgvector compatible)

BEGIN;

-- Helper function to compute sum of squares for arrays
CREATE OR REPLACE FUNCTION array_sum_squares(arr FLOAT[])
RETURNS FLOAT AS $$
DECLARE
    result FLOAT := 0;
    i INT;
BEGIN
    FOR i IN 1..array_length(arr, 1) LOOP
        result := result + arr[i] * arr[i];
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Poincaré distance function for hyperbolic space
-- d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))
CREATE OR REPLACE FUNCTION poincare_distance(a FLOAT[], b FLOAT[])
RETURNS FLOAT AS $$
DECLARE
    diff_sq FLOAT := 0;
    norm_a_sq FLOAT := 0;
    norm_b_sq FLOAT := 0;
    i INT;
    denom FLOAT;
BEGIN
    -- Validate array lengths match
    IF array_length(a, 1) != array_length(b, 1) THEN
        RAISE EXCEPTION 'Array lengths must match';
    END IF;

    FOR i IN 1..array_length(a, 1) LOOP
        diff_sq := diff_sq + (a[i] - b[i]) * (a[i] - b[i]);
        norm_a_sq := norm_a_sq + a[i] * a[i];
        norm_b_sq := norm_b_sq + b[i] * b[i];
    END LOOP;

    -- Clamp norms to stay inside the Poincaré ball
    IF norm_a_sq >= 1 THEN norm_a_sq := 0.9999; END IF;
    IF norm_b_sq >= 1 THEN norm_b_sq := 0.9999; END IF;

    denom := (1 - norm_a_sq) * (1 - norm_b_sq);
    IF denom <= 0 THEN denom := 0.0001; END IF;

    RETURN acosh(1 + 2 * diff_sq / denom);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Spherical (great circle) distance function
-- d(x,y) = arccos(x·y / (||x|| ||y||))
CREATE OR REPLACE FUNCTION spherical_distance(a FLOAT[], b FLOAT[])
RETURNS FLOAT AS $$
DECLARE
    dot_product FLOAT := 0;
    norm_a FLOAT := 0;
    norm_b FLOAT := 0;
    i INT;
    cos_angle FLOAT;
BEGIN
    IF array_length(a, 1) != array_length(b, 1) THEN
        RAISE EXCEPTION 'Array lengths must match';
    END IF;

    FOR i IN 1..array_length(a, 1) LOOP
        dot_product := dot_product + a[i] * b[i];
        norm_a := norm_a + a[i] * a[i];
        norm_b := norm_b + b[i] * b[i];
    END LOOP;

    norm_a := sqrt(norm_a);
    norm_b := sqrt(norm_b);

    IF norm_a < 0.0001 OR norm_b < 0.0001 THEN
        RETURN 3.14159;  -- Max distance if either is zero vector
    END IF;

    cos_angle := dot_product / (norm_a * norm_b);
    -- Clamp to [-1, 1] for numerical stability
    IF cos_angle > 1 THEN cos_angle := 1; END IF;
    IF cos_angle < -1 THEN cos_angle := -1; END IF;

    RETURN acos(cos_angle);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

-- Main table for product manifold coordinates
CREATE TABLE IF NOT EXISTS product_manifold_coords (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,  -- segment, entity, claim, memory, cluster

    -- Hyperbolic coordinates (32d Poincaré ball)
    -- Points inside unit ball: ||x|| < 1
    -- Distance grows exponentially toward boundary
    hyperbolic_coords FLOAT[32] NOT NULL,

    -- Spherical coordinates (16d unit sphere)
    -- Points on unit sphere: ||x|| = 1
    -- Great circle distance
    spherical_coords FLOAT[16] NOT NULL,

    -- Euclidean coordinates (64d)
    -- Standard L2 distance, compatible with pgvector
    euclidean_coords vector(64),

    -- Metadata
    projection_version INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5,
    computed_at TIMESTAMPTZ DEFAULT NOW(),

    -- Unique constraint: one set of coords per target
    CONSTRAINT uq_product_manifold_coords UNIQUE(target_id, target_type)
);

-- Index for fast Euclidean ANN search (pre-filter stage)
-- This is the fast path for two-stage retrieval
CREATE INDEX IF NOT EXISTS idx_pmc_euclidean ON product_manifold_coords
    USING ivfflat (euclidean_coords vector_l2_ops) WITH (lists = 100);

-- Index for target lookups
CREATE INDEX IF NOT EXISTS idx_pmc_target ON product_manifold_coords(target_id, target_type);

-- Index for finding items by type
CREATE INDEX IF NOT EXISTS idx_pmc_type ON product_manifold_coords(target_type);

-- Index for finding recently computed coords
CREATE INDEX IF NOT EXISTS idx_pmc_computed ON product_manifold_coords(computed_at DESC);

-- Add comment explaining the table
COMMENT ON TABLE product_manifold_coords IS
'Product manifold coordinates H^32 × S^16 × E^64 for true manifold-based retrieval.
Hyperbolic space captures hierarchical structure, spherical space captures categories,
Euclidean space captures semantic similarity.';

COMMENT ON COLUMN product_manifold_coords.hyperbolic_coords IS
'32-dimensional Poincaré ball coordinates. ||x|| < 1. Distance grows exponentially toward boundary.';

COMMENT ON COLUMN product_manifold_coords.spherical_coords IS
'16-dimensional unit sphere coordinates. ||x|| = 1. Great circle distance.';

COMMENT ON COLUMN product_manifold_coords.euclidean_coords IS
'64-dimensional Euclidean coordinates. Standard L2 distance. Used for fast pgvector pre-filtering.';

COMMIT;
