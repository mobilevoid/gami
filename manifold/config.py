"""Configuration for multi-manifold memory system.

All tunable weights, thresholds, and parameters in one place.
Override via environment variables or config file.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import json


@dataclass
class ManifoldConfig:
    """Central configuration for manifold system."""

    # === Embedding Settings ===
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    ollama_url: str = "http://localhost:11434"

    # === Manifold Dimensions ===
    topic_dim: int = 768
    claim_dim: int = 768
    procedure_dim: int = 768
    temporal_features: int = 12
    evidence_features: int = 5

    # === Alpha Weight Defaults (per query mode) ===
    # These are the default weights; actual weights come from schemas.ALPHA_WEIGHTS
    alpha_topic_default: float = 0.25
    alpha_claim_default: float = 0.20
    alpha_procedure_default: float = 0.15
    alpha_relation_default: float = 0.15
    alpha_time_default: float = 0.10
    alpha_evidence_default: float = 0.15

    # === Secondary Signal Weights ===
    beta_lexical: float = 0.05
    beta_alias: float = 0.03
    beta_cache: float = 0.02

    # === Penalty Weights ===
    penalty_noise: float = 0.10
    penalty_duplicate: float = 0.15
    penalty_contradiction: float = 0.20
    penalty_low_confidence: float = 0.05

    # === Promotion Scoring ===
    promotion_threshold: float = 0.65
    demotion_threshold: float = 0.35
    # Factor weights (must sum to 1.0)
    promotion_weight_importance: float = 0.20
    promotion_weight_retrieval: float = 0.15
    promotion_weight_diversity: float = 0.10
    promotion_weight_confidence: float = 0.20
    promotion_weight_novelty: float = 0.10
    promotion_weight_centrality: float = 0.10
    promotion_weight_relevance: float = 0.15

    # === Retrieval Settings ===
    default_top_k: int = 20
    max_top_k: int = 100
    similarity_threshold: float = 0.3
    rerank_multiplier: int = 3  # Fetch rerank_multiplier * top_k for reranking

    # === Graph Expansion ===
    max_expansion_hops: int = 2
    max_fanout_per_hop: int = 10
    edge_weight_decay: float = 0.7  # Weight multiplier per hop

    # === Temporal Settings ===
    recency_halflife_days: float = 30.0
    temporal_bucket_hours: int = 6  # Granularity for time features

    # === Claim Processing ===
    min_claim_confidence: float = 0.5
    max_claims_per_segment: int = 10
    spo_extraction_model: str = "qwen3:8b"

    # === Procedure Processing ===
    min_steps_for_procedure: int = 2
    max_steps_per_procedure: int = 50

    # === Shadow Mode ===
    shadow_mode_enabled: bool = False
    shadow_sample_rate: float = 0.1  # Fraction of queries to shadow

    # === Cache Settings ===
    redis_url: str = "redis://localhost:6380/0"
    query_cache_ttl_seconds: int = 300
    embedding_cache_ttl_seconds: int = 3600

    # === Background Processing ===
    batch_size_embed: int = 100
    batch_size_extract: int = 50
    worker_concurrency: int = 4

    def validate(self) -> list:
        """Validate configuration, return list of errors."""
        errors = []

        # Check weight sums
        alpha_sum = (
            self.alpha_topic_default + self.alpha_claim_default +
            self.alpha_procedure_default + self.alpha_relation_default +
            self.alpha_time_default + self.alpha_evidence_default
        )
        if abs(alpha_sum - 1.0) > 0.01:
            errors.append(f"Alpha weights sum to {alpha_sum}, should be 1.0")

        promotion_sum = (
            self.promotion_weight_importance + self.promotion_weight_retrieval +
            self.promotion_weight_diversity + self.promotion_weight_confidence +
            self.promotion_weight_novelty + self.promotion_weight_centrality +
            self.promotion_weight_relevance
        )
        if abs(promotion_sum - 1.0) > 0.01:
            errors.append(f"Promotion weights sum to {promotion_sum}, should be 1.0")

        # Check thresholds
        if self.promotion_threshold <= self.demotion_threshold:
            errors.append("Promotion threshold must be > demotion threshold")

        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            errors.append("Similarity threshold must be in [0, 1]")

        return errors

    @classmethod
    def from_env(cls) -> "ManifoldConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Override from environment
        env_mappings = {
            "MANIFOLD_EMBEDDING_MODEL": "embedding_model",
            "MANIFOLD_EMBEDDING_DIM": ("embedding_dim", int),
            "MANIFOLD_OLLAMA_URL": "ollama_url",
            "MANIFOLD_PROMOTION_THRESHOLD": ("promotion_threshold", float),
            "MANIFOLD_DEMOTION_THRESHOLD": ("demotion_threshold", float),
            "MANIFOLD_DEFAULT_TOP_K": ("default_top_k", int),
            "MANIFOLD_MAX_TOP_K": ("max_top_k", int),
            "MANIFOLD_SHADOW_MODE": ("shadow_mode_enabled", lambda x: x.lower() == "true"),
            "MANIFOLD_SHADOW_SAMPLE_RATE": ("shadow_sample_rate", float),
        }

        for env_var, mapping in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if isinstance(mapping, tuple):
                    attr_name, converter = mapping
                    setattr(config, attr_name, converter(value))
                else:
                    setattr(config, mapping, value)

        return config

    @classmethod
    def from_file(cls, path: str) -> "ManifoldConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global config instance
_config: Optional[ManifoldConfig] = None


def get_config() -> ManifoldConfig:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = ManifoldConfig.from_env()
    return _config


def set_config(config: ManifoldConfig) -> None:
    """Set global config instance."""
    global _config
    _config = config
