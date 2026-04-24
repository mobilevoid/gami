"""
Hierarchical Configuration System v2 for Manifold.

Config precedence (highest wins):
    Code Defaults → Env Vars → Config File → Database → Tenant Override → Agent Override

Features:
- All scoring weights configurable
- Per-tenant and per-agent overrides
- Hot-reload without restart
- Validation of weight constraints
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
import os
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("manifold.config")


# =============================================================================
# SCORING WEIGHTS
# =============================================================================

@dataclass
class ScoringWeights:
    """All scoring weights — configurable, validated.

    These weights control the multi-manifold retrieval scoring formulas.
    Changes take effect on next retrieval (no restart required with hot-reload).
    """

    # --- Evidence Manifold (must sum to 1.0) ---
    evidence_authority: float = 0.25
    evidence_corroboration: float = 0.30
    evidence_recency: float = 0.15
    evidence_specificity: float = 0.10
    evidence_non_contradiction: float = 0.20
    evidence_contradiction_exponent: float = 1.5  # Penalty multiplier

    # --- Relation Manifold (must sum to 1.0) ---
    relation_structural: float = 0.40
    relation_neighbor: float = 0.30
    relation_path: float = 0.30

    # --- Promotion Scoring (must sum to 1.0) ---
    promotion_importance: float = 0.20
    promotion_retrieval: float = 0.15
    promotion_diversity: float = 0.10
    promotion_confidence: float = 0.20
    promotion_novelty: float = 0.10
    promotion_centrality: float = 0.10
    promotion_relevance: float = 0.15

    # --- Beta Weights (secondary signals, don't need to sum to 1.0) ---
    beta_lexical: float = 0.05
    beta_alias: float = 0.03
    beta_cache: float = 0.02
    beta_prior_importance: float = 0.02

    # --- Penalties ---
    penalty_noise: float = 0.10
    penalty_duplicate: float = 0.15
    penalty_contradiction: float = 0.20
    penalty_low_confidence: float = 0.05

    # --- Source Authority by Type ---
    authority_documentation: float = 0.90
    authority_configuration: float = 0.85
    authority_code_comment: float = 0.70
    authority_assistant_response: float = 0.60
    authority_conversation: float = 0.50
    authority_user_message: float = 0.40
    authority_log: float = 0.30
    authority_unknown: float = 0.30

    # --- Temporal Decay ---
    recency_halflife_days: float = 60.0  # Evidence manifold
    importance_halflife_days: float = 30.0  # Promotion scoring

    def validate(self) -> List[str]:
        """Validate that weight groups sum correctly."""
        errors = []

        # Evidence weights
        evidence_sum = (
            self.evidence_authority + self.evidence_corroboration +
            self.evidence_recency + self.evidence_specificity +
            self.evidence_non_contradiction
        )
        if abs(evidence_sum - 1.0) > 0.01:
            errors.append(f"Evidence weights sum to {evidence_sum:.3f}, expected 1.0")

        # Relation weights
        relation_sum = (
            self.relation_structural + self.relation_neighbor + self.relation_path
        )
        if abs(relation_sum - 1.0) > 0.01:
            errors.append(f"Relation weights sum to {relation_sum:.3f}, expected 1.0")

        # Promotion weights
        promotion_sum = (
            self.promotion_importance + self.promotion_retrieval +
            self.promotion_diversity + self.promotion_confidence +
            self.promotion_novelty + self.promotion_centrality +
            self.promotion_relevance
        )
        if abs(promotion_sum - 1.0) > 0.01:
            errors.append(f"Promotion weights sum to {promotion_sum:.3f}, expected 1.0")

        # Range checks
        for name, value in asdict(self).items():
            if isinstance(value, float):
                if "halflife" in name:
                    if value <= 0:
                        errors.append(f"{name} must be positive, got {value}")
                elif "exponent" in name:
                    if value < 0:
                        errors.append(f"{name} must be non-negative, got {value}")
                elif value < 0 or value > 1:
                    errors.append(f"{name} must be in [0, 1], got {value}")

        return errors

    def get_authority(self, source_type: str) -> float:
        """Get authority score for a source type."""
        attr_name = f"authority_{source_type}"
        return getattr(self, attr_name, self.authority_unknown)


# =============================================================================
# FEATURE CONFIGS
# =============================================================================

@dataclass
class LearningConfig:
    """Retrieval learning configuration.

    Controls how the system learns from user feedback on retrievals.
    """
    enabled: bool = True
    bandit_alpha: float = 0.1  # Learning rate for score adjustments
    positive_signal_weight: float = 1.0
    negative_signal_weight: float = 1.5  # Penalize negatives more
    decay_halflife_days: float = 30.0
    min_observations: int = 5  # Minimum signals before adjusting
    max_adjustment: float = 0.3  # Maximum single adjustment magnitude

    def validate(self) -> List[str]:
        errors = []
        if self.bandit_alpha <= 0 or self.bandit_alpha > 1:
            errors.append(f"bandit_alpha must be in (0, 1], got {self.bandit_alpha}")
        if self.min_observations < 1:
            errors.append(f"min_observations must be >= 1, got {self.min_observations}")
        if self.max_adjustment <= 0 or self.max_adjustment > 1:
            errors.append(f"max_adjustment must be in (0, 1], got {self.max_adjustment}")
        return errors


@dataclass
class ConsolidationConfig:
    """Memory consolidation configuration.

    Controls clustering, abstraction, decay, and inference generation.
    """
    enabled: bool = True
    cluster_similarity_threshold: float = 0.85
    min_cluster_size: int = 3
    max_cluster_size: int = 50
    abstraction_enabled: bool = True
    decay_enabled: bool = True
    decay_rate_per_day: float = 0.01
    archive_threshold: float = 0.1  # Decay below this = archive
    reinforcement_boost: float = 0.1
    inference_enabled: bool = True
    inference_confidence_threshold: float = 0.6

    def validate(self) -> List[str]:
        errors = []
        if self.cluster_similarity_threshold < 0.5 or self.cluster_similarity_threshold > 1:
            errors.append(f"cluster_similarity_threshold should be in [0.5, 1], got {self.cluster_similarity_threshold}")
        if self.min_cluster_size < 2:
            errors.append(f"min_cluster_size must be >= 2, got {self.min_cluster_size}")
        if self.archive_threshold < 0 or self.archive_threshold > 0.5:
            errors.append(f"archive_threshold should be in [0, 0.5], got {self.archive_threshold}")
        return errors


@dataclass
class CausalConfig:
    """Causal extraction configuration.

    Controls pattern-based and LLM-based causal relation extraction.
    """
    enabled: bool = True
    patterns: List[str] = field(default_factory=lambda: [
        r"because\s+(.+)",
        r"caused\s+by\s+(.+)",
        r"resulted\s+in\s+(.+)",
        r"led\s+to\s+(.+)",
        r"due\s+to\s+(.+)",
        r"therefore\s+(.+)",
        r"consequently\s+(.+)",
        r"as\s+a\s+result\s+of\s+(.+)",
        r"triggered\s+(.+)",
        r"enabled\s+(.+)",
        r"prevented\s+(.+)",
    ])
    require_temporal_validation: bool = True
    min_explicitness_score: float = 0.3
    llm_extraction_enabled: bool = True

    def validate(self) -> List[str]:
        errors = []
        if self.min_explicitness_score < 0 or self.min_explicitness_score > 1:
            errors.append(f"min_explicitness_score must be in [0, 1]")
        return errors


@dataclass
class SubconsciousConfig:
    """Subconscious daemon configuration.

    Controls the proactive context management daemon.
    """
    enabled: bool = False  # Off by default, enable after validation
    poll_interval_ms: int = 500
    state_classification_model: str = "qwen3:8b"
    state_classification_via_patterns: bool = True  # Fast path
    state_classification_via_llm: bool = True  # Fallback
    predictive_retrieval_enabled: bool = True
    hot_context_size: int = 20
    hot_context_ttl_seconds: int = 300
    injection_enabled: bool = True
    injection_threshold: float = 0.8
    injection_max_tokens: int = 500

    # State patterns (regex) - loaded from prompt_templates table in production
    state_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "debugging": [
            r"error", r"fail", r"bug", r"crash", r"exception",
            r"traceback", r"not working", r"broken"
        ],
        "planning": [
            r"how (should|do) (we|I)", r"plan", r"steps to",
            r"approach", r"strategy", r"design", r"implement"
        ],
        "recalling": [
            r"remember", r"last time", r"previously", r"what was",
            r"you told me", r"we discussed", r"earlier"
        ],
        "exploring": [
            r"what is", r"explain", r"how does", r"tell me about",
            r"describe", r"overview"
        ],
        "executing": [
            r"do it", r"run", r"execute", r"apply", r"make the change",
            r"go ahead"
        ],
    })

    def validate(self) -> List[str]:
        errors = []
        if self.poll_interval_ms < 100:
            errors.append(f"poll_interval_ms should be >= 100, got {self.poll_interval_ms}")
        if self.hot_context_size < 1 or self.hot_context_size > 100:
            errors.append(f"hot_context_size should be in [1, 100], got {self.hot_context_size}")
        return errors


# =============================================================================
# MAIN CONFIG
# =============================================================================

@dataclass
class ManifoldConfigV2:
    """Complete manifold configuration with all tunable parameters.

    This is the main configuration class. Use get_config() to get the
    current configuration with any overrides applied.
    """

    # --- Connection Settings ---
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    database_url: str = "postgresql://gami:gami@localhost:5433/gami"
    redis_url: str = "redis://localhost:6380/0"
    graph_name: str = "manifold_graph"

    # --- Retrieval Settings ---
    default_top_k: int = 20
    max_top_k: int = 100
    similarity_threshold: float = 0.3
    rerank_multiplier: int = 3

    # --- Graph Expansion ---
    graph_max_hops: int = 2
    graph_beam_width: int = 20
    graph_fanout_limit: int = 12
    graph_edge_decay: float = 0.7

    # --- Promotion Thresholds ---
    promotion_threshold: float = 0.65
    demotion_threshold: float = 0.35

    # --- Cache Settings ---
    cache_query_ttl: int = 300  # seconds
    cache_embedding_ttl: int = 3600  # seconds

    # --- Worker Settings ---
    worker_concurrency: int = 4
    worker_batch_size_embed: int = 100
    worker_batch_size_extract: int = 50

    # --- Feature Configs ---
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    learning: LearningConfig = field(default_factory=LearningConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    subconscious: SubconsciousConfig = field(default_factory=SubconsciousConfig)

    # --- Debug Settings ---
    shadow_mode: bool = False
    shadow_sample_rate: float = 0.1
    debug_scoring: bool = False

    @classmethod
    def from_env(cls) -> "ManifoldConfigV2":
        """Load configuration from environment variables.

        Supports both flat env vars and JSON-encoded nested configs.
        """
        config = cls()

        # Map of env vars to config fields
        env_map = {
            # Connection
            "MANIFOLD_EMBEDDING_MODEL": "embedding_model",
            "MANIFOLD_EMBEDDING_DIM": ("embedding_dim", int),
            "MANIFOLD_OLLAMA_URL": "ollama_url",
            "MANIFOLD_VLLM_URL": "vllm_url",
            "DATABASE_URL": "database_url",
            "REDIS_URL": "redis_url",
            "MANIFOLD_GRAPH_NAME": "graph_name",

            # Retrieval
            "MANIFOLD_DEFAULT_TOP_K": ("default_top_k", int),
            "MANIFOLD_MAX_TOP_K": ("max_top_k", int),
            "MANIFOLD_SIMILARITY_THRESHOLD": ("similarity_threshold", float),

            # Thresholds
            "MANIFOLD_PROMOTION_THRESHOLD": ("promotion_threshold", float),
            "MANIFOLD_DEMOTION_THRESHOLD": ("demotion_threshold", float),

            # Features
            "MANIFOLD_LEARNING_ENABLED": ("learning.enabled", _parse_bool),
            "MANIFOLD_CONSOLIDATION_ENABLED": ("consolidation.enabled", _parse_bool),
            "MANIFOLD_CAUSAL_ENABLED": ("causal.enabled", _parse_bool),
            "MANIFOLD_SUBCONSCIOUS_ENABLED": ("subconscious.enabled", _parse_bool),

            # Debug
            "MANIFOLD_SHADOW_MODE": ("shadow_mode", _parse_bool),
            "MANIFOLD_SHADOW_SAMPLE_RATE": ("shadow_sample_rate", float),
            "MANIFOLD_DEBUG_SCORING": ("debug_scoring", _parse_bool),
        }

        for env_key, field_spec in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                try:
                    if isinstance(field_spec, tuple):
                        field_name, converter = field_spec
                        value = converter(value)
                    else:
                        field_name = field_spec

                    # Handle nested fields (e.g., "learning.enabled")
                    if "." in field_name:
                        obj_name, attr_name = field_name.split(".", 1)
                        setattr(getattr(config, obj_name), attr_name, value)
                    else:
                        setattr(config, field_name, value)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse {env_key}={value}: {e}")

        # Check for JSON-encoded scoring overrides
        scoring_json = os.environ.get("MANIFOLD_SCORING_WEIGHTS")
        if scoring_json:
            try:
                scoring_data = json.loads(scoring_json)
                for key, value in scoring_data.items():
                    if hasattr(config.scoring, key):
                        setattr(config.scoring, key, float(value))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse MANIFOLD_SCORING_WEIGHTS: {e}")

        return config

    @classmethod
    def from_file(cls, path: Path) -> "ManifoldConfigV2":
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ManifoldConfigV2":
        """Create config from a dictionary, handling nested dataclasses."""
        # Extract nested configs
        scoring_data = data.pop("scoring", {})
        learning_data = data.pop("learning", {})
        consolidation_data = data.pop("consolidation", {})
        causal_data = data.pop("causal", {})
        subconscious_data = data.pop("subconscious", {})

        # Create nested configs
        scoring = ScoringWeights(**scoring_data) if scoring_data else ScoringWeights()
        learning = LearningConfig(**learning_data) if learning_data else LearningConfig()
        consolidation = ConsolidationConfig(**consolidation_data) if consolidation_data else ConsolidationConfig()
        causal = CausalConfig(**causal_data) if causal_data else CausalConfig()
        subconscious = SubconsciousConfig(**subconscious_data) if subconscious_data else SubconsciousConfig()

        return cls(
            **data,
            scoring=scoring,
            learning=learning,
            consolidation=consolidation,
            causal=causal,
            subconscious=subconscious,
        )

    def merge_overrides(self, overrides: Dict[str, Any]) -> "ManifoldConfigV2":
        """Create new config with overrides applied.

        Performs deep merge of nested configs.
        """
        base = asdict(self)
        _deep_merge(base, overrides)
        return self._from_dict(base)

    def validate(self) -> List[str]:
        """Validate all configuration settings."""
        errors = []

        # Validate nested configs
        errors.extend(self.scoring.validate())
        errors.extend(self.learning.validate())
        errors.extend(self.consolidation.validate())
        errors.extend(self.causal.validate())
        errors.extend(self.subconscious.validate())

        # Validate main config
        if self.promotion_threshold <= self.demotion_threshold:
            errors.append(
                f"promotion_threshold ({self.promotion_threshold}) must be > "
                f"demotion_threshold ({self.demotion_threshold})"
            )

        if self.default_top_k > self.max_top_k:
            errors.append(
                f"default_top_k ({self.default_top_k}) must be <= "
                f"max_top_k ({self.max_top_k})"
            )

        if self.embedding_dim not in (384, 768, 1024, 1536):
            errors.append(
                f"embedding_dim ({self.embedding_dim}) should be a standard dimension"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes", "on")


def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# =============================================================================
# GLOBAL CONFIG SINGLETON
# =============================================================================

_config: Optional[ManifoldConfigV2] = None
_config_loaded_at: Optional[datetime] = None


def get_config() -> ManifoldConfigV2:
    """Get the current configuration.

    Loads from environment on first call, caches thereafter.
    Use set_config() to override or reload_config() to refresh.
    """
    global _config, _config_loaded_at

    if _config is None:
        _config = ManifoldConfigV2.from_env()
        _config_loaded_at = datetime.now()

        # Validate and warn
        errors = _config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config validation: {error}")

    return _config


def set_config(config: ManifoldConfigV2) -> None:
    """Set the global configuration."""
    global _config, _config_loaded_at

    errors = config.validate()
    if errors:
        for error in errors:
            logger.warning(f"Config validation: {error}")

    _config = config
    _config_loaded_at = datetime.now()


def reload_config() -> ManifoldConfigV2:
    """Reload configuration from environment."""
    global _config
    _config = None
    return get_config()


def get_config_age_seconds() -> Optional[float]:
    """Get seconds since config was loaded."""
    if _config_loaded_at is None:
        return None
    return (datetime.now() - _config_loaded_at).total_seconds()
