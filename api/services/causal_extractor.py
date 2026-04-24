"""
Causal relationship extraction service.

Extracts cause-effect relationships from text using:
1. Pattern matching (fast, explicit causality)
2. LLM extraction (slower, implicit causality)

Validates temporal ordering when timestamps are available.
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import hashlib

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from manifold.config_v2 import ManifoldConfigV2, CausalConfig

logger = logging.getLogger("gami.causal_extractor")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CausalRelation:
    """Extracted causal relationship."""
    cause_text: str
    effect_text: str
    causal_type: str
    confidence: float = 0.5

    # Entity linkage
    cause_entity_id: Optional[str] = None
    effect_entity_id: Optional[str] = None

    # Temporal validation
    cause_timestamp: Optional[datetime] = None
    effect_timestamp: Optional[datetime] = None
    temporal_valid: Optional[bool] = None
    temporal_gap_hours: Optional[float] = None

    # Scoring
    explicitness_score: float = 0.5
    authority_score: float = 0.5

    # Provenance
    source_segment_id: Optional[str] = None
    extraction_pattern: Optional[str] = None
    extraction_method: str = "pattern"

    def causal_id(self) -> str:
        """Generate unique ID for this causal relation."""
        content = f"{self.cause_text}|{self.effect_text}|{self.causal_type}"
        return f"CAUS_{hashlib.md5(content.encode()).hexdigest()[:12]}"


@dataclass
class ExtractionContext:
    """Context for causal extraction."""
    segment_id: str
    tenant_id: str
    source_authority: float = 0.5
    entity_map: Dict[str, str] = field(default_factory=dict)  # name -> entity_id
    agent_id: Optional[str] = None
    timestamp: Optional[datetime] = None


# =============================================================================
# CAUSAL PATTERNS
# =============================================================================

# Pattern structure: (regex_pattern, causal_type, explicitness_score)
CAUSAL_PATTERNS: List[Tuple[str, str, float]] = [
    # Explicit causal connectors
    (r"(?:^|[.!?]\s*)([^.!?]+)\s+because\s+([^.!?]+)", "because", 0.95),
    (r"(?:^|[.!?]\s*)([^.!?]+)\s+since\s+([^.!?]+)", "because", 0.85),
    (r"([^.!?]+)\s+caused\s+by\s+([^.!?]+)", "caused_by", 0.95),
    (r"([^.!?]+)\s+was\s+caused\s+by\s+([^.!?]+)", "caused_by", 0.95),
    (r"([^.!?]+)\s+resulted\s+in\s+([^.!?]+)", "resulted_in", 0.90),
    (r"([^.!?]+)\s+results\s+in\s+([^.!?]+)", "resulted_in", 0.90),
    (r"([^.!?]+)\s+led\s+to\s+([^.!?]+)", "led_to", 0.85),
    (r"([^.!?]+)\s+leads\s+to\s+([^.!?]+)", "led_to", 0.85),
    (r"([^.!?]+)\s+due\s+to\s+([^.!?]+)", "due_to", 0.90),

    # Consequence connectors
    (r"(?:^|[.!?]\s*)therefore[,]?\s+([^.!?]+)", "therefore", 0.85),
    (r"(?:^|[.!?]\s*)consequently[,]?\s+([^.!?]+)", "consequently", 0.85),
    (r"(?:^|[.!?]\s*)as\s+a\s+result[,]?\s+([^.!?]+)", "as_result", 0.85),
    (r"(?:^|[.!?]\s*)thus[,]?\s+([^.!?]+)", "thus", 0.80),
    (r"(?:^|[.!?]\s*)hence[,]?\s+([^.!?]+)", "hence", 0.80),

    # Trigger/enable patterns
    (r"([^.!?]+)\s+triggered\s+([^.!?]+)", "triggered", 0.90),
    (r"([^.!?]+)\s+triggers\s+([^.!?]+)", "triggered", 0.90),
    (r"([^.!?]+)\s+enabled\s+([^.!?]+)", "enabled", 0.85),
    (r"([^.!?]+)\s+enables\s+([^.!?]+)", "enabled", 0.85),
    (r"([^.!?]+)\s+prevented\s+([^.!?]+)", "prevented", 0.85),
    (r"([^.!?]+)\s+prevents\s+([^.!?]+)", "prevented", 0.85),

    # Temporal-causal patterns
    (r"after\s+([^,]+),\s*([^.!?]+(?:broke|failed|stopped|crashed|errored))", "temporal_effect", 0.70),
    (r"when\s+(?:we\s+)?(?:changed|updated|modified|deleted|added)\s+([^,]+),\s*([^.!?]+)", "intervention", 0.75),
    (r"once\s+([^,]+),\s*([^.!?]+)", "temporal_effect", 0.65),

    # Diagnostic patterns (common in debugging)
    (r"([^.!?]+)\s+is\s+(?:caused|happening)\s+because\s+([^.!?]+)", "because", 0.90),
    (r"the\s+(?:reason|cause)\s+(?:for|of)\s+([^.!?]+)\s+is\s+([^.!?]+)", "caused_by", 0.90),
    (r"([^.!?]+)\s+(?:happened|occurred|started)\s+after\s+([^.!?]+)", "followed", 0.70),
    (r"if\s+([^,]+),\s+then\s+([^.!?]+)", "conditional", 0.60),
]


# =============================================================================
# CAUSAL EXTRACTOR SERVICE
# =============================================================================

class CausalExtractor:
    """Service for extracting causal relationships from text."""

    def __init__(self, config: Optional["ManifoldConfigV2"] = None):
        """Initialize the causal extractor.

        Args:
            config: Manifold configuration with causal settings
        """
        self._config = config
        self._compiled_patterns: Optional[List[Tuple[re.Pattern, str, float]]] = None

    @property
    def config(self) -> "CausalConfig":
        """Get causal config."""
        if self._config is None:
            from manifold.config_v2 import get_config
            self._config = get_config()
        return self._config.causal

    @property
    def patterns(self) -> List[Tuple[re.Pattern, str, float]]:
        """Get compiled causal patterns."""
        if self._compiled_patterns is None:
            self._compiled_patterns = [
                (re.compile(pattern, re.IGNORECASE | re.MULTILINE), causal_type, explicitness)
                for pattern, causal_type, explicitness in CAUSAL_PATTERNS
            ]
        return self._compiled_patterns

    async def extract(
        self,
        text: str,
        context: ExtractionContext,
    ) -> List[CausalRelation]:
        """Extract causal relationships from text.

        Args:
            text: Text to extract from
            context: Extraction context with metadata

        Returns:
            List of extracted CausalRelations
        """
        if not self.config.enabled:
            return []

        relations = []

        # Pattern-based extraction (fast)
        pattern_relations = self._extract_by_patterns(text, context)
        relations.extend(pattern_relations)

        # LLM extraction for implicit causality (if enabled and text is substantial)
        if self.config.llm_extraction_enabled and len(text) > 100:
            llm_relations = await self._extract_by_llm(text, context)

            # Deduplicate against pattern results
            for llm_rel in llm_relations:
                if not self._is_duplicate(llm_rel, relations):
                    relations.append(llm_rel)

        # Validate temporal ordering
        if self.config.require_temporal_validation:
            for rel in relations:
                self._validate_temporal(rel)

        # Filter by minimum explicitness
        min_explicitness = self.config.min_explicitness_score
        relations = [r for r in relations if r.explicitness_score >= min_explicitness]

        logger.debug(f"Extracted {len(relations)} causal relations from segment {context.segment_id}")
        return relations

    def _extract_by_patterns(
        self,
        text: str,
        context: ExtractionContext,
    ) -> List[CausalRelation]:
        """Extract causal relations using regex patterns."""
        relations = []

        for pattern, causal_type, explicitness in self.patterns:
            for match in pattern.finditer(text):
                groups = match.groups()

                if len(groups) < 1:
                    continue

                # Parse cause and effect from match groups based on pattern type
                if causal_type in ("therefore", "consequently", "as_result", "thus", "hence"):
                    # These patterns only capture effect, cause is in preceding context
                    effect_text = groups[0].strip()
                    cause_text = self._extract_preceding_context(text, match.start())
                elif causal_type in ("because", "due_to", "caused_by"):
                    # Effect is before the connector, cause is after
                    if len(groups) >= 2:
                        effect_text = groups[0].strip()
                        cause_text = groups[1].strip()
                    else:
                        continue
                else:
                    # Standard pattern: cause then effect
                    if len(groups) >= 2:
                        cause_text = groups[0].strip()
                        effect_text = groups[1].strip()
                    else:
                        continue

                # Skip empty or very short extractions
                if len(cause_text) < 5 or len(effect_text) < 5:
                    continue

                # Create relation
                relation = CausalRelation(
                    cause_text=cause_text,
                    effect_text=effect_text,
                    causal_type=causal_type,
                    confidence=explicitness * 0.9,  # Slightly discount for uncertainty
                    explicitness_score=explicitness,
                    authority_score=context.source_authority,
                    source_segment_id=context.segment_id,
                    extraction_pattern=pattern.pattern[:50],
                    extraction_method="pattern",
                )

                # Try to link to known entities
                relation.cause_entity_id = self._find_entity(cause_text, context.entity_map)
                relation.effect_entity_id = self._find_entity(effect_text, context.entity_map)

                relations.append(relation)

        return relations

    def _extract_preceding_context(self, text: str, position: int, max_chars: int = 200) -> str:
        """Extract preceding sentence as cause context."""
        start = max(0, position - max_chars)
        context = text[start:position]

        # Find last sentence boundary
        sentences = re.split(r'[.!?]\s+', context)
        if sentences:
            return sentences[-1].strip()
        return context.strip()

    def _find_entity(self, text: str, entity_map: Dict[str, str]) -> Optional[str]:
        """Find entity ID mentioned in text."""
        text_lower = text.lower()

        for entity_name, entity_id in entity_map.items():
            if entity_name.lower() in text_lower:
                return entity_id

        return None

    def _is_duplicate(self, new_rel: CausalRelation, existing: List[CausalRelation]) -> bool:
        """Check if relation is duplicate of existing one."""
        new_cause = new_rel.cause_text.lower()
        new_effect = new_rel.effect_text.lower()

        for rel in existing:
            existing_cause = rel.cause_text.lower()
            existing_effect = rel.effect_text.lower()

            # Check for high overlap
            cause_overlap = self._text_overlap(new_cause, existing_cause)
            effect_overlap = self._text_overlap(new_effect, existing_effect)

            if cause_overlap > 0.7 and effect_overlap > 0.7:
                return True

        return False

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union

    def _validate_temporal(self, relation: CausalRelation) -> None:
        """Validate temporal ordering of cause and effect."""
        if relation.cause_timestamp and relation.effect_timestamp:
            relation.temporal_valid = relation.cause_timestamp < relation.effect_timestamp

            if relation.temporal_valid:
                gap = relation.effect_timestamp - relation.cause_timestamp
                relation.temporal_gap_hours = gap.total_seconds() / 3600
            else:
                # Penalize confidence for invalid temporal order
                relation.confidence *= 0.5

    async def _extract_by_llm(
        self,
        text: str,
        context: ExtractionContext,
    ) -> List[CausalRelation]:
        """Extract implicit causal relations using LLM."""
        try:
            from .prompt_service import PromptService, get_default_prompt
            from .extraction import call_vllm

            # Get prompt (use default since we may not have async db session here)
            variables = {
                "text": text[:2000],  # Truncate for LLM
                "entities": list(context.entity_map.keys())[:20],
            }

            system_prompt, user_prompt, params = get_default_prompt(
                "causal_extraction",
                variables,
            )

            # Call LLM
            response = await call_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=params.get("temperature", 0.2),
                max_tokens=params.get("max_tokens", 2048),
            )

            # Parse response
            relations = self._parse_llm_response(response, context)
            return relations

        except Exception as e:
            logger.warning(f"LLM causal extraction failed: {e}")
            return []

    def _parse_llm_response(
        self,
        response: str,
        context: ExtractionContext,
    ) -> List[CausalRelation]:
        """Parse LLM response into CausalRelations."""
        import json

        relations = []

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                return []

            data = json.loads(json_match.group())

            for item in data:
                if not isinstance(item, dict):
                    continue

                cause = item.get("cause", "").strip()
                effect = item.get("effect", "").strip()

                if not cause or not effect:
                    continue

                relation = CausalRelation(
                    cause_text=cause,
                    effect_text=effect,
                    causal_type=item.get("causal_type", "inferred"),
                    confidence=float(item.get("confidence", 0.6)),
                    explicitness_score=0.5,  # LLM extractions are inherently less explicit
                    authority_score=context.source_authority,
                    source_segment_id=context.segment_id,
                    extraction_method="llm",
                )

                # Link entities
                if item.get("cause_entity"):
                    relation.cause_entity_id = context.entity_map.get(item["cause_entity"])
                else:
                    relation.cause_entity_id = self._find_entity(cause, context.entity_map)

                if item.get("effect_entity"):
                    relation.effect_entity_id = context.entity_map.get(item["effect_entity"])
                else:
                    relation.effect_entity_id = self._find_entity(effect, context.entity_map)

                relations.append(relation)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse LLM causal response: {e}")

        return relations


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def save_causal_relations(
    db: AsyncSession,
    relations: List[CausalRelation],
    tenant_id: str,
    agent_id: Optional[str] = None,
) -> int:
    """Save causal relations to database.

    Args:
        db: Database session
        relations: Relations to save
        tenant_id: Owner tenant
        agent_id: Creating agent

    Returns:
        Number of relations saved
    """
    if not relations:
        return 0

    saved = 0

    for rel in relations:
        try:
            # Check for existing relation
            existing = await db.execute(text("""
                SELECT 1 FROM causal_relations
                WHERE cause_text = :cause AND effect_text = :effect
                AND owner_tenant_id = :tenant_id
                AND status = 'active'
            """), {
                "cause": rel.cause_text[:500],
                "effect": rel.effect_text[:500],
                "tenant_id": tenant_id,
            })

            if existing.fetchone():
                # Update corroboration count
                await db.execute(text("""
                    UPDATE causal_relations SET
                        corroboration_count = corroboration_count + 1,
                        authority_score = GREATEST(authority_score, :authority)
                    WHERE cause_text = :cause AND effect_text = :effect
                    AND owner_tenant_id = :tenant_id AND status = 'active'
                """), {
                    "cause": rel.cause_text[:500],
                    "effect": rel.effect_text[:500],
                    "tenant_id": tenant_id,
                    "authority": rel.authority_score,
                })
                continue

            # Insert new relation
            await db.execute(text("""
                INSERT INTO causal_relations (
                    causal_id, owner_tenant_id,
                    cause_entity_id, cause_text,
                    effect_entity_id, effect_text,
                    causal_type, cause_timestamp, effect_timestamp,
                    temporal_valid, temporal_gap_hours,
                    explicitness_score, authority_score, corroboration_count,
                    source_segment_id, extraction_pattern, extraction_method,
                    created_by_agent_id
                ) VALUES (
                    :causal_id, :tenant_id,
                    :cause_entity, :cause_text,
                    :effect_entity, :effect_text,
                    :causal_type, :cause_ts, :effect_ts,
                    :temporal_valid, :temporal_gap,
                    :explicitness, :authority, 0,
                    :segment_id, :pattern, :method,
                    :agent_id
                )
            """), {
                "causal_id": rel.causal_id(),
                "tenant_id": tenant_id,
                "cause_entity": rel.cause_entity_id,
                "cause_text": rel.cause_text[:500],
                "effect_entity": rel.effect_entity_id,
                "effect_text": rel.effect_text[:500],
                "causal_type": rel.causal_type,
                "cause_ts": rel.cause_timestamp,
                "effect_ts": rel.effect_timestamp,
                "temporal_valid": rel.temporal_valid,
                "temporal_gap": rel.temporal_gap_hours,
                "explicitness": rel.explicitness_score,
                "authority": rel.authority_score,
                "segment_id": rel.source_segment_id,
                "pattern": rel.extraction_pattern,
                "method": rel.extraction_method,
                "agent_id": agent_id,
            })

            saved += 1

        except Exception as e:
            logger.error(f"Failed to save causal relation: {e}")

    await db.commit()
    logger.info(f"Saved {saved} causal relations for tenant {tenant_id}")
    return saved


async def get_causal_chain(
    db: AsyncSession,
    entity_id: str,
    direction: str = "both",
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """Get causal chain starting from an entity.

    Args:
        db: Database session
        entity_id: Starting entity
        direction: "causes" (effects of entity), "effects" (causes of entity), or "both"
        max_depth: Maximum chain depth

    Returns:
        List of causal relations in the chain
    """
    chain = []
    visited = set()

    async def _traverse(current_id: str, depth: int, dir: str):
        if depth > max_depth or current_id in visited:
            return
        visited.add(current_id)

        if dir in ("causes", "both"):
            # Find effects of current entity
            result = await db.execute(text("""
                SELECT causal_id, cause_text, effect_text, effect_entity_id,
                       causal_type, strength_score, temporal_valid
                FROM causal_relations
                WHERE cause_entity_id = :entity_id AND status = 'active'
                ORDER BY strength_score DESC
                LIMIT 10
            """), {"entity_id": current_id})

            for row in result.fetchall():
                chain.append({
                    "causal_id": row.causal_id,
                    "cause_text": row.cause_text,
                    "effect_text": row.effect_text,
                    "effect_entity_id": row.effect_entity_id,
                    "causal_type": row.causal_type,
                    "strength_score": row.strength_score,
                    "temporal_valid": row.temporal_valid,
                    "direction": "causes",
                    "depth": depth,
                })

                if row.effect_entity_id:
                    await _traverse(row.effect_entity_id, depth + 1, "causes")

        if dir in ("effects", "both"):
            # Find causes of current entity
            result = await db.execute(text("""
                SELECT causal_id, cause_text, effect_text, cause_entity_id,
                       causal_type, strength_score, temporal_valid
                FROM causal_relations
                WHERE effect_entity_id = :entity_id AND status = 'active'
                ORDER BY strength_score DESC
                LIMIT 10
            """), {"entity_id": current_id})

            for row in result.fetchall():
                chain.append({
                    "causal_id": row.causal_id,
                    "cause_text": row.cause_text,
                    "effect_text": row.effect_text,
                    "cause_entity_id": row.cause_entity_id,
                    "causal_type": row.causal_type,
                    "strength_score": row.strength_score,
                    "temporal_valid": row.temporal_valid,
                    "direction": "effects",
                    "depth": depth,
                })

                if row.cause_entity_id:
                    await _traverse(row.cause_entity_id, depth + 1, "effects")

    await _traverse(entity_id, 0, direction)
    return chain


# =============================================================================
# GLOBAL SERVICE INSTANCE
# =============================================================================

_extractor: Optional[CausalExtractor] = None


def get_causal_extractor() -> CausalExtractor:
    """Get global causal extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = CausalExtractor()
    return _extractor
