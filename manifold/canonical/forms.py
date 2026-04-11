"""Canonical form templates for manifold embeddings.

Defines the standardized text formats that get embedded into each manifold.
Different manifolds require different canonical representations of the same
underlying data.
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CanonicalClaimForm:
    """Canonical form for claim manifold embedding.

    Format:
        [SUBJECT] | [PREDICATE] | [OBJECT] | modality=X | qualifiers=[...] | time=[...]

    Example:
        [Vietnamese Communist Party] | [exercised control over] | [CPK] |
        modality=factual | qualifiers=[significant, formative period] | time=[through 1973]
    """
    subject: str
    predicate: str
    object: Optional[str] = None
    modality: str = "factual"
    qualifiers: List[str] = field(default_factory=list)
    temporal_scope: Optional[str] = None

    def to_text(self) -> str:
        """Generate canonical text for embedding."""
        parts = [f"[{self.subject}]", "|", f"[{self.predicate}]"]
        if self.object:
            parts.extend(["|", f"[{self.object}]"])
        parts.append(f"| modality={self.modality}")
        if self.qualifiers:
            parts.append(f"| qualifiers={self.qualifiers}")
        if self.temporal_scope:
            parts.append(f"| time=[{self.temporal_scope}]")
        return " ".join(parts)

    @classmethod
    def from_prose(
        cls,
        text: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> "CanonicalClaimForm":
        """Create from prose, using provided or extracted SPO.

        If subject/predicate/object not provided, uses the full text as subject.
        """
        return cls(
            subject=subject or text[:200],
            predicate=predicate or "states",
            object=object,
        )

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class CanonicalProcedureForm:
    """Canonical form for procedure manifold embedding.

    Format:
        title=[...] | prerequisites=[...] | steps=[...] | outcome=[...]

    Example:
        title=[Deploy Application to Production] |
        prerequisites=[PostgreSQL 16, Redis 7, Python 3.11] |
        steps=[1. Run migrations; 2. Start API; 3. Verify health] |
        outcome=[Application accessible on :9000]
    """
    title: str
    prerequisites: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_outcome: Optional[str] = None

    def to_text(self) -> str:
        """Generate canonical text for embedding."""
        parts = [f"title=[{self.title}]"]
        if self.prerequisites:
            parts.append(f"prerequisites=[{', '.join(self.prerequisites)}]")
        if self.steps:
            step_texts = []
            for i, step in enumerate(self.steps):
                order = step.get("order", i + 1)
                text = step.get("text", str(step))
                step_texts.append(f"{order}. {text}")
            parts.append(f"steps=[{'; '.join(step_texts)}]")
        if self.expected_outcome:
            parts.append(f"outcome=[{self.expected_outcome}]")
        return " | ".join(parts)

    @classmethod
    def from_steps(
        cls,
        title: str,
        steps: List[str],
        prerequisites: Optional[List[str]] = None,
        outcome: Optional[str] = None,
    ) -> "CanonicalProcedureForm":
        """Create from a list of step strings."""
        step_dicts = [{"order": i + 1, "text": step} for i, step in enumerate(steps)]
        return cls(
            title=title,
            prerequisites=prerequisites or [],
            steps=step_dicts,
            expected_outcome=outcome,
        )

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class CanonicalEntityForm:
    """Canonical form for entity in relation manifold.

    Format:
        entity=[name] | type=[type] | relations=[top N typed relations]

    Example:
        entity=[PostgreSQL] | type=[technology] |
        relations=[hosts: Manifold, used_by: server-01, version: 16]
    """
    name: str
    entity_type: str
    relations: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Generate canonical text for embedding."""
        parts = [
            f"entity=[{self.name}]",
            f"type=[{self.entity_type}]",
        ]
        if self.aliases:
            parts.append(f"aliases=[{', '.join(self.aliases[:5])}]")
        if self.relations:
            parts.append(f"relations=[{', '.join(self.relations[:10])}]")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class CanonicalEventForm:
    """Canonical form for events in time manifold.

    Format:
        event=[what] | when=[temporal] | where=[location] | actors=[who]

    Example:
        event=[Edge failover triggered] | when=[March 28, 2026] |
        where=[Walter gateway] | actors=[AT&T WAN, Stargate backup]
    """
    event: str
    when: Optional[str] = None
    where: Optional[str] = None
    actors: List[str] = field(default_factory=list)
    precedes: Optional[str] = None
    follows: Optional[str] = None

    def to_text(self) -> str:
        """Generate canonical text for embedding."""
        parts = [f"event=[{self.event}]"]
        if self.when:
            parts.append(f"when=[{self.when}]")
        if self.where:
            parts.append(f"where=[{self.where}]")
        if self.actors:
            parts.append(f"actors=[{', '.join(self.actors[:5])}]")
        if self.follows:
            parts.append(f"follows=[{self.follows}]")
        if self.precedes:
            parts.append(f"precedes=[{self.precedes}]")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class CanonicalSummaryForm:
    """Canonical form for summaries in evidence manifold.

    Format:
        summary=[text] | based_on=[source IDs] | sources=[count] |
        confidence=[score] | provenance=[density]

    Example:
        summary=[Vietnamese influence on CPK was paramount...] |
        based_on=[SEG_abc123, SEG_def456] | sources=[3] |
        confidence=[0.85] | provenance=[0.9]
    """
    summary_text: str
    based_on: List[str] = field(default_factory=list)
    source_count: int = 1
    confidence: float = 0.5
    provenance_density: float = 0.5

    def to_text(self) -> str:
        """Generate canonical text for embedding."""
        parts = [f"summary=[{self.summary_text[:500]}]"]
        if self.based_on:
            parts.append(f"based_on=[{', '.join(self.based_on[:5])}]")
        parts.append(f"sources=[{self.source_count}]")
        parts.append(f"confidence=[{self.confidence:.2f}]")
        parts.append(f"provenance=[{self.provenance_density:.2f}]")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.to_text()


# ---------------------------------------------------------------------------
# Form Factories
# ---------------------------------------------------------------------------

def create_claim_form(
    subject: str,
    predicate: str,
    object: Optional[str] = None,
    **kwargs,
) -> CanonicalClaimForm:
    """Factory function for claim forms."""
    return CanonicalClaimForm(
        subject=subject,
        predicate=predicate,
        object=object,
        modality=kwargs.get("modality", "factual"),
        qualifiers=kwargs.get("qualifiers", []),
        temporal_scope=kwargs.get("temporal_scope"),
    )


def create_procedure_form(
    title: str,
    steps: List[str],
    **kwargs,
) -> CanonicalProcedureForm:
    """Factory function for procedure forms."""
    return CanonicalProcedureForm.from_steps(
        title=title,
        steps=steps,
        prerequisites=kwargs.get("prerequisites"),
        outcome=kwargs.get("outcome"),
    )


def create_entity_form(
    name: str,
    entity_type: str,
    **kwargs,
) -> CanonicalEntityForm:
    """Factory function for entity forms."""
    return CanonicalEntityForm(
        name=name,
        entity_type=entity_type,
        relations=kwargs.get("relations", []),
        aliases=kwargs.get("aliases", []),
    )


def create_event_form(
    event: str,
    **kwargs,
) -> CanonicalEventForm:
    """Factory function for event forms."""
    return CanonicalEventForm(
        event=event,
        when=kwargs.get("when"),
        where=kwargs.get("where"),
        actors=kwargs.get("actors", []),
        precedes=kwargs.get("precedes"),
        follows=kwargs.get("follows"),
    )


def create_summary_form(
    summary_text: str,
    **kwargs,
) -> CanonicalSummaryForm:
    """Factory function for summary forms."""
    return CanonicalSummaryForm(
        summary_text=summary_text,
        based_on=kwargs.get("based_on", []),
        source_count=kwargs.get("source_count", 1),
        confidence=kwargs.get("confidence", 0.5),
        provenance_density=kwargs.get("provenance_density", 0.5),
    )
