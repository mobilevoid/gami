"""Procedure normalizer — extracts structured procedures from text.

Takes instructional content like:
    "To deploy GAMI, first run the migrations, then start the API server,
     and finally verify the health endpoint."

And converts to structured form:
    title: "Deploy GAMI"
    prerequisites: []
    steps: [
        {"order": 1, "text": "run the migrations"},
        {"order": 2, "text": "start the API server"},
        {"order": 3, "text": "verify the health endpoint"}
    ]
    expected_outcome: None

The canonical form is then used to generate the embedding for the procedure manifold.
"""
import logging
import re
from typing import Optional, List, Dict, Any

from ..models.schemas import (
    CanonicalProcedureSchema,
    ProcedureStep,
)

logger = logging.getLogger("manifold.canonical.procedure")


# Patterns for detecting procedural content
INSTRUCTION_STARTERS = re.compile(
    r"^(to |how to |steps to |in order to |for |when you want to )",
    re.IGNORECASE,
)
NUMBERED_STEP = re.compile(
    r"^\s*(\d+)[.):]\s*(.+)$",
    re.MULTILINE,
)
BULLETED_STEP = re.compile(
    r"^\s*[-*•]\s*(.+)$",
    re.MULTILINE,
)
SEQUENCE_WORDS = re.compile(
    r"\b(first|second|third|fourth|fifth|then|next|after that|finally|lastly|"
    r"to begin|to start|to finish|in conclusion)\b",
    re.IGNORECASE,
)
PREREQUISITE_PATTERNS = re.compile(
    r"\b(requires?|needs?|must have|before you|prerequisites?|ensure that|"
    r"make sure|you will need|dependencies)\b",
    re.IGNORECASE,
)
OUTCOME_PATTERNS = re.compile(
    r"\b(result|outcome|you will have|you should see|this will|"
    r"after completing|when done|upon completion)\b",
    re.IGNORECASE,
)


class ProcedureNormalizer:
    """Extracts structured procedures from text.

    Uses pattern-based extraction for lists and LLM for prose instructions.
    """

    def __init__(self, use_llm: bool = True, vllm_url: Optional[str] = None):
        """Initialize the procedure normalizer.

        Args:
            use_llm: Whether to use LLM for complex extractions.
            vllm_url: URL of the vLLM server (default: from settings).
        """
        self.use_llm = use_llm
        self.vllm_url = vllm_url

    def normalize(
        self,
        text: str,
        source_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        owner_tenant_id: str = "shared",
    ) -> Optional[CanonicalProcedureSchema]:
        """Extract a procedure from text.

        Args:
            text: The source text containing procedure.
            source_id: Source ID for provenance.
            segment_id: Segment ID for provenance.
            owner_tenant_id: Tenant that owns this procedure.

        Returns:
            CanonicalProcedureSchema if a procedure was extracted, None otherwise.
        """
        if not text or len(text.strip()) < 20:
            return None

        # Check if this looks like procedural content
        if not self._is_procedural(text):
            return None

        # Try numbered list extraction
        steps = self._extract_numbered_steps(text)
        if not steps:
            # Try bulleted list extraction
            steps = self._extract_bulleted_steps(text)
        if not steps:
            # Try sequence word extraction
            steps = self._extract_sequence_steps(text)

        if not steps or len(steps) < 2:
            # Not enough steps to be a procedure
            if self.use_llm:
                return self._llm_extract(text, source_id, segment_id, owner_tenant_id)
            return None

        # Extract title
        title = self._extract_title(text)

        # Extract prerequisites
        prerequisites = self._extract_prerequisites(text)

        # Extract outcome
        outcome = self._extract_outcome(text)

        procedure = CanonicalProcedureSchema(
            source_id=source_id,
            segment_id=segment_id,
            title=title,
            prerequisites=prerequisites,
            steps=steps,
            expected_outcome=outcome,
            owner_tenant_id=owner_tenant_id,
            confidence=0.7,
        )

        return procedure

    def _is_procedural(self, text: str) -> bool:
        """Check if text appears to contain procedural content."""
        # Check for instruction starters
        if INSTRUCTION_STARTERS.search(text):
            return True
        # Check for numbered or bulleted lists
        if NUMBERED_STEP.search(text) or BULLETED_STEP.search(text):
            return True
        # Check for sequence words
        sequence_matches = SEQUENCE_WORDS.findall(text)
        if len(sequence_matches) >= 2:
            return True
        return False

    def _extract_numbered_steps(self, text: str) -> List[ProcedureStep]:
        """Extract steps from numbered list."""
        matches = NUMBERED_STEP.findall(text)
        steps = []
        for num, step_text in matches:
            step_text = step_text.strip()
            if len(step_text) > 5:  # Filter out very short items
                steps.append(ProcedureStep(
                    order=int(num),
                    text=step_text[:500],  # Limit step length
                    optional=self._is_optional_step(step_text),
                ))
        return steps

    def _extract_bulleted_steps(self, text: str) -> List[ProcedureStep]:
        """Extract steps from bulleted list."""
        matches = BULLETED_STEP.findall(text)
        steps = []
        for i, step_text in enumerate(matches, 1):
            step_text = step_text.strip()
            if len(step_text) > 5:
                steps.append(ProcedureStep(
                    order=i,
                    text=step_text[:500],
                    optional=self._is_optional_step(step_text),
                ))
        return steps

    def _extract_sequence_steps(self, text: str) -> List[ProcedureStep]:
        """Extract steps based on sequence words."""
        # Split by sequence words
        parts = re.split(
            r"\b(first|then|next|after that|finally|lastly)\b",
            text,
            flags=re.IGNORECASE,
        )

        steps = []
        order = 1
        for i, part in enumerate(parts):
            part = part.strip()
            # Skip the sequence words themselves
            if SEQUENCE_WORDS.fullmatch(part):
                continue
            # Clean up the part
            part = re.sub(r"^[,.:;]\s*", "", part)
            part = re.sub(r"[,.:;]\s*$", "", part)
            if len(part) > 10:
                steps.append(ProcedureStep(
                    order=order,
                    text=part[:500],
                    optional=self._is_optional_step(part),
                ))
                order += 1

        return steps

    def _is_optional_step(self, step_text: str) -> bool:
        """Check if a step is optional."""
        optional_patterns = re.compile(
            r"\b(optional|optionally|if needed|if desired|you may|"
            r"you can also|alternatively)\b",
            re.IGNORECASE,
        )
        return bool(optional_patterns.search(step_text))

    def _extract_title(self, text: str) -> str:
        """Extract a title for the procedure."""
        # Look for "How to X" or "To X" at the start
        match = re.match(
            r"^(?:how to |to |steps to |instructions for )(.+?)(?:[.:\n]|$)",
            text.strip(),
            re.IGNORECASE,
        )
        if match:
            title = match.group(1).strip()
            return title[:100]

        # Look for a heading-like first line
        first_line = text.strip().split("\n")[0]
        if len(first_line) < 100 and not first_line.endswith((".", "?", "!")):
            return first_line

        # Generate from content
        return "Procedure"

    def _extract_prerequisites(self, text: str) -> List[str]:
        """Extract prerequisites from text."""
        prerequisites = []

        # Find prerequisite sections
        prereq_match = re.search(
            r"(?:prerequisites?|requirements?|you will need|before you begin)[:\s]*(.+?)(?:\n\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if prereq_match:
            prereq_text = prereq_match.group(1)
            # Extract items
            items = re.findall(r"[-*•]\s*(.+)", prereq_text)
            if items:
                prerequisites = [item.strip()[:100] for item in items]
            else:
                # Split by commas or "and"
                parts = re.split(r",\s*|\s+and\s+", prereq_text)
                prerequisites = [p.strip()[:100] for p in parts if len(p.strip()) > 2]

        return prerequisites[:10]  # Limit to 10 prerequisites

    def _extract_outcome(self, text: str) -> Optional[str]:
        """Extract expected outcome from text."""
        # Look for outcome patterns
        match = re.search(
            r"(?:result|outcome|you will have|you should see|this will|"
            r"after completing|when done)[:\s]*(.+?)(?:[.\n]|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            outcome = match.group(1).strip()
            return outcome[:200]
        return None

    def _llm_extract(
        self,
        text: str,
        source_id: Optional[str],
        segment_id: Optional[str],
        owner_tenant_id: str,
    ) -> Optional[CanonicalProcedureSchema]:
        """Use LLM to extract procedure structure.

        NOTE: This method requires vLLM to be running.
        In the isolated manifold module, this is a stub that returns None.
        """
        # STUB: LLM extraction not connected in isolated module
        logger.debug("LLM extraction not available in isolated module")
        return None


def normalize_procedure(
    text: str,
    source_id: Optional[str] = None,
    segment_id: Optional[str] = None,
    owner_tenant_id: str = "shared",
    use_llm: bool = False,
) -> Optional[CanonicalProcedureSchema]:
    """Convenience function for procedure normalization.

    Args:
        text: Text containing procedure.
        source_id: Source ID for provenance.
        segment_id: Segment ID for provenance.
        owner_tenant_id: Owning tenant.
        use_llm: Whether to use LLM (disabled by default in isolated module).

    Returns:
        CanonicalProcedureSchema if successful, None otherwise.
    """
    normalizer = ProcedureNormalizer(use_llm=use_llm)
    return normalizer.normalize(text, source_id, segment_id, owner_tenant_id)


def batch_normalize_procedures(
    segments: List[Dict[str, Any]],
    owner_tenant_id: str = "shared",
    use_llm: bool = False,
) -> List[Optional[CanonicalProcedureSchema]]:
    """Normalize procedures from multiple segments.

    Args:
        segments: List of dicts with 'text', 'source_id', 'segment_id'.
        owner_tenant_id: Owning tenant.
        use_llm: Whether to use LLM.

    Returns:
        List of CanonicalProcedureSchema or None for each segment.
    """
    normalizer = ProcedureNormalizer(use_llm=use_llm)
    results = []
    for seg in segments:
        result = normalizer.normalize(
            seg.get("text", ""),
            seg.get("source_id"),
            seg.get("segment_id"),
            owner_tenant_id,
        )
        results.append(result)
    return results
