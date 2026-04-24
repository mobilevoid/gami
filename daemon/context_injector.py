"""
Context injector for proactive context management.

Prepares context for injection into agent responses:
1. Formats hot context for inclusion
2. Manages injection thresholds
3. Tracks injection acceptance
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .state_classifier import ConversationState

logger = logging.getLogger("gami.daemon.context_injector")


@dataclass
class InjectionContext:
    """Prepared context for injection."""
    text: str
    token_count: int
    segment_ids: List[str]
    relevance_score: float
    injection_type: str  # "proactive", "requested", "background"


@dataclass
class InjectionResult:
    """Result of an injection attempt."""
    injected: bool
    context: Optional[InjectionContext]
    reason: str
    session_id: str


class ContextInjector:
    """Service for preparing and injecting context."""

    def __init__(
        self,
        max_tokens: int = 500,
        injection_threshold: float = 0.7,
        format_template: Optional[str] = None,
    ):
        """Initialize context injector.

        Args:
            max_tokens: Maximum tokens for injection
            injection_threshold: Minimum relevance to inject
            format_template: Template for formatting context
        """
        self.max_tokens = max_tokens
        self.injection_threshold = injection_threshold
        self.format_template = format_template or self._default_template()

        # Track injection history per session
        self._injection_history: Dict[str, List[InjectionResult]] = {}

    def _default_template(self) -> str:
        """Default template for context formatting."""
        return """<relevant_context>
{context_items}
</relevant_context>"""

    async def prepare_injection(
        self,
        session_id: str,
        hot_context: List[Dict[str, Any]],
        current_state: ConversationState,
        agent_id: Optional[str] = None,
    ) -> Optional[InjectionContext]:
        """Prepare context for injection.

        Args:
            session_id: Session ID
            hot_context: Pre-loaded context segments
            current_state: Current conversation state
            agent_id: Agent ID for customization

        Returns:
            InjectionContext if threshold met, None otherwise
        """
        if not hot_context:
            return None

        # Score and filter context
        scored_context = self._score_context(hot_context, current_state)

        # Filter by threshold
        relevant = [
            (c, score) for c, score in scored_context
            if score >= self.injection_threshold
        ]

        if not relevant:
            logger.debug(f"No context meets threshold {self.injection_threshold} for {session_id}")
            return None

        # Sort by score
        relevant.sort(key=lambda x: x[1], reverse=True)

        # Build injection text within token budget
        context_text, segment_ids, token_count = self._build_injection_text(
            relevant,
            self.max_tokens,
        )

        if not context_text:
            return None

        avg_score = sum(s for _, s in relevant) / len(relevant)

        return InjectionContext(
            text=context_text,
            token_count=token_count,
            segment_ids=segment_ids,
            relevance_score=avg_score,
            injection_type="proactive",
        )

    def _score_context(
        self,
        context: List[Dict[str, Any]],
        state: ConversationState,
    ) -> List[tuple]:
        """Score context items based on state relevance."""
        # State-specific scoring adjustments
        state_weights = {
            ConversationState.DEBUGGING: {
                "error": 0.3, "log": 0.2, "config": 0.1, "diagnostic": 0.3
            },
            ConversationState.PLANNING: {
                "architecture": 0.2, "design": 0.2, "how_to": 0.2, "procedure": 0.1
            },
            ConversationState.RECALLING: {
                "conversation": 0.3, "decision": 0.2, "context": 0.2
            },
            ConversationState.EXPLORING: {
                "documentation": 0.3, "explanation": 0.2, "overview": 0.2
            },
            ConversationState.EXECUTING: {
                "procedure": 0.3, "command": 0.2, "config": 0.2
            },
            ConversationState.CONFIRMING: {
                "verification": 0.3, "test": 0.2, "result": 0.2
            },
        }

        weights = state_weights.get(state, {})
        scored = []

        for item in context:
            base_score = item.get("score", 0.5)
            segment_type = item.get("segment_type", "").lower()
            source_type = item.get("source_type", "").lower()

            # Apply state-specific boosts
            boost = 0.0
            for keyword, weight in weights.items():
                if keyword in segment_type or keyword in source_type:
                    boost += weight

            final_score = min(1.0, base_score + boost)
            scored.append((item, final_score))

        return scored

    def _build_injection_text(
        self,
        scored_context: List[tuple],
        max_tokens: int,
    ) -> tuple:
        """Build injection text within token budget.

        Returns: (text, segment_ids, token_count)
        """
        items = []
        segment_ids = []
        total_tokens = 0

        # Estimate tokens (rough: 4 chars per token)
        chars_per_token = 4

        for item, score in scored_context:
            text = item.get("text", "")
            segment_id = item.get("segment_id", "")

            # Estimate tokens for this item
            item_tokens = len(text) // chars_per_token

            if total_tokens + item_tokens > max_tokens:
                # Truncate to fit
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 50:
                    truncate_chars = remaining_tokens * chars_per_token
                    text = text[:truncate_chars] + "..."
                    item_tokens = remaining_tokens
                else:
                    break

            # Format item
            source = item.get("source", "unknown")
            formatted = f"[{source}] {text}"
            items.append(formatted)
            segment_ids.append(segment_id)
            total_tokens += item_tokens

            if total_tokens >= max_tokens:
                break

        if not items:
            return "", [], 0

        context_text = self.format_template.format(
            context_items="\n\n".join(items)
        )

        return context_text, segment_ids, total_tokens

    def record_injection(
        self,
        session_id: str,
        context: Optional[InjectionContext],
        accepted: bool,
        reason: str = "",
    ) -> InjectionResult:
        """Record an injection attempt.

        Args:
            session_id: Session ID
            context: Injected context (or None if skipped)
            accepted: Whether injection was accepted
            reason: Reason for outcome

        Returns:
            InjectionResult
        """
        result = InjectionResult(
            injected=accepted and context is not None,
            context=context,
            reason=reason,
            session_id=session_id,
        )

        if session_id not in self._injection_history:
            self._injection_history[session_id] = []

        self._injection_history[session_id].append(result)

        # Keep last 50 per session
        self._injection_history[session_id] = self._injection_history[session_id][-50:]

        return result

    def get_injection_stats(self, session_id: str) -> Dict[str, Any]:
        """Get injection statistics for a session."""
        history = self._injection_history.get(session_id, [])

        if not history:
            return {
                "total_attempts": 0,
                "successful": 0,
                "acceptance_rate": 0.0,
            }

        successful = sum(1 for r in history if r.injected)

        return {
            "total_attempts": len(history),
            "successful": successful,
            "acceptance_rate": successful / len(history) if history else 0.0,
            "recent_reasons": [r.reason for r in history[-5:]],
        }

    def adjust_threshold(
        self,
        session_id: str,
        acceptance_rate_target: float = 0.7,
    ) -> float:
        """Dynamically adjust injection threshold based on acceptance rate.

        Args:
            session_id: Session ID
            acceptance_rate_target: Target acceptance rate

        Returns:
            New threshold value
        """
        stats = self.get_injection_stats(session_id)

        if stats["total_attempts"] < 5:
            return self.injection_threshold

        current_rate = stats["acceptance_rate"]

        if current_rate < acceptance_rate_target - 0.1:
            # Too many rejections, raise threshold
            self.injection_threshold = min(0.95, self.injection_threshold + 0.05)
        elif current_rate > acceptance_rate_target + 0.1:
            # High acceptance, can lower threshold
            self.injection_threshold = max(0.3, self.injection_threshold - 0.05)

        return self.injection_threshold

    def clear_session(self, session_id: str) -> None:
        """Clear session history."""
        if session_id in self._injection_history:
            del self._injection_history[session_id]


class ContextFormatter:
    """Utility for formatting context for different agents."""

    @staticmethod
    def format_for_claude(context: InjectionContext) -> str:
        """Format context for Claude-style agents."""
        return f"""<system_context source="proactive_memory" relevance="{context.relevance_score:.2f}">
{context.text}
</system_context>"""

    @staticmethod
    def format_for_openai(context: InjectionContext) -> str:
        """Format context for OpenAI-style agents."""
        return f"""[Relevant Context - Relevance: {context.relevance_score:.0%}]
{context.text}
[End Context]"""

    @staticmethod
    def format_for_local(context: InjectionContext) -> str:
        """Format context for local models (simpler format)."""
        return f"""Context:
{context.text}

---"""

    @staticmethod
    def format_by_agent_type(
        context: InjectionContext,
        agent_type: str,
    ) -> str:
        """Format context based on agent type."""
        formatters = {
            "claude": ContextFormatter.format_for_claude,
            "anthropic": ContextFormatter.format_for_claude,
            "openai": ContextFormatter.format_for_openai,
            "gpt": ContextFormatter.format_for_openai,
            "local": ContextFormatter.format_for_local,
            "ollama": ContextFormatter.format_for_local,
            "vllm": ContextFormatter.format_for_local,
        }

        formatter = formatters.get(agent_type.lower(), ContextFormatter.format_for_local)
        return formatter(context)
