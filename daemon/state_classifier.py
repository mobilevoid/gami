"""
Conversation state classifier.

Classifies conversation state from recent messages using:
1. Pattern matching (fast, high confidence for explicit patterns)
2. LLM classification (slower, for ambiguous cases)
"""

import re
import logging
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger("gami.daemon.state_classifier")


class ConversationState(Enum):
    """Possible conversation states."""
    IDLE = "idle"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    RECALLING = "recalling"
    EXPLORING = "exploring"
    EXECUTING = "executing"
    CONFIRMING = "confirming"


# Default state patterns (can be overridden by config)
DEFAULT_STATE_PATTERNS: Dict[str, List[str]] = {
    "debugging": [
        r"\b(?:error|fail|bug|crash|exception|traceback|broken|issue|problem)\b",
        r"\b(?:not working|doesn't work|won't start|can't connect)\b",
        r"\b(?:debug|investigate|diagnose|troubleshoot|fix)\b",
        r"(?:why|what) (?:is|are|did) .+ (?:fail|error|crash)",
        r"stack\s*trace",
        r"(?:got|getting|seeing|have) (?:an? )?error",
    ],
    "planning": [
        r"\b(?:how (?:should|do|would|can) (?:we|I|you))\b",
        r"\b(?:plan|approach|strategy|design|architect)\b",
        r"\b(?:steps? to|roadmap|outline)\b",
        r"\b(?:implement|build|create|develop|make)\b.+\b(?:how|way|method)\b",
        r"what(?:'s| is) the best (?:way|approach|method)",
        r"(?:let's|let us) (?:think|plan|consider|discuss)",
    ],
    "recalling": [
        r"\b(?:remember|recall|mentioned|said|told me)\b",
        r"\b(?:last time|previously|earlier|before|yesterday)\b",
        r"\b(?:what was|where was|when did)\b",
        r"\b(?:you told me|we discussed|we talked about)\b",
        r"(?:do you|can you) remember",
        r"(?:what|where) did (?:we|you|I) .+ (?:say|do|put|use)",
    ],
    "exploring": [
        r"\b(?:what is|what are|explain|describe|tell me about)\b",
        r"\b(?:how does|how do|how is)\b",
        r"\b(?:overview|introduction|summary)\b",
        r"\b(?:learn|understand|know) (?:more )?about\b",
        r"(?:can you|could you) (?:explain|describe|tell)",
        r"what(?:'s| is) (?:a|an|the) .+\?",
    ],
    "executing": [
        r"\b(?:do it|run it|execute|apply|make the change)\b",
        r"\b(?:go ahead|proceed|start|begin|continue)\b",
        r"\b(?:deploy|push|commit|save|update)\b",
        r"(?:please|just|now) (?:do|run|execute|apply|make)",
        r"^(?:yes|ok|okay|sure|go)(?:[,.]|$)",
        r"that(?:'s| is) (?:fine|good|correct|right)",
    ],
    "confirming": [
        r"\b(?:is that (?:right|correct)|does that (?:work|look))\b",
        r"\b(?:verify|confirm|check|validate)\b",
        r"\b(?:are you sure|did it work|is it working)\b",
        r"(?:did|does|is) (?:that|this|it) .+ (?:right|correct|work)",
        r"(?:looks? |seems? )(?:good|correct|right)",
    ],
}


@dataclass
class StateClassification:
    """Result of state classification."""
    state: ConversationState
    confidence: float
    method: str  # "pattern" or "llm"
    patterns_matched: List[str]
    raw_scores: Dict[str, float]


class StateClassifier:
    """Classify conversation state from messages."""

    def __init__(
        self,
        patterns: Optional[Dict[str, List[str]]] = None,
        pattern_confidence_threshold: float = 0.5,
        use_llm_fallback: bool = True,
    ):
        """Initialize state classifier.

        Args:
            patterns: State patterns (uses defaults if not provided)
            pattern_confidence_threshold: Min confidence to use pattern result
            use_llm_fallback: Whether to use LLM for ambiguous cases
        """
        self.patterns = patterns or DEFAULT_STATE_PATTERNS
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.use_llm_fallback = use_llm_fallback

        # Compile patterns
        self._compiled: Dict[str, List[re.Pattern]] = {}
        for state, pattern_list in self.patterns.items():
            self._compiled[state] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]

    async def classify(
        self,
        messages: List[str],
        history: Optional[List[ConversationState]] = None,
    ) -> StateClassification:
        """Classify conversation state.

        Args:
            messages: Recent messages (most recent last)
            history: Previous states for context

        Returns:
            StateClassification with state and confidence
        """
        # Combine recent messages for analysis
        combined_text = " ".join(messages[-5:])  # Last 5 messages

        # Pattern-based classification
        pattern_result = self._classify_by_patterns(combined_text)

        if pattern_result.confidence >= self.pattern_confidence_threshold:
            return pattern_result

        # LLM fallback for ambiguous cases
        if self.use_llm_fallback:
            try:
                llm_result = await self._classify_by_llm(messages[-3:])
                if llm_result.confidence > pattern_result.confidence:
                    return llm_result
            except Exception as e:
                logger.debug(f"LLM classification failed: {e}")

        # Fall back to pattern result or IDLE
        if pattern_result.confidence > 0.2:
            return pattern_result

        return StateClassification(
            state=ConversationState.IDLE,
            confidence=0.3,
            method="default",
            patterns_matched=[],
            raw_scores={},
        )

    def _classify_by_patterns(self, text: str) -> StateClassification:
        """Classify using pattern matching."""
        scores: Dict[str, float] = {}
        matched_patterns: Dict[str, List[str]] = {}

        for state, patterns in self._compiled.items():
            state_score = 0.0
            state_matches = []

            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    # Weight by number of matches
                    state_score += len(matches) * 0.2
                    state_matches.append(pattern.pattern[:50])

            scores[state] = min(1.0, state_score)
            matched_patterns[state] = state_matches

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for state in scores:
                scores[state] /= total

        # Find best state
        if scores:
            best_state = max(scores, key=scores.get)
            best_confidence = scores[best_state]

            return StateClassification(
                state=ConversationState(best_state),
                confidence=best_confidence,
                method="pattern",
                patterns_matched=matched_patterns.get(best_state, []),
                raw_scores=scores,
            )

        return StateClassification(
            state=ConversationState.IDLE,
            confidence=0.0,
            method="pattern",
            patterns_matched=[],
            raw_scores=scores,
        )

    async def _classify_by_llm(self, messages: List[str]) -> StateClassification:
        """Classify using LLM."""
        import json

        try:
            from api.services.prompt_service import get_default_prompt
            from api.services.extraction import call_vllm

            variables = {
                "messages": [{"role": "user", "content": m} for m in messages]
            }

            system_prompt, user_prompt, params = get_default_prompt(
                "state_classification",
                variables,
            )

            response = await call_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=100,
            )

            # Parse response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                state_str = data.get("state", "idle").lower()
                confidence = float(data.get("confidence", 0.5))

                # Map to enum
                try:
                    state = ConversationState(state_str)
                except ValueError:
                    state = ConversationState.IDLE

                return StateClassification(
                    state=state,
                    confidence=confidence,
                    method="llm",
                    patterns_matched=[],
                    raw_scores={state_str: confidence},
                )

        except Exception as e:
            logger.debug(f"LLM state classification failed: {e}")

        return StateClassification(
            state=ConversationState.IDLE,
            confidence=0.3,
            method="llm_failed",
            patterns_matched=[],
            raw_scores={},
        )

    def get_state_description(self, state: ConversationState) -> str:
        """Get human-readable description of a state."""
        descriptions = {
            ConversationState.IDLE: "No active task or direction",
            ConversationState.DEBUGGING: "Investigating errors or unexpected behavior",
            ConversationState.PLANNING: "Discussing approach, design, or strategy",
            ConversationState.RECALLING: "Retrieving past information or context",
            ConversationState.EXPLORING: "Learning about concepts or systems",
            ConversationState.EXECUTING: "Actively performing requested actions",
            ConversationState.CONFIRMING: "Verifying understanding or results",
        }
        return descriptions.get(state, "Unknown state")

    def get_relevant_query_mode(self, state: ConversationState) -> str:
        """Get the query mode most relevant for a state."""
        mode_mapping = {
            ConversationState.IDLE: "general",
            ConversationState.DEBUGGING: "diagnostic",
            ConversationState.PLANNING: "how_to",
            ConversationState.RECALLING: "recall",
            ConversationState.EXPLORING: "exploration",
            ConversationState.EXECUTING: "procedural",
            ConversationState.CONFIRMING: "verification",
        }
        return mode_mapping.get(state, "general")
