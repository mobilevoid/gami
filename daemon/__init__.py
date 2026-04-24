"""
GAMI Subconscious Daemon.

Proactive context management for AI agents:
- State classification
- Predictive retrieval
- Hot context management
- Proactive injection
"""

from .state_classifier import StateClassifier, ConversationState
from .predictive_retriever import PredictiveRetriever
from .context_injector import ContextInjector

__all__ = [
    "StateClassifier",
    "ConversationState",
    "PredictiveRetriever",
    "ContextInjector",
]
