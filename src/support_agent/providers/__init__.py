"""Model provider abstraction and implementations."""

from support_agent.providers.base import (
    ChatProvider,
    GenerationRequest,
    GenerationResponse,
    PromptSegment,
)
from support_agent.providers.registry import build_chat_provider

__all__ = [
    "ChatProvider",
    "GenerationRequest",
    "GenerationResponse",
    "PromptSegment",
    "build_chat_provider",
]
