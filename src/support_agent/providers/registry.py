"""Chat provider construction.

The agent depends on the :class:`~support_agent.providers.base.ChatProvider`
protocol and nothing else, so changing model vendor is a configuration change.

Composition is the only place a model is used at all — classification, planning,
policy and verification are all deterministic — so the provider's blast radius
is one sentence of phrasing. That is why the ``template`` backend is a viable
default rather than a stub: with no model configured the agent still classifies,
gathers, decides, verifies and answers.
"""

from __future__ import annotations

from support_agent.config import ChatBackend, Settings
from support_agent.providers.base import ChatProvider
from support_agent.providers.ollama import OllamaChatProvider
from support_agent.providers.openai import OpenAIChatProvider


def build_chat_provider(settings: Settings) -> ChatProvider | None:
    """Construct the configured provider, or ``None`` for the template backend."""
    config = settings.chat
    match config.backend:
        case ChatBackend.TEMPLATE:
            return None
        case ChatBackend.OLLAMA:
            return OllamaChatProvider(
                base_url=settings.providers.ollama_base_url,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        case ChatBackend.OPENAI:
            return OpenAIChatProvider(
                base_url=settings.providers.openai_base_url,
                api_key=settings.providers.openai_api_key,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )


__all__ = ["build_chat_provider"]
