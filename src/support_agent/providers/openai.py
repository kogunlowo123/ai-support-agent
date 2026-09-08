"""OpenAI-compatible provider.

Written against the OpenAI REST surface rather than the vendor SDK so that any
compatible endpoint — Azure OpenAI, vLLM, LiteLLM, Together, a local
llama.cpp server — works by changing ``AGENT_PROVIDERS__OPENAI_BASE_URL``. The
SDK would add a dependency and a vendor coupling for functionality this module
already needs from :mod:`support_agent.providers.transport`.

The API key is held as a :class:`~pydantic.SecretStr` in configuration and is
only unwrapped when the ``Authorization`` header is built, so it cannot be
printed by an accidental ``repr`` of the settings object. The header name is on
the logging redaction list.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import SecretStr

from support_agent.errors import ConfigurationError, ProviderError
from support_agent.providers.base import GenerationRequest, GenerationResponse
from support_agent.providers.transport import ProviderTransport

PROVIDER_NAME = "openai"


def _auth_headers(api_key: SecretStr | None) -> dict[str, str]:
    if api_key is None:
        raise ConfigurationError(
            "the openai backend is selected but AGENT_PROVIDERS__OPENAI_API_KEY is not set"
        )
    return {"authorization": f"Bearer {api_key.get_secret_value()}"}


class OpenAIChatProvider:
    """Answer generation from an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: ProviderTransport | None = None,
    ) -> None:
        """Configure the endpoint, credentials and model."""
        self._model = model
        self._transport = transport or ProviderTransport(
            base_url=base_url,
            provider_name=PROVIDER_NAME,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            headers=_auth_headers(api_key),
        )

    @property
    def name(self) -> str:
        """Provider identifier recorded on every answer."""
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        """Model identifier recorded on every answer."""
        return self._model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate an answer, mapping trust levels onto chat roles."""
        messages: list[dict[str, str]] = []
        if system := request.system_text():
            messages.append({"role": "system", "content": system})

        user_parts: list[str] = []
        if evidence := request.render_untrusted():
            user_parts.append(evidence)
        if user := request.user_text():
            user_parts.append(user)
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_completion_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.stop:
            body["stop"] = list(request.stop)

        started = time.perf_counter()
        payload = await self._transport.post_json("/chat/completions", body)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("openai returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderError("openai returned a choice without message content")

        raw_usage = payload.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return GenerationResponse(
            text=message["content"],
            model=str(payload.get("model", self._model)),
            provider=PROVIDER_NAME,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            latency_ms=elapsed_ms,
            finish_reason=str(choices[0].get("finish_reason", "stop")),
        )

    async def health(self) -> bool:
        """Whether the endpoint responds to a model listing."""
        return await self._transport.get_ok("/models")

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._transport.aclose()


__all__ = ["PROVIDER_NAME", "OpenAIChatProvider"]
