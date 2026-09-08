"""Domain exception hierarchy.

Every deliberate error derives from :class:`AgentError` and carries an HTTP
status and a stable ``code``, so the API layer can translate exceptions without
knowing about individual failure modes and clients can branch on ``code``.

Messages are safe to return: they never embed customer data, tool payloads,
credentials or internal identifiers beyond those the caller already has.
"""

from __future__ import annotations

from http import HTTPStatus


class AgentError(Exception):
    """Base class for every deliberate application error."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, detail: dict[str, object] | None = None) -> None:
        """Record a client-safe message and optional structured detail."""
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class ConfigurationError(AgentError):
    """The application is configured in a way that cannot work."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "configuration_error"


class ValidationError(AgentError):
    """Caller-supplied input failed validation."""

    status_code = HTTPStatus.BAD_REQUEST
    code = "validation_error"


class AuthorizationError(AgentError):
    """The caller is not permitted to perform this action."""

    status_code = HTTPStatus.FORBIDDEN
    code = "forbidden"


class NotFoundError(AgentError):
    """The requested resource does not exist, or is not visible to the caller."""

    status_code = HTTPStatus.NOT_FOUND
    code = "not_found"


class ToolError(AgentError):
    """A tool could not be executed."""

    status_code = HTTPStatus.BAD_GATEWAY
    code = "tool_error"


class ToolNotFoundError(ToolError):
    """No tool with that name is registered."""

    status_code = HTTPStatus.NOT_FOUND
    code = "tool_not_found"


class ToolPermissionError(ToolError):
    """The caller or the current state may not invoke this tool."""

    status_code = HTTPStatus.FORBIDDEN
    code = "tool_forbidden"


class ToolArgumentError(ToolError):
    """The arguments did not match the tool's schema."""

    status_code = HTTPStatus.BAD_REQUEST
    code = "tool_invalid_arguments"


class ToolTimeoutError(ToolError):
    """A tool exceeded its execution budget."""

    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code = "tool_timeout"


class CircuitOpenError(ToolError):
    """A tool is failing and its circuit breaker is open."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "tool_circuit_open"


class ProviderError(AgentError):
    """An upstream model provider failed."""

    status_code = HTTPStatus.BAD_GATEWAY
    code = "provider_error"


class ProviderTimeoutError(ProviderError):
    """An upstream model provider did not respond within its budget."""

    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code = "provider_timeout"


class ProviderUnavailableError(ProviderError):
    """An upstream model provider is not reachable or not configured."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "provider_unavailable"


class BudgetExceededError(AgentError):
    """A run exceeded its step or time budget.

    Not a failure of the request so much as a refusal to keep spending on it.
    The run escalates rather than continuing.
    """

    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code = "budget_exceeded"


class InvalidTransitionError(AgentError):
    """The state machine was asked to make a transition it does not permit.

    This is always a programming error: the transition table is the design, and
    reaching this exception means code tried to step outside it.
    """

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "invalid_transition"


class PolicyViolationError(AgentError):
    """A request was refused by a security or business policy."""

    status_code = HTTPStatus.FORBIDDEN
    code = "policy_violation"


__all__ = [
    "AgentError",
    "AuthorizationError",
    "BudgetExceededError",
    "CircuitOpenError",
    "ConfigurationError",
    "InvalidTransitionError",
    "NotFoundError",
    "PolicyViolationError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ToolArgumentError",
    "ToolError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolTimeoutError",
    "ValidationError",
]
