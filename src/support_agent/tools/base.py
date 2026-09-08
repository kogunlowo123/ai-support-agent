"""Tool contracts.

A tool is the only way this agent affects or observes anything outside its own
process. That makes the tool boundary the place where authority is granted, so
the declaration of a tool is as important as its implementation:

* **Arguments are a Pydantic model**, which is both the validator and the JSON
  Schema shown to a planner. There is one definition, so the schema cannot drift
  from what the tool accepts.
* **Returns are a Pydantic model**, so the verifier can check the agent's claims
  against structured data rather than parsing prose.
* **Every tool declares its risk, its required scopes, whether it needs a
  verified identity, and which intents may reach it.** Permission is a property
  of the tool, not a check someone remembered to write at the call site.

There is no general-purpose tool. No shell, no HTTP fetch, no SQL. Each tool
answers one question or performs one action, with a typed surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from support_agent.domain.models import Intent, ToolRisk
from support_agent.errors import ToolArgumentError

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from support_agent.security.authz import Principal


class ToolArguments(BaseModel):
    """Base for tool argument models: strict, so an unknown field is an error."""

    model_config = {"extra": "forbid", "frozen": True}


class ToolReturns(BaseModel):
    """Base for tool return models."""

    model_config = {"extra": "forbid", "frozen": True}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything the registry needs to decide whether a call is permitted."""

    name: str
    description: str
    risk: ToolRisk
    arguments: type[ToolArguments]
    returns: type[ToolReturns]

    #: Scopes the calling principal must hold. Empty means any authenticated
    #: caller, which is only appropriate for public knowledge lookups.
    required_scopes: frozenset[str] = frozenset()

    #: Intents from which this tool may be reached. A tool that is legitimate
    #: for a refund request is not automatically legitimate for a shipping
    #: question; restricting by intent shrinks what a hijacked plan can do.
    allowed_intents: frozenset[Intent] = frozenset()

    #: Whether the customer's identity must be verified first. Account data and
    #: every write require it.
    requires_identity: bool = False

    #: Whether repeating the call is harmless. Only idempotent tools are
    #: retried, and only idempotent writes accept an idempotency key.
    idempotent: bool = True

    #: Per-tool timeout override. ``None`` uses the configured default.
    timeout_seconds: float | None = None

    #: Whether the returned payload may contain text written by a third party
    #: and must therefore be scanned before it reaches a prompt.
    returns_untrusted_text: bool = False

    def json_schema(self) -> dict[str, Any]:
        """Return the argument schema, as published in the API."""
        return self.arguments.model_json_schema()

    def describe(self) -> dict[str, Any]:
        """Describe the tool and its constraints for machine consumers."""
        return {
            "name": self.name,
            "description": self.description,
            "risk": str(self.risk),
            "requires_identity": self.requires_identity,
            "required_scopes": sorted(self.required_scopes),
            "allowed_intents": sorted(str(intent) for intent in self.allowed_intents),
            "idempotent": self.idempotent,
            "parameters": self.json_schema(),
        }


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to know about the run invoking it.

    Deliberately narrow. A tool receives the principal, the conversation
    identifiers, a database session and a deadline — not the agent, not the
    registry, and not the conversation text. A tool that cannot see the prompt
    cannot be steered by it.
    """

    principal: Principal
    session: AsyncSession
    conversation_id: str
    identity_verified: bool
    deadline: datetime
    idempotency_key: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """A callable unit of authority."""

    @property
    def spec(self) -> ToolSpec:
        """The declaration the registry enforces."""
        ...

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> ToolReturns:
        """Execute the tool. Raises a domain error on failure; never returns prose."""
        ...


def expect[ArgumentsT: ToolArguments](
    arguments: ToolArguments, expected: type[ArgumentsT]
) -> ArgumentsT:
    """Narrow validated arguments to the type a tool declared.

    The registry validates arguments against the tool's own model before
    invoking it, so this should never fail. It is a real check rather than an
    ``assert`` because an assert is removed under ``python -O``, which would
    leave the only guard in the tool body doing nothing in exactly the build
    someone runs in production.
    """
    if not isinstance(arguments, expected):
        raise ToolArgumentError(
            "the tool received arguments of an unexpected type",
            detail={"expected": expected.__name__, "received": type(arguments).__name__},
        )
    return arguments


__all__ = ["Tool", "ToolArguments", "ToolContext", "ToolReturns", "ToolSpec", "expect"]
