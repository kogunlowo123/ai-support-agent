"""Dependency wiring for the HTTP layer.

Long-lived collaborators — the engine, the chat provider, the circuit breakers —
are built once during startup and held on :class:`Services`. Per-request objects
— the session, the repositories and the machine bound to it — are created per
request and torn down with it.

The circuit breakers are deliberately on the long-lived side. A breaker rebuilt
per request would never accumulate enough evidence to open, which is the same as
not having one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from support_agent.agent.composer import Composer
from support_agent.agent.machine import AgentMachine
from support_agent.config import Settings
from support_agent.providers.base import ChatProvider
from support_agent.security.authz import ApiKeyAuthenticator, Principal
from support_agent.storage.repositories import (
    AuditRepository,
    ConversationRepository,
    CustomerRepository,
    FeedbackRepository,
    KnowledgeRepository,
    OrderRepository,
    RunRepository,
    TicketRepository,
)
from support_agent.tools.breaker import CircuitBreakerRegistry
from support_agent.tools.builtin import build_default_tools
from support_agent.tools.registry import ToolRegistry


@dataclass
class Services:
    """Process-lifetime collaborators, built during startup."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    chat: ChatProvider | None
    composer: Composer
    authenticator: ApiKeyAuthenticator
    breakers: CircuitBreakerRegistry

    def registry(self) -> ToolRegistry:
        """Build a tool registry sharing the process-wide breakers."""
        registry = ToolRegistry(limits=self.settings.limits, breakers=self.breakers)
        for tool in build_default_tools(self.settings.policy):
            registry.register(tool)
        return registry

    async def aclose(self) -> None:
        """Release every long-lived resource."""
        if self.chat is not None:
            await self.chat.aclose()
        await self.engine.dispose()


@dataclass
class RequestContext:
    """Per-request collaborators bound to one database session."""

    session: AsyncSession
    conversations: ConversationRepository
    customers: CustomerRepository
    orders: OrderRepository
    tickets: TicketRepository
    knowledge: KnowledgeRepository
    runs: RunRepository
    feedback: FeedbackRepository
    audit: AuditRepository
    machine: AgentMachine
    registry: ToolRegistry


def get_services(request: Request) -> Services:
    """Return the process-lifetime services from application state."""
    services: Services = request.app.state.services
    return services


def get_settings_dep(request: Request) -> Settings:
    """Return the active settings."""
    return get_services(request).settings


async def get_context(request: Request) -> AsyncIterator[RequestContext]:
    """Build the per-request context, committing or rolling back around the handler.

    The transaction wraps the whole request rather than each repository call, so
    a run that fails part-way leaves no half-written conversation behind.
    """
    services = get_services(request)
    session = services.session_factory()
    registry = services.registry()
    try:
        yield RequestContext(
            session=session,
            conversations=ConversationRepository(session),
            customers=CustomerRepository(session),
            orders=OrderRepository(session),
            tickets=TicketRepository(session),
            knowledge=KnowledgeRepository(session),
            runs=RunRepository(session),
            feedback=FeedbackRepository(session),
            audit=AuditRepository(session),
            machine=AgentMachine(
                settings=services.settings,
                registry=registry,
                composer=services.composer,
                session=session,
            ),
            registry=registry,
        )
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_principal(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """Authenticate the caller from ``X-API-Key`` or a bearer token."""
    services = get_services(request)
    presented = x_api_key
    if presented is None and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[len("bearer ") :]
    # AuthorizationError propagates to the domain error handler, which renders
    # it as a 403 with the standard error envelope.
    return services.authenticator.authenticate(presented)


ServicesDep = Annotated[Services, Depends(get_services)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ContextDep = Annotated[RequestContext, Depends(get_context)]
PrincipalDep = Annotated[Principal, Depends(get_principal)]

__all__ = [
    "ContextDep",
    "PrincipalDep",
    "RequestContext",
    "Services",
    "ServicesDep",
    "SettingsDep",
    "get_context",
    "get_principal",
    "get_services",
]
