"""In-process runtime.

The HTTP layer is one way to drive this agent; the CLI and the scenario suite
are others. All three need the same object graph, so it is assembled here once.

Without this, the scenario suite would have to go through HTTP, which would make
evaluating the agent depend on a running server and would measure the network as
well as the behaviour under test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from support_agent.agent.composer import Composer, ModelComposer, TemplateComposer
from support_agent.agent.machine import AgentMachine
from support_agent.config import Settings, get_settings
from support_agent.observability.logging import configure_logging
from support_agent.providers.base import ChatProvider
from support_agent.providers.registry import build_chat_provider
from support_agent.storage.db import create_engine, create_schema, create_session_factory
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


@dataclass(frozen=True, slots=True)
class UnitOfWork:
    """Repositories and the machine, all bound to one transaction."""

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


class Runtime:
    """Owns the process-lifetime object graph outside the HTTP layer."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the engine, providers, tools and breakers from configuration."""
        self.settings: Settings = settings or get_settings()
        self.engine: AsyncEngine = create_engine(self.settings)
        self.session_factory: async_sessionmaker[AsyncSession] = create_session_factory(self.engine)
        self.chat: ChatProvider | None = build_chat_provider(self.settings)
        self.composer: Composer = (
            ModelComposer(self.chat) if self.chat is not None else TemplateComposer()
        )
        # Breakers are process-lifetime on purpose: a dependency that failed for
        # one conversation is the same dependency for the next one, and a
        # per-request breaker would never accumulate enough evidence to open.
        self.breakers = CircuitBreakerRegistry(
            failure_threshold=self.settings.circuit_breaker.failure_threshold,
            reset_seconds=self.settings.circuit_breaker.reset_seconds,
            half_open_successes=self.settings.circuit_breaker.half_open_successes,
        )

    def _registry(self) -> ToolRegistry:
        registry = ToolRegistry(limits=self.settings.limits, breakers=self.breakers)
        for tool in build_default_tools(self.settings.policy):
            registry.register(tool)
        return registry

    async def start(self) -> None:
        """Create the schema if it does not exist."""
        await create_schema(self.engine)

    @asynccontextmanager
    async def unit_of_work(self) -> AsyncIterator[UnitOfWork]:
        """Open a transaction and yield everything bound to it."""
        session = self.session_factory()
        try:
            yield UnitOfWork(
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
                    settings=self.settings,
                    registry=self._registry(),
                    composer=self.composer,
                    session=session,
                ),
            )
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def aclose(self) -> None:
        """Release the provider and the database engine."""
        if self.chat is not None:
            await self.chat.aclose()
        await self.engine.dispose()

    async def __aenter__(self) -> Self:
        """Start the runtime for use as an async context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the runtime on context exit."""
        await self.aclose()


async def build_runtime(settings: Settings | None = None) -> Runtime:
    """Create and start a runtime with logging configured."""
    active = settings or get_settings()
    configure_logging(
        level=active.observability.log_level,
        fmt=active.observability.log_format,
        service_name=active.observability.service_name,
    )
    runtime = Runtime(active)
    await runtime.start()
    return runtime


__all__ = ["Runtime", "UnitOfWork", "build_runtime"]
