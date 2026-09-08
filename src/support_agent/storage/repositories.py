"""Repositories: the only place that knows SQL.

Every method takes an explicit ``tenant_id`` and every query filters on it. The
repetition is deliberate — it makes tenant isolation reviewable by reading this
one file rather than trusting each call site to have remembered.

Customer-scoped reads take a ``customer_id`` as well, and filter on both. An
order belongs to a customer, and the agent must not be able to reach one
customer's order by supplying another customer's reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from support_agent.domain.models import Conversation, Turn, new_id
from support_agent.storage.schema import (
    AuditRow,
    ConversationRow,
    CustomerRow,
    FeedbackRow,
    KnowledgeArticleRow,
    OrderRow,
    RunRow,
    StepRow,
    TicketRow,
)

if TYPE_CHECKING:
    from support_agent.domain.models import Answer, Step

#: Conversation turns kept in the row. Bounded so a long-running thread cannot
#: grow the JSON column without limit.
MAX_STORED_TURNS = 60


async def _scalar[RowT](session: AsyncSession, statement: Select[tuple[RowT]]) -> RowT | None:
    """Run a select and return the first row, or ``None``.

    SQLAlchemy types ``session.scalar`` as returning ``Any``, so calling it
    directly makes every repository read lose its row type at the boundary —
    exactly where a typo in a column name would otherwise be caught.
    """
    row: RowT | None = await session.scalar(statement)
    return row


class CustomerRepository:
    """Customer lookups."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def by_id(self, tenant_id: str, customer_id: str) -> CustomerRow | None:
        """Fetch one customer within a tenant."""
        return await _scalar(
            self._session,
            select(CustomerRow).where(
                CustomerRow.id == customer_id, CustomerRow.tenant_id == tenant_id
            ),
        )

    async def by_email(self, tenant_id: str, email: str) -> CustomerRow | None:
        """Fetch one customer by email within a tenant."""
        return await _scalar(
            self._session,
            select(CustomerRow).where(
                func.lower(CustomerRow.email) == email.strip().lower(),
                CustomerRow.tenant_id == tenant_id,
            ),
        )

    async def add(self, row: CustomerRow) -> CustomerRow:
        """Insert a customer."""
        self._session.add(row)
        await self._session.flush()
        return row


class OrderRepository:
    """Order lookups, always scoped to a tenant and a customer."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def by_reference(
        self, tenant_id: str, customer_id: str, reference: str
    ) -> OrderRow | None:
        """Fetch one order by its customer-facing reference.

        Filtered on the customer as well as the tenant: an order reference is
        guessable, and it must not be a way to read someone else's order.
        """
        return await _scalar(
            self._session,
            select(OrderRow).where(
                OrderRow.tenant_id == tenant_id,
                OrderRow.customer_id == customer_id,
                func.upper(OrderRow.reference) == reference.strip().upper(),
            ),
        )

    async def recent_for_customer(
        self, tenant_id: str, customer_id: str, limit: int = 5
    ) -> list[OrderRow]:
        """Return the most recent orders for a customer."""
        rows = await self._session.scalars(
            select(OrderRow)
            .where(OrderRow.tenant_id == tenant_id, OrderRow.customer_id == customer_id)
            .order_by(OrderRow.placed_at.desc())
            .limit(limit)
        )
        return list(rows.all())

    async def mark_refunded(self, tenant_id: str, order_id: str) -> bool:
        """Record that an order was refunded. Returns whether a row changed."""
        result = await self._session.execute(
            update(OrderRow)
            .where(
                OrderRow.id == order_id,
                OrderRow.tenant_id == tenant_id,
                OrderRow.refunded_at.is_(None),
            )
            .values(refunded_at=datetime.now(UTC), status="refunded")
        )
        return bool(result.rowcount)  # type: ignore[attr-defined]

    async def add(self, row: OrderRow) -> OrderRow:
        """Insert an order."""
        self._session.add(row)
        await self._session.flush()
        return row


class TicketRepository:
    """Ticket creation and lookup, with durable idempotency."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def create(
        self,
        *,
        tenant_id: str,
        subject: str,
        body: str,
        category: str,
        priority: str,
        customer_id: str | None,
        conversation_id: str | None,
        idempotency_key: str | None,
    ) -> tuple[TicketRow, bool]:
        """Create a ticket, or return the existing one for the same key.

        Returns ``(ticket, created)``. Idempotency is enforced by a unique
        constraint rather than a read-then-write, so two concurrent requests
        cannot both pass the check and both insert.
        """
        if idempotency_key:
            existing = await self._session.scalar(
                select(TicketRow).where(
                    TicketRow.tenant_id == tenant_id,
                    TicketRow.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                return existing, False

        row = TicketRow(
            id=new_id("tkt"),
            tenant_id=tenant_id,
            customer_id=customer_id,
            conversation_id=conversation_id,
            subject=subject[:200],
            body=body,
            category=category,
            priority=priority,
            status="open",
            idempotency_key=idempotency_key,
        )
        self._session.add(row)
        try:
            await self._session.flush()
        except IntegrityError:
            # Lost the race. The winner's row is the answer.
            await self._session.rollback()
            existing = await self._session.scalar(
                select(TicketRow).where(
                    TicketRow.tenant_id == tenant_id,
                    TicketRow.idempotency_key == idempotency_key,
                )
            )
            if existing is None:
                raise
            return existing, False
        return row, True

    async def by_id(self, tenant_id: str, ticket_id: str) -> TicketRow | None:
        """Fetch one ticket within a tenant."""
        return await _scalar(
            self._session,
            select(TicketRow).where(TicketRow.id == ticket_id, TicketRow.tenant_id == tenant_id),
        )

    async def recent(self, tenant_id: str, limit: int = 50) -> list[TicketRow]:
        """Return the most recent tickets for a tenant."""
        rows = await self._session.scalars(
            select(TicketRow)
            .where(TicketRow.tenant_id == tenant_id)
            .order_by(TicketRow.created_at.desc())
            .limit(limit)
        )
        return list(rows.all())


class KnowledgeRepository:
    """Knowledge article search.

    Ranking is term overlap over the title and body, computed in Python after a
    bounded candidate fetch. That is honest for a corpus of tens to hundreds of
    published support articles, which is what a support knowledge base is.
    Beyond that, PostgreSQL full-text search replaces this method and nothing
    else.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def add(self, row: KnowledgeArticleRow) -> KnowledgeArticleRow:
        """Insert an article."""
        self._session.add(row)
        await self._session.flush()
        return row

    async def by_slug(self, tenant_id: str, slug: str) -> KnowledgeArticleRow | None:
        """Fetch one article by slug."""
        return await _scalar(
            self._session,
            select(KnowledgeArticleRow).where(
                KnowledgeArticleRow.tenant_id == tenant_id,
                KnowledgeArticleRow.slug == slug,
            ),
        )

    async def candidates(
        self, tenant_id: str, terms: Sequence[str], limit: int = 50
    ) -> list[KnowledgeArticleRow]:
        """Articles whose title or body contains any of the terms."""
        statement = select(KnowledgeArticleRow).where(KnowledgeArticleRow.tenant_id == tenant_id)
        if terms:
            clauses = [
                or_(
                    func.lower(KnowledgeArticleRow.title).contains(term),
                    func.lower(KnowledgeArticleRow.body).contains(term),
                )
                for term in terms[:12]
            ]
            statement = statement.where(or_(*clauses))
        rows = await self._session.scalars(statement.limit(limit))
        return list(rows.all())

    async def count(self, tenant_id: str) -> int:
        """Count the articles held for a tenant."""
        total = await self._session.scalar(
            select(func.count())
            .select_from(KnowledgeArticleRow)
            .where(KnowledgeArticleRow.tenant_id == tenant_id)
        )
        return int(total or 0)


class ConversationRepository:
    """Conversation persistence."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    @staticmethod
    def _to_domain(row: ConversationRow) -> Conversation:
        turns = tuple(
            Turn(role=str(turn.get("role", "customer")), text=str(turn.get("text", "")))
            for turn in (row.turns or [])
            if isinstance(turn, dict)
        )
        return Conversation(
            id=row.id,
            tenant_id=row.tenant_id,
            customer_id=row.customer_id,
            identity_verified=row.identity_verified,
            turns=turns,
            summary=row.summary,
            escalated=row.escalated,
            created_at=row.created_at,
        )

    async def get(self, tenant_id: str, conversation_id: str) -> Conversation | None:
        """Fetch a conversation within a tenant."""
        row = await self._session.scalar(
            select(ConversationRow).where(
                ConversationRow.id == conversation_id, ConversationRow.tenant_id == tenant_id
            )
        )
        return self._to_domain(row) if row else None

    async def upsert(
        self, conversation: Conversation, *, verification_method: str = "none"
    ) -> None:
        """Create or update a conversation, bounding the stored turns."""
        turns = [
            {"role": turn.role, "text": turn.text, "at": turn.at.isoformat()}
            for turn in conversation.turns[-MAX_STORED_TURNS:]
        ]
        row = await self._session.get(ConversationRow, conversation.id)
        if row is None:
            self._session.add(
                ConversationRow(
                    id=conversation.id,
                    tenant_id=conversation.tenant_id,
                    customer_id=conversation.customer_id,
                    identity_verified=conversation.identity_verified,
                    verification_method=verification_method,
                    turns=turns,
                    summary=conversation.summary,
                    escalated=conversation.escalated,
                )
            )
        else:
            if row.tenant_id != conversation.tenant_id:
                msg = "conversation belongs to a different tenant"
                raise ValueError(msg)
            row.customer_id = conversation.customer_id
            row.identity_verified = conversation.identity_verified
            row.turns = turns
            row.summary = conversation.summary
            row.escalated = conversation.escalated
            if verification_method != "none":
                row.verification_method = verification_method
        await self._session.flush()


class RunRepository:
    """Run and step persistence. Together these are the audit trace."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def record(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        answer: Answer,
        intent_confidence: float,
    ) -> str:
        """Persist one run and every step it took."""
        run_id = new_id("run")
        self._session.add(
            RunRow(
                id=run_id,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                intent=str(answer.intent),
                intent_confidence=intent_confidence,
                final_state=str(answer.state),
                escalation_reason=(
                    str(answer.escalation_reason) if answer.escalation_reason else None
                ),
                refusal_reason=str(answer.refusal_reason) if answer.refusal_reason else None,
                tool_calls=answer.tool_calls,
                steps_used=len(answer.steps),
                duration_ms=answer.duration_ms,
                provider=answer.provider,
                verified=not answer.unverified_claims,
            )
        )
        self._session.add_all([self._step_row(run_id, step) for step in answer.steps])
        await self._session.flush()
        return run_id

    @staticmethod
    def _step_row(run_id: str, step: Step) -> StepRow:
        return StepRow(
            run_id=run_id,
            index=step.index,
            kind=str(step.kind),
            state=str(step.state),
            summary=step.summary[:500],
            detail=step.detail,
            duration_ms=step.duration_ms,
            at=step.at,
        )

    async def recent(self, tenant_id: str, limit: int = 50) -> list[RunRow]:
        """Return the most recent runs for a tenant, newest first."""
        rows = await self._session.scalars(
            select(RunRow)
            .where(RunRow.tenant_id == tenant_id)
            .order_by(RunRow.created_at.desc())
            .limit(limit)
        )
        return list(rows.all())

    async def steps_for(self, tenant_id: str, run_id: str) -> list[StepRow]:
        """Every step of one run, in order."""
        run = await self._session.scalar(
            select(RunRow).where(RunRow.id == run_id, RunRow.tenant_id == tenant_id)
        )
        if run is None:
            return []
        rows = await self._session.scalars(
            select(StepRow).where(StepRow.run_id == run_id).order_by(StepRow.index)
        )
        return list(rows.all())


class FeedbackRepository:
    """Feedback capture."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def record(
        self,
        *,
        tenant_id: str,
        conversation_id: str,
        run_id: str | None,
        helpful: bool,
        comment: str,
    ) -> None:
        """Record one verdict."""
        self._session.add(
            FeedbackRow(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
                helpful=helpful,
                comment=comment[:2000],
            )
        )
        await self._session.flush()

    async def summary(self, tenant_id: str) -> dict[str, int]:
        """Count helpful and unhelpful verdicts for a tenant."""
        rows = await self._session.execute(
            select(FeedbackRow.helpful, func.count())
            .where(FeedbackRow.tenant_id == tenant_id)
            .group_by(FeedbackRow.helpful)
        )
        counts = {"helpful": 0, "unhelpful": 0}
        for helpful, count in rows.all():
            counts["helpful" if helpful else "unhelpful"] = int(count)
        return counts


class AuditRepository:
    """Appends security-relevant events."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a session."""
        self._session = session

    async def record(
        self,
        *,
        tenant_id: str,
        actor_key_id: str,
        event: str,
        outcome: str,
        request_id: str | None = None,
        conversation_id: str | None = None,
        subject_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit event."""
        self._session.add(
            AuditRow(
                tenant_id=tenant_id,
                actor_key_id=actor_key_id,
                event=event,
                outcome=outcome,
                request_id=request_id,
                conversation_id=conversation_id,
                subject_id=subject_id,
                attributes=attributes or {},
            )
        )
        await self._session.flush()

    async def recent(self, tenant_id: str, *, limit: int = 100) -> list[AuditRow]:
        """Recent audit events for a tenant, newest first."""
        rows = await self._session.scalars(
            select(AuditRow)
            .where(AuditRow.tenant_id == tenant_id)
            .order_by(AuditRow.created_at.desc())
            .limit(limit)
        )
        return list(rows.all())


__all__ = [
    "MAX_STORED_TURNS",
    "AuditRepository",
    "ConversationRepository",
    "CustomerRepository",
    "FeedbackRepository",
    "KnowledgeRepository",
    "OrderRepository",
    "RunRepository",
    "TicketRepository",
]
