"""Relational schema.

Two groups of tables, kept apart because they have different lifetimes and
different sensitivity:

**Business records** — customers, orders, tickets — model the systems a real
deployment would integrate with. They are here so the tools have something real
to read and write, and so authorisation, idempotency and policy can be tested
against actual rows rather than stubs.

**Agent records** — conversations, runs, steps and audit events — are what the
agent produced. `agent_runs` and `agent_steps` together are the trace: for any
answer the customer received, they say which tools ran, what policy decided,
what the verifier found, and why the run ended where it did.

The audit table carries no message text. It records identifiers, decisions and
rule ids, so it can be retained under a longer policy than the conversation it
describes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for every mapped class."""


# ---------------------------------------------------------------------------
# Business records
# ---------------------------------------------------------------------------


class CustomerRow(Base):
    """A customer of the tenant."""

    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_customers_tenant_email"),
        Index("ix_customers_tenant", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    tier: Mapped[str] = mapped_column(String(32), nullable=False, default="standard")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    orders: Mapped[list[OrderRow]] = relationship(
        back_populates="customer", cascade="all, delete-orphan", passive_deletes=True
    )


class OrderRow(Base):
    """An order, with the fields refund policy actually depends on.

    Amounts are integer minor units. Floating-point money is a defect waiting
    for a rounding edge case, and refund thresholds are exactly the place it
    would surface.
    """

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("tenant_id", "reference", name="uq_orders_tenant_reference"),
        Index("ix_orders_customer", "customer_id"),
        Index("ix_orders_tenant_status", "tenant_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    reference: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="placed")
    item_name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_digital: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    downloaded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    amount_minor: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="GBP")
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    refunded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    carrier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tracking_number: Mapped[str | None] = mapped_column(String(128), nullable=True)
    #: Free text written by warehouse or support staff. Third-party content, so
    #: it is scanned for injection before it can reach a prompt.
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    customer: Mapped[CustomerRow] = relationship(back_populates="orders")


class TicketRow(Base):
    """A support ticket raised by the agent or by a human."""

    __tablename__ = "tickets"
    __table_args__ = (
        UniqueConstraint("tenant_id", "idempotency_key", name="uq_tickets_idempotency"),
        Index("ix_tickets_tenant_status", "tenant_id", "status"),
        Index("ix_tickets_conversation", "conversation_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    priority: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    #: Durable idempotency. The in-process store absorbs a retry inside one run;
    #: this unique constraint is what stops a duplicate across processes.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class KnowledgeArticleRow(Base):
    """A published support article the agent may quote."""

    __tablename__ = "knowledge_articles"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_articles_tenant_slug"),
        Index("ix_articles_tenant", "tenant_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )


# ---------------------------------------------------------------------------
# Agent records
# ---------------------------------------------------------------------------


class ConversationRow(Base):
    """A conversation thread with one customer."""

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    identity_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verification_method: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    turns: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now
    )

    runs: Mapped[list[RunRow]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True
    )


class RunRow(Base):
    """One turn of the agent: from a customer message to an answer."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_runs_tenant_created", "tenant_id", "created_at"),
        Index("ix_runs_state", "final_state"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    intent: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    intent_confidence: Mapped[float] = mapped_column(nullable=False, default=0.0)
    final_state: Mapped[str] = mapped_column(String(16), nullable=False)
    escalation_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    refusal_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    steps_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[float] = mapped_column(nullable=False, default=0.0)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    conversation: Mapped[ConversationRow] = relationship(back_populates="runs")
    steps: Mapped[list[StepRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class StepRow(Base):
    """One recorded step of a run. Together, the steps are the trace."""

    __tablename__ = "agent_steps"
    __table_args__ = (
        UniqueConstraint("run_id", "index", name="uq_steps_run_index"),
        Index("ix_steps_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    detail: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    duration_ms: Mapped[float] = mapped_column(nullable=False, default=0.0)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)

    run: Mapped[RunRow] = relationship(back_populates="steps")


class FeedbackRow(Base):
    """A customer's verdict on one answer."""

    __tablename__ = "feedback"
    __table_args__ = (Index("ix_feedback_conversation", "conversation_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class AuditRow(Base):
    """Append-only record of security-relevant events.

    Carries no message, ticket or article text — only identifiers, rule ids,
    outcomes and decisions.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
        Index("ix_audit_event", "event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_key_id: Mapped[str] = mapped_column(String(32), nullable=False, default="anonymous")
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attributes: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


__all__ = [
    "AuditRow",
    "Base",
    "ConversationRow",
    "CustomerRow",
    "FeedbackRow",
    "KnowledgeArticleRow",
    "OrderRow",
    "RunRow",
    "StepRow",
    "TicketRow",
]
