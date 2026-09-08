"""Storage, with tenant isolation as the property under test.

Every repository method takes an explicit tenant, and the repetition only pays
off if something checks it. These tests seed two tenants and assert that neither
can see the other, which is the failure that would matter most and the one least
likely to be noticed by a test that seeds only one.
"""

from __future__ import annotations

import pytest

from support_agent.domain.models import (
    AgentState,
    Answer,
    Conversation,
    Intent,
    Step,
    StepKind,
    new_id,
)
from support_agent.knowledge.seed import seed_demo_account, seed_knowledge
from support_agent.storage.repositories import MAX_STORED_TURNS
from support_agent.storage.schema import CustomerRow

pytestmark = pytest.mark.integration

OTHER_TENANT = "globex"


@pytest.fixture
async def two_tenants(runtime):
    """Seed the same demonstration data under two different tenants."""
    async with runtime.unit_of_work() as work:
        for tenant in ("acme", OTHER_TENANT):
            await seed_knowledge(work.knowledge, tenant)
            await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id=tenant,
                email="ada@example.com",
            )
    return runtime


class TestTenantIsolation:
    async def test_a_customer_is_invisible_to_another_tenant(self, two_tenants):
        async with two_tenants.unit_of_work() as work:
            mine = await work.customers.by_email("acme", "ada@example.com")
            theirs = await work.customers.by_email(OTHER_TENANT, "ada@example.com")
        assert mine is not None
        assert theirs is not None
        assert mine.id != theirs.id

    async def test_an_order_cannot_be_read_across_tenants(self, two_tenants):
        async with two_tenants.unit_of_work() as work:
            mine = await work.customers.by_email("acme", "ada@example.com")
            assert mine is not None
            crossed = await work.orders.by_reference(OTHER_TENANT, mine.id, "ORD-1001")
        assert crossed is None

    async def test_an_order_cannot_be_read_across_customers(self, two_tenants):
        """Same tenant, wrong customer: a guessed reference must find nothing."""
        async with two_tenants.unit_of_work() as work:
            found = await work.orders.by_reference("acme", "cus_someone_else", "ORD-1001")
        assert found is None

    async def test_knowledge_is_per_tenant(self, two_tenants):
        async with two_tenants.unit_of_work() as work:
            mine = await work.knowledge.by_slug("acme", "returns-policy")
            theirs = await work.knowledge.by_slug(OTHER_TENANT, "returns-policy")
        assert mine is not None
        assert theirs is not None
        assert mine.id != theirs.id

    async def test_a_tenant_that_seeded_nothing_finds_nothing(self, two_tenants):
        async with two_tenants.unit_of_work() as work:
            assert await work.knowledge.by_slug("initech", "returns-policy") is None
            assert await work.customers.by_email("initech", "ada@example.com") is None


class TestSeeding:
    async def test_seeding_is_idempotent(self, runtime):
        """Restarting a container must not duplicate the knowledge base."""
        async with runtime.unit_of_work() as work:
            first = await seed_knowledge(work.knowledge, "acme")
            second = await seed_knowledge(work.knowledge, "acme")
            total = await work.knowledge.count("acme")
        assert first > 0
        assert second == 0
        assert total == first

    async def test_the_demo_account_is_not_duplicated(self, runtime):
        async with runtime.unit_of_work() as work:
            first, _ = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id="acme",
                email="ada@example.com",
            )
            second, _ = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id="acme",
                email="ada@example.com",
            )
        assert first == second

    async def test_the_seeded_orders_cover_every_policy_branch(self, runtime):
        async with runtime.unit_of_work() as work:
            _, references = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id="acme",
                email="ada@example.com",
            )
        assert {
            "recent_delivered",
            "outside_window",
            "digital_downloaded",
            "high_value",
            "in_transit",
            "already_refunded",
            "poisoned_note",
        } <= set(references)


class TestConversations:
    async def test_a_conversation_round_trips(self, runtime):
        conversation = Conversation(id=new_id("conv"), tenant_id="acme").with_turn(
            "customer", "hello"
        )
        async with runtime.unit_of_work() as work:
            await work.conversations.upsert(conversation)
        async with runtime.unit_of_work() as work:
            loaded = await work.conversations.get("acme", conversation.id)
        assert loaded is not None
        assert len(loaded.turns) == 1

    async def test_stored_turns_are_bounded(self, runtime):
        """A long-running thread must not grow the row without limit."""
        conversation = Conversation(id=new_id("conv"), tenant_id="acme")
        for index in range(MAX_STORED_TURNS + 20):
            conversation = conversation.with_turn("customer", f"message {index}")
        async with runtime.unit_of_work() as work:
            await work.conversations.upsert(conversation)
        async with runtime.unit_of_work() as work:
            loaded = await work.conversations.get("acme", conversation.id)
        assert loaded is not None
        assert len(loaded.turns) <= MAX_STORED_TURNS

    async def test_a_conversation_is_not_readable_by_another_tenant(self, runtime):
        conversation = Conversation(id=new_id("conv"), tenant_id="acme")
        async with runtime.unit_of_work() as work:
            await work.conversations.upsert(conversation)
        async with runtime.unit_of_work() as work:
            assert await work.conversations.get(OTHER_TENANT, conversation.id) is None


class TestRunsAndTickets:
    async def test_a_run_and_its_trace_are_recorded_together(self, runtime):
        """A run row references the conversation it belongs to, so one must exist.

        The foreign key is the point: a trace pointing at nothing is a trace
        nobody can join back to what the customer actually asked.
        """
        conversation = Conversation(id=new_id("conv"), tenant_id="acme")
        async with runtime.unit_of_work() as work:
            await work.conversations.upsert(conversation)

        answer = Answer(
            text="Order ORD-1001 is delivered.",
            intent=Intent.ORDER_STATUS,
            state=AgentState.ANSWERED,
            steps=(
                Step(
                    index=0,
                    kind=StepKind.TOOL_CALL,
                    state=AgentState.GATHERING,
                    summary="lookup_order -> ok",
                ),
            ),
        )
        async with runtime.unit_of_work() as work:
            run_id = await work.runs.record(
                tenant_id="acme",
                conversation_id=conversation.id,
                answer=answer,
                intent_confidence=0.9,
            )
        async with runtime.unit_of_work() as work:
            steps = await work.runs.steps_for("acme", run_id)
            recent = await work.runs.recent("acme", limit=5)
            crossed = await work.runs.steps_for(OTHER_TENANT, run_id)
        assert len(steps) == 1
        assert steps[0].summary == "lookup_order -> ok"
        assert any(run.id == run_id for run in recent)
        assert crossed == []

    async def test_a_ticket_round_trips(self, runtime):
        async with runtime.unit_of_work() as work:
            ticket, created = await work.tickets.create(
                tenant_id="acme",
                subject="Refund query",
                body="Please help",
                category="refund",
                priority="normal",
                customer_id=None,
                conversation_id=None,
                idempotency_key=None,
            )
            ticket_id = ticket.id
        assert created is True
        async with runtime.unit_of_work() as work:
            loaded = await work.tickets.by_id("acme", ticket_id)
            assert await work.tickets.by_id(OTHER_TENANT, ticket_id) is None
        assert loaded is not None
        assert loaded.subject == "Refund query"


class TestAuditAndFeedback:
    async def test_an_audit_event_is_recorded(self, runtime):
        async with runtime.unit_of_work() as work:
            await work.audit.record(
                tenant_id="acme",
                actor_key_id="abcd1234",
                event="agent.run",
                outcome="answered",
                conversation_id="conv_1",
            )
        async with runtime.unit_of_work() as work:
            events = await work.audit.recent("acme", limit=5)
        assert events
        assert events[0].event == "agent.run"

    async def test_a_ticket_is_created_once_for_one_idempotency_key(self, runtime):
        """Enforced by a unique constraint, not a read-then-write.

        Two concurrent requests can both pass a read-then-write check and both
        insert; only the database can settle it.
        """
        async with runtime.unit_of_work() as work:
            first, created_first = await work.tickets.create(
                tenant_id="acme",
                subject="Refund query",
                body="Please help",
                category="refund",
                priority="normal",
                customer_id=None,
                conversation_id=None,
                idempotency_key="same-key",
            )
            second, created_second = await work.tickets.create(
                tenant_id="acme",
                subject="A different subject entirely",
                body="Different body",
                category="general",
                priority="high",
                customer_id=None,
                conversation_id=None,
                idempotency_key="same-key",
            )
        assert created_first is True
        assert created_second is False
        assert first.id == second.id


class TestAuditAndFeedbackTotals:
    async def test_feedback_totals_are_per_tenant(self, runtime):
        async with runtime.unit_of_work() as work:
            await work.feedback.record(
                tenant_id="acme", conversation_id="conv_1", run_id=None, helpful=True, comment=""
            )
            await work.feedback.record(
                tenant_id="acme", conversation_id="conv_1", run_id=None, helpful=False, comment=""
            )
        async with runtime.unit_of_work() as work:
            mine = await work.feedback.summary("acme")
            theirs = await work.feedback.summary(OTHER_TENANT)
        assert mine == {"helpful": 1, "unhelpful": 1}
        assert theirs == {"helpful": 0, "unhelpful": 0}


class TestConstraints:
    async def test_a_duplicate_customer_email_is_rejected(self, runtime):
        """One account per address per tenant, enforced by the database."""
        async with runtime.unit_of_work() as work:
            await work.customers.add(
                CustomerRow(
                    id=new_id("cus"),
                    tenant_id="acme",
                    email="dup@example.com",
                    full_name="First",
                )
            )
        with pytest.raises(Exception, match=r"[Uu]nique|IntegrityError|constraint"):
            async with runtime.unit_of_work() as work:
                await work.customers.add(
                    CustomerRow(
                        id=new_id("cus"),
                        tenant_id="acme",
                        email="dup@example.com",
                        full_name="Second",
                    )
                )
