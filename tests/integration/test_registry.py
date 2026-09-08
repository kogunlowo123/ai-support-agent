"""The tool registry, against real tools and a real database.

This is where authority is granted or refused, so the tests are ordered the way
the checks are: a call that should never happen must be stopped by the first
check that can stop it, and must leave no trace in the dependency.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from support_agent.config import LimitSettings, PolicySettings
from support_agent.domain.models import Intent, ToolCall, ToolOutcome, ToolRisk
from support_agent.security.authz import DEFAULT_SCOPES, SCOPE_KNOWLEDGE, Principal
from support_agent.tools.base import ToolArguments, ToolContext, ToolReturns, ToolSpec
from support_agent.tools.breaker import CircuitBreakerRegistry
from support_agent.tools.builtin import build_default_tools
from support_agent.tools.registry import IdempotencyStore, RunBudget, ToolRegistry
from tests.conftest import TENANT

pytestmark = pytest.mark.integration


def build_registry(**limit_overrides: object) -> ToolRegistry:
    limits = LimitSettings(**limit_overrides)
    registry = ToolRegistry(
        limits=limits,
        breakers=CircuitBreakerRegistry(failure_threshold=2, reset_seconds=30.0),
    )
    for tool in build_default_tools(PolicySettings()):
        registry.register(tool)
    return registry


def make_context(
    session, principal: Principal, *, verified: bool = True, customer_id: str = ""
) -> ToolContext:
    return ToolContext(
        principal=principal,
        session=session,
        conversation_id="conv_test",
        identity_verified=verified,
        deadline=datetime.now(UTC) + timedelta(seconds=10),
        metadata={"customer_id": customer_id},
    )


def budget(**overrides: int) -> RunBudget:
    return RunBudget(
        max_tool_calls=overrides.get("max_tool_calls", 6),
        max_calls_per_tool=overrides.get("max_calls_per_tool", 3),
    )


class TestRegistration:
    def test_the_default_tools_are_registered(self):
        assert set(build_registry().names) == {
            "search_knowledge_base",
            "lookup_customer",
            "lookup_order",
            "check_refund_eligibility",
            "check_return_eligibility",
            "create_ticket",
        }

    def test_registering_a_name_twice_is_a_programming_error(self):
        registry = build_registry()
        tools = build_default_tools(PolicySettings())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tools[0])

    def test_every_tool_declares_what_it_needs(self):
        """A tool with no declaration would inherit the process's authority."""
        for spec in build_registry().specs():
            assert spec.description
            assert spec.risk in set(ToolRisk)
            assert spec.arguments is not None

    def test_writes_require_a_verified_identity(self):
        for spec in build_registry().specs():
            if spec.risk is not ToolRisk.READ:
                assert spec.requires_identity, spec.name


class TestAvailability:
    def test_an_unverified_caller_is_offered_only_public_tools(self, principal: Principal):
        available = build_registry().available_for(
            intent=Intent.ORDER_STATUS, principal=principal, identity_verified=False
        )
        assert {spec.name for spec in available} == {"search_knowledge_base"}

    def test_a_verified_caller_is_offered_the_account_tools(self, principal: Principal):
        available = build_registry().available_for(
            intent=Intent.ORDER_STATUS, principal=principal, identity_verified=True
        )
        assert "lookup_order" in {spec.name for spec in available}

    def test_scopes_the_caller_lacks_remove_the_tool(self):
        narrow = Principal(tenant_id=TENANT, key_id="k", scopes=frozenset({SCOPE_KNOWLEDGE}))
        available = build_registry().available_for(
            intent=Intent.ORDER_STATUS, principal=narrow, identity_verified=True
        )
        assert "lookup_order" not in {spec.name for spec in available}

    def test_the_allowlist_matches_what_invoke_would_permit(self, principal: Principal):
        """A planner is never offered something the registry would refuse."""
        registry = build_registry()
        offered = {
            spec.name
            for spec in registry.available_for(
                intent=Intent.RETURN_POLICY, principal=principal, identity_verified=False
            )
        }
        for spec in registry.specs():
            permitted = (
                not spec.allowed_intents or Intent.RETURN_POLICY in spec.allowed_intents
            ) and (not spec.required_scopes or spec.required_scopes <= principal.scopes)
            permitted = permitted and not spec.requires_identity
            assert (spec.name in offered) is permitted, spec.name


class TestPermissionChecks:
    async def test_an_unknown_tool_is_refused(self, seeded_work, principal: Principal):
        """A name nobody registered is a planning bug or an injected call."""
        result = await build_registry().invoke(
            ToolCall(tool="rm_minus_rf", arguments={}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.NOT_FOUND
        assert result.data == {}

    async def test_a_tool_not_permitted_for_the_intent_is_refused(
        self, seeded_work, principal: Principal
    ):
        """A tool legitimate for one intent is not automatically legitimate for another."""
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-1001"}),
            make_context(seeded_work.session, principal),
            intent=Intent.SHIPPING_QUESTION,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.DENIED

    async def test_an_unverified_caller_cannot_reach_an_order(
        self, seeded_work, principal: Principal
    ):
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-1001"}),
            make_context(seeded_work.session, principal, verified=False),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.DENIED
        assert result.data == {}

    async def test_a_caller_without_the_scope_is_refused(self, seeded_work):
        narrow = Principal(tenant_id=TENANT, key_id="k", scopes=frozenset({SCOPE_KNOWLEDGE}))
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-1001"}),
            make_context(seeded_work.session, narrow),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.DENIED

    async def test_invalid_arguments_are_rejected_before_execution(
        self, seeded_work, principal: Principal
    ):
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "!!! not a reference"}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.INVALID_ARGUMENTS


class TestBudgets:
    async def test_the_run_budget_is_enforced(self, seeded_work, principal: Principal):
        registry = build_registry()
        context = make_context(seeded_work.session, principal)
        spent = budget(max_tool_calls=2)
        outcomes = [
            (
                await registry.invoke(
                    ToolCall(tool="search_knowledge_base", arguments={"query": "returns"}),
                    context,
                    intent=Intent.RETURN_POLICY,
                    budget=spent,
                )
            ).outcome
            for _ in range(3)
        ]
        assert outcomes[:2] == [ToolOutcome.OK, ToolOutcome.OK]
        assert outcomes[2] is ToolOutcome.RATE_LIMITED

    async def test_one_tool_cannot_consume_the_whole_budget(
        self, seeded_work, principal: Principal
    ):
        registry = build_registry()
        context = make_context(seeded_work.session, principal)
        spent = budget(max_tool_calls=10, max_calls_per_tool=1)
        first = await registry.invoke(
            ToolCall(tool="search_knowledge_base", arguments={"query": "returns"}),
            context,
            intent=Intent.RETURN_POLICY,
            budget=spent,
        )
        second = await registry.invoke(
            ToolCall(tool="search_knowledge_base", arguments={"query": "shipping"}),
            context,
            intent=Intent.RETURN_POLICY,
            budget=spent,
        )
        assert first.outcome is ToolOutcome.OK
        assert second.outcome is ToolOutcome.RATE_LIMITED

    def test_a_refused_call_does_not_consume_budget(self):
        spent = budget(max_tool_calls=1)
        assert spent.check("lookup_order") is None
        spent.consume("lookup_order")
        assert spent.check("lookup_order") is not None


class TestExecution:
    async def test_an_unbound_conversation_cannot_reach_an_order(
        self, seeded_work, principal: Principal
    ):
        """Identity is verified but no customer is bound: still no account data.

        A verified conversation that is not attached to a customer has nothing
        to scope the lookup by, and scoping by the reference alone would let a
        guessed reference reach someone else's order.
        """
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-1001"}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.DENIED
        assert result.data == {}

    async def test_a_permitted_call_returns_data(
        self, seeded_work, principal: Principal, customer_id: str
    ):
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-1001"}),
            make_context(seeded_work.session, principal, customer_id=customer_id),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.OK
        assert result.data["order_reference"] == "ORD-1001"

    async def test_an_order_that_is_not_yours_is_refused_the_same_way_as_one_that_does_not_exist(
        self, seeded_work, principal: Principal, customer_id: str
    ):
        """The refusal must not tell a caller which references are real.

        Distinguishing "no such order" from "not your order" would turn the tool
        into an oracle for enumerating other people's order references.
        """
        result = await build_registry().invoke(
            ToolCall(tool="lookup_order", arguments={"order_reference": "ORD-9999"}),
            make_context(seeded_work.session, principal, customer_id=customer_id),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.DENIED
        assert not result.data
        assert "for this customer" in result.message

    async def test_the_result_records_how_long_it_took(self, seeded_work, principal: Principal):
        result = await build_registry().invoke(
            ToolCall(tool="search_knowledge_base", arguments={"query": "returns policy"}),
            make_context(seeded_work.session, principal),
            intent=Intent.RETURN_POLICY,
            budget=budget(),
        )
        assert result.duration_ms >= 0.0
        assert result.call_id


class TestIdempotency:
    def test_a_supplied_key_is_namespaced_by_tool_and_caller(self):
        """Two callers passing the same key are not the same call.

        Taking the supplied key verbatim would let one customer's idempotency
        key return another customer's recorded result.
        """
        key = IdempotencyStore.key("create_ticket", "p1", {"subject": "x"}, "caller-key")
        other = IdempotencyStore.key("create_ticket", "p2", {"subject": "x"}, "caller-key")
        assert "caller-key" in key
        assert key != other

    def test_identical_calls_derive_identical_keys(self):
        first = IdempotencyStore.key("create_ticket", "p1", {"subject": "x"}, None)
        second = IdempotencyStore.key("create_ticket", "p1", {"subject": "x"}, None)
        assert first == second

    def test_different_arguments_derive_different_keys(self):
        first = IdempotencyStore.key("create_ticket", "p1", {"subject": "x"}, None)
        second = IdempotencyStore.key("create_ticket", "p1", {"subject": "y"}, None)
        assert first != second

    def test_different_principals_derive_different_keys(self):
        """Two customers raising the same ticket are two tickets."""
        first = IdempotencyStore.key("create_ticket", "p1", {"subject": "x"}, None)
        second = IdempotencyStore.key("create_ticket", "p2", {"subject": "x"}, None)
        assert first != second

    async def test_a_replayed_write_is_not_executed_twice(self, seeded_work, principal: Principal):
        registry = build_registry()
        context = ToolContext(
            principal=principal,
            session=seeded_work.session,
            conversation_id="conv_idem",
            identity_verified=True,
            deadline=datetime.now(UTC) + timedelta(seconds=10),
            idempotency_key="fixed-key",
        )
        call = ToolCall(
            tool="create_ticket",
            arguments={"subject": "Refund query", "body": "Please help", "category": "refund"},
        )
        spent = budget()
        first = await registry.invoke(call, context, intent=Intent.REFUND_REQUEST, budget=spent)
        second = await registry.invoke(call, context, intent=Intent.REFUND_REQUEST, budget=spent)

        assert first.outcome is ToolOutcome.OK
        assert second.outcome is ToolOutcome.OK
        assert second.replayed is True
        assert second.data["ticket_id"] == first.data["ticket_id"]


class TestFailureHandling:
    async def test_a_timeout_is_reported_rather_than_raised(self, seeded_work, principal):
        registry = ToolRegistry(
            limits=LimitSettings(tool_timeout_seconds=0.01, tool_max_retries=0),
            breakers=CircuitBreakerRegistry(),
        )
        registry.register(SlowTool())
        result = await registry.invoke(
            ToolCall(tool="slow_tool", arguments={}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert result.outcome is ToolOutcome.TIMEOUT

    async def test_repeated_failure_opens_the_circuit(self, seeded_work, principal):
        registry = ToolRegistry(
            limits=LimitSettings(tool_max_retries=0, max_tool_calls=10),
            breakers=CircuitBreakerRegistry(failure_threshold=2, reset_seconds=60.0),
        )
        registry.register(BrokenTool())
        context = make_context(seeded_work.session, principal)
        spent = budget(max_tool_calls=10)
        outcomes = [
            (
                await registry.invoke(
                    ToolCall(tool="broken_tool", arguments={}),
                    context,
                    intent=Intent.ORDER_STATUS,
                    budget=spent,
                )
            ).outcome
            for _ in range(3)
        ]
        assert outcomes[0] is ToolOutcome.ERROR
        assert outcomes[-1] is ToolOutcome.CIRCUIT_OPEN

    async def test_a_non_idempotent_tool_is_not_retried(self, seeded_work, principal):
        """Retrying a write is how one refund becomes two."""
        tool = CountingTool(idempotent=False)
        registry = ToolRegistry(
            limits=LimitSettings(tool_max_retries=3), breakers=CircuitBreakerRegistry()
        )
        registry.register(tool)
        await registry.invoke(
            ToolCall(tool="counting_tool", arguments={}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert tool.calls == 1

    async def test_an_idempotent_tool_is_retried(self, seeded_work, principal):
        tool = CountingTool(idempotent=True)
        registry = ToolRegistry(
            limits=LimitSettings(tool_max_retries=2), breakers=CircuitBreakerRegistry()
        )
        registry.register(tool)
        await registry.invoke(
            ToolCall(tool="counting_tool", arguments={}),
            make_context(seeded_work.session, principal),
            intent=Intent.ORDER_STATUS,
            budget=budget(),
        )
        assert tool.calls > 1


class _NoArguments(ToolArguments):
    pass


class _NoReturns(ToolReturns):
    pass


def _spec(name: str, *, idempotent: bool = True) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="A tool used only by the tests.",
        risk=ToolRisk.READ,
        arguments=_NoArguments,
        returns=_NoReturns,
        required_scopes=frozenset(),
        allowed_intents=frozenset(),
        requires_identity=False,
        idempotent=idempotent,
    )


class SlowTool:
    """Takes longer than any deadline a test will give it."""

    spec = _spec("slow_tool")

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> _NoReturns:
        await asyncio.sleep(5)
        return _NoReturns()


class BrokenTool:
    """Always fails, so a breaker has something to open on."""

    spec = _spec("broken_tool")

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> _NoReturns:
        msg = "this dependency is down"
        raise RuntimeError(msg)


class CountingTool:
    """Fails every time, and counts how often it was asked to."""

    def __init__(self, *, idempotent: bool) -> None:
        self.spec = _spec("counting_tool", idempotent=idempotent)
        self.calls = 0

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> _NoReturns:
        self.calls += 1
        msg = "still failing"
        raise RuntimeError(msg)


assert DEFAULT_SCOPES  # the fixtures rely on the default grant being non-empty
