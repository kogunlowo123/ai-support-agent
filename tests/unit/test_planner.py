"""Planning.

The plan is a function of the intent and what is already known. These tests
pin the two properties that matter: a step whose preconditions are unmet is
never proposed, and no plan reaches account data without a verified identity.
"""

from __future__ import annotations

import pytest

from support_agent.agent.planner import (
    ALWAYS_ESCALATE,
    REQUIRE_IDENTITY,
    PlanContext,
    missing_prerequisite,
    plan,
    produced_data,
)
from support_agent.domain.models import Intent, ToolOutcome, ToolResult

pytestmark = pytest.mark.unit

ACCOUNT_TOOLS = {"lookup_order", "lookup_customer", "check_refund_eligibility"}


def context(intent: Intent, **overrides: object) -> PlanContext:
    base: dict[str, object] = {
        "intent": intent,
        "message": "test message",
        "identity_verified": True,
        "customer_id": "cus_1",
        "order_reference": "ORD-1001",
    }
    base.update(overrides)
    return PlanContext(**base)  # type: ignore[arg-type]


def tools(intent: Intent, **overrides: object) -> list[str]:
    return [call.tool for call in plan(context(intent, **overrides))]


class TestPlanShape:
    def test_a_policy_question_reads_the_knowledge_base(self):
        assert "search_knowledge_base" in tools(Intent.RETURN_POLICY)

    def test_an_order_question_looks_the_order_up(self):
        assert "lookup_order" in tools(Intent.ORDER_STATUS)

    def test_a_refund_request_evaluates_eligibility(self):
        assert "check_refund_eligibility" in tools(Intent.REFUND_REQUEST)

    def test_asking_for_a_human_needs_no_tools(self):
        assert tools(Intent.SPEAK_TO_HUMAN) == []

    def test_an_unknown_intent_plans_nothing(self):
        """A plan for a message nobody understood would be a guess."""
        assert tools(Intent.UNKNOWN) == []

    def test_every_intent_has_a_declared_plan(self):
        for intent in Intent:
            plan(context(intent))

    def test_the_plan_is_deterministic(self):
        first = tools(Intent.REFUND_REQUEST)
        assert all(tools(Intent.REFUND_REQUEST) == first for _ in range(10))


class TestPreconditions:
    def test_account_tools_are_dropped_without_a_verified_identity(self):
        planned = set(tools(Intent.ORDER_STATUS, identity_verified=False))
        assert not planned & ACCOUNT_TOOLS

    def test_order_tools_are_dropped_without_a_reference(self):
        planned = set(tools(Intent.ORDER_STATUS, order_reference=None))
        assert "lookup_order" not in planned

    def test_the_customer_lookup_is_dropped_without_a_customer(self):
        planned = tools(Intent.ACCOUNT_QUESTION, customer_id=None)
        assert "lookup_customer" not in planned

    @pytest.mark.parametrize("intent", list(Intent))
    def test_no_plan_reaches_account_data_unverified(self, intent):
        """The property the identity boundary rests on, checked exhaustively."""
        planned = set(tools(intent, identity_verified=False))
        assert not planned & ACCOUNT_TOOLS, f"{intent} planned {planned}"

    def test_the_order_reference_is_passed_to_the_tools_that_need_it(self):
        calls = plan(context(Intent.ORDER_STATUS))
        lookup = next(call for call in calls if call.tool == "lookup_order")
        assert lookup.arguments["order_reference"] == "ORD-1001"

    def test_the_message_is_passed_to_the_knowledge_search(self):
        calls = plan(context(Intent.RETURN_POLICY, message="how do I return a lamp?"))
        search = next(call for call in calls if call.tool == "search_knowledge_base")
        assert "lamp" in str(search.arguments["query"])

    def test_calls_are_attributed_to_the_planner(self):
        """Provenance for the call itself: a model did not ask for this."""
        assert all(call.requested_by == "planner" for call in plan(context(Intent.ORDER_STATUS)))


class TestPrerequisites:
    def test_an_unverified_account_question_asks_for_identity_first(self):
        assert missing_prerequisite(context(Intent.ORDER_STATUS, identity_verified=False)) == (
            "identity"
        )

    def test_a_missing_order_reference_is_asked_for(self):
        assert missing_prerequisite(context(Intent.ORDER_STATUS, order_reference=None)) == (
            "order_reference"
        )

    def test_identity_is_asked_for_before_a_reference(self):
        """One precise question at a time, and the security one comes first."""
        unknown = context(Intent.ORDER_STATUS, identity_verified=False, order_reference=None)
        assert missing_prerequisite(unknown) == "identity"

    def test_a_policy_question_has_no_prerequisites(self):
        assert missing_prerequisite(context(Intent.RETURN_POLICY, identity_verified=False)) is None

    def test_a_complete_context_has_no_prerequisites(self):
        assert missing_prerequisite(context(Intent.ORDER_STATUS)) is None


class TestIntentSets:
    def test_money_and_relationship_intents_always_reach_a_person(self):
        assert Intent.BILLING_DISPUTE in ALWAYS_ESCALATE
        assert Intent.COMPLAINT in ALWAYS_ESCALATE
        assert Intent.SPEAK_TO_HUMAN in ALWAYS_ESCALATE

    def test_account_intents_require_identity(self):
        assert Intent.ORDER_STATUS in REQUIRE_IDENTITY
        assert Intent.REFUND_REQUEST in REQUIRE_IDENTITY
        assert Intent.ACCOUNT_QUESTION in REQUIRE_IDENTITY

    def test_published_policy_questions_do_not_require_identity(self):
        """A public article stays readable without an account."""
        assert Intent.RETURN_POLICY not in REQUIRE_IDENTITY
        assert Intent.SHIPPING_QUESTION not in REQUIRE_IDENTITY


class TestProducedData:
    def test_a_successful_call_with_data_counts(self):
        result = ToolResult(
            call_id="c1", tool="lookup_order", outcome=ToolOutcome.OK, data={"status": "delivered"}
        )
        assert produced_data((result,)) is True

    def test_a_successful_call_with_nothing_in_it_does_not(self):
        result = ToolResult(call_id="c1", tool="lookup_order", outcome=ToolOutcome.OK, data={})
        assert produced_data((result,)) is False

    def test_a_failed_call_does_not(self):
        result = ToolResult(
            call_id="c1", tool="lookup_order", outcome=ToolOutcome.ERROR, data={"x": 1}
        )
        assert produced_data((result,)) is False

    def test_no_calls_at_all_does_not(self):
        assert produced_data(()) is False
