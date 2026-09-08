"""Planning: deciding which tools a turn needs.

The plan is a function of the classified intent and what is already known. A
model does not choose the tools.

Why the model does not plan
---------------------------
A model-chosen plan is the standard agent design and it is the wrong one here.
It makes tool selection an attack surface — anything that reaches the prompt can
propose a call — and it makes behaviour non-reproducible, so "why did it look up
that order?" has no answer beyond a sampled token. The plans in this file are
short and knowable: three or four calls, declared per intent, reviewable in one
screen.

What a fixed plan gives up is the long tail. A question that needs an
unanticipated combination of tools will not get one; it gets a clarifying
question or a human, which is the correct outcome for a support agent that must
never invent an answer.

The plan is still only a proposal. Every step passes through the registry, which
checks permission, scope, identity, budget and the circuit breaker, and may
refuse. The planner suggests; the registry decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from support_agent.domain.models import Intent, ToolCall

if TYPE_CHECKING:
    from support_agent.domain.models import ToolResult


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One intended tool call, with the condition under which it applies."""

    tool: str
    #: Arguments drawn from what is already known. Keys map to tool arguments.
    arguments: dict[str, object] = field(default_factory=dict)
    #: Skip this step unless an order reference is known.
    needs_order_reference: bool = False
    #: Skip this step unless the customer's identity has been verified.
    needs_identity: bool = False
    #: Skip this step when an earlier step already produced this data.
    optional: bool = False


@dataclass(frozen=True, slots=True)
class PlanContext:
    """What the planner knows before it decides."""

    intent: Intent
    message: str
    identity_verified: bool
    customer_id: str | None = None
    order_reference: str | None = None


#: The plan for each intent, in order. Empty means no tool is needed: a pure
#: policy question is answered from the knowledge base alone, and a request for
#: a human needs no lookup at all.
_PLANS: dict[Intent, tuple[PlanStep, ...]] = {
    Intent.ORDER_STATUS: (
        PlanStep(tool="lookup_order", needs_order_reference=True, needs_identity=True),
        PlanStep(tool="search_knowledge_base", optional=True),
    ),
    Intent.REFUND_REQUEST: (
        PlanStep(tool="lookup_order", needs_order_reference=True, needs_identity=True),
        PlanStep(tool="check_refund_eligibility", needs_order_reference=True, needs_identity=True),
        PlanStep(tool="search_knowledge_base", optional=True),
    ),
    Intent.RETURN_POLICY: (
        PlanStep(tool="search_knowledge_base"),
        PlanStep(
            tool="check_return_eligibility",
            needs_order_reference=True,
            needs_identity=True,
            optional=True,
        ),
    ),
    Intent.SHIPPING_QUESTION: (PlanStep(tool="search_knowledge_base"),),
    Intent.ACCOUNT_QUESTION: (
        PlanStep(tool="search_knowledge_base"),
        PlanStep(tool="lookup_customer", needs_identity=True, optional=True),
    ),
    Intent.BILLING_DISPUTE: (
        PlanStep(tool="lookup_order", needs_order_reference=True, needs_identity=True),
        PlanStep(tool="search_knowledge_base", optional=True),
    ),
    Intent.TECHNICAL_ISSUE: (PlanStep(tool="search_knowledge_base"),),
    Intent.COMPLAINT: (PlanStep(tool="search_knowledge_base", optional=True),),
    Intent.SPEAK_TO_HUMAN: (),
    Intent.UNKNOWN: (),
}

#: Intents that always end with a human, whatever the tools return. A billing
#: dispute is a money question and a complaint is a relationship question;
#: neither is closed by an agent quoting a policy.
ALWAYS_ESCALATE: frozenset[Intent] = frozenset(
    {Intent.SPEAK_TO_HUMAN, Intent.BILLING_DISPUTE, Intent.COMPLAINT}
)

#: Intents that cannot be answered at all without account access.
REQUIRE_IDENTITY: frozenset[Intent] = frozenset(
    {Intent.ORDER_STATUS, Intent.REFUND_REQUEST, Intent.ACCOUNT_QUESTION, Intent.BILLING_DISPUTE}
)


def plan(context: PlanContext) -> tuple[ToolCall, ...]:
    """Build the ordered tool calls for one turn.

    Steps whose preconditions are unmet are dropped rather than attempted, so
    the run does not spend budget on calls the registry would refuse anyway.
    """
    calls: list[ToolCall] = []

    for step in _PLANS.get(context.intent, ()):
        if step.needs_identity and not context.identity_verified:
            continue
        if step.needs_order_reference and not context.order_reference:
            continue

        arguments = dict(step.arguments)
        if step.tool in {"lookup_order", "check_refund_eligibility", "check_return_eligibility"}:
            arguments["order_reference"] = context.order_reference
        elif step.tool == "search_knowledge_base":
            arguments["query"] = context.message[:500]
        elif step.tool == "lookup_customer":
            if not context.customer_id:
                continue
            arguments["customer_id"] = context.customer_id

        calls.append(ToolCall(tool=step.tool, arguments=arguments, requested_by="planner"))

    return tuple(calls)


def needs_identity(intent: Intent) -> bool:
    """Whether this intent cannot be served without a verified customer."""
    return intent in REQUIRE_IDENTITY


def always_escalates(intent: Intent) -> bool:
    """Whether this intent ends with a human regardless of what tools returned."""
    return intent in ALWAYS_ESCALATE


def missing_prerequisite(context: PlanContext) -> str | None:
    """Return what the agent must ask for before it can act, if anything.

    Asking one precise question is better than running a plan that will produce
    nothing and then apologising.
    """
    if context.intent in REQUIRE_IDENTITY and not context.identity_verified:
        return "identity"
    if (
        context.intent in {Intent.ORDER_STATUS, Intent.REFUND_REQUEST}
        and not context.order_reference
    ):
        return "order_reference"
    return None


def produced_data(results: tuple[ToolResult, ...]) -> bool:
    """Whether any call in the plan returned usable data."""
    return any(result.succeeded and result.data for result in results)


__all__ = [
    "ALWAYS_ESCALATE",
    "REQUIRE_IDENTITY",
    "PlanContext",
    "PlanStep",
    "always_escalates",
    "missing_prerequisite",
    "needs_identity",
    "plan",
    "produced_data",
]
