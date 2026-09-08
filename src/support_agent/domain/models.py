"""Core domain types for the support agent.

Two ideas run through everything here.

**Provenance.** Every fact the agent can state carries a
:class:`Provenance` describing where it came from: a tool result, a knowledge
article, or the customer's own message. A claim with no provenance is not a
claim the agent is allowed to make, and :mod:`support_agent.agent.verifier`
enforces that after generation rather than trusting the model to comply.

**Bounded execution.** The agent is a state machine with a fixed transition
table, a step budget and a wall-clock budget. It cannot loop, cannot invent a
step, and cannot exceed either budget. :class:`AgentState` is that table.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, computed_field

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


def utc_now() -> datetime:
    """Timezone-aware current time."""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Generate a prefixed identifier, so ids are self-describing in logs."""
    return f"{prefix}_{uuid.uuid4().hex}"


class DomainModel(BaseModel):
    """Base for domain models: immutable, strict, no extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Trust and provenance
# ---------------------------------------------------------------------------


class TrustLevel(StrEnum):
    """How much authority a piece of text carries when it reaches a prompt.

    ``SYSTEM`` text is authored by this application. ``USER`` text is authored
    by the customer and may express intent but never policy. ``UNTRUSTED`` text
    comes from a tool result, a knowledge article, or a ticket written by
    someone else; it is evidence only.
    """

    SYSTEM = "system"
    USER = "user"
    UNTRUSTED = "untrusted"


class ProvenanceKind(StrEnum):
    """Where a fact in a draft answer came from."""

    TOOL_RESULT = "tool_result"
    KNOWLEDGE_ARTICLE = "knowledge_article"
    CUSTOMER_MESSAGE = "customer_message"
    POLICY_DECISION = "policy_decision"


class Provenance(DomainModel):
    """A pointer from a statement in the answer to the evidence behind it."""

    kind: ProvenanceKind
    reference: NonEmptyStr = Field(description="Tool call id, article id, or decision id.")
    excerpt: str = Field(default="", max_length=500)


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------


class Intent(StrEnum):
    """What the customer is asking for.

    The set is closed. An utterance that matches nothing is ``UNKNOWN``, which
    routes to a clarification or a human — never to a guess.
    """

    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    RETURN_POLICY = "return_policy"
    SHIPPING_QUESTION = "shipping_question"
    ACCOUNT_QUESTION = "account_question"
    BILLING_DISPUTE = "billing_dispute"
    TECHNICAL_ISSUE = "technical_issue"
    COMPLAINT = "complaint"
    SPEAK_TO_HUMAN = "speak_to_human"
    UNKNOWN = "unknown"


#: Confidence below which a classification is not acted on, and the margin the
#: winner must hold over the runner-up. Two intents scoring almost the same is
#: a message to ask about, not one to act on. Kept here rather than in the
#: classifier because they define what "confident" means for every caller;
#: support_agent.agent.intent ties its score ceiling to them.
MIN_INTENT_CONFIDENCE: float = 0.35
MIN_INTENT_MARGIN: float = 0.1


class IntentResult(DomainModel):
    """A classified utterance, with the evidence for the classification."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    alternatives: tuple[tuple[Intent, float], ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_confident(self) -> bool:
        """Whether the classification is clear enough to act on without asking."""
        if self.intent is Intent.UNKNOWN:
            return False
        runner_up = self.alternatives[0][1] if self.alternatives else 0.0
        return (
            self.confidence >= MIN_INTENT_CONFIDENCE
            and (self.confidence - runner_up) >= MIN_INTENT_MARGIN
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolRisk(StrEnum):
    """How much damage a tool can do, which decides what it takes to call it."""

    READ = "read"
    WRITE = "write"
    IRREVERSIBLE = "irreversible"


class ToolCall(DomainModel):
    """A request to invoke one tool."""

    id: str = Field(default_factory=lambda: new_id("call"))
    tool: NonEmptyStr
    arguments: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = Field(default="planner")


class ToolOutcome(StrEnum):
    """How a tool call ended."""

    OK = "ok"
    DENIED = "denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"


class ToolResult(DomainModel):
    """The outcome of one tool call.

    ``data`` is structured and schema-validated. The agent may only state facts
    that appear in some ``data`` payload, which is what makes "never invent
    account information" enforceable rather than aspirational.
    """

    call_id: NonEmptyStr
    tool: NonEmptyStr
    outcome: ToolOutcome
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    duration_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)
    idempotency_key: str | None = None
    replayed: bool = Field(
        default=False,
        description="True when a prior identical call was returned instead of re-executing.",
    )

    @property
    def succeeded(self) -> bool:
        """Whether the call produced usable data."""
        return self.outcome is ToolOutcome.OK


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


class AgentState(StrEnum):
    """States of the agent.

    The agent is not a loop that decides what to do next. It is a machine whose
    transitions are declared in :data:`ALLOWED_TRANSITIONS` and checked on every
    step, so a run cannot reach a state the designer did not sanction.
    """

    RECEIVED = "received"
    CLASSIFIED = "classified"
    GATHERING = "gathering"
    DECIDING = "deciding"
    COMPOSING = "composing"
    VERIFYING = "verifying"
    ANSWERED = "answered"
    CLARIFYING = "clarifying"
    ESCALATED = "escalated"
    REFUSED = "refused"
    FAILED = "failed"


#: The complete transition table. Terminal states map to the empty set.
ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.RECEIVED: frozenset({AgentState.CLASSIFIED, AgentState.REFUSED, AgentState.FAILED}),
    AgentState.CLASSIFIED: frozenset(
        {
            AgentState.GATHERING,
            AgentState.DECIDING,
            AgentState.CLARIFYING,
            AgentState.ESCALATED,
            AgentState.REFUSED,
            AgentState.FAILED,
        }
    ),
    AgentState.GATHERING: frozenset(
        {AgentState.GATHERING, AgentState.DECIDING, AgentState.ESCALATED, AgentState.FAILED}
    ),
    AgentState.DECIDING: frozenset(
        {AgentState.COMPOSING, AgentState.ESCALATED, AgentState.REFUSED, AgentState.FAILED}
    ),
    # ESCALATED is reachable from COMPOSING because composition can end with
    # nothing to say: a template that matched no shape, or a model that replied
    # INSUFFICIENT_EVIDENCE. There is no text to verify in that case, and the
    # only safe destination is a person. Handing over must be reachable from
    # every state that can discover it is needed, or the run crashes at exactly
    # the moment it decided to be careful.
    AgentState.COMPOSING: frozenset(
        {AgentState.VERIFYING, AgentState.ESCALATED, AgentState.FAILED}
    ),
    AgentState.VERIFYING: frozenset(
        {AgentState.ANSWERED, AgentState.ESCALATED, AgentState.REFUSED, AgentState.FAILED}
    ),
    AgentState.ANSWERED: frozenset(),
    AgentState.CLARIFYING: frozenset(),
    AgentState.ESCALATED: frozenset(),
    AgentState.REFUSED: frozenset(),
    AgentState.FAILED: frozenset(),
}

TERMINAL_STATES: frozenset[AgentState] = frozenset(
    state for state, nxt in ALLOWED_TRANSITIONS.items() if not nxt
)


def can_transition(current: AgentState, target: AgentState) -> bool:
    """Whether the machine may move from ``current`` to ``target``."""
    return target in ALLOWED_TRANSITIONS[current]


class StepKind(StrEnum):
    """What a recorded step did."""

    TRANSITION = "transition"
    TOOL_CALL = "tool_call"
    POLICY_DECISION = "policy_decision"
    GENERATION = "generation"
    VERIFICATION = "verification"
    GUARD = "guard"


class Step(DomainModel):
    """One recorded step of a run. The trace is the audit trail."""

    index: int = Field(ge=0)
    kind: StepKind
    state: AgentState
    summary: NonEmptyStr
    detail: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(default=0.0, ge=0.0)
    at: datetime = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Escalation, policy and answers
# ---------------------------------------------------------------------------


class EscalationReason(StrEnum):
    """Why a conversation was handed to a human.

    Enumerated rather than free text so escalation rate can be broken down by
    cause, which is the number that tells you whether the agent is improving.
    """

    CUSTOMER_REQUEST = "customer_request"
    LOW_CONFIDENCE = "low_confidence"
    POLICY_REQUIRES_HUMAN = "policy_requires_human"
    UNVERIFIABLE_ANSWER = "unverifiable_answer"
    TOOL_FAILURE = "tool_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REPEATED_FAILURE = "repeated_failure"
    SENSITIVE_TOPIC = "sensitive_topic"


class RefusalReason(StrEnum):
    """Why the agent declined to answer."""

    OUT_OF_SCOPE = "out_of_scope"
    IDENTITY_NOT_VERIFIED = "identity_not_verified"
    PROMPT_INJECTION = "prompt_injection"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    POLICY_PROHIBITED = "policy_prohibited"


class PolicyDecision(DomainModel):
    """A deterministic decision made by code, never by a model.

    Eligibility, entitlement and limits are computed here so that the answer to
    "can I have a refund?" is a function of the order data and the written
    policy, not of what a language model found plausible.
    """

    id: str = Field(default_factory=lambda: new_id("dec"))
    rule: NonEmptyStr
    allowed: bool
    reason: NonEmptyStr
    facts: dict[str, Any] = Field(default_factory=dict)
    requires_human: bool = False


class Answer(DomainModel):
    """What the agent says, plus everything needed to audit why."""

    text: str
    intent: Intent
    state: AgentState
    provenance: tuple[Provenance, ...] = ()
    escalated: bool = False
    escalation_reason: EscalationReason | None = None
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    ticket_id: str | None = None
    unverified_claims: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    steps: tuple[Step, ...] = ()
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    provider: str = ""
    model: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_terminal(self) -> bool:
        """Whether the run reached a state from which nothing follows."""
        return self.state in TERMINAL_STATES


class Turn(DomainModel):
    """One exchange in a conversation."""

    role: NonEmptyStr
    text: str
    at: datetime = Field(default_factory=utc_now)


class Conversation(DomainModel):
    """A conversation thread with one customer."""

    id: str = Field(default_factory=lambda: new_id("conv"))
    tenant_id: NonEmptyStr
    customer_id: str | None = None
    identity_verified: bool = False
    turns: tuple[Turn, ...] = ()
    summary: str = ""
    escalated: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    def with_turn(self, role: str, text: str) -> Self:
        """Return a copy with one more turn appended."""
        return self.model_copy(update={"turns": (*self.turns, Turn(role=role, text=text))})

    def recent(self, limit: int = 6) -> tuple[Turn, ...]:
        """Return the most recent turns, oldest first."""
        return self.turns[-limit:] if limit > 0 else ()


class Feedback(DomainModel):
    """A customer's verdict on one answer."""

    conversation_id: NonEmptyStr
    answer_index: int = Field(ge=0)
    helpful: bool
    comment: str = Field(default="", max_length=2000)
    at: datetime = Field(default_factory=utc_now)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "AgentState",
    "Answer",
    "Conversation",
    "DomainModel",
    "EscalationReason",
    "Feedback",
    "Intent",
    "IntentResult",
    "NonEmptyStr",
    "PolicyDecision",
    "Provenance",
    "ProvenanceKind",
    "RefusalReason",
    "Step",
    "StepKind",
    "ToolCall",
    "ToolOutcome",
    "ToolResult",
    "ToolRisk",
    "TrustLevel",
    "Turn",
    "can_transition",
    "new_id",
    "utc_now",
]
