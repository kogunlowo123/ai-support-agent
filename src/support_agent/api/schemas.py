"""HTTP request and response models.

Separate from the domain models on purpose. The domain is free to change shape
as the agent evolves; the wire contract is not. Keeping them apart also means an
internal field cannot become part of the public API by accident — a tool result
payload is never serialised to a client, only the provenance reference that
points at it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from support_agent.domain.models import (
    AgentState,
    Answer,
    EscalationReason,
    Intent,
    RefusalReason,
)

MessageText = Annotated[
    str, StringConstraints(min_length=1, max_length=4000, strip_whitespace=True)
]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]


class ApiModel(BaseModel):
    """Base for wire models: unknown fields are rejected rather than ignored."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(ApiModel):
    """Liveness response."""

    status: str = "ok"
    version: str


class ComponentStatus(ApiModel):
    """Readiness of one dependency."""

    name: str
    ready: bool
    detail: str = ""


class ReadinessResponse(ApiModel):
    """Readiness response, including circuit-breaker state.

    A tool whose breaker is open is not a reason to fail readiness — the agent
    degrades by escalating — but it is something an operator must be able to see
    without reading logs.
    """

    status: str
    version: str
    components: list[ComponentStatus]
    open_circuits: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StartConversationRequest(ApiModel):
    """Open a conversation thread."""

    customer_email: str | None = Field(
        default=None,
        max_length=320,
        description="Optional. Binds the thread to a customer before verification.",
    )


class ConversationResponse(ApiModel):
    """A conversation thread."""

    conversation_id: str
    identity_verified: bool
    customer_id: str | None
    turns: int
    escalated: bool
    created_at: datetime


class VerifyIdentityRequest(ApiModel):
    """Answer the identity challenge for a conversation."""

    email: Annotated[str, StringConstraints(min_length=3, max_length=320)]


class VerifyIdentityResponse(ApiModel):
    """Outcome of an identity check."""

    conversation_id: str
    verified: bool
    message: str


class MessageRequest(ApiModel):
    """Send a customer message to the agent."""

    message: MessageText
    include_trace: bool = Field(
        default=False,
        description="Return the step-by-step trace. Echoes decisions back, so it is opt-in.",
    )


class ProvenanceResponse(ApiModel):
    """Where a statement in the reply came from."""

    kind: str
    reference: str


class StepResponse(ApiModel):
    """One recorded step of the run."""

    index: int
    kind: str
    state: str
    summary: str
    detail: dict[str, Any]
    duration_ms: float


class MessageResponse(ApiModel):
    """The agent's reply and everything needed to audit it."""

    conversation_id: str
    reply: str
    intent: Intent
    state: AgentState
    escalated: bool
    escalation_reason: EscalationReason | None
    refused: bool
    refusal_reason: RefusalReason | None
    ticket_id: str | None
    provenance: list[ProvenanceResponse]
    unverified_claims: list[str]
    warnings: list[str]
    tool_calls: int
    steps_used: int
    duration_ms: float
    provider: str
    model: str
    trace: list[StepResponse] | None = None

    @classmethod
    def from_domain(
        cls, answer: Answer, *, conversation_id: str, include_trace: bool
    ) -> MessageResponse:
        """Project a domain answer onto the wire model."""
        return cls(
            conversation_id=conversation_id,
            reply=answer.text,
            intent=answer.intent,
            state=answer.state,
            escalated=answer.escalated,
            escalation_reason=answer.escalation_reason,
            refused=answer.refused,
            refusal_reason=answer.refusal_reason,
            ticket_id=answer.ticket_id,
            provenance=[
                ProvenanceResponse(kind=str(item.kind), reference=item.reference)
                for item in answer.provenance
            ],
            unverified_claims=list(answer.unverified_claims),
            warnings=list(answer.warnings),
            tool_calls=answer.tool_calls,
            steps_used=len(answer.steps),
            duration_ms=answer.duration_ms,
            provider=answer.provider,
            model=answer.model,
            trace=(
                [
                    StepResponse(
                        index=step.index,
                        kind=str(step.kind),
                        state=str(step.state),
                        summary=step.summary,
                        detail=step.detail,
                        duration_ms=step.duration_ms,
                    )
                    for step in answer.steps
                ]
                if include_trace
                else None
            ),
        )


class FeedbackRequest(ApiModel):
    """A customer's verdict on the last reply."""

    helpful: bool
    comment: str = Field(default="", max_length=2000)


class ToolDescription(ApiModel):
    """A registered tool and the constraints on calling it."""

    name: str
    description: str
    risk: str
    requires_identity: bool
    required_scopes: list[str]
    allowed_intents: list[str]
    idempotent: bool
    parameters: dict[str, Any]


class ToolListResponse(ApiModel):
    """Every registered tool, and which are reachable right now."""

    tools: list[ToolDescription]
    open_circuits: list[str] = Field(default_factory=list)


class TicketResponse(ApiModel):
    """A support ticket."""

    ticket_id: str
    subject: str
    category: str
    priority: str
    status: str
    conversation_id: str | None
    created_at: datetime


class TicketListResponse(ApiModel):
    """A page of tickets."""

    tickets: list[TicketResponse]


class RunSummaryResponse(ApiModel):
    """One recorded run."""

    run_id: str
    conversation_id: str
    intent: str
    intent_confidence: float
    final_state: str
    escalation_reason: str | None
    refusal_reason: str | None
    tool_calls: int
    steps_used: int
    duration_ms: float
    verified: bool
    created_at: datetime


class RunListResponse(ApiModel):
    """A page of runs."""

    runs: list[RunSummaryResponse]


class RunTraceResponse(ApiModel):
    """The full step trace of one run."""

    run_id: str
    steps: list[StepResponse]


class AuditEventResponse(ApiModel):
    """One audit event."""

    event: str
    outcome: str
    actor_key_id: str
    request_id: str | None
    conversation_id: str | None
    subject_id: str | None
    attributes: dict[str, Any]
    created_at: datetime


class AuditListResponse(ApiModel):
    """A page of audit events."""

    events: list[AuditEventResponse]


class FeedbackSummaryResponse(ApiModel):
    """Helpful and unhelpful counts."""

    helpful: int
    unhelpful: int


class ErrorResponse(ApiModel):
    """The single error shape returned by every failing endpoint."""

    code: str
    message: str
    request_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ApiModel",
    "AuditEventResponse",
    "AuditListResponse",
    "ComponentStatus",
    "ConversationResponse",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackSummaryResponse",
    "HealthResponse",
    "MessageRequest",
    "MessageResponse",
    "ProvenanceResponse",
    "ReadinessResponse",
    "RunListResponse",
    "RunSummaryResponse",
    "RunTraceResponse",
    "StartConversationRequest",
    "StepResponse",
    "TicketListResponse",
    "TicketResponse",
    "ToolDescription",
    "ToolListResponse",
    "VerifyIdentityRequest",
    "VerifyIdentityResponse",
]
