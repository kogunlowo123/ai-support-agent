"""Operational endpoints: tickets, run traces, audit and feedback.

These are what an operator uses to answer "what did the agent do, and why?".
The run trace is the important one: for any reply a customer received, it lists
every tool call, every policy decision, what the verifier found and the state
the run ended in.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from support_agent.api.dependencies import ContextDep, PrincipalDep
from support_agent.api.schemas import (
    AuditEventResponse,
    AuditListResponse,
    FeedbackSummaryResponse,
    RunListResponse,
    RunSummaryResponse,
    RunTraceResponse,
    StepResponse,
    TicketListResponse,
    TicketResponse,
)
from support_agent.errors import NotFoundError

router = APIRouter(prefix="/v1", tags=["operations"])


@router.get("/tickets", response_model=TicketListResponse, summary="Recent tickets")
async def list_tickets(
    context: ContextDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> TicketListResponse:
    """Return the tenant's most recent tickets, newest first."""
    rows = await context.tickets.recent(principal.tenant_id, limit=limit)
    return TicketListResponse(
        tickets=[
            TicketResponse(
                ticket_id=row.id,
                subject=row.subject,
                category=row.category,
                priority=row.priority,
                status=row.status,
                conversation_id=row.conversation_id,
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.get("/runs", response_model=RunListResponse, summary="Recent agent runs")
async def list_runs(
    context: ContextDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> RunListResponse:
    """Return the tenant's most recent runs, newest first."""
    rows = await context.runs.recent(principal.tenant_id, limit=limit)
    return RunListResponse(
        runs=[
            RunSummaryResponse(
                run_id=row.id,
                conversation_id=row.conversation_id,
                intent=row.intent,
                intent_confidence=row.intent_confidence,
                final_state=row.final_state,
                escalation_reason=row.escalation_reason,
                refusal_reason=row.refusal_reason,
                tool_calls=row.tool_calls,
                steps_used=row.steps_used,
                duration_ms=row.duration_ms,
                verified=row.verified,
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.get(
    "/runs/{run_id}/trace",
    response_model=RunTraceResponse,
    summary="The step-by-step trace of one run",
)
async def run_trace(run_id: str, context: ContextDep, principal: PrincipalDep) -> RunTraceResponse:
    """Return every step the run took, in order."""
    steps = await context.runs.steps_for(principal.tenant_id, run_id)
    if not steps:
        raise NotFoundError("run not found or not accessible")
    return RunTraceResponse(
        run_id=run_id,
        steps=[
            StepResponse(
                index=step.index,
                kind=step.kind,
                state=step.state,
                summary=step.summary,
                detail=dict(step.detail or {}),
                duration_ms=step.duration_ms,
            )
            for step in steps
        ],
    )


@router.get("/audit", response_model=AuditListResponse, summary="Recent audit events")
async def list_audit(
    context: ContextDep,
    principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> AuditListResponse:
    """Return the tenant's recent audit events, newest first."""
    rows = await context.audit.recent(principal.tenant_id, limit=limit)
    return AuditListResponse(
        events=[
            AuditEventResponse(
                event=row.event,
                outcome=row.outcome,
                actor_key_id=row.actor_key_id,
                request_id=row.request_id,
                conversation_id=row.conversation_id,
                subject_id=row.subject_id,
                attributes=dict(row.attributes or {}),
                created_at=row.created_at,
            )
            for row in rows
        ]
    )


@router.get(
    "/feedback/summary",
    response_model=FeedbackSummaryResponse,
    summary="Helpful and unhelpful counts",
)
async def feedback_summary(context: ContextDep, principal: PrincipalDep) -> FeedbackSummaryResponse:
    """Return the tenant's feedback totals."""
    counts = await context.feedback.summary(principal.tenant_id)
    return FeedbackSummaryResponse(helpful=counts["helpful"], unhelpful=counts["unhelpful"])


__all__ = ["router"]
