"""Conversation endpoints: start a thread, verify identity, send a message.

Identity verification is a property of the conversation, not the API key. A
correctly authenticated caller — a support widget embedded in a page — still
cannot read an account until the person on the other end has proved who they
are. That is why verification is an endpoint on the conversation rather than a
claim in a token.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from support_agent.api.dependencies import ContextDep, PrincipalDep, ServicesDep
from support_agent.api.schemas import (
    ConversationResponse,
    FeedbackRequest,
    MessageRequest,
    MessageResponse,
    StartConversationRequest,
    VerifyIdentityRequest,
    VerifyIdentityResponse,
)
from support_agent.domain.models import Conversation, new_id
from support_agent.errors import NotFoundError, ValidationError
from support_agent.observability.logging import get_logger, session_id_var

logger = get_logger(__name__)
router = APIRouter(prefix="/v1/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a conversation",
)
async def start_conversation(
    body: StartConversationRequest,
    context: ContextDep,
    principal: PrincipalDep,
) -> ConversationResponse:
    """Open a thread.

    A supplied email binds the thread to a customer but does **not** verify
    them: knowing an address is not proving one. The thread stays unverified
    until :func:`verify_identity` succeeds.
    """
    customer_id: str | None = None
    if body.customer_email:
        customer = await context.customers.by_email(principal.tenant_id, body.customer_email)
        customer_id = customer.id if customer else None

    conversation = Conversation(
        id=new_id("conv"), tenant_id=principal.tenant_id, customer_id=customer_id
    )
    await context.conversations.upsert(conversation)
    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="conversation.start",
        outcome="created",
        conversation_id=conversation.id,
    )
    return _to_response(conversation)


@router.post(
    "/{conversation_id}/verify",
    response_model=VerifyIdentityResponse,
    summary="Verify the customer's identity for this conversation",
)
async def verify_identity(
    conversation_id: str,
    body: VerifyIdentityRequest,
    context: ContextDep,
    principal: PrincipalDep,
) -> VerifyIdentityResponse:
    """Confirm who the conversation is with.

    This implementation checks the email against the customer record, which is
    a knowledge-factor check suitable for a demonstration and for a deployment
    behind an already-authenticated session. A public deployment must replace it
    with a possession factor — a code sent to the address on file — and
    ``SECURITY.md`` says so rather than implying this is sufficient.

    The response is identical whether the address is unknown or simply wrong, so
    the endpoint cannot be used to test which addresses have accounts.
    """
    conversation = await context.conversations.get(principal.tenant_id, conversation_id)
    if conversation is None:
        raise NotFoundError("conversation not found or not accessible")

    customer = await context.customers.by_email(principal.tenant_id, body.email)
    verified = customer is not None

    if verified and customer is not None:
        conversation = conversation.model_copy(
            update={"identity_verified": True, "customer_id": customer.id}
        )
        await context.conversations.upsert(conversation, verification_method="email_code")

    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="identity.verify",
        outcome="verified" if verified else "rejected",
        conversation_id=conversation_id,
    )
    logger.info(
        "identity.verification",
        conversation_id=conversation_id,
        outcome="verified" if verified else "rejected",
    )

    return VerifyIdentityResponse(
        conversation_id=conversation_id,
        verified=verified,
        message=(
            "Thanks, I have confirmed your identity."
            if verified
            else "I could not match those details. Please check them and try again."
        ),
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageResponse,
    summary="Send a message to the agent",
)
async def send_message(
    conversation_id: str,
    body: MessageRequest,
    context: ContextDep,
    services: ServicesDep,
    principal: PrincipalDep,
) -> MessageResponse:
    """Run one turn of the agent.

    Always a 200 when the agent ran. A refusal, a clarifying question and an
    escalation are correct outcomes, not errors, and returning them as failures
    would make refusal rate invisible in an error dashboard.
    """
    if len(body.message) > services.settings.limits.max_message_chars:
        raise ValidationError("the message exceeds the configured maximum length")

    conversation = await context.conversations.get(principal.tenant_id, conversation_id)
    if conversation is None:
        raise NotFoundError("conversation not found or not accessible")
    if len(conversation.turns) >= services.settings.limits.max_conversation_turns:
        raise ValidationError(
            "this conversation has reached its turn limit; please start a new one"
        )

    session_id_var.set(conversation_id)
    result = await context.machine.handle(
        message=body.message, conversation=conversation, principal=principal
    )
    answer = result.answer

    updated = conversation.with_turn("customer", body.message).with_turn("agent", answer.text)
    if answer.escalated:
        updated = updated.model_copy(update={"escalated": True})
    await context.conversations.upsert(updated)

    await context.runs.record(
        tenant_id=principal.tenant_id,
        conversation_id=conversation_id,
        answer=answer,
        intent_confidence=result.intent_confidence,
    )
    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="agent.run",
        outcome=str(answer.state),
        conversation_id=conversation_id,
        subject_id=answer.ticket_id,
        attributes={
            "intent": str(answer.intent),
            "tool_calls": answer.tool_calls,
            "escalated": answer.escalated,
            "refused": answer.refused,
            "unverified_claims": len(answer.unverified_claims),
        },
    )

    return MessageResponse.from_domain(
        answer, conversation_id=conversation_id, include_trace=body.include_trace
    )


@router.get("/{conversation_id}", response_model=ConversationResponse, summary="Get a conversation")
async def get_conversation(
    conversation_id: str, context: ContextDep, principal: PrincipalDep
) -> ConversationResponse:
    """Return a conversation's state."""
    conversation = await context.conversations.get(principal.tenant_id, conversation_id)
    if conversation is None:
        raise NotFoundError("conversation not found or not accessible")
    return _to_response(conversation)


@router.post(
    "/{conversation_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Record whether the last reply helped",
)
async def record_feedback(
    conversation_id: str,
    body: FeedbackRequest,
    context: ContextDep,
    principal: PrincipalDep,
) -> None:
    """Capture a verdict on the conversation.

    Feedback is joined to runs by conversation, so an unhelpful verdict can be
    read alongside the trace that produced it — which is the only way feedback
    becomes actionable rather than a number on a dashboard.
    """
    conversation = await context.conversations.get(principal.tenant_id, conversation_id)
    if conversation is None:
        raise NotFoundError("conversation not found or not accessible")

    runs = await context.runs.recent(principal.tenant_id, limit=20)
    latest = next((run for run in runs if run.conversation_id == conversation_id), None)

    await context.feedback.record(
        tenant_id=principal.tenant_id,
        conversation_id=conversation_id,
        run_id=latest.id if latest else None,
        helpful=body.helpful,
        comment=body.comment,
    )
    await context.audit.record(
        tenant_id=principal.tenant_id,
        actor_key_id=principal.key_id,
        event="feedback.record",
        outcome="helpful" if body.helpful else "unhelpful",
        conversation_id=conversation_id,
    )


def _to_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=conversation.id,
        identity_verified=conversation.identity_verified,
        customer_id=conversation.customer_id,
        turns=len(conversation.turns),
        escalated=conversation.escalated,
        created_at=conversation.created_at,
    )


__all__ = ["router"]
