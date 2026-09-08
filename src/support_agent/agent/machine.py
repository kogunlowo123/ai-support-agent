"""The agent, as a state machine.

This is the file that makes the difference between an agent and a loop. There is
no "decide what to do next" step. There is a transition table
(:data:`~support_agent.domain.models.ALLOWED_TRANSITIONS`), a step budget, a
wall-clock budget, and a fixed sequence of stages, each of which can only move
to a state the table permits.

```
RECEIVED ──► CLASSIFIED ──┬──► GATHERING ──► DECIDING ──► COMPOSING ──► VERIFYING ──► ANSWERED
                          │                                                 │
                          ├──► CLARIFYING (terminal)                        ├──► ESCALATED
                          ├──► ESCALATED  (terminal)                        └──► REFUSED
                          └──► REFUSED    (terminal)
```

Every stage appends a :class:`~support_agent.domain.models.Step`, so the trace
is a byproduct of running rather than something added for observability. For any
answer a customer received, the trace says which tools ran, what policy decided,
what the verifier found, and why the run ended where it did.

Exhausting a budget is not an error. It escalates: a run that has spent its
allowance without reaching an answer is exactly the situation a human should
take over.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from support_agent.agent import planner
from support_agent.agent.composer import SYSTEM_POLICY, CompositionRequest, evidence_summary
from support_agent.agent.intent import classify, extract_order_reference
from support_agent.agent.verifier import Evidence, strip_unsupported, verify
from support_agent.domain.models import (
    AgentState,
    Answer,
    EscalationReason,
    Intent,
    PolicyDecision,
    Provenance,
    RefusalReason,
    Step,
    StepKind,
    ToolCall,
    ToolResult,
    can_transition,
)
from support_agent.errors import InvalidTransitionError
from support_agent.observability.logging import get_logger
from support_agent.observability.tracing import (
    escalations,
    get_tracer,
    injection_findings,
    run_latency,
    runs_completed,
    steps_used,
    verification_failures,
)
from support_agent.policy.rules import evaluate_account_access
from support_agent.security import injection
from support_agent.security.disclosure import DisclosureDetector
from support_agent.tools.base import ToolContext
from support_agent.tools.registry import RunBudget

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from support_agent.agent.composer import Composer
    from support_agent.config import Settings
    from support_agent.domain.models import Conversation
    from support_agent.security.authz import Principal
    from support_agent.tools.registry import ToolRegistry

logger = get_logger(__name__)

#: Built once: shingling the policy on every request would put a measurable cost
#: on the hot path for a check whose input never changes.
_DISCLOSURE: Final[DisclosureDetector] = DisclosureDetector(SYSTEM_POLICY)
tracer = get_tracer()

#: Tools whose output is a policy decision rather than data.
_DECISION_TOOLS = frozenset({"check_refund_eligibility", "check_return_eligibility"})

#: What the agent says when it cannot proceed without knowing who it is talking
#: to. Deliberately fixed text: a refusal is not a place for a model to improvise.
_IDENTITY_PROMPT = (
    "Before I can look at your account or your orders, I need to check who you are. "
    "Could you confirm the email address on the account?"
)
_ORDER_REFERENCE_PROMPT = (
    "I can look that up for you. What is the order reference? It looks like ORD-1234 "
    "and is at the top of your confirmation email."
)
_CLARIFY_PROMPT = (
    "I want to make sure I help with the right thing. Could you tell me a little more "
    "about what you need?"
)
_INJECTION_REFUSAL = (
    "I can't act on that. If you have a support question about your orders or your "
    "account, tell me what you need and I will help."
)
_UNVERIFIABLE_REFUSAL = (
    "I could not confirm the details needed to answer that reliably, so I am passing "
    "this to a colleague rather than guessing."
)


@dataclass
class RunResult:
    """The answer, plus what the caller needs to persist alongside it."""

    answer: Answer
    intent_confidence: float
    ticket_id: str | None = None


#: Steps the machine is allowed to record *after* its budget is spent, for the
#: bookkeeping that ends a run: the transition into a terminal state and the
#: record of why. Without this allowance a run that exhausts its budget cannot
#: write down that it did; without the cap the budget is advisory, and a budget
#: that can be overshot by an unbounded amount is not a budget.
_CLOSING_STEP_ALLOWANCE: Final[int] = 3


@dataclass
class _Trace:
    """Accumulates steps and enforces the step budget."""

    max_steps: int
    steps: list[Step] = field(default_factory=list)
    truncated: bool = False

    def add(
        self,
        *,
        kind: StepKind,
        state: AgentState,
        summary: str,
        detail: dict[str, object] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record one step, up to the hard ceiling."""
        if len(self.steps) >= self.ceiling:
            self.truncated = True
            return
        self.steps.append(
            Step(
                index=len(self.steps),
                kind=kind,
                state=state,
                summary=summary,
                detail=detail or {},
                duration_ms=round(duration_ms, 3),
            )
        )

    @property
    def ceiling(self) -> int:
        """The most steps this run may ever record."""
        return self.max_steps + _CLOSING_STEP_ALLOWANCE

    @property
    def exhausted(self) -> bool:
        """Whether the step budget has been spent."""
        return len(self.steps) >= self.max_steps


class AgentMachine:
    """Runs one turn of the conversation."""

    def __init__(
        self,
        *,
        settings: Settings,
        registry: ToolRegistry,
        composer: Composer,
        session: AsyncSession,
    ) -> None:
        """Wire the machine's collaborators."""
        self._settings = settings
        self._registry = registry
        self._composer = composer
        self._session = session

    # Each early return is one documented way a run can end: refused, asked
    # about, escalated, answered. Splitting them into helpers to satisfy the
    # complexity limits would scatter the run across call sites and make the
    # order of the stages — which is the safety argument — impossible to read
    # in one place. The step and wall-clock budgets bound it instead.
    async def handle(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        *,
        message: str,
        conversation: Conversation,
        principal: Principal,
    ) -> RunResult:
        """Handle one customer message and return the answer plus its trace."""
        started = time.perf_counter()
        limits = self._settings.limits
        trace = _Trace(max_steps=limits.max_steps)
        state = AgentState.RECEIVED
        deadline = datetime.now(UTC).timestamp() + limits.max_run_seconds

        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.conversation_id", conversation.id)

            # -- 1. the message itself is untrusted ------------------------
            scan = injection.scan(message)
            if scan.is_suspicious:
                for finding in scan.findings:
                    injection_findings.add(
                        1, {"rule": finding.rule_id, "source": "customer_message"}
                    )
                trace.add(
                    kind=StepKind.GUARD,
                    state=state,
                    summary="instruction-like text in the customer message",
                    detail={
                        "risk": scan.risk,
                        "rules": [finding.rule_id for finding in scan.findings],
                    },
                )
                logger.warning(
                    "security.injection_in_message",
                    conversation_id=conversation.id,
                    risk=scan.risk,
                    rules=[finding.rule_id for finding in scan.findings],
                )
                if (
                    self._settings.security.injection_action == "refuse"
                    or scan.risk >= self._settings.security.injection_refuse_threshold
                ):
                    return self._terminate(
                        trace,
                        state,
                        AgentState.REFUSED,
                        text=_INJECTION_REFUSAL,
                        intent=Intent.UNKNOWN,
                        refusal=RefusalReason.PROMPT_INJECTION,
                        started=started,
                        confidence=0.0,
                    )

            # -- 2. classify ------------------------------------------------
            stage = time.perf_counter()
            classification = classify(message)
            state = self._transition(trace, state, AgentState.CLASSIFIED)
            trace.add(
                kind=StepKind.TRANSITION,
                state=state,
                summary=f"classified as {classification.intent}",
                detail={
                    "intent": str(classification.intent),
                    "confidence": classification.confidence,
                    "matched": list(classification.matched_terms),
                },
                duration_ms=(time.perf_counter() - stage) * 1000.0,
            )

            if not classification.is_confident:
                return self._terminate(
                    trace,
                    state,
                    AgentState.CLARIFYING,
                    text=_CLARIFY_PROMPT,
                    intent=classification.intent,
                    started=started,
                    confidence=classification.confidence,
                )

            intent = classification.intent

            if planner.always_escalates(intent) and intent is Intent.SPEAK_TO_HUMAN:
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.CUSTOMER_REQUEST,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                )

            # -- 3. prerequisites -------------------------------------------
            context = planner.PlanContext(
                intent=intent,
                message=message,
                identity_verified=conversation.identity_verified,
                customer_id=conversation.customer_id,
                order_reference=extract_order_reference(message),
            )
            if missing := planner.missing_prerequisite(context):
                decision = (
                    evaluate_account_access(
                        identity_verified=conversation.identity_verified,
                        settings=self._settings.policy,
                    )
                    if missing == "identity"
                    else None
                )
                trace.add(
                    kind=StepKind.POLICY_DECISION,
                    state=state,
                    summary=f"cannot proceed without {missing}",
                    detail={"missing": missing, "rule": decision.rule if decision else ""},
                )
                return self._terminate(
                    trace,
                    state,
                    AgentState.CLARIFYING,
                    text=(_IDENTITY_PROMPT if missing == "identity" else _ORDER_REFERENCE_PROMPT),
                    intent=intent,
                    started=started,
                    confidence=classification.confidence,
                )

            # -- 4. gather ---------------------------------------------------
            state = self._transition(trace, state, AgentState.GATHERING)
            results, decisions, gather_failed = await self._gather(
                trace=trace,
                state=state,
                calls=planner.plan(context),
                intent=intent,
                conversation=conversation,
                principal=principal,
                deadline=deadline,
            )

            if gather_failed:
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.TOOL_FAILURE,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                    results=results,
                )

            # -- 5. decide ---------------------------------------------------
            state = self._transition(trace, state, AgentState.DECIDING)
            trace.add(
                kind=StepKind.POLICY_DECISION,
                state=state,
                summary=evidence_summary(results, decisions),
                detail={"decisions": [decision.rule for decision in decisions]},
            )

            needs_human = planner.always_escalates(intent) or any(
                decision.requires_human for decision in decisions
            )

            if not results and not decisions:
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.LOW_CONFIDENCE,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                )

            ticket_id: str | None = None
            if needs_human:
                ticket_id = await self._raise_ticket(
                    trace=trace,
                    state=state,
                    intent=intent,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    deadline=deadline,
                    decisions=decisions,
                )

            if trace.exhausted:
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.BUDGET_EXHAUSTED,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                    results=results,
                    ticket_id=ticket_id,
                )

            # -- 6. compose ---------------------------------------------------
            state = self._transition(trace, state, AgentState.COMPOSING)
            stage = time.perf_counter()
            response = await self._composer.compose(
                CompositionRequest(
                    message=message,
                    intent=intent,
                    results=results,
                    decisions=decisions,
                    history=tuple((turn.role, turn.text) for turn in conversation.recent()),
                    escalating=needs_human,
                    ticket_id=ticket_id,
                )
            )
            trace.add(
                kind=StepKind.GENERATION,
                state=state,
                summary=f"composed with {response.provider}",
                detail={
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "degraded": response.metadata.get("degraded", "false"),
                },
                duration_ms=(time.perf_counter() - stage) * 1000.0,
            )

            if not response.text.strip():
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.LOW_CONFIDENCE,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                    results=results,
                    ticket_id=ticket_id,
                )

            # -- 7. verify -----------------------------------------------------
            state = self._transition(trace, state, AgentState.VERIFYING)
            text, provenance, unverified = self._verify(
                trace=trace,
                state=state,
                text=response.text,
                message=message,
                results=results,
                decisions=decisions,
                ticket_id=ticket_id,
            )

            if unverified and self._settings.verification.on_failure == "escalate":
                return await self._escalate(
                    trace,
                    state,
                    intent=intent,
                    reason=EscalationReason.UNVERIFIABLE_ANSWER,
                    message=message,
                    conversation=conversation,
                    principal=principal,
                    started=started,
                    confidence=classification.confidence,
                    results=results,
                    ticket_id=ticket_id,
                    text=_UNVERIFIABLE_REFUSAL,
                )
            if unverified and self._settings.verification.on_failure == "refuse":
                return self._terminate(
                    trace,
                    state,
                    AgentState.REFUSED,
                    text=_UNVERIFIABLE_REFUSAL,
                    intent=intent,
                    refusal=RefusalReason.UNSUPPORTED_CLAIM,
                    started=started,
                    confidence=classification.confidence,
                    unverified=unverified,
                )

            warnings: list[str] = []
            if response.metadata.get("degraded") == "true":
                warnings.append(
                    "the configured language model was unavailable; this reply was composed "
                    "from the tool results directly"
                )
            if unverified:
                warnings.append(
                    f"{len(unverified)} sentence(s) could not be traced to evidence and were "
                    "removed"
                )

            duration = (time.perf_counter() - started) * 1000.0
            state = self._transition(trace, state, AgentState.ANSWERED)
            runs_completed.add(1, {"state": "answered", "intent": str(intent)})
            run_latency.record(duration, {"intent": str(intent)})
            steps_used.record(len(trace.steps), {"intent": str(intent)})

            return RunResult(
                answer=Answer(
                    text=text,
                    intent=intent,
                    state=state,
                    provenance=provenance,
                    escalated=needs_human,
                    escalation_reason=(
                        EscalationReason.POLICY_REQUIRES_HUMAN if needs_human else None
                    ),
                    ticket_id=ticket_id,
                    unverified_claims=unverified,
                    warnings=tuple(warnings),
                    steps=tuple(trace.steps),
                    tool_calls=len(results),
                    duration_ms=round(duration, 3),
                    provider=response.provider,
                    model=response.model,
                ),
                intent_confidence=classification.confidence,
                ticket_id=ticket_id,
            )

    # -- stages -------------------------------------------------------------

    @staticmethod
    def _transition(trace: _Trace, current: AgentState, target: AgentState) -> AgentState:
        """Move the machine, refusing any transition the table does not permit.

        Reaching the exception is always a programming error: the table is the
        design, and this is the assertion that the code has not stepped outside
        it.
        """
        if not can_transition(current, target):
            raise InvalidTransitionError(
                "the agent attempted a transition its state machine does not allow",
                detail={"from": str(current), "to": str(target)},
            )
        return target

    async def _gather(
        self,
        *,
        trace: _Trace,
        state: AgentState,
        calls: tuple[ToolCall, ...],
        intent: Intent,
        conversation: Conversation,
        principal: Principal,
        deadline: float,
    ) -> tuple[tuple[ToolResult, ...], tuple[PolicyDecision, ...], bool]:
        """Run the plan, collecting results and policy decisions."""
        limits = self._settings.limits
        budget = RunBudget(
            max_tool_calls=limits.max_tool_calls,
            max_calls_per_tool=limits.max_tool_calls_per_tool,
        )
        results: list[ToolResult] = []
        decisions: list[PolicyDecision] = []
        hard_failure = False

        for call in calls:
            if trace.exhausted or datetime.now(UTC).timestamp() >= deadline:
                break

            context = ToolContext(
                principal=principal,
                session=self._session,
                conversation_id=conversation.id,
                identity_verified=conversation.identity_verified,
                deadline=datetime.fromtimestamp(deadline, tz=UTC),
                idempotency_key=f"{conversation.id}:{call.tool}",
                metadata={"customer_id": conversation.customer_id or ""},
            )
            result = await self._registry.invoke(call, context, intent=intent, budget=budget)
            if result.data.get("notes_flagged"):
                logger.warning(
                    "security.neutralised_tool_output",
                    conversation_id=conversation.id,
                    tool=result.tool,
                )
            trace.add(
                kind=StepKind.TOOL_CALL,
                state=state,
                summary=f"{result.tool} -> {result.outcome}",
                detail={
                    "tool": result.tool,
                    "outcome": str(result.outcome),
                    "attempts": result.attempts,
                    "replayed": result.replayed,
                    # Tools that return third-party free text say whether any of
                    # it was neutralised. Surfaced on the step so a run trace
                    # shows the defence acting; neutralising only in a log line
                    # makes it invisible to anyone auditing the answer.
                    "untrusted_text_neutralised": bool(result.data.get("notes_flagged")),
                },
                duration_ms=result.duration_ms,
            )

            if result.succeeded:
                results.append(result)
                if result.tool in _DECISION_TOOLS:
                    decisions.append(self._decision_from(result))
            elif result.outcome.value in {"timeout", "error", "circuit_open"}:
                # A denied or not-found call is information. A dependency that
                # broke is a reason to stop and hand over.
                hard_failure = True

        return tuple(results), tuple(decisions), hard_failure

    @staticmethod
    def _decision_from(result: ToolResult) -> PolicyDecision:
        data = result.data
        return PolicyDecision(
            rule=str(data.get("rule", "policy.unknown")),
            allowed=bool(data.get("allowed", False)),
            reason=str(data.get("reason", "")) or "No reason was recorded.",
            facts={
                key: value
                for key, value in data.items()
                if key not in {"rule", "allowed", "reason", "requires_human"}
            },
            requires_human=bool(data.get("requires_human", False)),
        )

    def _verify(
        self,
        *,
        trace: _Trace,
        state: AgentState,
        text: str,
        message: str,
        results: tuple[ToolResult, ...],
        decisions: tuple[PolicyDecision, ...],
        ticket_id: str | None = None,
    ) -> tuple[str, tuple[Provenance, ...], tuple[str, ...]]:
        """Check the draft against the evidence and apply the configured action."""
        config = self._settings.verification
        if not config.enabled:
            trace.add(kind=StepKind.VERIFICATION, state=state, summary="verification disabled")
            return text, (), ()

        stage = time.perf_counter()

        # Checked before the evidence check, because it is not an evidence
        # question. A reply repeating the system policy asserts no amount and
        # names no order, so nothing about it is unsupported — and it is still
        # the one thing the policy tells the model never to do.
        if leaked := _DISCLOSURE.matches(text):
            trace.add(
                kind=StepKind.VERIFICATION,
                state=state,
                summary="the draft repeated the agent's own instructions",
                detail={"repeated_phrases": len(leaked)},
                duration_ms=(time.perf_counter() - stage) * 1000.0,
            )
            logger.warning("security.prompt_disclosure", phrases=len(leaked))
            return "", (), ("the draft repeated the agent's own instructions",)

        evidence = [
            *(Evidence.from_tool_result(result) for result in results if result.succeeded),
            *(Evidence.from_decision(decision) for decision in decisions),
            Evidence.from_customer(message),
        ]
        if ticket_id:
            # The ticket this run raised is a fact the system produced, so the
            # sentence quoting its reference is supported.
            evidence.append(Evidence.from_system("ticket", f"ticket {ticket_id} raised"))
        report = verify(text, evidence, check_numbers=config.forbid_unsupported_numbers)

        unverified = (
            *report.unsupported,
            *(f"unsupported figure: {n}" for n in report.unsupported_numbers),
            *(f"unknown identifier: {i}" for i in report.unsupported_identifiers),
        )
        trace.add(
            kind=StepKind.VERIFICATION,
            state=state,
            summary=(
                f"{report.supported_sentences}/{report.factual_sentences} factual sentences "
                "traced to evidence"
            ),
            detail={
                "ratio": round(report.supported_ratio, 4),
                "unsupported": len(report.unsupported),
                "unsupported_numbers": list(report.unsupported_numbers),
                "unsupported_identifiers": list(report.unsupported_identifiers),
            },
            duration_ms=(time.perf_counter() - stage) * 1000.0,
        )

        if (
            report.supported_ratio >= config.min_supported_ratio
            and not report.unsupported_numbers
            and not report.unsupported_identifiers
        ):
            return text, report.provenance, ()

        verification_failures.add(
            1,
            {
                "cause": (
                    "unsupported_number"
                    if report.unsupported_numbers
                    else "unknown_identifier"
                    if report.unsupported_identifiers
                    else "unsupported_claim"
                )
            },
        )
        logger.warning(
            "verification.failed",
            ratio=round(report.supported_ratio, 4),
            unsupported=len(report.unsupported),
            unsupported_numbers=len(report.unsupported_numbers),
            action=config.on_failure,
        )

        if config.on_failure == "strip":
            return strip_unsupported(text, report), report.provenance, unverified
        return text, report.provenance, unverified

    async def _raise_ticket(
        self,
        *,
        trace: _Trace,
        state: AgentState,
        intent: Intent,
        message: str,
        conversation: Conversation,
        principal: Principal,
        deadline: float,
        decisions: tuple[PolicyDecision, ...],
    ) -> str | None:
        """Raise a ticket for a human, if the registry permits it."""
        summary = "; ".join(decision.reason for decision in decisions)[:1500]
        call = ToolCall(
            tool="create_ticket",
            requested_by="machine",
            arguments={
                "subject": f"{str(intent).replace('_', ' ').title()} needs a colleague",
                "body": (
                    f"Customer said: {message[:1500]}\n\nPolicy decisions: {summary or 'none'}"
                ),
                "category": _ticket_category(intent),
                "priority": "normal",
            },
        )
        context = ToolContext(
            principal=principal,
            session=self._session,
            conversation_id=conversation.id,
            identity_verified=conversation.identity_verified,
            deadline=datetime.fromtimestamp(deadline, tz=UTC),
            idempotency_key=f"{conversation.id}:escalation",
            metadata={"customer_id": conversation.customer_id or ""},
        )
        budget = RunBudget(max_tool_calls=1, max_calls_per_tool=1)
        result = await self._registry.invoke(call, context, intent=intent, budget=budget)
        trace.add(
            kind=StepKind.TOOL_CALL,
            state=state,
            summary=f"create_ticket -> {result.outcome}",
            detail={"outcome": str(result.outcome)},
            duration_ms=result.duration_ms,
        )
        return str(result.data.get("ticket_id")) if result.succeeded else None

    async def _escalate(
        self,
        trace: _Trace,
        state: AgentState,
        *,
        intent: Intent,
        reason: EscalationReason,
        message: str,
        conversation: Conversation,
        principal: Principal,
        started: float,
        confidence: float,
        results: tuple[ToolResult, ...] = (),
        ticket_id: str | None = None,
        text: str | None = None,
    ) -> RunResult:
        """Hand the conversation to a human, raising a ticket if possible."""
        deadline = datetime.now(UTC).timestamp() + self._settings.limits.max_run_seconds
        if ticket_id is None:
            ticket_id = await self._raise_ticket(
                trace=trace,
                state=state,
                intent=intent,
                message=message,
                conversation=conversation,
                principal=principal,
                deadline=deadline,
                decisions=(),
            )

        escalations.add(1, {"reason": str(reason), "intent": str(intent)})
        logger.info(
            "agent.escalated",
            conversation_id=conversation.id,
            intent=str(intent),
            reason=str(reason),
            ticket_id=ticket_id,
        )

        body = text or (
            f"I have passed this to a colleague who can help. Your reference is {ticket_id}."
            if ticket_id
            else "I have passed this to a colleague who can help."
        )
        if text and ticket_id:
            body = f"{text} Your reference is {ticket_id}."

        duration = (time.perf_counter() - started) * 1000.0
        final = self._transition(trace, state, AgentState.ESCALATED)
        trace.add(
            kind=StepKind.TRANSITION,
            state=final,
            summary=f"escalated: {reason}",
            detail={"reason": str(reason), "ticket_id": ticket_id or ""},
        )
        runs_completed.add(1, {"state": "escalated", "intent": str(intent)})
        run_latency.record(duration, {"intent": str(intent)})
        steps_used.record(len(trace.steps), {"intent": str(intent)})

        return RunResult(
            answer=Answer(
                text=body,
                intent=intent,
                state=final,
                escalated=True,
                escalation_reason=reason,
                ticket_id=ticket_id,
                steps=tuple(trace.steps),
                tool_calls=len(results),
                duration_ms=round(duration, 3),
                provider="escalation",
            ),
            intent_confidence=confidence,
            ticket_id=ticket_id,
        )

    def _terminate(
        self,
        trace: _Trace,
        state: AgentState,
        target: AgentState,
        *,
        text: str,
        intent: Intent,
        started: float,
        confidence: float,
        refusal: RefusalReason | None = None,
        unverified: tuple[str, ...] = (),
    ) -> RunResult:
        """End the run in a terminal state without escalating."""
        duration = (time.perf_counter() - started) * 1000.0
        final = self._transition(trace, state, target)
        trace.add(
            kind=StepKind.TRANSITION,
            state=final,
            summary=f"terminated in {final}",
            detail={"refusal": str(refusal) if refusal else ""},
        )
        runs_completed.add(1, {"state": str(final), "intent": str(intent)})
        run_latency.record(duration, {"intent": str(intent)})
        steps_used.record(len(trace.steps), {"intent": str(intent)})

        return RunResult(
            answer=Answer(
                text=text,
                intent=intent,
                state=final,
                refused=final is AgentState.REFUSED,
                refusal_reason=refusal,
                unverified_claims=unverified,
                steps=tuple(trace.steps),
                duration_ms=round(duration, 3),
                provider="policy",
            ),
            intent_confidence=confidence,
        )


def _ticket_category(intent: Intent) -> str:
    """Map an intent onto the ticket categories the tool accepts."""
    match intent:
        case Intent.REFUND_REQUEST:
            return "refund"
        case Intent.RETURN_POLICY:
            return "return"
        case Intent.SHIPPING_QUESTION | Intent.ORDER_STATUS:
            return "shipping"
        case Intent.BILLING_DISPUTE:
            return "billing"
        case Intent.TECHNICAL_ISSUE:
            return "technical"
        case _:
            return "general"


__all__ = ["AgentMachine", "RunResult"]
