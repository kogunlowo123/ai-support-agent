"""The tool registry: where authority is granted, and where it is refused.

Every tool call passes through :meth:`ToolRegistry.invoke`, and it is the only
path. The checks below run in a fixed order, cheapest and most decisive first,
so that a call which should never happen costs nothing and leaves no trace in a
dependency:

1. **Registered?** An unknown name is a planning bug or an injection attempt.
2. **Permitted for this intent?** A tool legitimate for a refund request is not
   automatically legitimate for a shipping question.
3. **Scopes held?** Deny-by-default against the principal's granted scopes.
4. **Identity verified?** Account data and every write require it.
5. **Budget remaining?** Per-run totals and a per-tool ceiling, so one tool
   cannot consume the whole budget.
6. **Circuit closed?** A failing dependency is not called again yet.
7. **Arguments valid?** Validated against the tool's own model, so the schema
   shown to a planner and the schema enforced here cannot diverge.
8. **Already executed?** An idempotent write with a replayed key returns the
   recorded result instead of executing twice.
9. **Execute** under a timeout, retrying only what is safe to retry.

Nothing here trusts the planner. The planner proposes; the registry decides.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from support_agent.domain.models import Intent, ToolCall, ToolOutcome, ToolResult, ToolRisk
from support_agent.errors import (
    CircuitOpenError,
    ToolArgumentError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
)
from support_agent.observability.logging import get_logger
from support_agent.observability.tracing import get_tracer, tool_calls, tool_latency
from support_agent.tools.base import Tool, ToolContext, ToolSpec
from support_agent.tools.breaker import CircuitBreakerRegistry

if TYPE_CHECKING:
    from support_agent.config import LimitSettings
    from support_agent.security.authz import Principal

logger = get_logger(__name__)
tracer = get_tracer()

#: Base backoff for tool retries. Short: a tool is an internal dependency, not
#: an external API, and a run has a wall-clock budget to respect.
_BASE_BACKOFF_SECONDS = 0.05
_MAX_BACKOFF_SECONDS = 0.8


@dataclass
class RunBudget:
    """Per-run consumption, checked before every call.

    Held on the run rather than the registry so two concurrent conversations
    cannot exhaust each other's budgets.
    """

    max_tool_calls: int
    max_calls_per_tool: int
    used: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)

    def check(self, tool: str) -> str | None:
        """Return a refusal reason, or ``None`` when the call may proceed."""
        if self.used >= self.max_tool_calls:
            return f"the run has used its budget of {self.max_tool_calls} tool calls"
        if self.per_tool.get(tool, 0) >= self.max_calls_per_tool:
            return f"'{tool}' has already been called {self.max_calls_per_tool} times in this run"
        return None

    def consume(self, tool: str) -> None:
        """Record one call against the budget."""
        self.used += 1
        self.per_tool[tool] = self.per_tool.get(tool, 0) + 1


@dataclass(frozen=True, slots=True)
class _Recorded:
    """A completed idempotent call, kept so a replay does not execute twice."""

    result: ToolResult
    at: float


class IdempotencyStore:
    """In-process record of completed idempotent writes.

    Scoped to the process on purpose. It removes the common failure — a retry
    within one run, or a client resubmitting the same request — without
    pretending to give a distributed guarantee it cannot provide. Durable
    idempotency belongs in the downstream system that owns the side effect, and
    the ticket tool enforces it there as well.
    """

    def __init__(self, ttl_seconds: float = 900.0, max_entries: int = 4096) -> None:
        """Configure how long and how many records to keep."""
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, _Recorded] = {}

    @staticmethod
    def key(
        tool: str, principal_id: str, arguments: dict[str, object], supplied: str | None
    ) -> str:
        """Derive the idempotency key for a call.

        A caller-supplied key wins. Otherwise the key is the tool, the principal
        and the canonical arguments, so an accidental repeat inside one run is
        absorbed while two genuinely different requests are not conflated.
        """
        if supplied:
            return f"{tool}:{principal_id}:{supplied}"
        canonical = json.dumps(arguments, sort_keys=True, default=str)
        return f"{tool}:{principal_id}:{canonical}"

    def get(self, key: str) -> ToolResult | None:
        """Return a recorded result if one is still valid."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry.at > self._ttl:
            self._entries.pop(key, None)
            return None
        return entry.result

    def put(self, key: str, result: ToolResult) -> None:
        """Record a result, evicting the oldest entries when full."""
        if len(self._entries) >= self._max:
            oldest = sorted(self._entries.items(), key=lambda item: item[1].at)[: self._max // 4]
            for stale_key, _ in oldest:
                self._entries.pop(stale_key, None)
        self._entries[key] = _Recorded(result=result, at=time.monotonic())

    def clear(self) -> None:
        """Forget every record. Used between test cases."""
        self._entries.clear()


class ToolRegistry:
    """Holds the tools and enforces every constraint on calling them."""

    def __init__(
        self,
        *,
        limits: LimitSettings,
        breakers: CircuitBreakerRegistry,
        idempotency: IdempotencyStore | None = None,
    ) -> None:
        """Build an empty registry with the configured limits."""
        self._tools: dict[str, Tool] = {}
        self._limits = limits
        self._breakers = breakers
        self._idempotency = idempotency or IdempotencyStore()

    # -- registration -------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a tool. Registering the same name twice is a programming error."""
        name = tool.spec.name
        if name in self._tools:
            msg = f"a tool named {name!r} is already registered"
            raise ValueError(msg)
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool, or raise :class:`ToolNotFoundError`."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                "no such tool", detail={"tool": name, "available": sorted(self._tools)}
            )
        return tool

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered tool name."""
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        """Every registered specification."""
        return tuple(self._tools[name].spec for name in sorted(self._tools))

    def available_for(
        self, *, intent: Intent, principal: Principal, identity_verified: bool
    ) -> tuple[ToolSpec, ...]:
        """Return the tools reachable right now.

        This is the allowlist a planner is shown. A tool the caller could not
        successfully invoke is not offered, so the planner is never in a
        position to propose something the registry will refuse.
        """
        return tuple(
            spec
            for spec in self.specs()
            if self._permission_error(
                spec, intent=intent, principal=principal, identity_verified=identity_verified
            )
            is None
        )

    # -- permission ---------------------------------------------------------

    @staticmethod
    def _permission_error(
        spec: ToolSpec, *, intent: Intent, principal: Principal, identity_verified: bool
    ) -> str | None:
        """Return why the tool may not be called, or ``None`` when it may."""
        if spec.allowed_intents and intent not in spec.allowed_intents:
            return f"'{spec.name}' is not permitted for the {intent} intent"
        if spec.required_scopes and not spec.required_scopes <= principal.scopes:
            missing = sorted(spec.required_scopes - principal.scopes)
            return f"'{spec.name}' requires scopes not held by the caller: {missing}"
        if spec.requires_identity and not identity_verified:
            return f"'{spec.name}' requires a verified customer identity"
        return None

    # -- invocation ---------------------------------------------------------

    # One early return per check that can refuse a call. They are deliberately
    # in one function and in a fixed order: this is the list a reviewer reads
    # to answer "what has to be true before a tool runs?".
    async def invoke(  # noqa: PLR0911
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        intent: Intent,
        budget: RunBudget,
    ) -> ToolResult:
        """Execute one tool call, or return a refusal describing why not.

        Refusals are returned as :class:`ToolResult` rather than raised: a
        denied call is information the agent must reason about, not an error
        that should unwind the run.
        """
        started = time.perf_counter()

        try:
            tool = self.get(call.tool)
        except ToolNotFoundError as exc:
            return self._refuse(call, ToolOutcome.NOT_FOUND, exc.message, started)

        spec = tool.spec

        if reason := self._permission_error(
            spec,
            intent=intent,
            principal=context.principal,
            identity_verified=context.identity_verified,
        ):
            logger.warning(
                "tool.denied",
                tool=spec.name,
                intent=str(intent),
                actor=context.principal.key_id,
                reason=reason,
            )
            return self._refuse(call, ToolOutcome.DENIED, reason, started)

        if reason := budget.check(spec.name):
            return self._refuse(call, ToolOutcome.RATE_LIMITED, reason, started)

        if not self._breakers.allows(spec.name):
            return self._refuse(
                call,
                ToolOutcome.CIRCUIT_OPEN,
                f"'{spec.name}' is temporarily unavailable after repeated failures",
                started,
            )

        try:
            arguments = spec.arguments.model_validate(call.arguments)
        except PydanticValidationError as exc:
            # The raw value is excluded: arguments can carry customer data.
            fields = [
                {"field": ".".join(str(part) for part in err["loc"]), "rule": err["type"]}
                for err in exc.errors()[:8]
            ]
            return self._refuse(
                call,
                ToolOutcome.INVALID_ARGUMENTS,
                "the arguments did not match the tool's schema",
                started,
                detail={"fields": fields},
            )

        key: str | None = None
        if spec.risk is not ToolRisk.READ and spec.idempotent:
            key = IdempotencyStore.key(
                spec.name,
                context.principal.key_id,
                arguments.model_dump(mode="json"),
                context.idempotency_key,
            )
            if recorded := self._idempotency.get(key):
                budget.consume(spec.name)
                logger.info("tool.replayed", tool=spec.name, call_id=call.id)
                return recorded.model_copy(update={"call_id": call.id, "replayed": True})

        budget.consume(spec.name)
        result = await self._execute(tool, arguments, context, call, started)

        if key is not None and result.succeeded:
            self._idempotency.put(key, result)
        return result

    async def _execute(
        self,
        tool: Tool,
        arguments: object,
        context: ToolContext,
        call: ToolCall,
        started: float,
    ) -> ToolResult:
        spec = tool.spec
        timeout = spec.timeout_seconds or self._limits.tool_timeout_seconds
        # Only idempotent tools are retried. Retrying a non-idempotent write is
        # how one customer gets two refunds.
        attempts_allowed = self._limits.tool_max_retries + 1 if spec.idempotent else 1

        last_error: str = ""
        last_outcome = ToolOutcome.ERROR

        for attempt in range(1, attempts_allowed + 1):
            remaining = (context.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                return self._finish(
                    call, ToolOutcome.TIMEOUT, "the run budget elapsed", started, attempt
                )

            with tracer.start_as_current_span("agent.tool") as span:
                span.set_attribute("tool.name", spec.name)
                span.set_attribute("tool.attempt", attempt)
                try:
                    payload = await asyncio.wait_for(
                        tool(arguments, context),  # type: ignore[arg-type]
                        timeout=min(timeout, remaining),
                    )
                except TimeoutError:
                    last_outcome, last_error = ToolOutcome.TIMEOUT, "the tool timed out"
                except ToolTimeoutError as exc:
                    last_outcome, last_error = ToolOutcome.TIMEOUT, exc.message
                except ToolPermissionError as exc:
                    self._breakers.record_success(spec.name)
                    return self._finish(call, ToolOutcome.DENIED, exc.message, started, attempt)
                except ToolArgumentError as exc:
                    self._breakers.record_success(spec.name)
                    return self._finish(
                        call, ToolOutcome.INVALID_ARGUMENTS, exc.message, started, attempt
                    )
                except CircuitOpenError as exc:
                    return self._finish(
                        call, ToolOutcome.CIRCUIT_OPEN, exc.message, started, attempt
                    )
                except Exception as exc:
                    last_outcome = ToolOutcome.ERROR
                    # The exception text is logged in full but never returned:
                    # it may quote a query, a row or a downstream response.
                    last_error = "the tool failed"
                    logger.exception(
                        "tool.failed", tool=spec.name, attempt=attempt, error=type(exc).__name__
                    )
                else:
                    self._breakers.record_success(spec.name)
                    duration = (time.perf_counter() - started) * 1000.0
                    tool_calls.add(1, {"tool": spec.name, "outcome": "ok"})
                    tool_latency.record(duration, {"tool": spec.name})
                    return ToolResult(
                        call_id=call.id,
                        tool=spec.name,
                        outcome=ToolOutcome.OK,
                        data=payload.model_dump(mode="json"),
                        duration_ms=round(duration, 3),
                        attempts=attempt,
                        idempotency_key=context.idempotency_key,
                    )

            self._breakers.record_failure(spec.name)
            if attempt < attempts_allowed:
                await asyncio.sleep(_backoff(attempt))

        return self._finish(call, last_outcome, last_error, started, attempts_allowed)

    @staticmethod
    def _refuse(
        call: ToolCall,
        outcome: ToolOutcome,
        message: str,
        started: float,
        detail: dict[str, object] | None = None,
    ) -> ToolResult:
        tool_calls.add(1, {"tool": call.tool, "outcome": str(outcome)})
        return ToolResult(
            call_id=call.id,
            tool=call.tool,
            outcome=outcome,
            message=message,
            data=detail or {},
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    @staticmethod
    def _finish(
        call: ToolCall, outcome: ToolOutcome, message: str, started: float, attempts: int
    ) -> ToolResult:
        duration = (time.perf_counter() - started) * 1000.0
        tool_calls.add(1, {"tool": call.tool, "outcome": str(outcome)})
        tool_latency.record(duration, {"tool": call.tool})
        return ToolResult(
            call_id=call.id,
            tool=call.tool,
            outcome=outcome,
            message=message,
            duration_ms=round(duration, 3),
            attempts=attempts,
        )

    def reset(self) -> None:
        """Clear breaker and idempotency state. Used between test cases."""
        self._breakers.reset()
        self._idempotency.clear()


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter.

    Jittered because several concurrent runs retrying a recovering dependency in
    lockstep is how a partial outage becomes a total one.
    """
    ceiling: float = min(_MAX_BACKOFF_SECONDS, _BASE_BACKOFF_SECONDS * (2**attempt))
    return ceiling * (secrets.randbelow(1000) / 1000.0)


def deadline_from(seconds: float) -> datetime:
    """Return a wall-clock deadline ``seconds`` from now."""
    return datetime.now(UTC) + timedelta(seconds=seconds)


__all__ = [
    "IdempotencyStore",
    "RunBudget",
    "ToolRegistry",
    "deadline_from",
]
