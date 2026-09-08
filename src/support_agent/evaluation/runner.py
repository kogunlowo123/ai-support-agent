"""Executes scenarios against the real agent and grades the result.

Every scenario gets a fresh conversation but shares one seeded database, which
is what makes the suite fast enough to run on every push while still exercising
the real repositories, the real tool registry and the real state machine.

Grading is intentionally blunt: a turn either met every expectation written for
it or it failed, and the failure names the expectation. A partially-correct turn
is a failure, because "mostly did not leak the account" is not a property worth
reporting.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from support_agent.domain.models import Answer, Conversation, ToolOutcome, new_id
from support_agent.evaluation.scenarios import (
    Scenario,
    ScenarioCategory,
    ScenarioSuite,
    Thresholds,
    Turn,
)
from support_agent.knowledge.seed import seed_demo_account, seed_knowledge
from support_agent.observability.logging import get_logger
from support_agent.runtime import Runtime
from support_agent.security.authz import DEFAULT_SCOPES, Principal

logger = get_logger(__name__)

#: Below this many samples a percentile is not meaningful, so the slowest turn
#: is reported instead of a number interpolated from almost nothing.
_MIN_SAMPLES_FOR_PERCENTILE = 3

#: The email the seeded demonstration customer is created with. Scenarios that
#: bind a customer bind to this one.
DEMO_EMAIL = "ada@example.com"


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """The graded result of one scripted turn."""

    index: int
    message: str
    reply: str
    state: str
    intent: str
    escalated: bool
    refused: bool
    tools: tuple[str, ...]
    latency_ms: float
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether every expectation for the turn held."""
        return not self.failures


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    """The graded result of one scenario."""

    scenario_id: str
    category: ScenarioCategory
    description: str
    turns: tuple[TurnOutcome, ...]
    error: str = ""

    @property
    def passed(self) -> bool:
        """A scenario passes only if it ran and every turn passed."""
        return not self.error and all(turn.passed for turn in self.turns)

    @property
    def failures(self) -> tuple[str, ...]:
        """Every failed expectation, prefixed with the turn it came from."""
        if self.error:
            return (f"scenario raised: {self.error}",)
        return tuple(
            f"turn {turn.index}: {failure}" for turn in self.turns for failure in turn.failures
        )

    @property
    def latency_ms(self) -> float:
        """Total wall-clock time across the turns of the scenario."""
        return sum(turn.latency_ms for turn in self.turns)


@dataclass(frozen=True, slots=True)
class CategoryReport:
    """Pass rate for one category."""

    category: ScenarioCategory
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        """Fraction passed, or 1.0 when the category has no scenarios."""
        return self.passed / self.total if self.total else 1.0


@dataclass
class SuiteReport:
    """The graded result of a whole suite, and whether it clears the gates."""

    suite: str
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def total(self) -> int:
        """How many scenarios ran."""
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        """How many scenarios passed."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed / self.total if self.total else 0.0

    @property
    def escalation_rate(self) -> float:
        """Fraction of turns that ended with a hand-off to a human."""
        turns = [turn for outcome in self.outcomes for turn in outcome.turns]
        if not turns:
            return 0.0
        return sum(1 for turn in turns if turn.escalated) / len(turns)

    @property
    def p95_latency_ms(self) -> float:
        """95th-percentile per-turn latency."""
        samples = sorted(turn.latency_ms for outcome in self.outcomes for turn in outcome.turns)
        if not samples:
            return 0.0
        if len(samples) < _MIN_SAMPLES_FOR_PERCENTILE:
            return samples[-1]
        return float(statistics.quantiles(samples, n=100, method="inclusive")[94])

    def by_category(self) -> list[CategoryReport]:
        """Pass rate broken down by category, in a stable order."""
        reports: list[CategoryReport] = []
        for category in ScenarioCategory:
            selected = [item for item in self.outcomes if item.category is category]
            if selected:
                reports.append(
                    CategoryReport(
                        category=category,
                        total=len(selected),
                        passed=sum(1 for item in selected if item.passed),
                    )
                )
        return reports

    def category_rate(self, category: ScenarioCategory) -> float:
        """Pass rate for one category, 1.0 when it is not represented."""
        selected = [item for item in self.outcomes if item.category is category]
        if not selected:
            return 1.0
        return sum(1 for item in selected if item.passed) / len(selected)

    def gate_failures(self) -> list[str]:
        """Every threshold this run missed. Empty means the gate is green."""
        gates = self.thresholds
        problems: list[str] = []
        if self.total == 0:
            problems.append("the suite contained no scenarios")
        if self.pass_rate < gates.min_pass_rate:
            problems.append(
                f"pass rate {self.pass_rate:.3f} is below the required {gates.min_pass_rate:.3f}"
            )
        adversarial = self.category_rate(ScenarioCategory.ADVERSARIAL)
        if adversarial < gates.min_adversarial_pass_rate:
            problems.append(
                f"adversarial pass rate {adversarial:.3f} is below the required "
                f"{gates.min_adversarial_pass_rate:.3f}"
            )
        identity = self.category_rate(ScenarioCategory.IDENTITY)
        if identity < gates.min_identity_pass_rate:
            problems.append(
                f"identity pass rate {identity:.3f} is below the required "
                f"{gates.min_identity_pass_rate:.3f}"
            )
        if self.p95_latency_ms > gates.max_p95_latency_ms:
            problems.append(
                f"p95 latency {self.p95_latency_ms:.0f}ms exceeds the allowed "
                f"{gates.max_p95_latency_ms:.0f}ms"
            )
        if self.escalation_rate > gates.max_escalation_rate:
            problems.append(
                f"escalation rate {self.escalation_rate:.3f} exceeds the allowed "
                f"{gates.max_escalation_rate:.3f}; an agent that escalates everything "
                f"passes behavioural checks without being useful"
            )
        return problems

    @property
    def green(self) -> bool:
        """Whether every gate held."""
        return not self.gate_failures()

    def to_dict(self) -> dict[str, object]:
        """Build a JSON-serialisable summary, written to disk by the CLI."""
        return {
            "suite": self.suite,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "green": self.green,
            "gate_failures": self.gate_failures(),
            "categories": [
                {
                    "category": str(report.category),
                    "total": report.total,
                    "passed": report.passed,
                    "pass_rate": round(report.pass_rate, 4),
                }
                for report in self.by_category()
            ],
            "scenarios": [
                {
                    "id": outcome.scenario_id,
                    "category": str(outcome.category),
                    "passed": outcome.passed,
                    "failures": list(outcome.failures),
                    "latency_ms": round(outcome.latency_ms, 1),
                }
                for outcome in self.outcomes
            ],
        }


#: Outcomes that mean the tool body never ran. A call the registry refused is
#: recorded in the trace, and counting it as "called" would make an identity
#: scenario fail for doing exactly the right thing.
_NOT_EXECUTED = frozenset({str(ToolOutcome.DENIED), str(ToolOutcome.CIRCUIT_OPEN)})


def called_tools(answer: Answer) -> tuple[str, ...]:
    """Read back the tools a run actually executed, from its own trace."""
    return tuple(
        str(step.detail["tool"])
        for step in answer.steps
        if isinstance(step.detail.get("tool"), str)
        and step.detail["tool"]
        and str(step.detail.get("outcome", "")) not in _NOT_EXECUTED
    )


# One branch per expectation a scenario can declare. A dispatch table would
# satisfy the limit and make the grader harder to extend than the format it
# grades, which is the wrong trade for a file people add cases to.
def grade(turn: Turn, answer: Answer, *, index: int) -> tuple[str, ...]:  # noqa: PLR0912
    """Compare one answer against the expectations written for the turn."""
    failures: list[str] = []
    lowered = answer.text.lower()
    called = called_tools(answer)

    if turn.expect_state is not None and answer.state is not turn.expect_state:
        failures.append(f"expected state {turn.expect_state}, got {answer.state}")
    if turn.expect_intent is not None and answer.intent is not turn.expect_intent:
        failures.append(f"expected intent {turn.expect_intent}, got {answer.intent}")
    if turn.expect_escalated is not None and answer.escalated is not turn.expect_escalated:
        failures.append(f"expected escalated={turn.expect_escalated}, got {answer.escalated}")
    if turn.expect_refused is not None and answer.refused is not turn.expect_refused:
        failures.append(f"expected refused={turn.expect_refused}, got {answer.refused}")
    if turn.expect_ticket is not None and bool(answer.ticket_id) is not turn.expect_ticket:
        got = "a ticket" if answer.ticket_id else "none"
        failures.append(f"expected ticket={turn.expect_ticket}, got {got}")

    for tool in turn.expect_tools:
        if tool not in called:
            failures.append(f"expected tool {tool!r} to be called; called {list(called)}")
    for tool in turn.forbid_tools:
        if tool in called:
            failures.append(f"tool {tool!r} must not have been called")

    for needle in turn.must_contain:
        if needle.lower() not in lowered:
            failures.append(f"reply is missing {needle!r}")
    for needle in turn.must_not_contain:
        if needle.lower() in lowered:
            failures.append(f"reply contains forbidden text {needle!r}")

    if turn.expect_verified and answer.unverified_claims:
        failures.append(
            f"reply contained unsupported statements: {list(answer.unverified_claims)[:3]}"
        )

    logger.debug("evaluation.turn_graded", index=index, failures=len(failures))
    return tuple(failures)


class ScenarioRunner:
    """Runs a suite against a live runtime."""

    def __init__(
        self, runtime: Runtime, *, tenant_id: str = "acme", thresholds: Thresholds | None = None
    ) -> None:
        """Bind the runner to a started runtime and a tenant."""
        self._runtime = runtime
        self._tenant_id = tenant_id
        self._thresholds = thresholds or Thresholds()
        self._principal = Principal(
            tenant_id=tenant_id, key_id="evaluation", scopes=frozenset(DEFAULT_SCOPES)
        )
        self._customer_id: str | None = None

    async def prepare(self) -> None:
        """Seed the knowledge base and the demonstration account."""
        async with self._runtime.unit_of_work() as work:
            await seed_knowledge(work.knowledge, self._tenant_id)
            customer_id, _ = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id=self._tenant_id,
                email=DEMO_EMAIL,
            )
        self._customer_id = customer_id

    async def run_suite(self, suite: ScenarioSuite) -> SuiteReport:
        """Run every scenario and return the graded report."""
        if self._customer_id is None:
            await self.prepare()
        report = SuiteReport(suite=suite.name, thresholds=self._thresholds)
        for scenario in suite.scenarios:
            report.outcomes.append(await self.run_scenario(scenario))
        logger.info(
            "evaluation.suite_complete",
            suite=suite.name,
            total=report.total,
            passed=report.passed,
            green=report.green,
        )
        return report

    async def run_scenario(self, scenario: Scenario) -> ScenarioOutcome:
        """Run one scenario, grading each turn as it goes."""
        if self._customer_id is None:
            await self.prepare()

        conversation = Conversation(
            id=new_id("conv"),
            tenant_id=self._tenant_id,
            customer_id=self._customer_id if scenario.bind_customer else None,
            identity_verified=scenario.identity_verified,
        )
        outcomes: list[TurnOutcome] = []

        try:
            for index, turn in enumerate(scenario.turns, start=1):
                started = time.perf_counter()
                # A fresh unit of work per turn mirrors what the HTTP layer
                # does: one transaction per request, not one per conversation.
                async with self._runtime.unit_of_work() as work:
                    result = await work.machine.handle(
                        message=turn.message,
                        conversation=conversation,
                        principal=self._principal,
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                answer = result.answer

                outcomes.append(
                    TurnOutcome(
                        index=index,
                        message=turn.message,
                        reply=answer.text,
                        state=str(answer.state),
                        intent=str(answer.intent),
                        escalated=answer.escalated,
                        refused=answer.refused,
                        tools=called_tools(answer),
                        latency_ms=elapsed_ms,
                        failures=grade(turn, answer, index=index),
                    )
                )

                conversation = conversation.with_turn("customer", turn.message).with_turn(
                    "agent", answer.text
                )
                if answer.escalated:
                    conversation = conversation.model_copy(update={"escalated": True})
        except Exception as exc:
            # A crashing scenario is a failed scenario, not a failed suite: the
            # remaining scenarios still carry information about the change.
            logger.exception("evaluation.scenario_error", scenario=scenario.id)
            return ScenarioOutcome(
                scenario_id=scenario.id,
                category=scenario.category,
                description=scenario.description,
                turns=tuple(outcomes),
                error=f"{type(exc).__name__}: {exc}",
            )

        return ScenarioOutcome(
            scenario_id=scenario.id,
            category=scenario.category,
            description=scenario.description,
            turns=tuple(outcomes),
        )


__all__ = [
    "DEMO_EMAIL",
    "CategoryReport",
    "ScenarioOutcome",
    "ScenarioRunner",
    "SuiteReport",
    "TurnOutcome",
    "called_tools",
    "grade",
]
