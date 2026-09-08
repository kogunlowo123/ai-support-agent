"""Command-line interface.

Four commands, each corresponding to something a person actually does with this
service: start it, put data in it, talk to it, and find out whether a change
made it worse.

``evaluate`` is the one that matters for the pipeline. It exits non-zero when
the suite misses a threshold, which is what turns the scenario file into a gate
rather than a report nobody reads.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Final

from support_agent import __version__
from support_agent.config import Settings, get_settings
from support_agent.domain.models import Conversation, new_id
from support_agent.evaluation.runner import DEMO_EMAIL, ScenarioRunner, SuiteReport
from support_agent.evaluation.scenarios import ScenarioSuite, Thresholds
from support_agent.knowledge.seed import seed_demo_account, seed_knowledge
from support_agent.observability.logging import configure_logging
from support_agent.runtime import Runtime, build_runtime
from support_agent.security.authz import DEFAULT_SCOPES, Principal

#: Where the checked-in scenario suite lives, relative to the repository root.
DEFAULT_SUITE: Final[Path] = Path("data/scenarios/support.jsonl")

_EXIT_OK: Final[int] = 0
_EXIT_GATE_FAILED: Final[int] = 1
_EXIT_USAGE: Final[int] = 2


def _write(line: str = "") -> None:
    """Write one line to stdout.

    The CLI talks to a person on a terminal, so its output goes to stdout
    directly rather than through the structured logger, which is aimed at a log
    aggregator and would wrap every line in JSON.
    """
    sys.stdout.write(f"{line}\n")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def _serve(arguments: argparse.Namespace, settings: Settings) -> int:
    """Run the HTTP service."""
    import uvicorn

    uvicorn.run(
        "support_agent.api.app:create_app",
        factory=True,
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        log_level=settings.observability.log_level.lower(),
        access_log=False,
        server_header=False,
    )
    return _EXIT_OK


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


async def _seed_async(arguments: argparse.Namespace, settings: Settings) -> int:
    runtime = await build_runtime(settings)
    try:
        async with runtime.unit_of_work() as work:
            articles = await seed_knowledge(work.knowledge, arguments.tenant)
            customer_id, references = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id=arguments.tenant,
                email=arguments.email,
            )
    finally:
        await runtime.aclose()

    _write(f"tenant           {arguments.tenant}")
    _write(f"articles added   {articles}")
    _write(f"customer         {customer_id} ({arguments.email})")
    for purpose, reference in sorted(references.items()):
        _write(f"  {purpose:<20} {reference}")
    return _EXIT_OK


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


async def _chat_async(arguments: argparse.Namespace, settings: Settings) -> int:
    runtime = await build_runtime(settings)
    principal = Principal(
        tenant_id=arguments.tenant, key_id="cli", scopes=frozenset(DEFAULT_SCOPES)
    )
    try:
        async with runtime.unit_of_work() as work:
            await seed_knowledge(work.knowledge, arguments.tenant)
            customer_id, _ = await seed_demo_account(
                customers=work.customers,
                orders=work.orders,
                tenant_id=arguments.tenant,
                email=arguments.email,
            )

        conversation = Conversation(
            id=new_id("conv"),
            tenant_id=arguments.tenant,
            customer_id=customer_id,
            identity_verified=arguments.verified,
        )
        _write(f"conversation {conversation.id}")
        _write(
            "identity verified"
            if arguments.verified
            else "identity NOT verified — account questions will be declined"
        )
        _write("type a message, or an empty line to quit")
        _write()

        while True:
            try:
                # input() blocks the event loop, and the loop is what the
                # agent runs on. Reading on a worker thread keeps the
                # runtime responsive while a person is typing.
                raw = await asyncio.to_thread(input, "you > ")
                message = raw.strip()
            except (EOFError, KeyboardInterrupt):
                _write()
                break
            if not message:
                break

            async with runtime.unit_of_work() as work:
                result = await work.machine.handle(
                    message=message, conversation=conversation, principal=principal
                )
            answer = result.answer
            _write(f"agent > {answer.text}")
            _write(
                f"        [{answer.state}] intent={answer.intent} "
                f"tools={answer.tool_calls} escalated={answer.escalated} "
                f"refused={answer.refused}"
            )
            if answer.ticket_id:
                _write(f"        ticket {answer.ticket_id}")
            if arguments.trace:
                for step in answer.steps:
                    _write(f"        {step.index:>2}. {step.kind}: {step.summary}")
            _write()

            conversation = conversation.with_turn("customer", message).with_turn(
                "agent", answer.text
            )
    finally:
        await runtime.aclose()
    return _EXIT_OK


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _render_report(report: SuiteReport, *, verbose: bool) -> None:
    """Print the report in a form a person reads in a CI log."""
    _write(f"suite            {report.suite}")
    _write(f"scenarios        {report.passed}/{report.total} passed")
    _write(f"pass rate        {report.pass_rate:.3f}")
    _write(f"escalation rate  {report.escalation_rate:.3f}")
    _write(f"p95 latency      {report.p95_latency_ms:.0f}ms")
    _write()
    for category in report.by_category():
        _write(
            f"  {category.category!s:<12} {category.passed}/{category.total} "
            f"({category.pass_rate:.3f})"
        )
    _write()

    for outcome in report.outcomes:
        if outcome.passed and not verbose:
            continue
        marker = "PASS" if outcome.passed else "FAIL"
        _write(f"{marker} {outcome.scenario_id} [{outcome.category}]")
        if outcome.description:
            _write(f"     {outcome.description}")
        for failure in outcome.failures:
            _write(f"     - {failure}")
        if verbose:
            for turn in outcome.turns:
                _write(f"     {turn.index}. > {turn.message[:110]}")
                _write(f"        < {turn.reply[:110]}")
    if not report.green:
        _write()
        _write("gate failures:")
        for problem in report.gate_failures():
            _write(f"  - {problem}")


async def _evaluate_async(arguments: argparse.Namespace, settings: Settings) -> int:
    suite = ScenarioSuite.load(arguments.suite)
    thresholds = Thresholds(
        min_pass_rate=arguments.min_pass_rate,
        max_escalation_rate=arguments.max_escalation_rate,
    )

    runtime = Runtime(settings)
    await runtime.start()
    try:
        runner = ScenarioRunner(runtime, tenant_id=arguments.tenant, thresholds=thresholds)
        report = await runner.run_suite(suite)
    finally:
        await runtime.aclose()

    _render_report(report, verbose=arguments.verbose)

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        _write(f"\nwrote {arguments.report}")

    return _EXIT_OK if report.green else _EXIT_GATE_FAILED


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="support-agent",
        description="A bounded customer support agent: serve it, seed it, talk to it, grade it.",
    )
    parser.add_argument("--version", action="version", version=f"support-agent {__version__}")
    parser.add_argument(
        "--tenant",
        default=None,
        help=(
            "Tenant to operate on. Defaults to AGENT_TENANT, then to "
            "security.default_tenant — which is also the tenant an unauthenticated "
            "caller gets, so seeding and serving agree on a local instance."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", metavar="command")

    serve = subcommands.add_parser("serve", help="Run the HTTP service.")
    serve.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Loopback by default; the container overrides it.",
    )
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true", help="Reload on source changes.")
    serve.set_defaults(handler=_serve, is_async=False)

    seed = subcommands.add_parser(
        "seed", help="Insert the knowledge base and the demonstration account."
    )
    seed.add_argument("--email", default=DEMO_EMAIL)
    seed.set_defaults(handler=_seed_async, is_async=True)

    chat = subcommands.add_parser("chat", help="Talk to the agent from the terminal.")
    chat.add_argument("--email", default=DEMO_EMAIL)
    chat.add_argument(
        "--verified",
        action="store_true",
        help="Start with identity verified, so account questions are answerable.",
    )
    chat.add_argument("--trace", action="store_true", help="Print the step trace of each run.")
    chat.set_defaults(handler=_chat_async, is_async=True)

    evaluate = subcommands.add_parser(
        "evaluate", help="Run the scenario suite. Exits non-zero when a gate fails."
    )
    evaluate.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    evaluate.add_argument("--report", type=Path, default=None, help="Write a JSON report here.")
    evaluate.add_argument("--min-pass-rate", type=float, default=0.9)
    evaluate.add_argument("--max-escalation-rate", type=float, default=0.6)
    evaluate.add_argument(
        "-v", "--verbose", action="store_true", help="Show passing scenarios and replies."
    )
    evaluate.set_defaults(handler=_evaluate_async, is_async=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``support-agent`` command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return _EXIT_USAGE

    settings = get_settings()
    if arguments.tenant is None:
        arguments.tenant = os.environ.get("AGENT_TENANT") or settings.security.default_tenant
    configure_logging(
        level=settings.observability.log_level,
        fmt=settings.observability.log_format,
        service_name=settings.observability.service_name,
    )

    if arguments.is_async:
        return int(asyncio.run(arguments.handler(arguments, settings)))
    return int(arguments.handler(arguments, settings))


if __name__ == "__main__":
    sys.exit(main())
