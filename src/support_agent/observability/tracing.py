"""OpenTelemetry tracing and metrics setup.

Tracing is opt-in. When it is disabled the module still returns a working
tracer and metric handles, because the no-op implementations shipped with the
OpenTelemetry API cost almost nothing and keep call sites free of conditionals.

An agent run is hard to debug from logs alone: a slow answer might be slow in
classification, in one of several tool calls, at the model, or in verification.
Each is a span, so the breakdown is visible without adding timing code to every
function, and a trace shows the shape of the run as well as its duration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from opentelemetry import metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

if TYPE_CHECKING:
    from support_agent.config import Settings

INSTRUMENTATION_NAME: Final[str] = "support_agent"

_provider: TracerProvider | None = None


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider when tracing is enabled.

    An OTLP endpoint is optional: without one, spans are still created and
    sampled, which keeps trace ids flowing into log records for correlation
    even in environments with no collector.
    """
    global _provider  # noqa: PLW0603 - one provider per process is the OTel contract

    if not settings.observability.tracing_enabled:
        return
    if _provider is not None:
        return

    resource = Resource.create(
        {
            "service.name": settings.observability.service_name,
            "service.version": _version(),
            "deployment.environment": str(settings.environment),
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.observability.trace_sample_ratio)),
    )

    if settings.observability.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint))
        )

    trace.set_tracer_provider(provider)
    _provider = provider


def shutdown_tracing() -> None:
    """Flush and tear down the tracer provider during application shutdown."""
    global _provider  # noqa: PLW0603
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def get_tracer(name: str = INSTRUMENTATION_NAME) -> trace.Tracer:
    """Return a tracer. Yields a no-op tracer when tracing is disabled."""
    return trace.get_tracer(name)


def _version() -> str:
    from support_agent import __version__

    return __version__


_meter = metrics.get_meter(INSTRUMENTATION_NAME)

#: Conversations handled, labelled by terminal state and intent.
runs_completed = _meter.create_counter(
    "agent.runs.completed",
    unit="1",
    description="Agent runs, labelled by terminal state and intent.",
)

#: Tool invocations, labelled by tool and outcome.
tool_calls = _meter.create_counter(
    "agent.tool.calls",
    unit="1",
    description="Tool invocations, labelled by tool and outcome.",
)

#: Escalations to a human, labelled by reason. The number that says whether the
#: agent is improving is not the raw rate but its breakdown by cause.
escalations = _meter.create_counter(
    "agent.escalations",
    unit="1",
    description="Handovers to a human, labelled by reason.",
)

#: Answers that failed verification. A rising value means the composer is
#: asserting things the tools did not return.
verification_failures = _meter.create_counter(
    "agent.verification.failures",
    unit="1",
    description="Draft answers rejected by the verifier, labelled by cause.",
)

#: Prompt-injection findings in customer messages and tool output.
injection_findings = _meter.create_counter(
    "agent.security.injection_findings",
    unit="1",
    description="Injection findings in untrusted content, labelled by rule and action.",
)

#: End-to-end run latency.
run_latency = _meter.create_histogram(
    "agent.run.duration",
    unit="ms",
    description="End-to-end agent run latency in milliseconds.",
)

#: Tool latency, the usual first suspect when a run is slow.
tool_latency = _meter.create_histogram(
    "agent.tool.duration",
    unit="ms",
    description="Tool execution latency in milliseconds, labelled by tool.",
)

#: Steps consumed per run, against the configured budget.
steps_used = _meter.create_histogram(
    "agent.run.steps",
    unit="1",
    description="State transitions consumed per run.",
)

__all__ = [
    "INSTRUMENTATION_NAME",
    "configure_tracing",
    "escalations",
    "get_tracer",
    "injection_findings",
    "run_latency",
    "runs_completed",
    "shutdown_tracing",
    "steps_used",
    "tool_calls",
    "tool_latency",
    "verification_failures",
]
