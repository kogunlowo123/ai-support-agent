"""Liveness and readiness endpoints.

``/healthz`` answers "is the process alive?" and never touches a dependency. A
health check that fails because the database is slow causes an orchestrator to
restart a healthy process during an incident, turning a degradation into an
outage.

``/readyz`` answers "should this instance receive traffic?" and does check
dependencies. It also reports which tool circuit breakers are open: an agent
whose order lookup is failing still serves policy questions and escalates
everything else, so an open breaker is a warning rather than a readiness
failure — but it must be visible without reading logs.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from sqlalchemy import text

from support_agent import __version__
from support_agent.api.dependencies import ContextDep, ServicesDep
from support_agent.api.schemas import ComponentStatus, HealthResponse, ReadinessResponse
from support_agent.config import ChatBackend
from support_agent.observability.logging import get_logger
from support_agent.tools.breaker import BreakerState

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz() -> HealthResponse:
    """Report that the process is running. Checks nothing external."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
async def readyz(
    services: ServicesDep, context: ContextDep, response: Response
) -> ReadinessResponse:
    """Report whether this instance can serve traffic."""
    components: list[ComponentStatus] = []
    warnings: list[str] = []

    try:
        await context.session.execute(text("SELECT 1"))
        components.append(ComponentStatus(name="database", ready=True))
    except Exception as exc:
        logger.warning("readiness.database_unavailable", error=type(exc).__name__)
        components.append(
            ComponentStatus(name="database", ready=False, detail="the database is unreachable")
        )

    backend = services.settings.chat.backend
    if backend is ChatBackend.TEMPLATE:
        components.append(ComponentStatus(name="composer", ready=True, detail="backend=template"))
        warnings.append(
            "The 'template' composer renders tool results directly and cannot rephrase. "
            "Set AGENT_CHAT__BACKEND=ollama or openai for natural replies."
        )
    else:
        healthy = services.chat is not None and await services.chat.health()
        components.append(
            ComponentStatus(
                name="composer", ready=True, detail=f"backend={backend} reachable={healthy}"
            )
        )
        if not healthy:
            warnings.append(
                "the configured language model is not reachable; replies will be composed "
                "from tool results directly until it recovers"
            )

    components.append(
        ComponentStatus(
            name="tools",
            ready=True,
            detail=f"registered={len(context.registry.names)}",
        )
    )

    open_circuits = [
        tool
        for tool, state in services.breakers.snapshot().items()
        if state == str(BreakerState.OPEN)
    ]
    if open_circuits:
        warnings.append(
            f"{len(open_circuits)} tool circuit breaker(s) are open; those requests will "
            "escalate to a human"
        )

    if not services.settings.verification.enabled:
        warnings.append(
            "answer verification is disabled; the agent may state facts no tool returned"
        )

    ready = all(component.ready for component in components)
    if not ready:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        version=__version__,
        components=components,
        open_circuits=open_circuits,
        warnings=warnings,
    )


__all__ = ["router"]
