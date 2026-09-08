"""Application factory.

Builds the FastAPI application, installs middleware in a fixed order, wires the
exception handlers that turn domain errors into the single documented error
shape, and manages the lifecycle of process-lifetime resources.

Startup deliberately fails loudly. Configuration invariants are checked before
the first request, so a deployment with verification disabled in production, or
with identity checks turned off, refuses to serve rather than serving in that
state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from support_agent import __version__
from support_agent.agent.composer import ModelComposer, TemplateComposer
from support_agent.api.dependencies import Services
from support_agent.api.middleware import (
    AccessLogMiddleware,
    BodyLimitMiddleware,
    CorrelationMiddleware,
    SecurityHeadersMiddleware,
)
from support_agent.api.routes import conversations, health, operations, tools
from support_agent.config import Settings, get_settings
from support_agent.errors import AgentError
from support_agent.observability.logging import configure_logging, get_logger
from support_agent.observability.tracing import configure_tracing, shutdown_tracing
from support_agent.providers.registry import build_chat_provider
from support_agent.security.authz import ApiKeyAuthenticator
from support_agent.storage.db import create_engine, create_schema, create_session_factory
from support_agent.tools.breaker import CircuitBreakerRegistry

logger = get_logger(__name__)

DESCRIPTION: Final[str] = """\
A customer support agent that is bounded by construction.

It is a state machine with a fixed transition table, a step budget and a
wall-clock budget — not a loop that decides what to do next. Tools are the only
way it observes or changes anything, and each declares the scopes, identity and
intents required to call it. Eligibility, entitlement and limits are computed by
code, never by a model. Every reply is checked against the evidence the run
actually gathered, and one that asserts a fact no tool returned is escalated
rather than sent.

A model is used for one thing: phrasing material that has already been decided.
"""


def _build_services(settings: Settings) -> Services:
    """Construct process-lifetime collaborators from configuration."""
    engine = create_engine(settings)
    chat = build_chat_provider(settings)
    return Services(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        chat=chat,
        composer=ModelComposer(chat) if chat is not None else TemplateComposer(),
        authenticator=ApiKeyAuthenticator(
            keys=settings.security.api_keys,
            require_key=settings.security.require_api_key,
            default_tenant=settings.security.default_tenant,
        ),
        breakers=CircuitBreakerRegistry(
            failure_threshold=settings.circuit_breaker.failure_threshold,
            reset_seconds=settings.circuit_breaker.reset_seconds,
            half_open_successes=settings.circuit_breaker.half_open_successes,
        ),
    )


def _error_response(
    request: Request, *, status_code: int, code: str, message: str, detail: dict[str, object]
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
            "detail": detail,
        },
    )


def _install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentError)
    async def _domain_error(request: Request, exc: AgentError) -> JSONResponse:
        # Client errors are expected traffic and log at info; server errors are
        # not, and carry a stack trace.
        log = (
            logger.info if exc.status_code < HTTPStatus.INTERNAL_SERVER_ERROR else logger.exception
        )
        log("api.domain_error", code=exc.code, status=exc.status_code)
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's raw error list echoes the submitted value, which may be a
        # customer message, so only the location and the rule are returned.
        fields = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "rule": error.get("type", ""),
            }
            for error in exc.errors()[:10]
        ]
        return _error_response(
            request,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="validation_error",
            message="the request did not match the expected schema",
            detail={"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            request,
            status_code=exc.status_code,
            code="http_error",
            message=str(exc.detail),
            detail={},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Logged in full; the client is told nothing about it. An unhandled
        # error's message is exactly the kind of internal detail that must not
        # cross the boundary.
        logger.exception("api.unhandled_error", error_type=type(exc).__name__)
        return _error_response(
            request,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="an internal error occurred",
            detail={},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings so tests can vary them."""
    active = settings or get_settings()
    configure_logging(
        level=active.observability.log_level,
        fmt=active.observability.log_format,
        service_name=active.observability.service_name,
    )
    active.enforce_environment_invariants()
    configure_tracing(active)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        services = _build_services(active)
        application.state.services = services
        await create_schema(services.engine)
        logger.info(
            "app.started",
            environment=str(active.environment),
            composer=str(active.chat.backend),
            verification=active.verification.enabled,
            max_steps=active.limits.max_steps,
            version=__version__,
        )
        try:
            yield
        finally:
            await services.aclose()
            shutdown_tracing()
            logger.info("app.stopped")

    app = FastAPI(
        title="AI Support Agent",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Outermost first. Correlation wraps everything so a rejection by the body
    # limit is still logged with a request id.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodyLimitMiddleware, max_bytes=active.security.max_request_bytes)
    app.add_middleware(CorrelationMiddleware)

    _install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(tools.router)
    app.include_router(operations.router)

    if active.observability.tracing_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")

    return app


__all__ = ["create_app"]
