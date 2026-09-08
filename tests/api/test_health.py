"""Liveness and readiness.

Readiness is where an operator finds out that the process is up but degraded —
the model is unreachable, a circuit is open, verification is off. Every one of
those states is asserted here, because a warning nobody ever produces is a
warning nobody will see when it matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from asgi_lifespan import LifespanManager

from support_agent.api.app import create_app
from support_agent.config import (
    ChatBackend,
    ChatSettings,
    Environment,
    ProviderEndpoints,
    SecuritySettings,
    Settings,
    StorageSettings,
    VerificationSettings,
)
from support_agent.tools.breaker import BreakerState

pytestmark = pytest.mark.integration


def settings_for(tmp_path: Path, name: str, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": Environment.TEST,
        "storage": StorageSettings(
            database_url=f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}"
        ),
        "chat": ChatSettings(backend=ChatBackend.TEMPLATE),
        "security": SecuritySettings(require_api_key=False),
    }
    base.update(overrides)
    return Settings(**base)


async def probe(settings: Settings, path: str = "/readyz") -> tuple[int, dict[str, Any]]:
    import httpx

    app = create_app(settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(path)
            return response.status_code, response.json()


class TestLiveness:
    async def test_it_reports_ok_without_touching_anything(self, tmp_path):
        """A liveness probe that queries the database restarts a healthy pod."""
        status, body = await probe(settings_for(tmp_path, "live.db"), "/healthz")
        assert status == 200
        assert body == {"status": "ok", "version": body["version"]}


class TestReadiness:
    async def test_a_healthy_process_is_ready(self, tmp_path):
        status, body = await probe(settings_for(tmp_path, "ready.db"))
        assert status == 200
        assert body["status"] == "ready"

    async def test_the_database_is_checked(self, tmp_path):
        _, body = await probe(settings_for(tmp_path, "db.db"))
        database = next(c for c in body["components"] if c["name"] == "database")
        assert database["ready"] is True

    async def test_the_registered_tools_are_reported(self, tmp_path):
        _, body = await probe(settings_for(tmp_path, "tools.db"))
        tools = next(c for c in body["components"] if c["name"] == "tools")
        assert "registered=6" in tools["detail"]

    async def test_the_template_composer_is_reported_as_a_limitation(self, tmp_path):
        """Running without a model is a supported state, but not a silent one."""
        _, body = await probe(settings_for(tmp_path, "template.db"))
        composer = next(c for c in body["components"] if c["name"] == "composer")
        assert composer["detail"] == "backend=template"
        assert any("cannot rephrase" in warning for warning in body["warnings"])

    async def test_an_unreachable_model_warns_without_failing_readiness(self, tmp_path):
        """The agent degrades to templates; it does not stop serving."""
        settings = settings_for(
            tmp_path,
            "ollama.db",
            chat=ChatSettings(backend=ChatBackend.OLLAMA, timeout_seconds=0.05),
            providers=ProviderEndpoints(ollama_base_url="http://127.0.0.1:9"),
        )
        status, body = await probe(settings)
        assert status == 200
        assert body["status"] == "ready"
        assert any("not reachable" in warning for warning in body["warnings"])

    async def test_disabled_verification_is_warned_about(self, tmp_path):
        settings = settings_for(
            tmp_path, "noverify.db", verification=VerificationSettings(enabled=False)
        )
        _, body = await probe(settings)
        assert any("verification is disabled" in warning for warning in body["warnings"])

    async def test_open_circuits_are_listed(self, tmp_path):
        """A degraded tool is visible without reading logs."""
        import httpx

        app = create_app(settings_for(tmp_path, "circuits.db"))
        async with LifespanManager(app):
            breakers = app.state.services.breakers
            for _ in range(breakers.failure_threshold):
                breakers.record_failure("lookup_order")
            assert breakers.state("lookup_order") is BreakerState.OPEN

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                body = (await client.get("/readyz")).json()

        assert body["open_circuits"] == ["lookup_order"]
        assert any("circuit breaker" in warning for warning in body["warnings"])
        assert body["status"] == "ready"

    async def test_readiness_needs_no_credential(self, tmp_path):
        settings = settings_for(
            tmp_path,
            "authed.db",
            security=SecuritySettings(require_api_key=True, api_keys=["acme:secret"]),
        )
        status, _ = await probe(settings)
        assert status == 200


class TestStartupInvariants:
    async def test_an_unsafe_production_configuration_refuses_to_start(self, tmp_path):
        """Startup fails loudly rather than serving in an unsafe state."""
        from support_agent.errors import ConfigurationError

        settings = settings_for(
            tmp_path,
            "prod.db",
            environment=Environment.PRODUCTION,
            security=SecuritySettings(require_api_key=False),
        )
        with pytest.raises(ConfigurationError):
            create_app(settings)
