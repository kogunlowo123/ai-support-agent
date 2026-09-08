"""Health, tool discovery and the operational endpoints.

An agent nobody can inspect after the fact is not operable, so what these
endpoints expose is part of the product: which tools exist and what it takes to
call them, what a run actually did, and who did what.
"""

from __future__ import annotations

import pytest

from tests.conftest import DEMO_EMAIL

pytestmark = pytest.mark.integration


async def conversation_with_a_run(client) -> str:
    response = await client.post("/v1/conversations", json={"customer_email": DEMO_EMAIL})
    conversation = str(response.json()["conversation_id"])
    await client.post(f"/v1/conversations/{conversation}/verify", json={"email": DEMO_EMAIL})
    await client.post(
        f"/v1/conversations/{conversation}/messages",
        json={"message": "Where is my order ORD-1005?"},
    )
    return conversation


class TestHealth:
    async def test_liveness_needs_no_credential(self, anonymous_client):
        """A liveness probe that can fail authentication is not a liveness probe."""
        response = await anonymous_client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_liveness_reports_the_version(self, anonymous_client):
        assert (await anonymous_client.get("/healthz")).json()["version"]

    async def test_readiness_reports_each_component(self, anonymous_client):
        body = (await anonymous_client.get("/readyz")).json()
        assert body["components"]
        assert {"name", "ready", "detail"} <= set(body["components"][0])

    async def test_readiness_reports_open_circuits(self, anonymous_client):
        """A degraded tool is not a reason to fail readiness, but an operator must see it."""
        body = (await anonymous_client.get("/readyz")).json()
        assert body["open_circuits"] == []

    async def test_readiness_is_ready_on_a_healthy_process(self, anonymous_client):
        assert (await anonymous_client.get("/readyz")).json()["status"] == "ready"


class TestToolDiscovery:
    async def test_every_tool_is_published_with_its_constraints(self, client):
        tools = (await client.get("/v1/tools")).json()["tools"]
        assert len(tools) == 6
        for tool in tools:
            assert tool["description"]
            assert isinstance(tool["required_scopes"], list)
            assert isinstance(tool["allowed_intents"], list)
            assert tool["parameters"]

    async def test_the_published_constraints_match_what_is_enforced(self, client):
        tools = {tool["name"]: tool for tool in (await client.get("/v1/tools")).json()["tools"]}
        assert tools["lookup_order"]["requires_identity"] is True
        assert tools["search_knowledge_base"]["requires_identity"] is False

    async def test_writes_are_published_as_writes(self, client):
        tools = {tool["name"]: tool for tool in (await client.get("/v1/tools")).json()["tools"]}
        assert tools["create_ticket"]["risk"] != "read"

    async def test_tool_discovery_requires_a_credential(self, anonymous_client):
        assert (await anonymous_client.get("/v1/tools")).status_code == 403


class TestRuns:
    async def test_a_run_is_recorded(self, client):
        await conversation_with_a_run(client)
        runs = (await client.get("/v1/runs")).json()["runs"]
        assert runs
        assert runs[0]["intent"]
        assert runs[0]["final_state"]

    async def test_the_run_trace_is_retrievable(self, client):
        await conversation_with_a_run(client)
        run_id = (await client.get("/v1/runs")).json()["runs"][0]["run_id"]
        trace = (await client.get(f"/v1/runs/{run_id}/trace")).json()
        assert trace["steps"]
        assert {"index", "kind", "state", "summary"} <= set(trace["steps"][0])

    async def test_an_unknown_run_is_a_404(self, client):
        assert (await client.get("/v1/runs/run_missing/trace")).status_code == 404

    async def test_runs_record_what_the_agent_did_not_what_was_said(self, client):
        """A run summary is operational data, not a transcript."""
        await conversation_with_a_run(client)
        runs = (await client.get("/v1/runs")).json()["runs"]
        assert "ORD-1005" not in str(runs)


class TestTickets:
    async def test_an_escalation_produces_a_visible_ticket(self, client):
        response = await client.post("/v1/conversations", json={"customer_email": DEMO_EMAIL})
        conversation = response.json()["conversation_id"]
        await client.post(f"/v1/conversations/{conversation}/verify", json={"email": DEMO_EMAIL})
        await client.post(
            f"/v1/conversations/{conversation}/messages",
            json={"message": "I want to speak to a human being"},
        )
        tickets = (await client.get("/v1/tickets")).json()["tickets"]
        assert tickets
        assert tickets[0]["status"]

    async def test_tickets_require_a_credential(self, anonymous_client):
        assert (await anonymous_client.get("/v1/tickets")).status_code == 403


class TestAudit:
    async def test_every_run_is_audited(self, client):
        await conversation_with_a_run(client)
        events = (await client.get("/v1/audit")).json()["events"]
        assert {event["event"] for event in events} >= {"conversation.start", "agent.run"}

    async def test_an_audit_event_names_the_key_that_acted(self, client):
        """The question "which credential did this?" must be answerable."""
        await conversation_with_a_run(client)
        events = (await client.get("/v1/audit")).json()["events"]
        assert all(event["actor_key_id"] for event in events)

    async def test_identity_verification_is_audited(self, client):
        await conversation_with_a_run(client)
        events = (await client.get("/v1/audit")).json()["events"]
        assert any(event["event"] == "identity.verify" for event in events)

    async def test_the_audit_log_does_not_contain_message_text(self, client):
        await conversation_with_a_run(client)
        body = (await client.get("/v1/audit")).text
        assert "Where is my order" not in body


class TestDocumentation:
    async def test_the_openapi_document_is_served(self, anonymous_client):
        document = (await anonymous_client.get("/openapi.json")).json()
        assert document["info"]["title"]
        assert "/v1/conversations" in document["paths"]

    async def test_every_endpoint_is_summarised(self, anonymous_client):
        document = (await anonymous_client.get("/openapi.json")).json()
        for path, operations in document["paths"].items():
            for method, operation in operations.items():
                assert operation.get("summary"), f"{method} {path} has no summary"


class TestSecurityHeaders:
    async def test_responses_carry_the_expected_headers(self, anonymous_client):
        headers = (await anonymous_client.get("/healthz")).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert "x-frame-options" in headers

    async def test_the_server_header_does_not_advertise_the_stack(self, anonymous_client):
        headers = (await anonymous_client.get("/healthz")).headers
        assert "uvicorn" not in headers.get("server", "").lower()

    async def test_an_oversized_body_is_rejected_before_it_is_read(self, client):
        response = await client.post(
            "/v1/conversations",
            content=b"x" * (300 * 1024),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["code"] == "request_too_large"
        # The envelope has to match every other error, including the request id
        # — an operator correlating a rejection needs it most for the ones the
        # application never saw.
        assert set(body) == {"code", "message", "request_id", "detail"}
        assert body["request_id"]
