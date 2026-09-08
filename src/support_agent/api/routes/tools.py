"""Tool introspection.

Publishing the tool surface is a deliberate choice. A tool registry that is only
visible in code is one nobody audits, and the constraints — which scopes, which
intents, whether identity is required — are exactly what a reviewer needs to
see. Nothing here is a secret: knowing that ``lookup_order`` requires a verified
identity does not help anyone bypass the requirement.
"""

from __future__ import annotations

from fastapi import APIRouter

from support_agent.api.dependencies import ContextDep, PrincipalDep, ServicesDep
from support_agent.api.schemas import ToolDescription, ToolListResponse
from support_agent.tools.breaker import BreakerState

router = APIRouter(prefix="/v1/tools", tags=["tools"])


@router.get("", response_model=ToolListResponse, summary="List the registered tools")
async def list_tools(
    context: ContextDep, services: ServicesDep, principal: PrincipalDep
) -> ToolListResponse:
    """Return every tool with its schema and the constraints on calling it."""
    described = [spec.describe() for spec in context.registry.specs()]
    return ToolListResponse(
        tools=[
            ToolDescription(
                name=str(item["name"]),
                description=str(item["description"]),
                risk=str(item["risk"]),
                requires_identity=bool(item["requires_identity"]),
                required_scopes=list(item["required_scopes"]),
                allowed_intents=list(item["allowed_intents"]),
                idempotent=bool(item["idempotent"]),
                parameters=dict(item["parameters"]),
            )
            for item in described
        ],
        open_circuits=[
            tool
            for tool, state in services.breakers.snapshot().items()
            if state == str(BreakerState.OPEN)
        ],
    )


__all__ = ["router"]
