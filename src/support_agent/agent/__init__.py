"""The agent: classification, planning, composition, verification and the machine."""

from support_agent.agent.composer import (
    Composer,
    CompositionRequest,
    ModelComposer,
    TemplateComposer,
    build_prompt,
)
from support_agent.agent.intent import classify, extract_order_reference
from support_agent.agent.machine import AgentMachine, RunResult
from support_agent.agent.planner import PlanContext, plan
from support_agent.agent.verifier import Evidence, VerificationReport, verify

__all__ = [
    "AgentMachine",
    "Composer",
    "CompositionRequest",
    "Evidence",
    "ModelComposer",
    "PlanContext",
    "RunResult",
    "TemplateComposer",
    "VerificationReport",
    "build_prompt",
    "classify",
    "extract_order_reference",
    "plan",
    "verify",
]
