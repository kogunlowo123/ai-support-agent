"""Tools: the only way the agent affects or observes anything outside itself."""

from support_agent.tools.base import Tool, ToolArguments, ToolContext, ToolReturns, ToolSpec
from support_agent.tools.breaker import BreakerState, CircuitBreakerRegistry
from support_agent.tools.builtin import build_default_tools
from support_agent.tools.registry import IdempotencyStore, RunBudget, ToolRegistry, deadline_from

__all__ = [
    "BreakerState",
    "CircuitBreakerRegistry",
    "IdempotencyStore",
    "RunBudget",
    "Tool",
    "ToolArguments",
    "ToolContext",
    "ToolRegistry",
    "ToolReturns",
    "ToolSpec",
    "build_default_tools",
    "deadline_from",
]
