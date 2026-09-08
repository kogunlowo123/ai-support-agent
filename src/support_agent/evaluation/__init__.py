"""Scenario-based evaluation of the agent.

The suite runs the real machine against a seeded database and asserts on
behaviour: the state a run reached, the tools it called, whether it escalated,
and what it must never say. It is the gate CI uses to decide whether a change
made the agent worse.
"""

from support_agent.evaluation.scenarios import (
    Scenario,
    ScenarioCategory,
    ScenarioSuite,
    Thresholds,
    Turn,
)

__all__ = ["Scenario", "ScenarioCategory", "ScenarioSuite", "Thresholds", "Turn"]
