"""Circuit breakers.

A breaker is only useful if it accumulates evidence across requests, so these
tests drive one instance through a whole failure-and-recovery cycle rather than
asserting on a single call.
"""

from __future__ import annotations

import pytest

from support_agent.tools.breaker import BreakerState, CircuitBreakerRegistry

pytestmark = pytest.mark.unit

TOOL = "lookup_order"


@pytest.fixture
def registry() -> CircuitBreakerRegistry:
    return CircuitBreakerRegistry(failure_threshold=3, reset_seconds=30.0, half_open_successes=2)


class TestClosedState:
    def test_an_unused_tool_is_allowed(self, registry):
        assert registry.allows(TOOL) is True
        assert registry.state(TOOL) is BreakerState.CLOSED

    def test_failures_below_the_threshold_do_not_open_it(self, registry):
        registry.record_failure(TOOL)
        registry.record_failure(TOOL)
        assert registry.allows(TOOL) is True

    def test_a_success_clears_accumulated_failures(self, registry):
        """Intermittent failures are not the same as a broken dependency."""
        registry.record_failure(TOOL)
        registry.record_failure(TOOL)
        registry.record_success(TOOL)
        registry.record_failure(TOOL)
        registry.record_failure(TOOL)
        assert registry.allows(TOOL) is True


class TestOpenState:
    def test_reaching_the_threshold_opens_it(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        assert registry.state(TOOL) is BreakerState.OPEN
        assert registry.allows(TOOL) is False

    def test_one_tool_opening_does_not_affect_another(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        assert registry.allows("search_knowledge_base") is True

    def test_it_half_opens_once_the_reset_window_passes(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        registry._get(TOOL).opened_at -= 31.0
        assert registry.state(TOOL) is BreakerState.HALF_OPEN
        assert registry.allows(TOOL) is True


class TestHalfOpenState:
    def test_enough_successes_close_it_again(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        registry._get(TOOL).opened_at -= 31.0
        registry.record_success(TOOL)
        registry.record_success(TOOL)
        assert registry.state(TOOL) is BreakerState.CLOSED

    def test_a_failure_while_half_open_reopens_it_immediately(self, registry):
        """One probe is enough. A still-broken dependency should not be hammered."""
        for _ in range(3):
            registry.record_failure(TOOL)
        registry._get(TOOL).opened_at -= 31.0
        assert registry.state(TOOL) is BreakerState.HALF_OPEN
        registry.record_failure(TOOL)
        assert registry.state(TOOL) is BreakerState.OPEN


class TestObservability:
    def test_the_snapshot_reports_every_known_tool(self, registry):
        registry.record_failure(TOOL)
        registry.record_success("search_knowledge_base")
        snapshot = registry.snapshot()
        assert set(snapshot) == {TOOL, "search_knowledge_base"}

    def test_the_snapshot_names_open_circuits(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        assert registry.snapshot()[TOOL] == str(BreakerState.OPEN)

    def test_reset_clears_everything(self, registry):
        for _ in range(3):
            registry.record_failure(TOOL)
        registry.reset()
        assert registry.allows(TOOL) is True
        assert registry.snapshot() == {}
