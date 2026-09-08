"""Per-tool circuit breaker.

A dependency that is failing should stop being called, not be retried into the
ground. Retrying a broken dependency turns one team's incident into everyone's:
each retry consumes a connection the dependency needs to recover.

Three states, which is the whole design:

```
        failures >= threshold
CLOSED ──────────────────────► OPEN
  ▲                              │ reset_seconds elapsed
  │ half_open_successes          ▼
  └────────────── HALF_OPEN ◄────┘
                     │ any failure
                     └──────────► OPEN
```

``HALF_OPEN`` admits calls again so recovery is detected without a deploy, and
a single failure there returns to ``OPEN`` rather than waiting for the full
threshold — a dependency that fails its first probe is not ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class BreakerState(StrEnum):
    """State of one tool's breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Breaker:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    half_open_successes: int = 0
    opened_at: float = 0.0


@dataclass
class CircuitBreakerRegistry:
    """Circuit breakers, one per tool, keyed by tool name.

    Time is injected as a callable so tests can advance the clock rather than
    sleep. A test that has to sleep for the reset window is a test nobody runs.
    """

    failure_threshold: int = 4
    reset_seconds: float = 30.0
    half_open_successes: int = 2
    clock: object = field(default=time.monotonic)
    _breakers: dict[str, _Breaker] = field(default_factory=dict)

    def _now(self) -> float:
        return float(self.clock())  # type: ignore[operator]

    def _get(self, tool: str) -> _Breaker:
        return self._breakers.setdefault(tool, _Breaker())

    def state(self, tool: str) -> BreakerState:
        """Report the current state, having first applied any elapsed reset window.

        A tool that has never been called is closed and is *not* recorded. A
        query that creates the thing it is asked about would make the readiness
        snapshot grow every time anything looked at it.
        """
        breaker = self._breakers.get(tool)
        if breaker is None:
            return BreakerState.CLOSED
        if (
            breaker.state is BreakerState.OPEN
            and self._now() - breaker.opened_at >= self.reset_seconds
        ):
            breaker.state = BreakerState.HALF_OPEN
            breaker.half_open_successes = 0
        return breaker.state

    def allows(self, tool: str) -> bool:
        """Whether a call to ``tool`` may proceed."""
        return self.state(tool) is not BreakerState.OPEN

    def record_success(self, tool: str) -> None:
        """Register a successful call."""
        breaker = self._get(tool)
        breaker.consecutive_failures = 0
        if breaker.state is BreakerState.HALF_OPEN:
            breaker.half_open_successes += 1
            if breaker.half_open_successes >= self.half_open_successes:
                breaker.state = BreakerState.CLOSED
                breaker.half_open_successes = 0
        else:
            breaker.state = BreakerState.CLOSED

    def record_failure(self, tool: str) -> None:
        """Register a failed call, opening the breaker if the threshold is met."""
        breaker = self._get(tool)
        if breaker.state is BreakerState.HALF_OPEN:
            # A dependency that fails its first probe is not ready. Reopen
            # immediately rather than spending the whole threshold again.
            breaker.state = BreakerState.OPEN
            breaker.opened_at = self._now()
            breaker.consecutive_failures = self.failure_threshold
            return

        breaker.consecutive_failures += 1
        if breaker.consecutive_failures >= self.failure_threshold:
            breaker.state = BreakerState.OPEN
            breaker.opened_at = self._now()

    def snapshot(self) -> dict[str, str]:
        """Report the state of every known breaker, for the readiness endpoint."""
        return {tool: str(self.state(tool)) for tool in self._breakers}

    def reset(self) -> None:
        """Forget every breaker. Used between test cases."""
        self._breakers.clear()


__all__ = ["BreakerState", "CircuitBreakerRegistry"]
