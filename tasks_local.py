"""Tasks specific to this repository.

``tasks.py`` holds what every project in the portfolio shares; this file holds
what only the agent has. Keeping them apart means the shared runner can be
updated without merging around per-project commands.
"""

from __future__ import annotations

import os

UV = "uv"
IMAGE = os.environ.get("IMAGE_NAME", "ai-support-agent")


def _run(*args: str) -> list[str]:
    return [UV, "run", *args]


TASKS = {
    "seed": (
        "Load the knowledge base and the demonstration account.",
        [_run("support-agent", "seed")],
    ),
    "serve": (
        "Run the HTTP service on http://127.0.0.1:8000.",
        [_run("support-agent", "serve")],
    ),
    "chat": (
        "Talk to the agent from the terminal, with identity already verified.",
        [_run("support-agent", "chat", "--verified", "--trace")],
    ),
    "evaluate": (
        "Run the scenario suite at the gate CI uses. Exits non-zero on a regression.",
        [
            _run(
                "support-agent",
                "evaluate",
                "--suite",
                "data/scenarios/support.jsonl",
                "--report",
                "reports/scenarios.json",
                "--min-pass-rate",
                "1.0",
            )
        ],
    ),
    "test-e2e": ("Run the end-to-end journeys only.", [_run("pytest", "-m", "e2e")]),
    "test-regression": (
        "Run the scenario suite as a test.",
        [_run("pytest", "-m", "regression")],
    ),
    "examples": (
        "Run every example, the way CI does.",
        [
            _run("python", "examples/quickstart.py"),
            _run("python", "examples/identity_boundary.py"),
            _run("python", "examples/injection_demo.py"),
        ],
    ),
    "smoke": (
        "Build the container and run the smoke test against it.",
        [
            ["docker", "build", "-t", f"{IMAGE}:local", "."],
            ["bash", "scripts/smoke-test.sh", f"{IMAGE}:local"],
        ],
    ),
    "up": (
        "Start the local stack with docker compose.",
        [["docker", "compose", "up", "--build", "-d"]],
    ),
    "down": (
        "Stop the local stack and remove its volumes.",
        [["docker", "compose", "down", "-v"]],
    ),
}
