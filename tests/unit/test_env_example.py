"""``.env.example`` is documentation that has to stay true.

A configuration example naming a setting that does not exist is worse than none:
an operator sets it, nothing happens, and the deployment runs on a default while
they believe otherwise. Now that unknown fields are rejected, a stale key here
would also stop the process from starting at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from support_agent.config import SecuritySettings, Settings

pytestmark = pytest.mark.unit

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
_ASSIGNMENT = re.compile(r"^#?\s*(AGENT_[A-Z0-9_]+)=(.*)$", re.MULTILINE)


def known_keys(model: type[BaseModel] = Settings, prefix: str = "AGENT_") -> set[str]:
    """Every environment variable the settings model actually reads."""
    keys: set[str] = set()
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            keys |= known_keys(annotation, f"{prefix}{name.upper()}__")
        else:
            keys.add(f"{prefix}{name.upper()}")
    return keys


@pytest.fixture(scope="module")
def documented() -> list[tuple[str, str]]:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return [(match.group(1), match.group(2)) for match in _ASSIGNMENT.finditer(text)]


class TestEnvExample:
    def test_it_is_checked_in(self):
        assert ENV_EXAMPLE.is_file()

    def test_every_documented_key_exists(self, documented):
        unknown = sorted({key for key, _ in documented} - known_keys())
        assert not unknown, f"documented but not read by Settings: {unknown}"

    def test_the_safety_critical_settings_are_documented(self, documented):
        """An operator must not have to read the source to find these."""
        keys = {key for key, _ in documented}
        assert {
            "AGENT_ENVIRONMENT",
            "AGENT_SECURITY__REQUIRE_API_KEY",
            "AGENT_VERIFICATION__ENABLED",
            "AGENT_POLICY__ALLOW_AGENT_INITIATED_REFUNDS",
            "AGENT_POLICY__REQUIRE_IDENTITY_FOR_ACCOUNT_DATA",
            "AGENT_OBSERVABILITY__LOG_MESSAGE_CONTENT",
            "AGENT_LIMITS__MAX_STEPS",
        } <= keys

    def test_the_dangerous_defaults_are_the_safe_ones(self, documented):
        values = dict(documented)
        assert values["AGENT_POLICY__ALLOW_AGENT_INITIATED_REFUNDS"] == "false"
        assert values["AGENT_VERIFICATION__ENABLED"] == "true"
        assert values["AGENT_OBSERVABILITY__LOG_MESSAGE_CONTENT"] == "false"
        # The example turns authentication off so a clean clone starts with no
        # editing, and says so. The *code* default is on, and production
        # refuses to start without it — that is the property that matters.
        assert values["AGENT_SECURITY__REQUIRE_API_KEY"] == "false"
        assert "LOCAL value" in ENV_EXAMPLE.read_text(encoding="utf-8")
        assert SecuritySettings().require_api_key is True

    def test_no_real_credential_is_committed(self, documented):
        """The file ships placeholders. A working key here would be a leak."""
        booleans = {"true", "false"}
        for key, value in documented:
            names_a_secret = key.endswith(("API_KEYS", "API_KEY", "SECRET", "PASSWORD"))
            if names_a_secret and value and value.lower() not in booleans:
                assert "replace" in value.lower(), f"{key} looks like a real credential"

    def test_the_active_lines_parse_as_settings(self, documented, monkeypatch):
        """Applying the uncommented values must produce a valid configuration."""
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("AGENT_") and "=" in line:
                key, value = line.split("=", 1)
                monkeypatch.setenv(key, value)
        Settings().enforce_environment_invariants()
