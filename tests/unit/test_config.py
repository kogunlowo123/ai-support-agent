"""Configuration, and the invariants that refuse to start an unsafe process.

The point of these tests is that a dangerous configuration cannot be deployed by
accident. Every production invariant has a case asserting that the process
refuses, because an invariant that is never exercised is a comment.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from support_agent.config import (
    Environment,
    LimitSettings,
    ObservabilitySettings,
    PolicySettings,
    SecuritySettings,
    Settings,
    StorageSettings,
    VerificationSettings,
    get_settings,
    reset_settings_cache,
)
from support_agent.errors import ConfigurationError

pytestmark = pytest.mark.unit


def production(**overrides: object) -> Settings:
    """A production configuration that is valid unless a test breaks it."""
    base: dict[str, object] = {
        "environment": Environment.PRODUCTION,
        "storage": StorageSettings(database_url="postgresql+psycopg://db/agent"),
        "security": SecuritySettings(require_api_key=True, api_keys=[SecretStr("acme:k")]),
        "verification": VerificationSettings(enabled=True),
        "policy": PolicySettings(require_identity_for_account_data=True),
        "observability": ObservabilitySettings(log_message_content=False),
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def problems(settings: Settings) -> str:
    """Run the invariants and return the reported problems as one string.

    The exception message is deliberately generic; the specific complaints live
    in its detail, which is what an operator reads and therefore what a test
    should assert on.
    """
    with pytest.raises(ConfigurationError) as raised:
        settings.enforce_environment_invariants()
    reported: object = raised.value.detail.get("problems", [])
    assert isinstance(reported, list)
    assert reported, "the configuration was refused without saying why"
    return " | ".join(str(item) for item in reported)


class TestDefaults:
    def test_the_default_environment_is_local(self):
        assert Settings().environment is Environment.LOCAL

    def test_the_agent_cannot_issue_refunds_by_default(self):
        """The most consequential default in the system."""
        assert PolicySettings().allow_agent_initiated_refunds is False

    def test_verification_is_on_by_default(self):
        assert VerificationSettings().enabled is True

    def test_message_content_is_not_logged_by_default(self):
        assert ObservabilitySettings().log_message_content is False

    def test_authentication_is_required_by_default(self):
        assert SecuritySettings().require_api_key is True


class TestLimits:
    def test_the_tool_budget_cannot_exceed_the_step_budget(self):
        """A plan that cannot fit in the step budget is a misconfiguration."""
        with pytest.raises(ValidationError, match="max_tool_calls"):
            Settings(limits=LimitSettings(max_steps=3, max_tool_calls=10))

    def test_a_zero_step_budget_is_rejected(self):
        with pytest.raises(ValidationError):
            LimitSettings(max_steps=0)

    def test_an_unbounded_run_is_rejected(self):
        with pytest.raises(ValidationError):
            LimitSettings(max_run_seconds=0)

    def test_the_defaults_are_bounded(self):
        limits = LimitSettings()
        assert limits.max_steps > 0
        assert limits.max_run_seconds > 0
        assert limits.tool_timeout_seconds < limits.max_run_seconds


class TestProductionInvariants:
    def test_a_valid_production_configuration_starts(self):
        production().enforce_environment_invariants()

    def test_authentication_cannot_be_disabled_in_production(self):
        settings = production(
            security=SecuritySettings(require_api_key=False, api_keys=[SecretStr("acme:k")])
        )
        assert "authentication cannot be disabled" in problems(settings)

    def test_verification_cannot_be_disabled_in_production(self):
        settings = production(verification=VerificationSettings(enabled=False))
        assert "verification.enabled cannot be false" in problems(settings)

    def test_identity_checks_cannot_be_disabled_in_production(self):
        settings = production(policy=PolicySettings(require_identity_for_account_data=False))
        assert "require_identity_for_account_data" in problems(settings)

    def test_message_content_cannot_be_logged_in_production(self):
        settings = production(observability=ObservabilitySettings(log_message_content=True))
        assert "log_message_content" in problems(settings)

    def test_sql_echoing_is_refused_in_production(self):
        """Echoed statements carry their parameters, which are customer data."""
        settings = production(
            storage=StorageSettings(database_url="postgresql+psycopg://db/agent", echo_sql=True)
        )
        assert "echo_sql" in problems(settings)

    def test_sqlite_is_refused_in_production(self):
        settings = production(storage=StorageSettings(database_url="sqlite+aiosqlite:///x.db"))
        assert "SQLite" in problems(settings)

    def test_the_production_check_compares_by_value_not_identity(self):
        """A settings object rebuilt by model_copy holds a plain string.

        An identity check reported "not production" for it and skipped every
        invariant above, which is the quietest possible way to lose them.
        """
        settings = production().model_copy(update={"environment": "production"})
        assert settings.is_production is True


class TestUniversalInvariants:
    def test_requiring_a_key_with_no_keys_configured_is_refused(self):
        """Nobody could ever authenticate, so the process would serve nothing."""
        settings = Settings(security=SecuritySettings(require_api_key=True, api_keys=[]))
        assert "api_keys is empty" in problems(settings)

    def test_agent_refunds_require_identity_checks_on_writes(self):
        settings = Settings(
            policy=PolicySettings(
                allow_agent_initiated_refunds=True, require_identity_for_writes=False
            ),
            security=SecuritySettings(require_api_key=False),
        )
        assert "require_identity_for_writes" in problems(settings)

    def test_local_development_needs_no_credentials(self):
        Settings(security=SecuritySettings(require_api_key=False)).enforce_environment_invariants()


class TestEnvironmentParsing:
    def test_collection_values_are_read_as_a_comma_separated_list(self, monkeypatch):
        """pydantic-settings would otherwise try to JSON-decode the value.

        A deployment writes AGENT_SECURITY__API_KEYS=a,b — not a JSON array —
        and without the explicit split that raises at startup.
        """
        monkeypatch.setenv("AGENT_SECURITY__API_KEYS", "acme:one,beta:two")
        reset_settings_cache()
        try:
            keys = get_settings().security.api_keys
            assert len(keys) == 2
            assert keys[0].get_secret_value() == "acme:one"
        finally:
            monkeypatch.delenv("AGENT_SECURITY__API_KEYS", raising=False)
            reset_settings_cache()

    def test_a_single_value_is_read_as_a_one_item_list(self, monkeypatch):
        monkeypatch.setenv("AGENT_SECURITY__API_KEYS", "acme:only")
        reset_settings_cache()
        try:
            assert len(get_settings().security.api_keys) == 1
        finally:
            monkeypatch.delenv("AGENT_SECURITY__API_KEYS", raising=False)
            reset_settings_cache()

    def test_nested_settings_are_read_with_the_documented_delimiter(self, monkeypatch):
        monkeypatch.setenv("AGENT_LIMITS__MAX_STEPS", "7")
        reset_settings_cache()
        try:
            assert get_settings().limits.max_steps == 7
        finally:
            monkeypatch.delenv("AGENT_LIMITS__MAX_STEPS", raising=False)
            reset_settings_cache()


class TestSecretHandling:
    def test_api_keys_are_secrets(self):
        settings = SecuritySettings(api_keys=[SecretStr("acme:very-secret")])
        assert "very-secret" not in repr(settings)

    def test_secrets_do_not_appear_in_the_serialised_settings(self):
        settings = Settings(security=SecuritySettings(api_keys=[SecretStr("acme:very-secret")]))
        assert "very-secret" not in str(settings.model_dump())
