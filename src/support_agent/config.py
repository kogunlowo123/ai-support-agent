"""Typed application configuration.

Environment variables prefixed ``AGENT_``, nested with a double underscore, so
``AGENT_LIMITS__MAX_STEPS=8`` sets ``settings.limits.max_steps``.

Two rules are enforced here rather than left to convention:

1. Secrets are :class:`~pydantic.SecretStr`. They never appear in a ``repr``, a
   log line or an error message.
2. Configurations that are unsafe for the declared environment fail at startup.
   An agent allowed to issue refunds without identity verification, or to skip
   answer verification, refuses to start in production.
"""

from __future__ import annotations

import functools
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from support_agent.errors import ConfigurationError


def _split_csv(value: object) -> object:
    """Accept a comma-separated environment variable as a collection.

    Collection-typed settings are annotated ``NoDecode``; without it
    pydantic-settings tries to JSON-decode the value and a perfectly ordinary
    ``AGENT_SECURITY__API_KEYS=acme:secret`` fails at startup with a JSON error
    that says nothing about the real problem.
    """
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class Environment(StrEnum):
    """Deployment environment. Controls which safety invariants are enforced."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class ChatBackend(StrEnum):
    """Selectable answer-composition implementations."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    TEMPLATE = "template"


class SettingsSection(BaseModel):
    """Base for a configuration section.

    ``extra="forbid"`` on purpose. Without it a mistyped environment variable —
    AGENT_POLICY__REFUND_AUTO_APPROVE_THRESHOLD_MINOR rather than
    AGENT_POLICY__REFUND_APPROVAL_THRESHOLD_MINOR_UNITS — is accepted, silently
    dropped, and the deployment runs on the default while its operator believes
    it is running on the value they set. A configuration mistake must fail at
    startup, which is the only moment anyone is looking.
    """

    model_config = ConfigDict(extra="forbid")


class StorageSettings(SettingsSection):
    """Conversation, ticket and audit persistence."""

    database_url: str = Field(default="sqlite+pysqlite:///./var/agent.db")
    echo_sql: bool = Field(default=False, description="Log every statement. Never in production.")
    pool_size: int = Field(default=5, ge=1, le=64)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)


class LimitSettings(SettingsSection):
    """The budgets that make the agent bounded.

    These are the difference between an agent and an uncontrolled loop. Every
    one is enforced, and exhausting any of them escalates to a human rather
    than continuing.
    """

    max_steps: int = Field(default=12, ge=1, le=100, description="State transitions per run.")
    max_tool_calls: int = Field(default=6, ge=0, le=50)
    max_tool_calls_per_tool: int = Field(default=3, ge=1, le=20)
    max_run_seconds: float = Field(default=30.0, gt=0, le=600)
    tool_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    tool_max_retries: int = Field(default=2, ge=0, le=5)
    max_message_chars: int = Field(default=4000, ge=16, le=100_000)
    max_conversation_turns: int = Field(default=40, ge=2, le=500)
    history_turns_in_prompt: int = Field(default=6, ge=0, le=50)


class CircuitBreakerSettings(SettingsSection):
    """Per-tool circuit breaker.

    A dependency that is failing should be stopped being called, not retried
    into the ground. The breaker opens after consecutive failures and half-opens
    after a cooldown so recovery is detected without a deploy.
    """

    failure_threshold: int = Field(default=4, ge=1, le=50)
    reset_seconds: float = Field(default=30.0, gt=0, le=3600)
    half_open_successes: int = Field(
        default=2, ge=1, le=10, description="Successes required to close a half-open breaker."
    )


class ChatSettings(SettingsSection):
    """Answer-composition provider selection."""

    backend: ChatBackend = ChatBackend.TEMPLATE
    model: str = Field(default="llama3.2:3b")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=500, ge=32, le=4096)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class ProviderEndpoints(SettingsSection):
    """Base URLs and credentials for outbound model providers."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: SecretStr | None = Field(default=None)


class PolicySettings(SettingsSection):
    """Business rules the agent enforces deterministically.

    These are settings rather than model behaviour on purpose: a refund window
    is a business decision, and a language model is the wrong place to keep it.
    """

    refund_window_days: int = Field(default=30, ge=0, le=3650)
    digital_refund_window_days: int = Field(default=14, ge=0, le=3650)
    refund_approval_threshold_minor_units: int = Field(
        default=50_000,
        ge=0,
        description=(
            "Refunds at or above this value require a human. Minor units, so 50000 means 500.00."
        ),
    )
    require_identity_for_account_data: bool = Field(default=True)
    require_identity_for_writes: bool = Field(default=True)
    allow_agent_initiated_refunds: bool = Field(
        default=False,
        description="Whether the agent may issue a refund itself rather than raising a ticket.",
    )


class VerificationSettings(SettingsSection):
    """How strictly a draft answer is checked before it is sent."""

    enabled: bool = Field(default=True)
    min_supported_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Fraction of factual sentences that must trace to evidence.",
    )
    on_failure: Literal["escalate", "refuse", "strip"] = Field(
        default="escalate",
        description="What to do with an answer that fails verification.",
    )
    forbid_unsupported_numbers: bool = Field(
        default=True,
        description="Refuse an answer containing a figure that appears in no tool result.",
    )


class SecuritySettings(SettingsSection):
    """Authentication and untrusted-content policy."""

    require_api_key: bool = Field(default=True)
    api_keys: Annotated[tuple[SecretStr, ...], NoDecode] = Field(default=())
    default_tenant: str = Field(
        default="default",
        min_length=1,
        description=(
            "Tenant assigned to a caller when require_api_key is false. It is also "
            "what the CLI seeds and evaluates against, so a local instance with "
            "authentication off reads the data it just loaded."
        ),
    )
    injection_action: Literal["annotate", "neutralise", "refuse"] = Field(
        default="neutralise",
        description="What to do with instruction-like text found in customer or tool content.",
    )
    injection_refuse_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_request_bytes: int = Field(default=256 * 1024, ge=1024)
    redact_pii_in_logs: bool = Field(default=True)

    _split_api_keys = field_validator("api_keys", mode="before")(_split_csv)


class ObservabilitySettings(SettingsSection):
    """Logging, tracing and metrics."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    service_name: str = "support-agent"
    tracing_enabled: bool = Field(default=False)
    otlp_endpoint: str | None = Field(default=None)
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    log_message_content: bool = Field(
        default=False,
        description="Include customer message text in logs. Off by default; it is user data.",
    )


class Settings(BaseSettings):
    """Root settings object. One instance per process, cached by :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    storage: StorageSettings = Field(default_factory=StorageSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    providers: ProviderEndpoints = Field(default_factory=ProviderEndpoints)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def is_production(self) -> bool:
        """Whether the process believes it is serving production traffic.

        Compared by value, not identity: ``Environment`` is a ``StrEnum`` and a
        settings object built by ``model_copy`` can hold the plain string. An
        identity check would silently report "not production" and skip every
        production invariant.
        """
        return self.environment == Environment.PRODUCTION

    @model_validator(mode="after")
    def _tool_budget_fits_step_budget(self) -> Settings:
        if self.limits.max_tool_calls > self.limits.max_steps:
            msg = "limits.max_tool_calls cannot exceed limits.max_steps"
            raise ValueError(msg)
        return self

    def enforce_environment_invariants(self) -> None:
        """Fail fast on configurations that are unsafe for the declared environment."""
        problems: list[str] = []

        if self.security.require_api_key and not self.security.api_keys:
            problems.append(
                "security.require_api_key is set but security.api_keys is empty; "
                "no caller could ever authenticate"
            )

        if (
            self.policy.allow_agent_initiated_refunds
            and not self.policy.require_identity_for_writes
        ):
            problems.append(
                "policy.allow_agent_initiated_refunds requires "
                "policy.require_identity_for_writes; issuing money to an unverified "
                "caller is never acceptable"
            )

        if self.is_production:
            if not self.security.require_api_key:
                problems.append("authentication cannot be disabled in production")
            if not self.verification.enabled:
                problems.append(
                    "verification.enabled cannot be false in production; it is the control "
                    "that stops the agent asserting facts no tool returned"
                )
            if not self.policy.require_identity_for_account_data:
                problems.append(
                    "policy.require_identity_for_account_data cannot be false in production"
                )
            if self.observability.log_message_content:
                problems.append("observability.log_message_content would log customer data")
            if self.storage.echo_sql:
                problems.append("storage.echo_sql would log query parameters in production")
            if self.storage.database_url.startswith("sqlite"):
                problems.append(
                    "storage.database_url points at SQLite; production requires a durable "
                    "multi-writer database such as PostgreSQL"
                )

        if problems:
            raise ConfigurationError(
                "invalid configuration for the declared environment",
                detail={"problems": problems, "environment": str(self.environment)},
            )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings. Used by tests that manipulate the environment."""
    get_settings.cache_clear()


__all__ = [
    "ChatBackend",
    "ChatSettings",
    "CircuitBreakerSettings",
    "Environment",
    "LimitSettings",
    "ObservabilitySettings",
    "PolicySettings",
    "ProviderEndpoints",
    "SecuritySettings",
    "Settings",
    "StorageSettings",
    "VerificationSettings",
    "get_settings",
    "reset_settings_cache",
]
