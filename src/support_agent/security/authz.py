"""Authentication, scopes and customer-identity verification.

Three distinct questions, deliberately kept apart because conflating them is how
agents end up handing account data to whoever asks:

**Who is calling the API?** An API key, compared in constant time against
SHA-256 digests. The key carries its tenant, so a caller cannot choose one.

**What is that caller allowed to do?** Scopes, held on the
:class:`Principal` and checked deny-by-default by the tool registry.

**Which customer are we talking about, and have they proved it?** Identity
verification is a property of the *conversation*, not the credential. A
correctly authenticated support widget still cannot read an account until the
customer on the other end has answered a verification challenge. Every tool
touching account data, and every write, declares ``requires_identity``.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from pydantic import SecretStr

from support_agent.errors import AuthorizationError

#: How many colon-separated fields a key definition has, by form:
#: ``tenant:secret`` and ``tenant:secret:scope|scope``. Split with maxsplit=2,
#: so a scope name containing a colon stays in the third field.
_KEY_FIELDS_WITH_TENANT: Final[int] = 2
_KEY_FIELDS_WITH_SCOPES: Final[int] = 3

#: Scopes a caller may hold. Enumerated rather than free-form so a typo in a key
#: definition cannot silently grant nothing, or something unintended.
SCOPE_KNOWLEDGE = "knowledge:read"
SCOPE_ACCOUNT = "account:read"
SCOPE_ORDER = "order:read"
SCOPE_TICKET = "ticket:write"
SCOPE_REFUND = "refund:write"
SCOPE_ADMIN = "admin"

ALL_SCOPES: frozenset[str] = frozenset(
    {SCOPE_KNOWLEDGE, SCOPE_ACCOUNT, SCOPE_ORDER, SCOPE_TICKET, SCOPE_REFUND, SCOPE_ADMIN}
)

#: What a normal customer-facing deployment grants. Notably absent:
#: ``refund:write``. Issuing money is a decision a human makes.
DEFAULT_SCOPES: frozenset[str] = frozenset(
    {SCOPE_KNOWLEDGE, SCOPE_ACCOUNT, SCOPE_ORDER, SCOPE_TICKET}
)


class VerificationMethod(StrEnum):
    """How a customer's identity was established."""

    NONE = "none"
    EMAIL_CODE = "email_code"
    ORDER_REFERENCE = "order_reference"
    UPSTREAM_SESSION = "upstream_session"


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller.

    ``key_id`` is the first eight characters of the key's digest: stable, safe
    to log, and enough to answer "which key did this?" without the key ever
    appearing in a log line.
    """

    tenant_id: str
    key_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_anonymous(self) -> bool:
        """Whether the caller presented no credential."""
        return self.key_id == "anonymous"

    def has(self, *scopes: str) -> bool:
        """Whether every named scope is held."""
        return set(scopes) <= self.scopes


def digest_key(raw: str) -> str:
    """Hex SHA-256 digest of an API key."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_id(raw: str) -> str:
    """Short, log-safe identifier derived from an API key."""
    return digest_key(raw)[:8]


class ApiKeyAuthenticator:
    """Constant-time API key authentication with per-key tenant and scopes.

    A key is written ``tenant:secret`` or ``tenant:secret:scope1|scope2``. The
    tenant and scopes come from the key rather than from the request, which is
    what makes them properties of the credential instead of client-supplied
    hints.
    """

    def __init__(
        self,
        *,
        keys: tuple[SecretStr, ...],
        require_key: bool,
        default_tenant: str = "default",
        default_scopes: frozenset[str] = DEFAULT_SCOPES,
    ) -> None:
        """Precompute digests, tenants and scopes for the configured keys."""
        self._require_key = require_key
        self._default_tenant = default_tenant
        self._default_scopes = default_scopes
        self._by_digest: dict[str, Principal] = {}

        for key in keys:
            raw = key.get_secret_value()
            # maxsplit=2: scope names contain colons themselves ("order:read"),
            # so only the first two separators delimit the key's own fields.
            parts = raw.split(":", 2)
            if len(parts) == _KEY_FIELDS_WITH_SCOPES:
                tenant, secret, scope_spec = parts[0], parts[1], parts[2]
                scopes = frozenset(s.strip() for s in scope_spec.split("|") if s.strip())
            elif len(parts) == _KEY_FIELDS_WITH_TENANT:
                tenant, secret, scopes = parts[0], parts[1], default_scopes
            else:
                tenant, secret, scopes = default_tenant, raw, default_scopes

            unknown = scopes - ALL_SCOPES
            if unknown:
                msg = f"unknown scopes in an API key definition: {sorted(unknown)}"
                raise ValueError(msg)

            self._by_digest[digest_key(secret)] = Principal(
                tenant_id=tenant, key_id=key_id(secret), scopes=scopes
            )

    @property
    def requires_key(self) -> bool:
        """Whether a credential is mandatory."""
        return self._require_key

    def authenticate(self, presented: str | None) -> Principal:
        """Resolve a presented credential to a :class:`Principal`.

        The refusal message is identical for a missing and a wrong key, so the
        response cannot be used as an oracle.
        """
        if not self._require_key:
            return Principal(
                tenant_id=self._default_tenant,
                key_id="anonymous",
                scopes=self._default_scopes,
            )

        if not presented:
            raise AuthorizationError("a valid API key is required")

        candidate = digest_key(presented.strip())
        for known_digest, principal in self._by_digest.items():
            if hmac.compare_digest(candidate, known_digest):
                return principal
        raise AuthorizationError("a valid API key is required")


@dataclass(frozen=True, slots=True)
class VerificationChallenge:
    """A pending identity check.

    The expected answer is stored as a digest, never in the clear, and the
    challenge expires. Comparison is constant time so a timing difference cannot
    reveal how much of a code was correct.
    """

    conversation_id: str
    method: VerificationMethod
    prompt: str
    expected_digest: str
    expires_at: datetime
    attempts_remaining: int = 3

    @classmethod
    def create(
        cls,
        *,
        conversation_id: str,
        method: VerificationMethod,
        prompt: str,
        expected: str,
        ttl_seconds: float = 600.0,
        attempts: int = 3,
    ) -> VerificationChallenge:
        """Build a challenge from a plaintext expected answer."""
        return cls(
            conversation_id=conversation_id,
            method=method,
            prompt=prompt,
            expected_digest=digest_key(expected.strip().lower()),
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
            attempts_remaining=attempts,
        )

    @property
    def expired(self) -> bool:
        """Whether the challenge is past its expiry."""
        return datetime.now(UTC) >= self.expires_at

    def check(self, answer: str) -> tuple[bool, VerificationChallenge]:
        """Compare an answer, returning the verdict and the updated challenge."""
        if self.expired or self.attempts_remaining <= 0:
            return False, self
        matched = hmac.compare_digest(digest_key(answer.strip().lower()), self.expected_digest)
        remaining = self.attempts_remaining if matched else self.attempts_remaining - 1
        return matched, replace(self, attempts_remaining=remaining)


def generate_verification_code(digits: int = 6) -> str:
    """Generate a numeric verification code.

    ``secrets`` rather than ``random``: this value gates access to account data,
    and a predictable code is not a control.
    """
    upper = 10**digits
    return str(secrets.randbelow(upper)).zfill(digits)


def require_scopes(principal: Principal, *scopes: str) -> None:
    """Raise unless the principal holds every named scope."""
    if not principal.has(*scopes):
        raise AuthorizationError(
            "the caller is not permitted to perform this action",
            detail={"missing": sorted(set(scopes) - principal.scopes)},
        )


__all__ = [
    "ALL_SCOPES",
    "DEFAULT_SCOPES",
    "SCOPE_ACCOUNT",
    "SCOPE_ADMIN",
    "SCOPE_KNOWLEDGE",
    "SCOPE_ORDER",
    "SCOPE_REFUND",
    "SCOPE_TICKET",
    "ApiKeyAuthenticator",
    "Principal",
    "VerificationChallenge",
    "VerificationMethod",
    "digest_key",
    "generate_verification_code",
    "key_id",
    "require_scopes",
]
