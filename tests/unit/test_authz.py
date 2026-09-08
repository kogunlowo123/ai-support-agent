"""Authentication and scopes.

Two properties matter here and both are security properties: a wrong key and a
missing key produce the same refusal, so the endpoint cannot be used to discover
which keys exist; and the tenant and scopes come from the credential, never from
the request, so a caller cannot claim authority by asking for it.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from support_agent.errors import AuthorizationError
from support_agent.security.authz import (
    ALL_SCOPES,
    DEFAULT_SCOPES,
    SCOPE_ACCOUNT,
    SCOPE_ADMIN,
    SCOPE_KNOWLEDGE,
    SCOPE_ORDER,
    SCOPE_REFUND,
    ApiKeyAuthenticator,
    Principal,
    digest_key,
    key_id,
)

pytestmark = pytest.mark.unit

KEY_PLAIN = "acme:plain-secret"
KEY_SCOPED = "beta:scoped-secret:knowledge:read|order:read"


def authenticator(*keys: str, require: bool = True) -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(keys=tuple(SecretStr(key) for key in keys), require_key=require)


class TestPrincipal:
    def test_has_requires_every_named_scope(self):
        principal = Principal(tenant_id="acme", key_id="k", scopes=frozenset({SCOPE_ORDER}))
        assert principal.has(SCOPE_ORDER) is True
        assert principal.has(SCOPE_ORDER, SCOPE_REFUND) is False

    def test_an_anonymous_principal_reports_itself(self):
        assert Principal(tenant_id="acme", key_id="anonymous").is_anonymous is True


class TestKeyDigests:
    def test_the_key_id_is_derived_from_the_key(self):
        assert key_id("secret") == digest_key("secret")[:8]

    def test_the_key_id_is_short_enough_to_log(self):
        assert len(key_id("secret")) == 8

    def test_the_key_id_does_not_contain_the_key(self):
        assert "secret" not in key_id("secret")


class TestAuthentication:
    def test_a_known_key_resolves_to_its_tenant(self):
        principal = authenticator(KEY_PLAIN).authenticate("plain-secret")
        assert principal.tenant_id == "acme"

    def test_a_key_without_scopes_gets_the_default_set(self):
        principal = authenticator(KEY_PLAIN).authenticate("plain-secret")
        assert principal.scopes == DEFAULT_SCOPES

    def test_scopes_declared_on_a_key_are_honoured(self):
        """Scope names contain colons, so the key format must split carefully."""
        principal = authenticator(KEY_SCOPED).authenticate("scoped-secret")
        assert principal.tenant_id == "beta"
        assert principal.scopes == frozenset({SCOPE_KNOWLEDGE, SCOPE_ORDER})

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(AuthorizationError):
            authenticator(KEY_PLAIN).authenticate("wrong-secret")

    def test_a_missing_key_is_refused(self):
        with pytest.raises(AuthorizationError):
            authenticator(KEY_PLAIN).authenticate(None)

    def test_a_wrong_key_and_a_missing_key_look_the_same(self):
        """The refusal must not be an oracle for which keys exist."""
        auth = authenticator(KEY_PLAIN)
        with pytest.raises(AuthorizationError) as missing:
            auth.authenticate(None)
        with pytest.raises(AuthorizationError) as wrong:
            auth.authenticate("wrong-secret")
        assert str(missing.value) == str(wrong.value)

    def test_surrounding_whitespace_is_tolerated(self):
        assert authenticator(KEY_PLAIN).authenticate("  plain-secret  ").tenant_id == "acme"

    def test_authentication_can_be_turned_off_for_local_use(self):
        principal = authenticator(require=False).authenticate(None)
        assert principal.is_anonymous is True
        assert principal.scopes == DEFAULT_SCOPES

    def test_an_unknown_scope_in_a_key_is_a_configuration_error(self):
        """A typo in a scope name must fail loudly, not grant nothing quietly."""
        with pytest.raises(ValueError, match="unknown scopes"):
            authenticator("acme:secret:order:reed")

    def test_several_keys_can_be_configured_at_once(self):
        auth = authenticator(KEY_PLAIN, KEY_SCOPED)
        assert auth.authenticate("plain-secret").tenant_id == "acme"
        assert auth.authenticate("scoped-secret").tenant_id == "beta"


class TestScopeSets:
    def test_the_default_set_does_not_include_refund_write(self):
        """Issuing money is not something a customer-facing key can do."""
        assert SCOPE_REFUND not in DEFAULT_SCOPES

    def test_the_default_set_does_not_include_admin(self):
        assert SCOPE_ADMIN not in DEFAULT_SCOPES

    def test_the_default_set_is_a_subset_of_everything_defined(self):
        assert DEFAULT_SCOPES <= ALL_SCOPES

    def test_reading_an_account_is_a_granted_default(self):
        assert SCOPE_ACCOUNT in DEFAULT_SCOPES
