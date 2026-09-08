"""Security controls: authentication, scopes, identity verification and injection defence."""

from support_agent.security.authz import (
    ALL_SCOPES,
    DEFAULT_SCOPES,
    ApiKeyAuthenticator,
    Principal,
    VerificationChallenge,
    VerificationMethod,
    require_scopes,
)
from support_agent.security.injection import (
    NEUTRALISED_MARKER,
    InjectionFinding,
    ScanResult,
    neutralise,
    scan,
)
from support_agent.security.normalization import normalize, strip_control_characters

__all__ = [
    "ALL_SCOPES",
    "DEFAULT_SCOPES",
    "NEUTRALISED_MARKER",
    "ApiKeyAuthenticator",
    "InjectionFinding",
    "Principal",
    "ScanResult",
    "VerificationChallenge",
    "VerificationMethod",
    "neutralise",
    "normalize",
    "require_scopes",
    "scan",
    "strip_control_characters",
]
