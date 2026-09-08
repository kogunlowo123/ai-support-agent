# Security policy

## Reporting a vulnerability

Please report privately, not in a public issue.

1. Go to the [Security tab](https://github.com/kogunlowo123/ai-support-agent/security/advisories)
   and choose **Report a vulnerability**.
2. Describe what you found, how to reproduce it, and what an attacker gains.

You will get an acknowledgement within three working days and an assessment
within ten. If a fix is warranted, you will be credited in the release notes
unless you would rather not be.

## Supported versions

The latest release on `main`. This is a portfolio project and does not carry a
long-term support commitment.

## What this project defends against

[THREAT-MODEL.md](THREAT-MODEL.md) is the full statement. In short: prompt
injection arriving directly or through data, cross-customer data access, acting
without a verified identity, a model inventing account facts, policy extraction,
unbounded execution, and unsafe configuration.

The design principle is that a defence a model can be talked out of is not a
defence. The controls are structural:

- **Tool selection is not a model decision.** Plans are declared per intent, and
  the registry decides what may run.
- **Authority is checked in one place.** Nine checks before any tool body runs.
- **Eligibility is computed in code.** A model never decides whether a refund is
  allowed.
- **Every reply is checked against the evidence the run gathered.** A sentence
  asserting a fact no tool returned is escalated, not sent.
- **Untrusted text is neutralised where it is read**, before it can reach a
  prompt.

## Known gaps

Stated plainly, because a security document that only lists strengths is not
useful.

### Identity verification is a knowledge factor

`POST /v1/conversations/{id}/verify` checks an email address against the customer
record. Anyone who knows the address passes it.

This is appropriate for a deployment behind an already-authenticated session,
where the surrounding application has established who the user is and this
endpoint only binds that identity to the conversation. It is **not appropriate
for a public deployment**, which needs a possession factor — a code sent to the
address on file, checked with a short expiry and a constant-time comparison.
`security/authz.py` contains `VerificationChallenge` for exactly that shape.

### There is no rate limiting

Budgets bound a single run; nothing bounds how many runs a caller starts. Put a
rate limiter in front of this service.

### The audit log is not tamper-evident

Audit rows live in the same database as application data. Ship them to
append-only storage if you need them to survive an attacker with database write
access.

### Verification is lexical, not semantic

The verifier checks that a sentence's terms, figures and identifiers trace to
evidence. A sentence whose words are all supported but whose meaning is wrong can
pass. The template composer, which cannot produce such a sentence because it only
formats values, is the default.

## Reporting scope

In scope: anything that lets a caller read another customer's data, act without a
verified identity, cause a write the policy forbids, extract the system policy,
make the agent state a fact no tool returned, or crash the service.

Out of scope: the knowledge-factor identity check and the missing rate limiter,
both documented above; findings that require write access to the database;
denial of service by volume alone.

## Handling secrets in this repository

- No credential is ever committed. `.env` is ignored, and
  [gitleaks](https://github.com/gitleaks/gitleaks) runs over the working tree and
  the full history on every push.
- API keys are held as `SecretStr` and are only unwrapped where a header is
  built. An accidental `repr` of the settings object prints nothing useful.
- Audit records name the first eight characters of a key's SHA-256 digest, never
  the key.
- `security/audit-exceptions.md` records every accepted scanner finding with a
  reason and a review date. An unexplained suppression is a bug.

## Automated checks

Every push runs:

| Check | What it covers |
|---|---|
| gitleaks | Secrets in the tree and in the full history |
| bandit | Python security anti-patterns |
| pip-audit | Known vulnerabilities in the locked dependency set |
| CodeQL | `security-extended` queries |
| Trivy | Filesystem and container image vulnerabilities |
| `pytest -m security` | Adversarial cases; a failure is a security regression |
| Scenario gate | Adversarial and identity categories at 100% |
