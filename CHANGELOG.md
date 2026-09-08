# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-07

First public release. The agent is feature-complete for its stated scope: answer
customer support questions about orders, refunds, returns, shipping and accounts,
without ever inventing an account fact, acting for an unverified customer, or
moving money.

### Added

**The agent**
- An explicit state machine with a declared transition table, checked on every
  move. Escalation is reachable from every working state; `ANSWERED` is reachable
  only through verification.
- Step, tool-call, per-tool, wall-clock and per-tool-timeout budgets. Exhausting
  any of them escalates rather than continuing. The step trace itself is capped.
- Deterministic intent classification over a closed set, with weighted phrase and
  keyword evidence, an order-reference signal, and a confidence margin the caller
  uses rather than the winner alone.
- Fixed per-intent plans. A model never selects a tool.
- Deterministic policy evaluation for refunds and returns, ordered so the
  customer is told the most specific reason.
- Answer verification: every factual sentence must trace to a tool result, a
  policy decision, a fact the system produced, or the customer's own words, with
  figures and identifiers checked separately.

**Tools**
- Six declared tools: knowledge search, customer lookup, order lookup, refund
  eligibility, return eligibility, ticket creation. No general-purpose tool.
- A registry running nine checks before any tool body executes, with per-tool
  circuit breakers, an idempotency store namespaced by tool and principal, and
  retries only for idempotent tools.
- `GET /v1/tools` publishes the same declarations the registry enforces.

**Security**
- Layered prompt-injection detection: Unicode normalisation with an offset map,
  14 pattern rules, four structural signals, noisy-OR risk aggregation, and
  neutralisation at the point untrusted text is read.
- Trust-separated prompt construction with per-request nonce fences. Policy is
  the only system content.
- Prompt-disclosure detection: a reply repeating six consecutive words of the
  system policy is withheld.
- Constant-time API key authentication with per-key tenant and scopes. Identical
  refusals for missing and wrong keys.
- Identity verification as a property of the conversation, enforced by the
  registry rather than by a prompt.
- Per-tenant isolation in every query; order reads additionally scoped to the
  verified customer, with a non-oracle refusal.
- Configuration invariants that refuse to start an unsafe production process.

**Interfaces**
- FastAPI service: conversations, identity verification, messages, feedback, tool
  discovery, runs and traces, tickets, audit, health and readiness.
- CLI: `serve`, `seed`, `chat`, `evaluate`.
- Three runnable examples, executed in CI.

**Evaluation**
- 31-scenario behavioural suite with six categories, gated in CI at a 100% pass
  rate, 100% on adversarial and identity, a p95 latency bound and a ceiling on
  escalation rate.
- 502 tests across unit, integration, API, security, e2e and regression layers;
  88% coverage against an 85% gate.
- Adversarial tests against a scripted provider that complies fully with each
  attack, so the verifier is exercised rather than assumed.

**Operations**
- Structured JSON logging with redaction as the final processor; message content
  never logged.
- OpenTelemetry traces and metrics for runs, steps, tools, escalations and
  injection findings.
- Multi-stage container running as a non-root user, with a smoke test that holds
  a real conversation over HTTP and asserts the identity boundary, the injection
  defence and the refund boundary.
- CI: lint, format, types, five test suites, coverage gate, scenario gate,
  examples, build. Security: gitleaks over tree and history, bandit, pip-audit,
  CodeQL, Trivy.

### Fixed during development

Recorded because each was a real defect that the tests caught, and the tests
exist because of them.

- Escalation was unreachable from `COMPOSING`, so a run whose composer produced
  nothing raised `InvalidTransitionError` at the moment it decided to be careful.
- The step budget was advisory: the trace appended past `max_steps`. It is now a
  hard ceiling with a bounded allowance for the steps that end a run.
- Injection scanning recorded one span per rule, so a second override attempt in
  the same text was detected but never neutralised.
- Injection rules missed named-persona jailbreaks, padded disclosure requests,
  "new instructions:" and "forget everything above".
- Knowledge relevance was divided by query length, so the same question padded
  with pleasantries fell below the floor. Now saturating and length-independent,
  with two negative controls.
- The intent score ceiling disagreed with the confidence threshold, so every
  single-keyword request fell through to a clarifying question it did not need.
- Refund *timing* classified as a refund *request*, putting a published article
  behind an identity check.
- Neutralised tool output was logged but never recorded on the run trace, so the
  defence was invisible to anyone auditing an answer.
- `CircuitBreakerRegistry.state()` created an entry for any tool it was asked
  about, so the readiness snapshot grew every time it was read.
- Unknown fields in a settings section were silently ignored, so a mistyped
  environment variable ran on the default while its operator believed otherwise.
- The seeding CLI and an unauthenticated API disagreed on which tenant to use, so
  a local instance served an empty database while `seed` reported success.
- The container shipped an empty virtual environment because `uv` resolves the
  project environment from `UV_PROJECT_ENVIRONMENT`, not `VIRTUAL_ENV`.
- `.dockerignore` excluded `data/`, so the scenario suite was missing from the
  image.
- Unused embedding providers and an unused `datetime` import were carried over
  from a sibling project and removed.

[Unreleased]: https://github.com/kogunlowo123/ai-support-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kogunlowo123/ai-support-agent/releases/tag/v0.1.0
