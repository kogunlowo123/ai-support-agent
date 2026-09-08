# HTTP API

Base path `/v1`. The interactive schema is served at `/docs` and the OpenAPI
document at `/openapi.json`; both are generated from the same models the service
validates against, so they cannot drift from what is enforced.

## Authentication

Present a key as `X-API-Key: <secret>` or `Authorization: Bearer <secret>`.

A key is *configured* as `tenant:secret` or `tenant:secret:scope|scope`; a client
presents only the secret half. The tenant and the scopes come from the
credential, never from the request, so a caller cannot claim authority by asking
for it.

A missing key and a wrong key produce byte-identical refusals, so the endpoint
cannot be used to discover which keys exist.

`/healthz` and `/readyz` need no credential — a liveness probe that can fail
authentication is not a liveness probe.

## Identity verification is separate from authentication

Authentication answers "which integration is calling?". Verification answers "has
the person on the other end proved who they are?". A correctly authenticated
support widget still cannot read an account until the customer has verified,
which is why verification is an endpoint on the conversation rather than a claim
in a token.

## Errors

Every failing endpoint returns the same envelope:

```json
{
  "code": "not_found",
  "message": "conversation not found or not accessible",
  "request_id": "9f2c1b7e4a0d4f9e8b3c1a2d5e6f7a8b",
  "detail": {}
}
```

| Status | `code` | When |
|---|---|---|
| 400 | `validation_error` | Input failed a documented rule |
| 403 | `forbidden` | Missing, wrong or insufficiently scoped credential |
| 404 | `not_found` | The resource does not exist *or* is not accessible to this tenant |
| 413 | `request_too_large` | Body above `security.max_request_bytes` |
| 422 | `validation_error` | Body did not match the schema |
| 502 | `provider_error` | The model provider failed and no fallback applied |
| 500 | `internal_error` | Unhandled. Logged in full; the client is told nothing else |

Validation errors report the field and the rule, never the submitted value —
that value may be a customer message.

`404` covers "does not exist" and "not yours" deliberately: distinguishing them
turns the endpoint into an oracle for enumerating identifiers.

**A refusal is not an error.** A run that refuses, asks a clarifying question or
escalates returns `200`. Those are correct outcomes, and returning them as
failures would hide the refusal rate in an error dashboard.

## Conversations

### `POST /v1/conversations` → 201

```json
{ "customer_email": "ada@example.com" }
```

`customer_email` is optional. Supplying it binds the thread to a customer; it
does **not** verify them, because knowing an address is not proving one. An
unknown address binds nothing and is not reported as an error.

```json
{
  "conversation_id": "conv_5a1f…",
  "identity_verified": false,
  "customer_id": "cus_9b2e…",
  "turns": 0,
  "escalated": false,
  "created_at": "2026-09-07T21:00:00Z"
}
```

### `POST /v1/conversations/{id}/verify` → 200

```json
{ "email": "ada@example.com" }
```

```json
{ "conversation_id": "conv_5a1f…", "verified": true, "message": "Thanks, I have confirmed your identity." }
```

The response is the same shape whether the address is unknown or simply wrong.

> This implementation checks a knowledge factor. A public deployment must replace
> it with a possession factor — see [SECURITY.md](../SECURITY.md).

### `POST /v1/conversations/{id}/messages` → 200

```json
{ "message": "Where is my order ORD-1005?", "include_trace": false }
```

```json
{
  "conversation_id": "conv_5a1f…",
  "reply": "Order ORD-1005 is currently shipped. It is with Evri under tracking number EVR2233445566.",
  "intent": "order_status",
  "state": "answered",
  "escalated": false,
  "escalation_reason": null,
  "refused": false,
  "refusal_reason": null,
  "ticket_id": null,
  "provenance": [{ "kind": "tool_result", "reference": "call_7c1a…" }],
  "unverified_claims": [],
  "warnings": [],
  "tool_calls": 1,
  "steps_used": 6,
  "duration_ms": 7.4,
  "provider": "template",
  "model": "template-v1",
  "trace": null
}
```

`include_trace` is opt-in because the trace echoes the decisions back.

**`state`** is the terminal state of the run:

| State | Meaning |
|---|---|
| `answered` | A verified answer was produced |
| `clarifying` | The agent needs one more thing before it can act |
| `escalated` | Handed to a person; `ticket_id` is set when one was raised |
| `refused` | Declined; `refusal_reason` says why |
| `failed` | Something broke. Logged; the customer is told a colleague will help |

**`escalation_reason`**: `customer_request`, `low_confidence`,
`policy_requires_human`, `unverifiable_answer`, `tool_failure`,
`budget_exhausted`, `repeated_failure`, `sensitive_topic`.

**`refusal_reason`**: `out_of_scope`, `identity_not_verified`,
`prompt_injection`, `unsupported_claim`, `policy_prohibited`.

**`provenance`** points every stated fact at the evidence behind it.
**`unverified_claims`** is what the verifier withheld — normally empty, and
non-empty is worth alerting on.

### `GET /v1/conversations/{id}` → 200

### `POST /v1/conversations/{id}/feedback` → 204

```json
{ "helpful": false, "comment": "did not answer my question" }
```

Feedback is joined to runs by conversation, so an unhelpful verdict can be read
next to the trace that produced it.

## Tools

### `GET /v1/tools` → 200

Publishes each registered tool with the constraints the registry enforces:
`requires_identity`, `required_scopes`, `allowed_intents`, `idempotent`, `risk`
and its JSON Schema. `open_circuits` names any tool currently unavailable.

This is generated from the same `ToolSpec` objects the registry checks, so the
published contract cannot drift from the enforced one.

## Operations

| Endpoint | Returns |
|---|---|
| `GET /v1/runs` | Recent runs: intent, confidence, final state, reasons, tool calls, steps, duration, whether verification passed |
| `GET /v1/runs/{id}/trace` | Every step of one run, in order |
| `GET /v1/tickets` | Recent tickets |
| `GET /v1/audit` | Recent audit events, each naming the key that acted |
| `GET /v1/feedback/summary` | Helpful and unhelpful totals |

None of these contain message text.

## Health

### `GET /healthz` → 200

Reports that the process is running. Checks nothing external, so a slow
dependency cannot cause a restart loop.

### `GET /readyz` → 200 / 503

Reports whether this instance can serve traffic, with a component breakdown,
`open_circuits`, and `warnings`. A degraded model or an open circuit produces a
warning, not a failure — the agent degrades by escalating rather than by
stopping. Only an unreachable database makes it `503`.

## Headers

Every response carries `X-Request-ID`, `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY` and a restrictive `Referrer-Policy`. The server header
does not advertise the stack.

## Limits

| Limit | Default | Setting |
|---|---|---|
| Request body | 256 KiB | `AGENT_SECURITY__MAX_REQUEST_BYTES` |
| Message length | 4000 chars | `AGENT_LIMITS__MAX_MESSAGE_CHARS` |
| Turns per conversation | 40 | `AGENT_LIMITS__MAX_CONVERSATION_TURNS` |

There is no built-in rate limiting; put a limiter in front of the service.
