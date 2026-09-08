# Configuration

Every setting is an environment variable prefixed `AGENT_`, with `__` separating
a section from a field. [`.env.example`](../.env.example) is the complete list
with defaults; this document explains the ones whose value is a decision rather
than a number.

Settings are read once per process and cached. Unknown keys **in a known section
are rejected at startup** — a mistyped variable used to be accepted, silently
dropped, and the deployment ran on the default while its operator believed
otherwise.

## Environment

`AGENT_ENVIRONMENT` is `local`, `test`, `staging` or `production`.

`production` is not a label. It turns on invariants that are checked before the
first request, and a process that fails any of them **refuses to start**:

| Invariant | Why |
|---|---|
| `security.require_api_key` must be true | An unauthenticated support API is an open account-lookup service |
| `verification.enabled` must be true | It is the control that stops the agent asserting facts no tool returned |
| `policy.require_identity_for_account_data` must be true | It is the identity boundary |
| `observability.log_message_content` must be false | Message text is customer data |
| `storage.echo_sql` must be false | Echoed statements carry their parameters |
| `storage.database_url` must not be SQLite | Single-writer storage behind more than one process silently corrupts or blocks |

Two more apply in every environment:

- Requiring an API key with none configured is refused — nobody could ever
  authenticate, so the process would serve nothing.
- `allow_agent_initiated_refunds` requires `require_identity_for_writes`.
  Issuing money to an unverified caller is never acceptable.

The failure names every problem at once, so a misconfigured deployment is fixed
in one pass rather than one restart per mistake.

## Authentication

```bash
AGENT_SECURITY__REQUIRE_API_KEY=true
AGENT_SECURITY__API_KEYS=acme:s3cr3t,beta:0th3r:knowledge:read|order:read
AGENT_SECURITY__DEFAULT_TENANT=default
```

A key is written `tenant:secret` or `tenant:secret:scope|scope`. Clients present
only the secret half; the tenant and scopes come from the credential.

Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

| Scope | Grants |
|---|---|
| `knowledge:read` | Search published articles |
| `account:read` | Read the verified customer's record |
| `order:read` | Read orders and evaluate eligibility |
| `ticket:write` | Raise a ticket |
| `refund:write` | Not used by any built-in tool. Reserved. |
| `admin` | Reserved for administrative surfaces |

A key with no scopes listed gets `knowledge:read`, `account:read`, `order:read`
and `ticket:write` — deliberately not `refund:write`.

`DEFAULT_TENANT` is the tenant assigned when authentication is off, and it is
what the CLI seeds and evaluates against. They have to agree, or a local instance
serves an empty database while `support-agent seed` reports success.

A typo in a scope name raises at startup rather than granting nothing quietly.

## The model

```bash
AGENT_CHAT__BACKEND=template   # template | ollama | openai
```

| Backend | Needs | Use |
|---|---|---|
| `template` | nothing | Default. Renders tool results and decisions directly. Cannot invent anything. |
| `ollama` | a local Ollama | Real generation with no credentials or per-token cost |
| `openai` | an API key | Any OpenAI-compatible endpoint — Azure, vLLM, LiteLLM, llama.cpp |

The model is used for one thing: phrasing material that has already been decided.
Its absence removes fluency and nothing else.

A provider failure falls back to template composition and marks the response
`degraded`, so the degradation is visible rather than silently producing a worse
answer.

## Budgets

```bash
AGENT_LIMITS__MAX_STEPS=12
AGENT_LIMITS__MAX_TOOL_CALLS=6
AGENT_LIMITS__MAX_TOOL_CALLS_PER_TOOL=3
AGENT_LIMITS__MAX_RUN_SECONDS=30
AGENT_LIMITS__TOOL_TIMEOUT_SECONDS=5
```

These are what make the agent bounded rather than a loop. Every one is enforced,
and exhausting any of them escalates to a human rather than continuing.

`max_tool_calls` may not exceed `max_steps`; a plan that cannot fit in the step
budget is a misconfiguration and is rejected at construction.

`max_tool_calls_per_tool` stops one tool consuming the whole budget.

## Policy

```bash
AGENT_POLICY__REFUND_WINDOW_DAYS=30
AGENT_POLICY__DIGITAL_REFUND_WINDOW_DAYS=14
AGENT_POLICY__REFUND_APPROVAL_THRESHOLD_MINOR_UNITS=50000
AGENT_POLICY__ALLOW_AGENT_INITIATED_REFUNDS=false
```

These are the product's promises to customers, expressed as configuration. The
threshold is in minor units — `50000` means 500.00 — because storing money as a
float is how rounding errors become refunds.

`ALLOW_AGENT_INITIATED_REFUNDS` is the most consequential setting in the file.
It is false, no built-in tool issues a refund, and turning it on is not covered
by the threat model.

## Verification

```bash
AGENT_VERIFICATION__ENABLED=true
AGENT_VERIFICATION__MIN_SUPPORTED_RATIO=0.8
AGENT_VERIFICATION__FORBID_UNSUPPORTED_NUMBERS=true
AGENT_VERIFICATION__ON_FAILURE=escalate   # escalate | strip | refuse
```

`escalate` hands the conversation to a person, which is the right default: the
customer asked something answerable and the agent could not answer it safely.
`strip` removes the unsupported sentences and sends the rest — useful when most
of a reply is grounded. `refuse` declines outright.

Production refuses to start with verification disabled.

## Injection handling

```bash
AGENT_SECURITY__INJECTION_ACTION=neutralise   # annotate | neutralise | refuse
AGENT_SECURITY__INJECTION_REFUSE_THRESHOLD=0.85
```

`neutralise` replaces matched spans with an explicit marker and keeps the
surrounding facts citable. `refuse` terminates the run when aggregate risk
exceeds the threshold; that is what happens to a direct override in a customer
message regardless of this setting, because the message is scanned before any
tool runs.

The threshold is on noisy-OR aggregate risk, not on a single rule. A single
high-severity match does not reach it — a document about prompt injection is a
legitimate thing for a support article to be.

## Observability

```bash
AGENT_OBSERVABILITY__LOG_FORMAT=json       # json | console
AGENT_OBSERVABILITY__LOG_MESSAGE_CONTENT=false
AGENT_OBSERVABILITY__TRACING_ENABLED=false
AGENT_OBSERVABILITY__OTLP_ENDPOINT=http://localhost:4318
```

`console` is readable in a terminal; `json` is what an aggregator wants.

`LOG_MESSAGE_CONTENT` includes customer text in logs. It is false everywhere and
production refuses to start with it on. Turn it on locally, briefly, to debug a
classification — never anywhere a log is retained.

Tracing is off by default because an OTLP exporter that cannot reach a collector
adds latency to every request for no benefit.

## Storage

```bash
AGENT_STORAGE__DATABASE_URL=sqlite+aiosqlite:///./var/agent.db
# AGENT_STORAGE__DATABASE_URL=postgresql+psycopg://agent:pass@host:5432/agent
```

SQLite runs with `foreign_keys=ON` and WAL. It is genuinely fine for one
process, and genuinely wrong for more than one, which is why production refuses
to start on it. PostgreSQL needs the extra:

```bash
uv sync --extra postgres
```

## Precedence

1. Environment variables
2. `.env` in the working directory
3. Defaults in `config.py`

Unrecognised `AGENT_*` variables that do not name a known section are ignored, so
an unrelated variable in a shared environment cannot stop the process starting.
An unrecognised *field* inside a known section is rejected.
