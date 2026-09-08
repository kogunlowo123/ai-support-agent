# Operations

## Running it

### Locally

```bash
uv sync --all-extras --dev
python tasks.py seed
python tasks.py serve          # http://127.0.0.1:8000/docs
```

### With Docker

```bash
docker compose up --build -d
docker compose exec agent support-agent seed
curl -s localhost:8000/readyz | python -m json.tool
```

Two optional profiles:

```bash
docker compose --profile ollama up -d      # add a local model
docker compose --profile postgres up -d    # swap SQLite for PostgreSQL
```

`docker-compose.yml` is a local stack, not a production manifest: no TLS, no
secret management, no replicas.

### A minimum deployment

```bash
AGENT_ENVIRONMENT=production
AGENT_STORAGE__DATABASE_URL=postgresql+psycopg://…
AGENT_SECURITY__REQUIRE_API_KEY=true
AGENT_SECURITY__API_KEYS=<from your secret store>
AGENT_OBSERVABILITY__LOG_FORMAT=json
AGENT_OBSERVABILITY__TRACING_ENABLED=true
AGENT_OBSERVABILITY__OTLP_ENDPOINT=http://collector:4318
```

Plus, from [THREAT-MODEL.md](../THREAT-MODEL.md), four things this service does
not provide: a possession-factor identity check, rate limiting, append-only audit
storage, and TLS termination.

The process refuses to start if any production invariant is unmet, and the error
names every problem at once.

## Probes

| Probe | Use for | Behaviour |
|---|---|---|
| `GET /healthz` | liveness | Reports the process is running. Touches nothing, so a slow dependency cannot cause a restart loop. |
| `GET /readyz` | readiness | Checks the database, the model provider and the tool registry. Reports open circuits and warnings. |

Only an unreachable database makes readiness fail. A degraded model or an open
circuit produces a warning, because the agent degrades by escalating rather than
by stopping — removing it from the load balancer would turn a partial outage into
a total one.

```json
{
  "status": "ready",
  "version": "0.1.0",
  "components": [
    {"name": "database", "ready": true, "detail": ""},
    {"name": "composer", "ready": true, "detail": "backend=ollama reachable=true"},
    {"name": "tools", "ready": true, "detail": "registered=6"}
  ],
  "open_circuits": [],
  "warnings": []
}
```

## What to watch

### Alert on these

| Signal | Meaning |
|---|---|
| `unverified_claims` non-empty on any response | The verifier withheld something a model said. Normally zero. A rise means the model is drifting from its evidence. |
| `security.prompt_disclosure` in logs | A reply repeated the system policy and was withheld. |
| Escalation rate rising with reason `unverifiable_answer` | The same signal, seen from the other side. |
| Escalation rate rising with reason `budget_exhausted` | Runs are hitting a limit. Either a dependency slowed down or a plan grew. |
| `open_circuits` non-empty for more than a few minutes | A tool's dependency is down; those conversations are escalating. |
| `security.injection_in_message` rate | Not itself a problem — the defence working. A step change is worth looking at. |
| `security.injection_in_tool_output` | More interesting: something wrote instruction-like text into your data. |

### Watch, do not alert

- **Overall escalation rate.** Escalating is safe. A slow rise usually means
  question mix changed, not that anything broke.
- **`clarifying` rate.** Rising usually means classification is missing a
  phrasing customers have started using. The fix is a lexicon entry and a
  scenario, not a threshold change.
- **p95 run duration.** Bounded by `max_run_seconds`; a rise inside the bound
  still means something got slower.

### Metrics emitted

`agent.runs_completed` (by state and intent), `agent.run_latency`,
`agent.steps_used`, `agent.escalations` (by reason), `agent.tool_calls` (by tool
and outcome), `agent.tool_latency`, `agent.verification_failures`,
`agent.injection_findings` (by rule and source).

## Logs

Structured JSON, one object per line, with `request_id` on everything inside a
request and `session_id` on everything inside a run.

**Customer message text is never logged.** The setting that would allow it is
false everywhere, and production refuses to start with it on. Redaction is the
final processor, so a field added later cannot bypass it.

Finding one conversation:

```bash
docker compose logs agent | grep '"session_id":"conv_5a1f…"'
```

## Answering "what happened?"

```bash
# Which runs escalated, and why
curl -s localhost:8000/v1/runs -H "X-API-Key: $KEY" | python -m json.tool

# Every step of one run, in order
curl -s localhost:8000/v1/runs/run_9c2e…/trace -H "X-API-Key: $KEY" | python -m json.tool

# Who did what
curl -s localhost:8000/v1/audit -H "X-API-Key: $KEY" | python -m json.tool
```

A trace shows each transition, each tool call with its outcome and duration,
each policy decision, the composition, and the verification result — including
whether third-party text was neutralised on the way through.

Audit rows name the first eight characters of the acting key's digest, never the
key.

## Common situations

### Every reply is escalating

Check `/readyz` for open circuits, then `/v1/runs` for the reason. If it is
`unverifiable_answer`, the model is stating things the tools did not return —
switching `AGENT_CHAT__BACKEND=template` restores correct (if plainer) answers
immediately while you investigate.

### Replies are correct but read like a form letter

`AGENT_CHAT__BACKEND=template`, or a configured provider is unreachable and the
service has fallen back. `/readyz` says which.

### A customer says the agent did not understand them

Look at the run: `intent` will be `unknown` or the confidence low. The fix is a
phrase in `agent/intent.py` and a scenario in `data/scenarios/support.jsonl`.
Both are data; neither needs a code change beyond adding the line.

### A legitimate article stopped being returned

The relevance floor (`_MIN_KNOWLEDGE_SCORE`) or an injection rule. Check the logs
for `security.injection_in_tool_output` naming the article. An article about
security wording can look like an attack — that is why the scanner requires
corroboration and why the suite carries negative controls.

### The container starts and immediately fails on import

`uv` installs a workspace project in editable mode by default, leaving a `.pth`
pointing at the build directory. The Dockerfile passes `--no-editable` and sets
`UV_PROJECT_ENVIRONMENT`; if you change either, this is the symptom.

## Backup and restore

The database holds conversations, tickets, runs, traces, audit and feedback. It
is the only stateful component.

```bash
# SQLite
docker compose exec agent sqlite3 /app/var/agent.db ".backup '/app/var/backup.db'"

# PostgreSQL
pg_dump --format=custom --file=agent.dump "$AGENT_STORAGE__DATABASE_URL"
```

The knowledge base and the demonstration account are seeded, not authored, so
they are reproducible with `support-agent seed`, which is idempotent.

## Upgrading

The schema is created on startup if missing and is not migrated. For a real
deployment, add Alembic before the first schema change — this is stated because
`create_all` on startup is convenient and is not a migration strategy.

## Releasing

CI must be green: lint, types, five test suites, the coverage gate, the scenario
gate, the examples, and the container smoke test. Then tag:

```bash
git tag -a v0.2.0 -m "…" && git push origin v0.2.0
```

The container workflow builds, scans with Trivy, smoke-tests the image and
publishes it.
