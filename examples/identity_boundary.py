"""Show that the same question is answered only after identity is proved.

This is the property most worth demonstrating, because it is the one an agent
built as a prompt cannot have: the refusal is not the model declining, it is the
tool registry never running the lookup at all.

Run it with ``python examples/identity_boundary.py``.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from support_agent.config import (
    ChatBackend,
    ChatSettings,
    Environment,
    SecuritySettings,
    Settings,
    StorageSettings,
)
from support_agent.domain.models import Conversation, new_id
from support_agent.knowledge.seed import seed_demo_account, seed_knowledge
from support_agent.runtime import Runtime
from support_agent.security.authz import DEFAULT_SCOPES, Principal

TENANT = "acme"
EMAIL = "ada@example.com"
QUESTION = "Where is my order ORD-1001?"

#: Values that exist in the seeded account and must never appear in a reply to
#: an unverified caller.
SECRETS = ("RM123456789GB", "Royal Mail", EMAIL, "Ada Lovelace")


async def ask(runtime: Runtime, conversation: Conversation, principal: Principal) -> str:
    """Run one turn and return the reply, printing what the agent did."""
    async with runtime.unit_of_work() as work:
        result = await work.machine.handle(
            message=QUESTION, conversation=conversation, principal=principal
        )
    answer = result.answer
    executed = [
        step.detail.get("tool")
        for step in answer.steps
        if step.detail.get("tool") and step.detail.get("outcome") == "ok"
    ]
    sys.stdout.write(f"  state          {answer.state}\n")
    sys.stdout.write(f"  tools executed {executed or 'none'}\n")
    sys.stdout.write(f"  reply          {answer.text}\n")
    return answer.text


async def main() -> int:
    """Ask the same question before and after verification."""
    with tempfile.TemporaryDirectory() as directory:
        runtime = Runtime(
            Settings(
                environment=Environment.LOCAL,
                storage=StorageSettings(
                    database_url=(
                        f"sqlite+aiosqlite:///{(Path(directory) / 'identity.db').as_posix()}"
                    )
                ),
                chat=ChatSettings(backend=ChatBackend.TEMPLATE),
                security=SecuritySettings(require_api_key=False),
            )
        )
        await runtime.start()
        try:
            async with runtime.unit_of_work() as work:
                await seed_knowledge(work.knowledge, TENANT)
                customer_id, _ = await seed_demo_account(
                    customers=work.customers,
                    orders=work.orders,
                    tenant_id=TENANT,
                    email=EMAIL,
                )

            principal = Principal(
                tenant_id=TENANT, key_id="example", scopes=frozenset(DEFAULT_SCOPES)
            )
            base = Conversation(id=new_id("conv"), tenant_id=TENANT, customer_id=customer_id)

            sys.stdout.write(f'customer > "{QUESTION}"\n\nBEFORE verification\n')
            before = await ask(runtime, base, principal)

            sys.stdout.write("\nAFTER verification\n")
            after = await ask(
                runtime, base.model_copy(update={"identity_verified": True}), principal
            )

            leaked = [secret for secret in SECRETS if secret.lower() in before.lower()]
            sys.stdout.write(
                f"\nAccount details leaked before verification: {leaked or 'none'}\n"
                f"Account details present after verification:  "
                f"{[s for s in SECRETS if s.lower() in after.lower()] or 'none'}\n"
            )
            if leaked:
                sys.stdout.write("FAILED: the unverified reply contained account data\n")
                return 1
        finally:
            await runtime.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
