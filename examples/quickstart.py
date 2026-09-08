"""Ask the agent three questions and print what it did.

Run it with ``python examples/quickstart.py``. It needs no configuration, no
model and no network: with no chat backend set the agent composes replies from
tool results directly, so a clean clone answers real questions immediately.

Every line printed comes from the run itself — the state it reached, the tools
it called, whether it escalated. That is the point of the example: the agent is
inspectable, not a black box that returns a paragraph.
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

QUESTIONS = [
    "What is your returns policy?",
    "Where is my order ORD-1005?",
    "I would like a refund for order ORD-1002 please",
]


def settings_in(directory: Path) -> Settings:
    """Configuration for a throwaway database in ``directory``."""
    return Settings(
        environment=Environment.LOCAL,
        storage=StorageSettings(
            database_url=f"sqlite+aiosqlite:///{(directory / 'quickstart.db').as_posix()}"
        ),
        chat=ChatSettings(backend=ChatBackend.TEMPLATE),
        security=SecuritySettings(require_api_key=False),
    )


async def main() -> int:
    """Seed a demonstration account and run three turns against it."""
    with tempfile.TemporaryDirectory() as directory:
        runtime = Runtime(settings_in(Path(directory)))
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
            # Verified, because two of the three questions are about an account.
            # The identity example shows what happens without this.
            conversation = Conversation(
                id=new_id("conv"),
                tenant_id=TENANT,
                customer_id=customer_id,
                identity_verified=True,
            )

            for question in QUESTIONS:
                async with runtime.unit_of_work() as work:
                    result = await work.machine.handle(
                        message=question, conversation=conversation, principal=principal
                    )
                answer = result.answer
                tools = [
                    step.detail.get("tool") for step in answer.steps if step.detail.get("tool")
                ]

                sys.stdout.write(f"\ncustomer > {question}\n")
                sys.stdout.write(f"agent    > {answer.text}\n")
                sys.stdout.write(
                    f"           state={answer.state} intent={answer.intent} "
                    f"escalated={answer.escalated} refused={answer.refused}\n"
                )
                sys.stdout.write(f"           tools={tools}\n")
                if answer.ticket_id:
                    sys.stdout.write(f"           ticket={answer.ticket_id}\n")

                conversation = conversation.with_turn("customer", question).with_turn(
                    "agent", answer.text
                )
        finally:
            await runtime.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
