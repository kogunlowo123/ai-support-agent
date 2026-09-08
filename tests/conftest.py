"""Shared fixtures.

Every fixture builds the real object. There is no mock of the database, the
registry, the machine or the policy engine, because the interesting failures in
this system live in how those pieces interact: a tool that runs when identity is
unverified, a reply that survives verification when it should not. A suite built
on mocks of those collaborators would pass through all of it.

The one thing that is substituted is the language model, and only where the test
is about the model's output — a real provider would make the suite non-
deterministic and dependent on a network.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

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
from support_agent.providers.base import GenerationRequest, GenerationResponse
from support_agent.runtime import Runtime, UnitOfWork
from support_agent.security.authz import ALL_SCOPES, DEFAULT_SCOPES, Principal

if TYPE_CHECKING:
    from pydantic import SecretStr

TENANT = "acme"
DEMO_EMAIL = "ada@example.com"

#: The key definition the test application is configured with, in the documented
#: ``tenant:secret`` form. A test credential in a test fixture, never read from
#: the environment and never valid anywhere.
TEST_KEY_DEFINITION = "acme:test-secret-value"

#: What a client actually presents. The tenant is a property of the credential,
#: read from its definition — the caller sends only the secret half.
TEST_API_KEY = "test-secret-value"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a database file unique to the test."""
    database = tmp_path / "agent.db"
    return Settings(
        environment=Environment.TEST,
        storage=StorageSettings(database_url=f"sqlite+aiosqlite:///{database.as_posix()}"),
        chat=ChatSettings(backend=ChatBackend.TEMPLATE),
        security=SecuritySettings(require_api_key=False),
    )


@pytest.fixture
async def runtime(settings: Settings) -> AsyncIterator[Runtime]:
    """A started runtime with an empty schema."""
    instance = Runtime(settings)
    await instance.start()
    try:
        yield instance
    finally:
        await instance.aclose()


@pytest.fixture
async def seeded(runtime: Runtime) -> Runtime:
    """A runtime with the knowledge base and the demonstration account loaded."""
    async with runtime.unit_of_work() as work:
        await seed_knowledge(work.knowledge, TENANT)
        await seed_demo_account(
            customers=work.customers, orders=work.orders, tenant_id=TENANT, email=DEMO_EMAIL
        )
    return runtime


@pytest.fixture
async def work(runtime: Runtime) -> AsyncIterator[UnitOfWork]:
    """One unit of work against an empty database."""
    async with runtime.unit_of_work() as unit:
        yield unit


@pytest.fixture
async def seeded_work(seeded: Runtime) -> AsyncIterator[UnitOfWork]:
    """One unit of work against the seeded database."""
    async with seeded.unit_of_work() as unit:
        yield unit


@pytest.fixture
async def customer_id(seeded: Runtime) -> str:
    """The identifier of the seeded demonstration customer."""
    async with seeded.unit_of_work() as unit:
        customer = await unit.customers.by_email(TENANT, DEMO_EMAIL)
    assert customer is not None
    return customer.id


@pytest.fixture
def principal() -> Principal:
    """A caller holding the scopes a customer-facing deployment grants."""
    return Principal(tenant_id=TENANT, key_id="test", scopes=frozenset(DEFAULT_SCOPES))


@pytest.fixture
def admin_principal() -> Principal:
    """A caller holding every scope, including the ones no widget should have."""
    return Principal(tenant_id=TENANT, key_id="admin", scopes=frozenset(ALL_SCOPES))


@pytest.fixture
def verified_conversation(customer_id: str) -> Conversation:
    """A conversation whose customer has proved who they are."""
    return Conversation(
        id=new_id("conv"), tenant_id=TENANT, customer_id=customer_id, identity_verified=True
    )


@pytest.fixture
def unverified_conversation(customer_id: str) -> Conversation:
    """A conversation bound to a customer who has not proved who they are."""
    return Conversation(
        id=new_id("conv"), tenant_id=TENANT, customer_id=customer_id, identity_verified=False
    )


class ScriptedProvider:
    """A chat provider that returns whatever the test tells it to.

    Used where the test is about what happens *after* a model speaks: a reply
    that asserts an amount no tool returned, or one that complies with an
    instruction hidden in an order note. Those paths cannot be reached with the
    template composer, which is incapable of saying anything a tool did not.
    """

    def __init__(self, *replies: str) -> None:
        """Queue the replies, in order. The last one repeats once exhausted."""
        self.replies = list(replies) or [""]
        self.requests: list[GenerationRequest] = []
        self.closed = False

    @property
    def name(self) -> str:
        """Identifier recorded on answers composed by this provider."""
        return "scripted"

    @property
    def model(self) -> str:
        """Model identifier recorded on answers composed by this provider."""
        return "scripted-v1"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Return the next scripted reply and record the prompt it was given."""
        self.requests.append(request)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return GenerationResponse(text=text, model=self.model, provider=self.name)

    async def health(self) -> bool:
        """A scripted provider is always reachable."""
        return True

    async def aclose(self) -> None:
        """Record that the runtime released the provider."""
        self.closed = True

    @property
    def last_prompt(self) -> str:
        """Everything the most recent prompt would put in front of a model."""
        request = self.requests[-1]
        return "\n\n".join((request.system_text(), request.render_untrusted(), request.user_text()))


@pytest.fixture
def scripted() -> ScriptedProvider:
    """A provider whose replies the test controls."""
    return ScriptedProvider("")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def api_settings(tmp_path: Path) -> Settings:
    """Settings for the HTTP application, with API key authentication on."""
    database = tmp_path / "api.db"
    secrets: list[Any] = [TEST_KEY_DEFINITION]
    return Settings(
        environment=Environment.TEST,
        storage=StorageSettings(database_url=f"sqlite+aiosqlite:///{database.as_posix()}"),
        chat=ChatSettings(backend=ChatBackend.TEMPLATE),
        security=SecuritySettings(require_api_key=True, api_keys=secrets),
    )


@pytest.fixture
async def client(api_settings: Settings) -> AsyncIterator[Any]:
    """An HTTP client bound to the application, with the schema created.

    ``asgi-lifespan`` runs the real startup and shutdown, so the test exercises
    the same wiring a deployment does — including the configuration invariants
    that refuse to start an unsafe process.
    """
    import httpx
    from asgi_lifespan import LifespanManager

    from support_agent.api.app import create_app

    app = create_app(api_settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"X-API-Key": TEST_API_KEY},
        ) as http:
            await _seed_via_app(app)
            yield http


@pytest.fixture
async def anonymous_client(api_settings: Settings) -> AsyncIterator[Any]:
    """An HTTP client that presents no credential."""
    import httpx
    from asgi_lifespan import LifespanManager

    from support_agent.api.app import create_app

    app = create_app(api_settings)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def _seed_via_app(app: Any) -> None:
    """Load the knowledge base and demonstration account into the running app."""
    services = app.state.services
    session = services.session_factory()
    try:
        from support_agent.storage.repositories import (
            CustomerRepository,
            KnowledgeRepository,
            OrderRepository,
        )

        await seed_knowledge(KnowledgeRepository(session), TENANT)
        await seed_demo_account(
            customers=CustomerRepository(session),
            orders=OrderRepository(session),
            tenant_id=TENANT,
            email=DEMO_EMAIL,
        )
        await session.commit()
    finally:
        await session.close()


@pytest.fixture
def unique_email() -> str:
    """An address that cannot collide with the seeded account."""
    return f"user-{uuid.uuid4().hex[:8]}@example.com"


def secret_value(secret: SecretStr | str) -> str:
    """Read a secret's value, accepting either a wrapper or a plain string."""
    return secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
