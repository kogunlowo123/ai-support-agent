"""The tools this agent actually has.

Six of them, each answering one question or performing one action. There is no
general-purpose tool — no shell, no HTTP fetch, no SQL — because a general
purpose tool hands the model whatever authority the process has, and the whole
point of a tool boundary is that it does not.

Each tool declares what it needs before it can be called: the scopes, whether
identity must be verified, and which intents may reach it. That declaration is
enforced by :class:`~support_agent.tools.registry.ToolRegistry`, not by the
tool, so a tool cannot forget to check.

Free text that a third party wrote — order notes, ticket bodies, article
content — is scanned before it is returned. A tool result reaches a prompt, and
a warehouse note is exactly the kind of place an indirect injection would live.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, StringConstraints

from support_agent.domain.models import Intent, ToolRisk
from support_agent.errors import ToolPermissionError
from support_agent.observability.logging import get_logger
from support_agent.policy.rules import OrderFacts, evaluate_refund, evaluate_return
from support_agent.security import injection
from support_agent.security.authz import (
    SCOPE_ACCOUNT,
    SCOPE_KNOWLEDGE,
    SCOPE_ORDER,
    SCOPE_TICKET,
)
from support_agent.storage.repositories import (
    CustomerRepository,
    KnowledgeRepository,
    OrderRepository,
    TicketRepository,
)
from support_agent.tools.base import (
    Tool,
    ToolArguments,
    ToolContext,
    ToolReturns,
    ToolSpec,
    expect,
)

if TYPE_CHECKING:
    from support_agent.config import PolicySettings
    from support_agent.storage.schema import KnowledgeArticleRow

logger = get_logger(__name__)

OrderReference = Annotated[
    str, StringConstraints(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9\-]+$")
]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=500, strip_whitespace=True)]

#: Length above which third-party free text is truncated before it enters a
#: prompt. A note field with no server-side limit is a context-flooding vector.
_MAX_UNTRUSTED_CHARS = 800

#: Raw evidence at which an article scores 0.5. Sets the shape of the saturating
#: curve in :meth:`SearchKnowledgeBase._score`.
_SCORE_SATURATION = 3.0

#: Shortest word treated as a content term. Two-letter tokens are almost all
#: function words, and the stop list cannot enumerate every one of them.
_MIN_TERM_LENGTH = 2

#: Minimum normalised score for a knowledge hit to be returned at all. Below
#: this a match is a coincidence of common words, and an agent that quotes a
#: coincidence is worse than one that says it does not know.
#:
#: With the saturation above, 0.35 requires raw evidence of about 1.6: a term in
#: the title, a term in the tags, or two distinct terms in the body. A single
#: common body word scores 0.25 and is rejected.
_MIN_KNOWLEDGE_SCORE = 0.35


def _sanitise(text: str, *, label: str) -> tuple[str, list[dict[str, object]]]:
    """Scan and neutralise third-party text before it can reach a prompt."""
    if not text.strip():
        return "", []
    result = injection.scan(text)
    if not result.is_suspicious:
        return text[:_MAX_UNTRUSTED_CHARS], []

    logger.warning(
        "security.injection_in_tool_output",
        field=label,
        risk=result.risk,
        rules=[finding.rule_id for finding in result.findings],
    )
    cleaned = injection.neutralise(text, result.spans)
    return cleaned[:_MAX_UNTRUSTED_CHARS], [finding.as_dict() for finding in result.findings]


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------


class SearchKnowledgeArguments(ToolArguments):
    """Arguments for the knowledge base search."""

    query: ShortText = Field(description="What the customer wants to know, in their words.")
    limit: int = Field(default=3, ge=1, le=5)


class KnowledgeHit(ToolReturns):
    """One matching article."""

    slug: str
    title: str
    excerpt: str
    score: float


class SearchKnowledgeReturns(ToolReturns):
    """Matching articles, best first."""

    hits: tuple[KnowledgeHit, ...] = ()
    searched_terms: tuple[str, ...] = ()


def _terms(text: str) -> list[str]:
    """Content words, lowercased, for article ranking."""
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
    words = [word.strip(".,?!:;'\"()") for word in text.lower().split()]
    return [word for word in words if len(word) > _MIN_TERM_LENGTH and word not in stop]


class SearchKnowledgeBase:
    """Search the published support knowledge base.

    The only tool that needs no verified identity: published articles are
    public, and forcing verification before answering "what is your returns
    policy?" would make the agent useless for the questions it handles best.
    """

    spec = ToolSpec(
        name="search_knowledge_base",
        description=(
            "Search published support articles for policy and how-to answers. "
            "Returns article excerpts with their slugs, which must be cited."
        ),
        risk=ToolRisk.READ,
        arguments=SearchKnowledgeArguments,
        returns=SearchKnowledgeReturns,
        required_scopes=frozenset({SCOPE_KNOWLEDGE}),
        allowed_intents=frozenset(
            {
                Intent.RETURN_POLICY,
                Intent.SHIPPING_QUESTION,
                Intent.REFUND_REQUEST,
                Intent.ACCOUNT_QUESTION,
                Intent.TECHNICAL_ISSUE,
                Intent.BILLING_DISPUTE,
                Intent.ORDER_STATUS,
            }
        ),
        requires_identity=False,
        idempotent=True,
        returns_untrusted_text=True,
    )

    async def __call__(
        self, arguments: ToolArguments, context: ToolContext
    ) -> SearchKnowledgeReturns:
        """Rank articles by term overlap and return the best excerpts."""
        args = expect(arguments, SearchKnowledgeArguments)

        terms = _terms(args.query)
        repository = KnowledgeRepository(context.session)
        candidates = await repository.candidates(context.principal.tenant_id, terms)

        scored = sorted(
            ((self._score(article, terms), article) for article in candidates),
            key=lambda pair: (-pair[0], pair[1].slug),
        )
        hits: list[KnowledgeHit] = []
        for score, article in scored[: args.limit]:
            if score < _MIN_KNOWLEDGE_SCORE:
                continue
            excerpt, _ = _sanitise(self._excerpt(article.body, terms), label="article.body")
            hits.append(
                KnowledgeHit(
                    slug=article.slug,
                    title=article.title,
                    excerpt=excerpt,
                    score=round(score, 4),
                )
            )
        return SearchKnowledgeReturns(hits=tuple(hits), searched_terms=tuple(terms))

    @staticmethod
    def _score(article: KnowledgeArticleRow, terms: list[str]) -> float:
        if not terms:
            return 0.0
        title = article.title.lower()
        body = article.body.lower()
        tags = " ".join(article.tags or []).lower()
        # A term in the title is worth more than one buried in the body: an
        # article titled "Returns policy" is about returns in a way that an
        # article merely mentioning the word is not.
        hits = sum(3.0 for term in terms if term in title)
        hits += sum(2.0 for term in terms if term in tags)
        hits += sum(1.0 for term in terms if term in body)
        # Saturating rather than divided by the number of query terms. Dividing
        # made relevance a function of message length: the same question padded
        # with pleasantries scored below the floor and the agent answered a
        # question it could have answered, because the padding words matched
        # nothing. What matters is how much evidence an article carries, not
        # what fraction of the customer's sentence it happened to cover.
        return hits / (hits + _SCORE_SATURATION)

    @staticmethod
    def _excerpt(body: str, terms: list[str], width: int = 400) -> str:
        """Cut a window around the first matching term, snapped to sentence bounds.

        Slicing on a raw offset produces excerpts that begin mid-word — the
        agent then quotes "asswords are reset from…", which reads like a defect
        because it is one.
        """
        lowered = body.lower()
        position = next((lowered.find(term) for term in terms if lowered.find(term) >= 0), 0)
        start = max(0, position - width // 3)

        if start > 0:
            # Prefer the start of a sentence; fall back to a word boundary.
            sentence_start = body.rfind(". ", 0, start)
            start = sentence_start + 2 if sentence_start != -1 else body.find(" ", start) + 1

        end = min(len(body), start + width)
        if end < len(body):
            sentence_end = body.rfind(". ", start, end)
            end = sentence_end + 1 if sentence_end > start else body.rfind(" ", start, end)
        return body[start:end].strip()


# ---------------------------------------------------------------------------
# lookup_customer
# ---------------------------------------------------------------------------


class LookupCustomerArguments(ToolArguments):
    """Arguments for the customer lookup."""

    customer_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class LookupCustomerReturns(ToolReturns):
    """The customer's own profile."""

    customer_id: str
    full_name: str
    email: str
    tier: str


class LookupCustomer:
    """Read the profile of the customer in this conversation.

    Takes a customer id but ignores any that does not match the conversation:
    the identifier that matters is the one identity verification established,
    not the one an argument supplied.
    """

    spec = ToolSpec(
        name="lookup_customer",
        description="Read the profile of the verified customer in this conversation.",
        risk=ToolRisk.READ,
        arguments=LookupCustomerArguments,
        returns=LookupCustomerReturns,
        required_scopes=frozenset({SCOPE_ACCOUNT}),
        allowed_intents=frozenset(
            {
                Intent.ACCOUNT_QUESTION,
                Intent.ORDER_STATUS,
                Intent.REFUND_REQUEST,
                Intent.BILLING_DISPUTE,
            }
        ),
        requires_identity=True,
        idempotent=True,
    )

    async def __call__(
        self, arguments: ToolArguments, context: ToolContext
    ) -> LookupCustomerReturns:
        """Return the verified customer's profile."""
        args = expect(arguments, LookupCustomerArguments)

        bound = context.metadata.get("customer_id")
        if not bound:
            raise ToolPermissionError("no customer is bound to this conversation")
        if args.customer_id != bound:
            # Not merely a mismatch: an attempt to read a different customer.
            logger.warning(
                "security.customer_id_mismatch",
                conversation_id=context.conversation_id,
                actor=context.principal.key_id,
            )
            raise ToolPermissionError(
                "this conversation may only access the customer it was verified for"
            )

        row = await CustomerRepository(context.session).by_id(context.principal.tenant_id, bound)
        if row is None:
            raise ToolPermissionError("the customer record is not accessible")
        return LookupCustomerReturns(
            customer_id=row.id, full_name=row.full_name, email=row.email, tier=row.tier
        )


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------


class LookupOrderArguments(ToolArguments):
    """Arguments for the order lookup."""

    order_reference: OrderReference = Field(description="The customer-facing order reference.")


class LookupOrderReturns(ToolReturns):
    """One order belonging to the verified customer."""

    order_reference: str
    status: str
    item_name: str
    is_digital: bool
    downloaded: bool
    amount_minor: int
    currency: str
    placed_at: str
    delivered_at: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    notes: str = ""
    notes_flagged: bool = False


class LookupOrder:
    """Read one of the verified customer's orders."""

    spec = ToolSpec(
        name="lookup_order",
        description=(
            "Read one order belonging to the verified customer, by its reference. "
            "Returns status, item, amount and delivery details."
        ),
        risk=ToolRisk.READ,
        arguments=LookupOrderArguments,
        returns=LookupOrderReturns,
        required_scopes=frozenset({SCOPE_ORDER}),
        allowed_intents=frozenset(
            {
                Intent.ORDER_STATUS,
                Intent.REFUND_REQUEST,
                Intent.RETURN_POLICY,
                Intent.SHIPPING_QUESTION,
                Intent.BILLING_DISPUTE,
                Intent.COMPLAINT,
            }
        ),
        requires_identity=True,
        idempotent=True,
        returns_untrusted_text=True,
    )

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> LookupOrderReturns:
        """Return the order, with third-party note text sanitised."""
        args = expect(arguments, LookupOrderArguments)

        customer_id = context.metadata.get("customer_id")
        if not customer_id:
            raise ToolPermissionError("no customer is bound to this conversation")

        row = await OrderRepository(context.session).by_reference(
            context.principal.tenant_id, customer_id, args.order_reference
        )
        if row is None:
            # Deliberately the same message whether the order does not exist or
            # belongs to someone else: otherwise the tool is an oracle for
            # enumerating order references.
            raise ToolPermissionError("no such order for this customer")

        notes, findings = _sanitise(row.notes, label="order.notes")
        return LookupOrderReturns(
            order_reference=row.reference,
            status=row.status,
            item_name=row.item_name,
            is_digital=row.is_digital,
            downloaded=row.downloaded,
            amount_minor=row.amount_minor,
            currency=row.currency,
            placed_at=row.placed_at.isoformat(),
            delivered_at=row.delivered_at.isoformat() if row.delivered_at else None,
            carrier=row.carrier,
            tracking_number=row.tracking_number,
            notes=notes,
            notes_flagged=bool(findings),
        )


# ---------------------------------------------------------------------------
# check_refund_eligibility
# ---------------------------------------------------------------------------


class CheckRefundArguments(ToolArguments):
    """Arguments for the refund eligibility check."""

    order_reference: OrderReference


class CheckRefundReturns(ToolReturns):
    """The decision, and the facts it was computed from."""

    order_reference: str
    rule: str
    allowed: bool
    reason: str
    requires_human: bool
    window_days: int | None = None
    age_days: int | None = None
    amount_minor: int | None = None
    currency: str | None = None


class CheckRefundEligibility:
    """Compute refund eligibility from the order and the written policy.

    A tool rather than a prompt instruction. The model is told the decision;
    it never makes it. That is what makes "the agent must never invent a
    refund policy" a property of the system rather than a hope about the model.
    """

    def __init__(self, policy: PolicySettings) -> None:
        """Bind the tool to the configured business policy."""
        self._policy = policy

    spec = ToolSpec(
        name="check_refund_eligibility",
        description=(
            "Determine whether an order can be refunded, applying the written refund "
            "policy to the order's age, type and status. Returns a decision and the "
            "facts behind it."
        ),
        risk=ToolRisk.READ,
        arguments=CheckRefundArguments,
        returns=CheckRefundReturns,
        required_scopes=frozenset({SCOPE_ORDER}),
        allowed_intents=frozenset({Intent.REFUND_REQUEST, Intent.RETURN_POLICY, Intent.COMPLAINT}),
        requires_identity=True,
        idempotent=True,
    )

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> CheckRefundReturns:
        """Evaluate the policy against the order."""
        args = expect(arguments, CheckRefundArguments)

        customer_id = context.metadata.get("customer_id")
        if not customer_id:
            raise ToolPermissionError("no customer is bound to this conversation")

        row = await OrderRepository(context.session).by_reference(
            context.principal.tenant_id, customer_id, args.order_reference
        )
        if row is None:
            raise ToolPermissionError("no such order for this customer")

        facts = OrderFacts(
            reference=row.reference,
            status=row.status,
            is_digital=row.is_digital,
            downloaded=row.downloaded,
            amount_minor=row.amount_minor,
            currency=row.currency,
            placed_at=row.placed_at.replace(tzinfo=row.placed_at.tzinfo or UTC),
            delivered_at=(
                row.delivered_at.replace(tzinfo=row.delivered_at.tzinfo or UTC)
                if row.delivered_at
                else None
            ),
            refunded_at=row.refunded_at,
        )
        decision = evaluate_refund(facts, self._policy)
        return CheckRefundReturns(
            order_reference=row.reference,
            rule=decision.rule,
            allowed=decision.allowed,
            reason=decision.reason,
            requires_human=decision.requires_human,
            window_days=decision.facts.get("window_days"),
            age_days=decision.facts.get("age_days"),
            amount_minor=row.amount_minor,
            currency=row.currency,
        )


class CheckReturnEligibility(CheckRefundEligibility):
    """Compute return eligibility, which is not the same question as a refund."""

    spec = ToolSpec(
        name="check_return_eligibility",
        description=(
            "Determine whether an item can be returned, applying the written returns "
            "policy to the order's delivery date, type and status."
        ),
        risk=ToolRisk.READ,
        arguments=CheckRefundArguments,
        returns=CheckRefundReturns,
        required_scopes=frozenset({SCOPE_ORDER}),
        allowed_intents=frozenset({Intent.RETURN_POLICY, Intent.REFUND_REQUEST}),
        requires_identity=True,
        idempotent=True,
    )

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> CheckRefundReturns:
        """Evaluate the returns policy against the order."""
        args = expect(arguments, CheckRefundArguments)

        customer_id = context.metadata.get("customer_id")
        if not customer_id:
            raise ToolPermissionError("no customer is bound to this conversation")

        row = await OrderRepository(context.session).by_reference(
            context.principal.tenant_id, customer_id, args.order_reference
        )
        if row is None:
            raise ToolPermissionError("no such order for this customer")

        facts = OrderFacts(
            reference=row.reference,
            status=row.status,
            is_digital=row.is_digital,
            downloaded=row.downloaded,
            amount_minor=row.amount_minor,
            currency=row.currency,
            placed_at=row.placed_at,
            delivered_at=row.delivered_at,
            refunded_at=row.refunded_at,
        )
        decision = evaluate_return(facts, self._policy)
        return CheckRefundReturns(
            order_reference=row.reference,
            rule=decision.rule,
            allowed=decision.allowed,
            reason=decision.reason,
            requires_human=decision.requires_human,
            window_days=decision.facts.get("window_days"),
            age_days=decision.facts.get("age_days"),
        )


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------


class CreateTicketArguments(ToolArguments):
    """Arguments for raising a support ticket."""

    subject: Annotated[str, StringConstraints(min_length=3, max_length=200, strip_whitespace=True)]
    body: Annotated[str, StringConstraints(min_length=3, max_length=4000)]
    category: Annotated[
        str, StringConstraints(pattern=r"^(refund|return|shipping|billing|technical|general)$")
    ] = "general"
    priority: Annotated[str, StringConstraints(pattern=r"^(low|normal|high)$")] = "normal"


class CreateTicketReturns(ToolReturns):
    """The ticket that now exists."""

    ticket_id: str
    subject: str
    category: str
    priority: str
    status: str
    created: bool = Field(description="False when an existing ticket was returned instead.")


class CreateTicket:
    """Raise a support ticket for a human to pick up.

    The only write this agent performs, and the only tool with a durable
    idempotency guarantee: the ticket table has a unique constraint on the
    idempotency key, so a retry inside a run and a retry across processes both
    produce one ticket.

    ``priority`` is capped at ``high``. There is no ``urgent``, because
    "urgent" is precisely what an injected instruction asks for.
    """

    spec = ToolSpec(
        name="create_ticket",
        description=(
            "Raise a support ticket for a human colleague. Use when the customer needs "
            "something the agent cannot do, or asks for a person."
        ),
        risk=ToolRisk.WRITE,
        arguments=CreateTicketArguments,
        returns=CreateTicketReturns,
        required_scopes=frozenset({SCOPE_TICKET}),
        allowed_intents=frozenset(
            {
                Intent.REFUND_REQUEST,
                Intent.BILLING_DISPUTE,
                Intent.TECHNICAL_ISSUE,
                Intent.COMPLAINT,
                Intent.SPEAK_TO_HUMAN,
                Intent.ACCOUNT_QUESTION,
                Intent.ORDER_STATUS,
                Intent.RETURN_POLICY,
                Intent.SHIPPING_QUESTION,
            }
        ),
        requires_identity=True,
        idempotent=True,
    )

    async def __call__(self, arguments: ToolArguments, context: ToolContext) -> CreateTicketReturns:
        """Create the ticket, or return the one an identical call already made."""
        args = expect(arguments, CreateTicketArguments)

        key = context.idempotency_key or f"{context.conversation_id}:{args.subject[:80]}"
        row, created = await TicketRepository(context.session).create(
            tenant_id=context.principal.tenant_id,
            subject=args.subject,
            body=args.body,
            category=args.category,
            priority=args.priority,
            customer_id=context.metadata.get("customer_id"),
            conversation_id=context.conversation_id,
            idempotency_key=key,
        )
        logger.info(
            "ticket.created" if created else "ticket.deduplicated",
            ticket_id=row.id,
            category=row.category,
            priority=row.priority,
            conversation_id=context.conversation_id,
        )
        return CreateTicketReturns(
            ticket_id=row.id,
            subject=row.subject,
            category=row.category,
            priority=row.priority,
            status=row.status,
            created=created,
        )


def build_default_tools(policy: PolicySettings) -> tuple[Tool, ...]:
    """Every tool a standard deployment registers."""
    return (
        SearchKnowledgeBase(),
        LookupCustomer(),
        LookupOrder(),
        CheckRefundEligibility(policy),
        CheckReturnEligibility(policy),
        CreateTicket(),
    )


__all__ = [
    "CheckRefundEligibility",
    "CheckReturnEligibility",
    "CreateTicket",
    "LookupCustomer",
    "LookupOrder",
    "SearchKnowledgeBase",
    "build_default_tools",
]
