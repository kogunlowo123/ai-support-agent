"""Deterministic business policy.

Every consequential decision this agent makes is computed here, by code, from
data. Not by a model, and not by a prompt that asks a model to apply a policy.

The reasoning is simple. "Is this order refundable?" has an answer that follows
from the order's age, its type, its download state and a written policy. That is
a function. Handing it to a language model replaces a function with a plausible
guess, makes the answer vary between identical requests, and puts the refund
window somewhere no one can grep for.

So the model never decides. It is given the decision this module produced, in
structured form, and its job is to explain it in a sentence. Every
:class:`~support_agent.domain.models.PolicyDecision` carries the facts it used,
so an answer a customer disputes can be reconstructed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from support_agent.domain.models import PolicyDecision

if TYPE_CHECKING:
    from support_agent.config import PolicySettings

#: Order statuses from which a refund is possible at all.
REFUNDABLE_STATUSES: frozenset[str] = frozenset({"placed", "shipped", "delivered", "returned"})

#: Statuses that make a refund question moot rather than refused.
TERMINAL_STATUSES: frozenset[str] = frozenset({"cancelled", "refunded"})


@dataclass(frozen=True, slots=True)
class OrderFacts:
    """The order attributes policy depends on.

    A narrow structure rather than the ORM row: policy should not be able to
    reach fields it has no business consulting, and a small explicit input is
    what makes these rules testable without a database.
    """

    reference: str
    status: str
    is_digital: bool
    downloaded: bool
    amount_minor: int
    currency: str
    placed_at: datetime
    delivered_at: datetime | None = None
    refunded_at: datetime | None = None

    def age_days(self, *, now: datetime | None = None) -> int:
        """Days since the order became eligible to be returned.

        Measured from delivery when known, and from placement otherwise: a
        customer cannot return an item they have not received, so counting the
        window from placement would penalise slow shipping.
        """
        reference_time = self.delivered_at or self.placed_at
        return max(0, ((now or datetime.now(UTC)) - reference_time).days)


# One return per policy rule, in the order the rules are checked. The order is
# the policy — the customer is told the most specific reason — so collapsing
# the returns would hide the thing this function exists to express.
def evaluate_refund(  # noqa: PLR0911
    facts: OrderFacts, settings: PolicySettings, *, now: datetime | None = None
) -> PolicyDecision:
    """Decide whether an order may be refunded, and by whom.

    The checks are ordered so that the most specific and least ambiguous reason
    is the one the customer is given. Being told "this was already refunded" is
    more useful than "you are outside the 30 day window", even when both are
    true.
    """
    common = {
        "order_reference": facts.reference,
        "status": facts.status,
        "is_digital": facts.is_digital,
        "amount_minor": facts.amount_minor,
        "currency": facts.currency,
        "age_days": facts.age_days(now=now),
    }

    if facts.refunded_at is not None or facts.status == "refunded":
        return PolicyDecision(
            rule="refund.already_refunded",
            allowed=False,
            reason="This order has already been refunded.",
            facts=common,
        )

    if facts.status == "cancelled":
        return PolicyDecision(
            rule="refund.cancelled",
            allowed=False,
            reason="This order was cancelled, so there is no payment to refund.",
            facts=common,
        )

    if facts.status not in REFUNDABLE_STATUSES:
        return PolicyDecision(
            rule="refund.status_not_eligible",
            allowed=False,
            reason=f"An order with status '{facts.status}' cannot be refunded automatically.",
            facts=common,
            requires_human=True,
        )

    if facts.is_digital and facts.downloaded:
        return PolicyDecision(
            rule="refund.digital_downloaded",
            allowed=False,
            reason=(
                "Digital items are not refundable once they have been downloaded, "
                "because the licence has been used."
            ),
            facts=common,
        )

    window = (
        settings.digital_refund_window_days if facts.is_digital else settings.refund_window_days
    )
    age = facts.age_days(now=now)
    if age > window:
        return PolicyDecision(
            rule="refund.outside_window",
            allowed=False,
            reason=(
                f"The refund window for this item is {window} days and this order is "
                f"{age} days old."
            ),
            facts={**common, "window_days": window},
        )

    if facts.amount_minor >= settings.refund_approval_threshold_minor_units:
        return PolicyDecision(
            rule="refund.requires_approval",
            allowed=False,
            reason=(
                "This refund is above the value a support agent can approve, so it needs "
                "a manager to review it."
            ),
            facts={
                **common,
                "threshold_minor": settings.refund_approval_threshold_minor_units,
                "window_days": window,
            },
            requires_human=True,
        )

    if not settings.allow_agent_initiated_refunds:
        return PolicyDecision(
            rule="refund.eligible_pending_human",
            allowed=True,
            reason=(
                f"This order is within the {window} day refund window and is eligible. "
                "A support colleague will process it."
            ),
            facts={**common, "window_days": window},
            requires_human=True,
        )

    return PolicyDecision(
        rule="refund.eligible",
        allowed=True,
        reason=f"This order is within the {window} day refund window and can be refunded.",
        facts={**common, "window_days": window},
    )


def evaluate_return(
    facts: OrderFacts, settings: PolicySettings, *, now: datetime | None = None
) -> PolicyDecision:
    """Decide whether an item may be returned.

    Distinct from a refund: a physical item can be returnable while the money
    is only released on receipt, and a digital item is never returnable at all.
    """
    common = {
        "order_reference": facts.reference,
        "status": facts.status,
        "is_digital": facts.is_digital,
        "age_days": facts.age_days(now=now),
    }

    if facts.is_digital:
        return PolicyDecision(
            rule="return.digital",
            allowed=False,
            reason="Digital items cannot be returned; a refund is the only remedy.",
            facts=common,
        )

    if facts.status in TERMINAL_STATUSES:
        return PolicyDecision(
            rule="return.terminal_status",
            allowed=False,
            reason=f"This order is {facts.status}, so there is nothing to return.",
            facts=common,
        )

    if facts.delivered_at is None:
        return PolicyDecision(
            rule="return.not_delivered",
            allowed=False,
            reason=(
                "This order has not been delivered yet. It can be cancelled instead of returned."
            ),
            facts=common,
            requires_human=True,
        )

    age = facts.age_days(now=now)
    if age > settings.refund_window_days:
        return PolicyDecision(
            rule="return.outside_window",
            allowed=False,
            reason=(
                f"Returns are accepted within {settings.refund_window_days} days of delivery "
                f"and this order was delivered {age} days ago."
            ),
            facts={**common, "window_days": settings.refund_window_days},
        )

    return PolicyDecision(
        rule="return.eligible",
        allowed=True,
        reason=(f"This order is within the {settings.refund_window_days} day return window."),
        facts={**common, "window_days": settings.refund_window_days},
    )


def evaluate_account_access(*, identity_verified: bool, settings: PolicySettings) -> PolicyDecision:
    """Decide whether account data may be disclosed in this conversation."""
    if identity_verified or not settings.require_identity_for_account_data:
        return PolicyDecision(
            rule="access.account_permitted",
            allowed=True,
            reason="The customer's identity has been verified for this conversation.",
            facts={"identity_verified": identity_verified},
        )
    return PolicyDecision(
        rule="access.identity_required",
        allowed=False,
        reason=("I can't look at account or order details until we've verified who you are."),
        facts={"identity_verified": False},
    )


def evaluate_write_permission(
    *, identity_verified: bool, settings: PolicySettings
) -> PolicyDecision:
    """Decide whether an action that changes state may be taken."""
    if identity_verified or not settings.require_identity_for_writes:
        return PolicyDecision(
            rule="access.write_permitted",
            allowed=True,
            reason="The customer's identity has been verified for this conversation.",
            facts={"identity_verified": identity_verified},
        )
    return PolicyDecision(
        rule="access.write_identity_required",
        allowed=False,
        reason="I can't make changes to an account until we've verified who you are.",
        facts={"identity_verified": False},
    )


__all__ = [
    "REFUNDABLE_STATUSES",
    "TERMINAL_STATUSES",
    "OrderFacts",
    "evaluate_account_access",
    "evaluate_refund",
    "evaluate_return",
    "evaluate_write_permission",
]
