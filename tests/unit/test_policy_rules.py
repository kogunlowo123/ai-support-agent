"""Policy is computed by code, never by a model. These tests are the policy.

Every branch of the refund and returns rules has a case here, and the ordering
of the checks is tested too: a customer told "this was already refunded" is
better served than one told "you are outside the window", even when both are
true of the same order.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from support_agent.config import PolicySettings
from support_agent.policy.rules import (
    OrderFacts,
    evaluate_account_access,
    evaluate_refund,
    evaluate_return,
    evaluate_write_permission,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


def facts(**overrides: object) -> OrderFacts:
    """An ordinary delivered order, five days old, refundable by default."""
    base: dict[str, object] = {
        "reference": "ORD-1001",
        "status": "delivered",
        "is_digital": False,
        "downloaded": False,
        "amount_minor": 4999,
        "currency": "GBP",
        "placed_at": NOW - timedelta(days=9),
        "delivered_at": NOW - timedelta(days=5),
        "refunded_at": None,
    }
    base.update(overrides)
    return OrderFacts(**base)  # type: ignore[arg-type]


@pytest.fixture
def policy() -> PolicySettings:
    return PolicySettings()


class TestOrderAge:
    def test_age_is_measured_from_delivery(self):
        assert facts().age_days(now=NOW) == 5

    def test_age_falls_back_to_placement_when_undelivered(self):
        """A customer cannot return what they have not received.

        Counting the window from placement for a delivered order would penalise
        slow shipping, so delivery is preferred whenever it is known.
        """
        assert facts(delivered_at=None).age_days(now=NOW) == 9

    def test_a_future_date_does_not_produce_a_negative_age(self):
        assert facts(delivered_at=NOW + timedelta(days=2)).age_days(now=NOW) == 0


class TestRefundEligibility:
    def test_an_ordinary_recent_order_is_eligible(self, policy):
        decision = evaluate_refund(facts(), policy, now=NOW)
        assert decision.allowed is True

    def test_an_order_outside_the_window_is_refused(self, policy):
        decision = evaluate_refund(facts(delivered_at=NOW - timedelta(days=90)), policy, now=NOW)
        assert decision.allowed is False
        assert decision.rule == "refund.outside_window"

    def test_a_downloaded_digital_item_is_refused(self, policy):
        decision = evaluate_refund(facts(is_digital=True, downloaded=True), policy, now=NOW)
        assert decision.allowed is False
        assert decision.rule == "refund.digital_downloaded"

    def test_an_undownloaded_digital_item_is_still_eligible(self, policy):
        """The licence is what cannot be returned, not the format."""
        decision = evaluate_refund(facts(is_digital=True, downloaded=False), policy, now=NOW)
        assert decision.allowed is True

    def test_an_already_refunded_order_is_refused(self, policy):
        decision = evaluate_refund(facts(refunded_at=NOW - timedelta(days=1)), policy, now=NOW)
        assert decision.allowed is False
        assert decision.rule == "refund.already_refunded"

    def test_already_refunded_outranks_outside_the_window(self, policy):
        """The most specific reason is the one the customer is told."""
        decision = evaluate_refund(
            facts(
                delivered_at=NOW - timedelta(days=90),
                refunded_at=NOW - timedelta(days=60),
                status="refunded",
            ),
            policy,
            now=NOW,
        )
        assert decision.rule == "refund.already_refunded"

    def test_a_cancelled_order_has_nothing_to_refund(self, policy):
        decision = evaluate_refund(facts(status="cancelled"), policy, now=NOW)
        assert decision.allowed is False
        assert decision.rule == "refund.cancelled"

    def test_an_unshippable_status_needs_a_person(self, policy):
        decision = evaluate_refund(facts(status="on_hold"), policy, now=NOW)
        assert decision.allowed is False
        assert decision.requires_human is True

    def test_a_high_value_order_requires_a_person(self, policy):
        decision = evaluate_refund(facts(amount_minor=89900), policy, now=NOW)
        assert decision.requires_human is True

    def test_the_threshold_is_configuration_not_a_constant(self):
        """Asserted on the rule, not on requires_human.

        Every eligible refund sets requires_human, because the agent never
        issues one — so asserting on that flag would pass whatever the
        threshold was, which is a test that cannot fail.
        """
        lenient = evaluate_refund(facts(amount_minor=4999), PolicySettings(), now=NOW)
        strict = evaluate_refund(
            facts(amount_minor=4999),
            PolicySettings(refund_approval_threshold_minor_units=1000),
            now=NOW,
        )
        assert lenient.rule == "refund.eligible_pending_human"
        assert strict.rule == "refund.requires_approval"
        assert strict.allowed is False

    def test_a_mistyped_setting_is_rejected_rather_than_ignored(self):
        """A typo that silently keeps the default is a deployment that lies."""
        with pytest.raises(ValidationError):
            # A plausible-looking name that is not the real field. Passed as a
            # mapping so the type checker does not reject the very mistake the
            # test exists to prove is rejected at runtime.
            PolicySettings(**{"refund_auto_approve_threshold_minor": 1000})

    def test_the_digital_window_is_separate_from_the_physical_one(self):
        """A digital item has a shorter window; the rules must not share one."""
        settings = PolicySettings(refund_window_days=30, digital_refund_window_days=14)
        decision = evaluate_refund(
            facts(is_digital=True, downloaded=False, delivered_at=NOW - timedelta(days=20)),
            settings,
            now=NOW,
        )
        assert decision.allowed is False

    def test_the_decision_carries_the_facts_it_was_made_from(self, policy):
        """An auditor must be able to see why, not just what."""
        decision = evaluate_refund(facts(), policy, now=NOW)
        assert decision.facts["order_reference"] == "ORD-1001"
        assert decision.facts["age_days"] == 5

    def test_the_agent_never_issues_the_refund_itself(self, policy):
        """Eligible is not the same as done. Money is a human decision."""
        decision = evaluate_refund(facts(), policy, now=NOW)
        assert decision.allowed is True
        assert policy.allow_agent_initiated_refunds is False


class TestReturnEligibility:
    def test_a_recent_physical_order_can_be_returned(self, policy):
        assert evaluate_return(facts(), policy, now=NOW).allowed is True

    def test_a_downloaded_digital_item_cannot_be_returned(self, policy):
        decision = evaluate_return(facts(is_digital=True, downloaded=True), policy, now=NOW)
        assert decision.allowed is False

    def test_an_order_outside_the_window_cannot_be_returned(self, policy):
        decision = evaluate_return(facts(delivered_at=NOW - timedelta(days=90)), policy, now=NOW)
        assert decision.allowed is False

    def test_an_undelivered_order_cannot_be_returned_yet(self, policy):
        decision = evaluate_return(facts(status="shipped", delivered_at=None), policy, now=NOW)
        assert decision.allowed is False


class TestAccountAccess:
    def test_a_verified_customer_may_read_their_account(self, policy):
        assert evaluate_account_access(identity_verified=True, settings=policy).allowed is True

    def test_an_unverified_customer_may_not(self, policy):
        decision = evaluate_account_access(identity_verified=False, settings=policy)
        assert decision.allowed is False

    def test_the_requirement_can_be_relaxed_only_by_configuration(self):
        """The setting exists for deployments behind an authenticated session.

        Production refuses to start with it off, which is asserted in the
        configuration tests; here it only has to behave as documented.
        """
        relaxed = PolicySettings(require_identity_for_account_data=False)
        assert evaluate_account_access(identity_verified=False, settings=relaxed).allowed is True


class TestWritePermission:
    def test_a_verified_customer_may_cause_a_write(self, policy):
        assert evaluate_write_permission(identity_verified=True, settings=policy).allowed is True

    def test_an_unverified_customer_may_not(self, policy):
        assert evaluate_write_permission(identity_verified=False, settings=policy).allowed is False
