"""The verifier is the last thing between a composed sentence and a customer.

Its job is narrow and it must do it without being clever: a factual sentence
whose terms, figures and identifiers do not trace back to something a tool
returned is unsupported, whatever it sounds like. These tests hold that line in
both directions — an invented amount must be caught, and a correct answer must
not be rejected for saying the same thing in different words.
"""

from __future__ import annotations

import pytest

from support_agent.agent.verifier import (
    Evidence,
    is_factual,
    split_sentences,
    strip_unsupported,
    verify,
)
from support_agent.domain.models import (
    PolicyDecision,
    ProvenanceKind,
    ToolOutcome,
    ToolResult,
)

pytestmark = pytest.mark.unit


def order_evidence() -> Evidence:
    return Evidence.from_tool_result(
        ToolResult(
            call_id="call_test_lookup",
            tool="lookup_order",
            outcome=ToolOutcome.OK,
            data={
                "order_reference": "ORD-1001",
                "status": "delivered",
                "amount_minor": 4999,
                "currency": "GBP",
                "carrier": "Royal Mail",
                "tracking_number": "RM123456789GB",
                "delivered_at": "2026-09-02T10:00:00Z",
            },
        )
    )


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_keeps_a_trailing_fragment(self):
        assert split_sentences("One. Two") == ["One.", "Two"]

    def test_empty_text_produces_no_sentences(self):
        assert split_sentences("   ") == []

    def test_a_decimal_point_does_not_end_a_sentence(self):
        sentences = split_sentences("The total was 49.99 GBP. It was delivered.")
        assert len(sentences) == 2
        assert "49.99" in sentences[0]


class TestFactualDetection:
    @pytest.mark.parametrize(
        "sentence",
        [
            "Order ORD-1001 was delivered on 2026-09-02.",
            "The order total was 49.99 GBP.",
            "Your refund will be processed within 5 business days.",
        ],
    )
    def test_sentences_carrying_claims_are_factual(self, sentence):
        assert is_factual(sentence) is True

    @pytest.mark.parametrize(
        "sentence",
        [
            "I am sorry to hear that.",
            "Thanks for getting in touch.",
            "Is there anything else I can help with?",
        ],
    )
    def test_courtesy_is_not_a_factual_claim(self, sentence):
        """Pleasantries carry no claim, so holding them to evidence is noise."""
        assert is_factual(sentence) is False


class TestVerification:
    def test_a_grounded_answer_passes(self):
        report = verify(
            "Order ORD-1001 is delivered. It was delivered on 2026-09-02.", [order_evidence()]
        )
        assert report.passed is True
        assert report.unsupported == ()

    def test_an_invented_amount_is_caught(self):
        """The archetypal hallucination: a figure that is nowhere in evidence."""
        report = verify("Your refund of 250.00 GBP has been sent.", [order_evidence()])
        assert report.passed is False
        assert "250.00" in " ".join(report.unsupported_numbers)

    def test_a_figure_present_in_evidence_is_accepted(self):
        report = verify("The order total was 49.99 GBP.", [order_evidence()])
        assert report.unsupported_numbers == ()

    def test_an_invented_identifier_is_caught(self):
        report = verify("I have opened ticket TKT-99999 for you.", [order_evidence()])
        assert report.passed is False
        assert any("TKT-99999" in item for item in report.unsupported_identifiers)

    def test_an_identifier_the_system_generated_is_accepted(self):
        """A ticket the run really created is evidence, not an invention."""
        evidence = [
            order_evidence(),
            Evidence.from_system("create_ticket", "Raised ticket TKT-99999 for this conversation."),
        ]
        report = verify("I have opened ticket TKT-99999 for you.", evidence)
        assert report.unsupported_identifiers == ()

    def test_a_policy_decision_supports_the_sentence_that_reports_it(self):
        decision = PolicyDecision(
            rule="refund.outside_window",
            allowed=False,
            reason="This order was delivered 90 days ago, outside the 30 day refund window.",
            facts={"age_days": 90, "window_days": 30},
        )
        report = verify(
            "This order was delivered 90 days ago, outside the 30 day refund window.",
            [Evidence.from_decision(decision)],
        )
        assert report.passed is True

    def test_a_courtesy_sentence_does_not_need_evidence(self):
        report = verify(
            "Thanks for getting in touch. Order ORD-1001 is delivered.", [order_evidence()]
        )
        assert report.passed is True
        assert report.factual_sentences < report.total_sentences

    def test_an_answer_with_no_evidence_at_all_fails(self):
        report = verify("Your order will arrive tomorrow.", [])
        assert report.passed is False

    def test_the_supported_ratio_is_one_when_nothing_is_claimed(self):
        report = verify("Thanks for getting in touch.", [])
        assert report.supported_ratio == 1.0

    def test_number_checking_can_be_disabled(self):
        """Deployments that phrase amounts loosely can turn the figure check off.

        The setting exists, so the behaviour is tested; the default stays on and
        production refuses to start with it disabled.
        """
        report = verify(
            "Your refund of 250.00 GBP has been sent.", [order_evidence()], check_numbers=False
        )
        assert report.unsupported_numbers == ()

    def test_provenance_points_at_the_evidence_that_supported_the_answer(self):
        """Provenance names the tool *call*, not the tool.

        A run may call the same tool more than once; only the call id says which
        result a sentence came from, which is what an auditor needs.
        """
        report = verify("Order ORD-1001 is delivered.", [order_evidence()])
        assert report.provenance
        assert any(item.reference == "call_test_lookup" for item in report.provenance)
        assert all(item.kind is ProvenanceKind.TOOL_RESULT for item in report.provenance)


class TestStripUnsupported:
    def test_unsupported_sentences_are_removed(self):
        answer = "Order ORD-1001 is delivered. Your refund of 250.00 GBP has been sent."
        report = verify(answer, [order_evidence()])
        stripped = strip_unsupported(answer, report)
        assert "ORD-1001" in stripped
        assert "250.00" not in stripped

    def test_a_fully_supported_answer_is_unchanged(self):
        answer = "Order ORD-1001 is delivered."
        report = verify(answer, [order_evidence()])
        assert strip_unsupported(answer, report) == answer


class TestCustomerEvidence:
    def test_a_customer_supplied_reference_is_evidence_it_was_asked_about(self):
        """Repeating a reference the customer wrote is not an invention.

        The lookup still happens against the verified account, so echoing the
        reference cannot leak anything; refusing to echo it would make every
        "I could not find ORD-9999" reply fail verification.
        """
        evidence = Evidence.from_customer("What is happening with ORD-9999?")
        report = verify("I could not find an order with the reference ORD-9999.", [evidence])
        assert report.unsupported_identifiers == ()
