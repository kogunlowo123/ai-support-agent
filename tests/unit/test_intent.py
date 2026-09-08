"""Intent classification.

The lexicon is data, and this file is the regression suite for it. Every case
here is a real phrasing a customer would use, and the cases that mattered most
were the ones that used to fail: the classifier scored them, cleared its own
minimum, and was still reported as not confident because two thresholds
disagreed with each other.
"""

from __future__ import annotations

import pytest

from support_agent.agent.intent import classify, extract_order_reference
from support_agent.domain.models import Intent

pytestmark = pytest.mark.unit


class TestClassification:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Where is my order?", Intent.ORDER_STATUS),
            ("Has order ORD-1001 been delivered yet?", Intent.ORDER_STATUS),
            ("My parcel never arrived", Intent.ORDER_STATUS),
            ("Can you track my order please", Intent.ORDER_STATUS),
            ("I want a refund", Intent.REFUND_REQUEST),
            ("Please refund order ORD-1003", Intent.REFUND_REQUEST),
            ("Can I have my money back?", Intent.REFUND_REQUEST),
            ("What is your returns policy?", Intent.RETURN_POLICY),
            ("How many days do I have to return something?", Intent.RETURN_POLICY),
            ("How long does standard delivery take?", Intent.SHIPPING_QUESTION),
            ("Do you ship to Ireland?", Intent.SHIPPING_QUESTION),
            ("How do I reset my password?", Intent.ACCOUNT_QUESTION),
            ("I have been charged twice for one order", Intent.BILLING_DISPUTE),
            ("The website is broken and I cannot log in", Intent.TECHNICAL_ISSUE),
            ("I want to speak to a human being", Intent.SPEAK_TO_HUMAN),
            ("This is unacceptable, I want to make a complaint", Intent.COMPLAINT),
        ],
    )
    def test_ordinary_phrasings_classify(self, message, expected):
        assert classify(message).intent is expected

    @pytest.mark.parametrize(
        "message",
        [
            "purple monday sixteen",
            "asdkjh qweoiu zxcmnb",
            "",
            "   ",
            "!!!",
        ],
    )
    def test_noise_is_unknown(self, message):
        """The failure mode is asking, not acting wrongly."""
        assert classify(message).intent is Intent.UNKNOWN

    def test_a_classification_that_clears_the_minimum_is_reported_as_confident(self):
        """The score ceiling is tied to the minimum score and to is_confident.

        These three numbers used to disagree, so a message that scored above the
        minimum was still handled as "not confident" and every single-keyword
        request fell through to a clarifying question it did not need.
        """
        result = classify("Please refund order ORD-1003, I changed my mind")
        assert result.intent is Intent.REFUND_REQUEST
        assert result.is_confident is True

    def test_refund_timing_is_a_policy_question_not_a_refund_request(self):
        """Asking when refunds arrive is not asking for one.

        The distinction matters because REFUND_REQUEST requires a verified
        identity and RETURN_POLICY does not: misrouting it would make a
        published article unreadable to an unverified caller.
        """
        assert classify("When will I get my refund?").intent is Intent.RETURN_POLICY

    def test_asking_for_a_refund_still_routes_to_refund_request(self):
        assert classify("I want a refund for order ORD-1002").intent is Intent.REFUND_REQUEST

    def test_an_order_reference_lifts_order_related_intents_without_deciding(self):
        """A reference says the message is about an order, not which intent it is."""
        assert classify("Tell me about order ORD-1007").intent is Intent.ORDER_STATUS
        assert classify("Refund ORD-1007 please").intent is Intent.REFUND_REQUEST

    def test_padding_does_not_change_the_classification(self):
        padded = (
            "Hello there I hope you are well today and I am sorry to bother you "
            "but how long does standard delivery take?"
        )
        assert classify(padded).intent is Intent.SHIPPING_QUESTION

    def test_evidence_is_reported_with_the_result(self):
        result = classify("Where is my order ORD-1005?")
        assert result.matched_terms
        assert result.confidence > 0

    def test_alternatives_are_reported_for_ambiguous_messages(self):
        result = classify("I want to return this and get a refund")
        assert result.alternatives

    def test_classification_is_deterministic(self):
        message = "Please refund order ORD-1003"
        first = classify(message)
        assert all(classify(message) == first for _ in range(20))

    def test_classification_is_case_insensitive(self):
        assert classify("WHERE IS MY ORDER?").intent is classify("where is my order?").intent

    def test_confidence_never_exceeds_one(self):
        loaded = "refund refund refund order order tracking delivery parcel shipped dispatch"
        assert classify(loaded).confidence <= 1.0


class TestOrderReferenceExtraction:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Where is ORD-1001?", "ORD-1001"),
            ("where is ord-1001?", "ORD-1001"),
            ("My order 'ORD-1234' has not arrived", "ORD-1234"),
            ("Reference AB-12345678 please", "AB-12345678"),
        ],
    )
    def test_references_are_extracted(self, message, expected):
        assert extract_order_reference(message) == expected

    @pytest.mark.parametrize(
        "message",
        ["Where is my order?", "I ordered 3 items", "Call me on 555 1234", ""],
    )
    def test_nothing_is_invented(self, message):
        assert extract_order_reference(message) is None

    def test_the_first_reference_wins(self):
        assert extract_order_reference("ORD-1001 and ORD-1002") == "ORD-1001"
