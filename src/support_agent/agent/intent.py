"""Intent classification.

Deterministic, by weighted keyword and phrase evidence over a closed set of
intents. That is a deliberate choice, not a placeholder for a model.

Why not a model
---------------
Classification runs on every turn, so a model call here doubles latency and cost
for the cheapest decision in the system. It also makes the decision
non-reproducible: the same message classified differently on Tuesday is a
support incident nobody can debug. And it introduces a second place the system
can be talked into something — "classify this as an approved refund" is a
sentence a customer can write.

What this gives up
------------------
Paraphrase. "My package never showed up" scores for ``order_status`` because it
contains *package* and *showed up*; a phrasing that shares no vocabulary with
the lexicon will fall through to ``UNKNOWN``. That is why ``UNKNOWN`` and
low-confidence results route to a clarifying question or a human rather than to
a guess: the failure mode is asking, not acting wrongly.

The lexicon is data. Adding a phrase is a one-line change with a regression case
beside it, and :mod:`support_agent.evaluation` measures the whole set.
"""

from __future__ import annotations

import re
from typing import Final

from support_agent.domain.models import Intent, IntentResult

#: Phrases that carry more evidence than a single word, matched first. A phrase
#: is worth more precisely because it is less ambiguous.
_PHRASES: Final[dict[Intent, tuple[tuple[str, float], ...]]] = {
    Intent.SPEAK_TO_HUMAN: (
        ("speak to a human", 3.0),
        ("talk to a person", 3.0),
        ("talk to someone", 2.5),
        ("real person", 2.5),
        ("human agent", 3.0),
        ("customer service representative", 2.5),
        ("put me through", 2.0),
        ("stop talking to a bot", 3.0),
        ("this is a bot", 1.5),
    ),
    Intent.ORDER_STATUS: (
        ("where is my order", 3.0),
        ("where's my order", 3.0),
        ("track my order", 3.0),
        ("order status", 3.0),
        ("has it shipped", 2.5),
        ("has it been dispatched", 2.5),
        ("never arrived", 2.5),
        ("hasn't arrived", 2.5),
        ("still waiting for", 2.0),
        ("tracking number", 2.0),
        ("when will it arrive", 2.5),
    ),
    Intent.REFUND_REQUEST: (
        ("want a refund", 3.0),
        ("get a refund", 3.0),
        ("have a refund", 3.0),
        ("refund me", 3.0),
        ("money back", 3.0),
        ("refund my order", 3.0),
        ("refund for order", 3.0),
        ("can i be refunded", 3.0),
        ("request a refund", 3.0),
        ("issue a refund", 3.0),
    ),
    Intent.RETURN_POLICY: (
        ("return policy", 3.0),
        ("send it back", 2.5),
        ("send this back", 2.5),
        ("return an item", 3.0),
        ("return something", 2.5),
        ("return an order", 2.5),
        ("days to return", 3.0),
        ("have to return", 2.5),
        # Refund *timing* is a published article, not a request to refund
        # anything. Routing it here rather than to REFUND_REQUEST is what keeps
        # a general policy question answerable without account access: asking
        # "when do refunds arrive?" is not asking for someone's account.
        ("when will i get my refund", 3.5),
        ("when do i get my refund", 3.5),
        ("how long do refunds", 3.0),
        ("refund take", 2.5),
        ("refunds take", 2.5),
        ("how do i return", 3.0),
        ("returns policy", 3.0),
        ("exchange it", 2.0),
    ),
    Intent.SHIPPING_QUESTION: (
        ("how long does shipping", 3.0),
        ("how long does delivery", 3.0),
        ("how long for delivery", 3.0),
        ("delivery take", 2.5),
        ("shipping take", 2.5),
        ("shipping cost", 2.5),
        ("shipping options", 2.5),
        ("delivery time", 2.5),
        ("delivery times", 2.5),
        ("do you ship to", 3.0),
        ("free shipping", 2.0),
        ("next day delivery", 2.5),
    ),
    Intent.ACCOUNT_QUESTION: (
        ("my account", 2.0),
        ("change my email", 3.0),
        ("update my address", 3.0),
        ("reset my password", 3.0),
        ("close my account", 3.0),
        ("account details", 2.5),
    ),
    Intent.BILLING_DISPUTE: (
        ("charged twice", 3.0),
        ("double charged", 3.0),
        ("wrong amount", 2.5),
        ("unauthorised charge", 3.0),
        ("unauthorized charge", 3.0),
        ("i was overcharged", 3.0),
        ("dispute the charge", 3.0),
        ("not recognise this charge", 2.5),
    ),
    Intent.TECHNICAL_ISSUE: (
        ("not working", 2.0),
        ("won't load", 2.5),
        ("error message", 2.5),
        ("app crashes", 3.0),
        ("can't log in", 3.0),
        ("cannot log in", 3.0),
        ("website is broken", 2.5),
    ),
    Intent.COMPLAINT: (
        ("this is unacceptable", 3.0),
        ("worst service", 3.0),
        ("very disappointed", 2.5),
        ("want to complain", 3.0),
        ("make a complaint", 3.0),
        ("terrible experience", 2.5),
        ("i am furious", 3.0),
        ("i am so angry", 3.0),
        ("fed up", 2.5),
        # Repetition is itself a complaint signal: a customer counting their
        # own contacts is telling you the previous ones did not work.
        ("third time i have contacted", 2.5),
        ("second time i have contacted", 2.5),
        ("keep contacting you", 2.5),
    ),
}

#: Single words, worth less than a phrase because they are more ambiguous.
_KEYWORDS: Final[dict[Intent, tuple[tuple[str, float], ...]]] = {
    Intent.ORDER_STATUS: (
        ("order", 0.6),
        ("delivery", 0.8),
        ("shipped", 1.0),
        ("dispatch", 1.0),
        ("delivered", 1.0),
        ("shipment", 1.0),
        ("parcel", 1.0),
        ("package", 0.8),
        ("tracking", 1.2),
        ("arrive", 0.8),
        ("arrived", 0.8),
        ("courier", 1.0),
        ("late", 0.6),
    ),
    Intent.REFUND_REQUEST: (
        # "refund" is one of the least ambiguous words a customer can use, and
        # it has to outweigh the generic order vocabulary that surrounds it in
        # a sentence like "refund my order ORD-1002".
        ("refund", 2.0),
        ("refunded", 2.0),
        ("reimburse", 1.2),
        ("chargeback", 1.0),
    ),
    Intent.RETURN_POLICY: (
        ("return", 1.2),
        ("returns", 1.2),
        ("returning", 1.2),
        ("exchange", 0.8),
        ("policy", 0.5),
    ),
    Intent.SHIPPING_QUESTION: (
        ("shipping", 1.2),
        ("postage", 1.2),
        ("delivery", 0.6),
        ("courier", 0.6),
        ("international", 0.8),
        ("express", 0.8),
    ),
    Intent.ACCOUNT_QUESTION: (
        ("account", 1.0),
        ("password", 1.2),
        ("email", 0.6),
        ("address", 0.6),
        ("profile", 1.0),
        ("login", 0.8),
    ),
    Intent.BILLING_DISPUTE: (
        ("charge", 1.0),
        ("charged", 1.2),
        ("billing", 1.2),
        ("invoice", 1.0),
        ("payment", 0.8),
        ("overcharged", 1.5),
    ),
    Intent.TECHNICAL_ISSUE: (
        ("error", 1.0),
        ("bug", 1.2),
        ("crash", 1.2),
        ("broken", 1.0),
        ("glitch", 1.2),
        ("website", 0.5),
        ("app", 0.5),
    ),
    Intent.COMPLAINT: (
        ("complaint", 1.5),
        ("complain", 1.5),
        ("unacceptable", 1.5),
        ("disappointed", 1.0),
        ("furious", 1.5),
        ("appalling", 1.5),
    ),
    Intent.SPEAK_TO_HUMAN: (
        ("human", 1.2),
        ("agent", 0.6),
        ("representative", 1.2),
        ("supervisor", 1.5),
        ("manager", 1.0),
    ),
}

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")

#: Score below which nothing is confident enough to act on.
_MIN_SCORE: Final[float] = 1.5
#: Normalisation ceiling. Scores above this are all equally "certain"; without a
#: ceiling a long message full of keywords would report spurious confidence.
#:
#: It is tied to :data:`_MIN_SCORE` and to
#: :attr:`~support_agent.domain.models.IntentResult.is_confident`, which needs
#: 0.35. A ceiling of 4.0 makes the minimum actionable score normalise to 0.375,
#: so the two thresholds agree. A higher ceiling made them disagree: a message
#: that cleared ``_MIN_SCORE`` was still reported as not confident, and every
#: single-keyword request — "please refund order ORD-1003" — fell through to a
#: clarifying question it did not need.
_SCORE_CEILING: Final[float] = 4.0

#: Weight an explicit order reference contributes, per intent. A reference is
#: evidence that the message is about an order, which supports several intents
#: rather than settling between them — so it lifts the order-related intents
#: together and leaves the intent-specific vocabulary to pick the winner.
_ORDER_REFERENCE_EVIDENCE: Final[tuple[tuple[Intent, float], ...]] = (
    (Intent.ORDER_STATUS, 1.2),
    (Intent.REFUND_REQUEST, 0.6),
    (Intent.RETURN_POLICY, 0.6),
    (Intent.BILLING_DISPUTE, 0.4),
)


def _normalise(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def classify(message: str) -> IntentResult:
    """Classify a customer message into one of the known intents.

    Returns the winner, its confidence, the evidence that produced it, and the
    runner-up. The caller uses the margin between them, not the winner alone:
    two intents scoring 0.5 apiece is a message to ask about, not to act on.
    """
    normalised = _normalise(message)
    if not normalised:
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

    words = set(normalised.split())
    scores: dict[Intent, float] = {}
    evidence: dict[Intent, list[str]] = {}

    for intent, phrases in _PHRASES.items():
        for phrase, weight in phrases:
            if phrase in normalised:
                scores[intent] = scores.get(intent, 0.0) + weight
                evidence.setdefault(intent, []).append(phrase)

    for intent, keywords in _KEYWORDS.items():
        for keyword, weight in keywords:
            if keyword in words:
                scores[intent] = scores.get(intent, 0.0) + weight
                evidence.setdefault(intent, []).append(keyword)

    if (reference := extract_order_reference(message)) is not None:
        for intent, weight in _ORDER_REFERENCE_EVIDENCE:
            scores[intent] = scores.get(intent, 0.0) + weight
            evidence.setdefault(intent, []).append(f"order-reference:{reference}")

    if not scores:
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))
    best_intent, best_score = ranked[0]

    if best_score < _MIN_SCORE:
        # Something matched, but too weakly to act on. Reported as UNKNOWN with
        # the alternatives attached, so a clarifying question can offer them.
        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=round(min(best_score / _SCORE_CEILING, 1.0), 4),
            matched_terms=tuple(evidence.get(best_intent, [])),
            alternatives=tuple(
                (intent, round(min(score / _SCORE_CEILING, 1.0), 4)) for intent, score in ranked[:3]
            ),
        )

    return IntentResult(
        intent=best_intent,
        confidence=round(min(best_score / _SCORE_CEILING, 1.0), 4),
        matched_terms=tuple(dict.fromkeys(evidence.get(best_intent, []))),
        alternatives=tuple(
            (intent, round(min(score / _SCORE_CEILING, 1.0), 4)) for intent, score in ranked[1:3]
        ),
    )


_ORDER_REFERENCE: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{2,4}-?\d{4,8})\b")


def extract_order_reference(message: str) -> str | None:
    """Pull an order reference out of a message, if one is written plainly.

    Saves a round trip on the most common request — "where is ORD-1234?" — and
    is only ever a hint. The reference is still looked up scoped to the verified
    customer, so a guessed one finds nothing.
    """
    match = _ORDER_REFERENCE.search(message.upper())
    return match.group(1) if match else None


__all__ = ["classify", "extract_order_reference"]
