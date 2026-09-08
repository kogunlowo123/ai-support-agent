"""Answer verification: the control that makes "never invent" enforceable.

An instruction in a prompt is a request. This module is the check.

Every draft answer is decomposed into sentences. Each sentence that makes a
*factual claim* — one containing a figure, a date, an identifier, a status word
or a policy term — must be traceable to something a tool actually returned. A
sentence that cannot be traced is an unsupported claim, and the configured
action decides what happens to the answer: escalate, refuse, or strip the
sentence.

Numbers get their own rule. A figure is the most consequential thing an agent
can get wrong — an amount, a window, a date — and it is the easiest thing for a
model to produce plausibly. Every number in the answer must appear in some tool
result or policy decision, in any of the forms that number is legitimately
written.

What this does not catch
------------------------
Term coverage is not entailment. A sentence that inverts a fact it otherwise
matches — "your refund was *not* approved" against a decision that approved it —
shares every content term with its evidence and passes. Negation and
quantity-comparison checks are the obvious next layer; today the evaluation
suite measures the gap rather than the README claiming it does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from support_agent.domain.models import Provenance, ProvenanceKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from support_agent.domain.models import PolicyDecision, ToolResult

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
#: A number as a customer would read it: 30, 8.99, 1,234, 50%.
#:
#: Bounded by "not adjacent to another digit" rather than by ``\b``. A word
#: boundary fails against an ISO timestamp — in ``2026-06-10T00:00:00`` there is
#: no boundary between ``10`` and ``T``, so ``\b`` cannot see the day. The
#: evidence side is full of ISO timestamps and the answer side is full of the
#: dates rendered from them, so that mismatch reported correct answers as
#: containing invented figures.
_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\d.])\d[\d,]*(?:\.\d+)?(?!\d)")
#: An identifier: a token that begins with a letter and contains a digit.
#: ORD-4417, TKT-9021, tkt_8a87f0, RM123456789GB, EVR2233445566.
#:
#: Identifiers are checked *as identifiers* — present in the evidence or not —
#: rather than decomposed into figures. A generated id is a run of hex, and
#: reading "tkt_8a875612" as the numbers 8, 875612 and 5612 reports every
#: escalation as containing invented amounts, which is how a verifier stops
#: being believed and then gets switched off.
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z][\w-]*\d[\w-]*\b")

#: Words that mark a sentence as asserting something about the world rather than
#: managing the conversation. "I'll check that for you" needs no evidence;
#: "your order was delivered on Tuesday" does.
_FACTUAL_MARKERS: Final[frozenset[str]] = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "shall",
        "can",
        "cannot",
        "delivered",
        "shipped",
        "dispatched",
        "refunded",
        "cancelled",
        "placed",
        "eligible",
        "ineligible",
        "approved",
        "rejected",
        "charged",
        "paid",
        "costs",
        "cost",
        "days",
        "day",
        "weeks",
        "week",
        "hours",
        "hour",
        "policy",
        "window",
        "status",
        "amount",
        "total",
        "order",
        "ticket",
        "account",
        "balance",
        "due",
        "within",
        "before",
        "after",
        "expires",
        "expired",
    ]
)

#: Conversational openers and offers, which assert nothing.
_NON_FACTUAL_PREFIXES: Final[tuple[str, ...]] = (
    "i'm sorry",
    "i am sorry",
    "sorry",
    "thanks",
    "thank you",
    "i'll",
    "i will",
    "let me",
    "i can help",
    "happy to",
    "of course",
    "certainly",
    "no problem",
    "would you",
    "could you",
    "can you",
    "please",
    "is there anything",
    "i've raised",
    "i have raised",
    "i've created",
    "i have created",
)

#: Stopwords for the coverage comparison.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "for",
        "from",
        "had",
        "has",
        "have",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "we",
        "us",
        "me",
        "my",
    ]
)

#: Fraction of a sentence's content terms that must appear in one evidence item.
SUPPORT_THRESHOLD: Final[float] = 0.55


def _without_identifiers(text: str) -> str:
    """Text with identifier tokens removed, for figure extraction."""
    return _IDENTIFIER_RE.sub(" ", text)


def _terms(text: str) -> set[str]:
    """Content words, excluding digits.

    Numbers are deliberately left out of the term comparison and checked
    separately by :func:`_number_forms`. Tokenising them here would compare
    "40.00" as the tokens "40" and "00" against a stored "4000" and report a
    correct sentence as unsupported — which is how a verifier earns a reputation
    for false alarms and then gets switched off.
    """
    return {
        word
        for word in _WORD_RE.findall(text.lower())
        if word not in _STOPWORDS and len(word) > 1 and not word.isdigit()
    }


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


def is_factual(sentence: str) -> bool:
    """Whether a sentence asserts something that needs evidence.

    Conservative in the safe direction: when unsure, a sentence is treated as
    factual and must be supported. The cost of that is an occasional
    conversational line needing evidence; the cost of the opposite is an
    invented fact passing unchecked.
    """
    lowered = sentence.lower().strip()
    if any(lowered.startswith(prefix) for prefix in _NON_FACTUAL_PREFIXES):
        # Still factual if it carries a number: "I'll refund you 40.00" is a
        # claim wearing a conversational hat.
        return bool(_NUMBER_RE.search(sentence))
    if lowered.endswith("?"):
        return False
    if _NUMBER_RE.search(sentence) or _IDENTIFIER_RE.search(sentence):
        return True
    return bool(_terms(sentence) & _FACTUAL_MARKERS)


def _number_forms(raw: str) -> set[str]:
    """Every way one number might legitimately be written.

    ``4000`` in a minor-units field is ``40.00`` to a customer, and ``1,234`` and
    ``1234`` are the same figure. Without this the verifier rejects correct
    answers for formatting, which trains everyone to disable it.
    """
    plain = raw.replace(",", "")
    forms = {raw, plain}
    try:
        value = float(plain)
    except ValueError:
        return forms

    if value.is_integer():
        integral = int(value)
        forms.update({str(integral), f"{integral:,}"})
        # Minor units seen as a major-unit amount, and the reverse.
        if integral % 100 == 0:
            forms.add(f"{integral // 100}")
        forms.add(f"{integral / 100:.2f}")
        forms.add(str(integral * 100))
    else:
        forms.add(f"{value:.2f}")
        forms.add(str(round(value * 100)))
    return forms


def _evidence_text(payload: Any) -> str:
    """Flatten a structured payload into searchable text."""
    if isinstance(payload, dict):
        return " ".join(f"{key} {_evidence_text(value)}" for key, value in payload.items())
    if isinstance(payload, list | tuple):
        return " ".join(_evidence_text(item) for item in payload)
    if payload is None:
        return ""
    return str(payload)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One thing the agent is allowed to have learned."""

    kind: ProvenanceKind
    reference: str
    text: str
    terms: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    identifiers: set[str] = field(default_factory=set)

    @staticmethod
    def _numbers(text: str) -> set[str]:
        stripped = _without_identifiers(text)
        return {form for raw in _NUMBER_RE.findall(stripped) for form in _number_forms(raw)}

    @staticmethod
    def _identifiers(text: str) -> set[str]:
        return {token.lower() for token in _IDENTIFIER_RE.findall(text)}

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> Evidence:
        """Build evidence from a successful tool call."""
        text = _evidence_text(result.data)
        return cls(
            kind=ProvenanceKind.TOOL_RESULT,
            reference=result.call_id,
            text=text,
            terms=_terms(text),
            numbers=cls._numbers(text),
            identifiers=cls._identifiers(text),
        )

    @classmethod
    def from_decision(cls, decision: PolicyDecision) -> Evidence:
        """Build evidence from a deterministic policy decision."""
        text = f"{decision.rule} {decision.reason} {_evidence_text(decision.facts)}"
        return cls(
            kind=ProvenanceKind.POLICY_DECISION,
            reference=decision.id,
            text=text,
            terms=_terms(text),
            numbers=cls._numbers(text),
            identifiers=cls._identifiers(text),
        )

    @classmethod
    def from_system(cls, reference: str, text: str) -> Evidence:
        """Build evidence from a fact this system itself produced.

        A ticket reference the run just created is not a claim about the world
        that needs corroborating; it is an outcome the run caused.
        """
        return cls(
            kind=ProvenanceKind.POLICY_DECISION,
            reference=reference,
            text=text,
            terms=_terms(text),
            numbers=cls._numbers(text),
            identifiers=cls._identifiers(text),
        )

    @classmethod
    def from_customer(cls, message: str) -> Evidence:
        """Build evidence from the customer's own words.

        Restating what the customer said is not a claim about the world, so it
        counts as supported. It is recorded with its own provenance kind so an
        auditor can see the difference.
        """
        return cls(
            kind=ProvenanceKind.CUSTOMER_MESSAGE,
            reference="customer",
            text=message,
            terms=_terms(message),
            numbers=cls._numbers(message),
            identifiers=cls._identifiers(message),
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """What the verifier concluded about one draft answer."""

    total_sentences: int
    factual_sentences: int
    supported_sentences: int
    unsupported: tuple[str, ...]
    unsupported_numbers: tuple[str, ...]
    unsupported_identifiers: tuple[str, ...]
    provenance: tuple[Provenance, ...]

    @property
    def supported_ratio(self) -> float:
        """Fraction of factual sentences that traced to evidence."""
        if self.factual_sentences == 0:
            return 1.0
        return self.supported_sentences / self.factual_sentences

    @property
    def passed(self) -> bool:
        """Whether nothing was left unsupported."""
        return not (self.unsupported or self.unsupported_numbers or self.unsupported_identifiers)


# One branch per way a sentence can fail to trace back to evidence. They are
# checked in sequence because the report names which check failed, and that is
# what an operator reads when an answer was withheld.
def verify(  # noqa: PLR0912
    answer: str,
    evidence: Sequence[Evidence],
    *,
    support_threshold: float = SUPPORT_THRESHOLD,
    check_numbers: bool = True,
) -> VerificationReport:
    """Check a draft answer against the evidence the run actually gathered."""
    sentences = split_sentences(answer)
    all_numbers: set[str] = set()
    all_identifiers: set[str] = set()
    for item in evidence:
        all_numbers |= item.numbers
        all_identifiers |= item.identifiers

    supported = 0
    factual = 0
    unsupported: list[str] = []
    provenance: dict[str, Provenance] = {}

    for sentence in sentences:
        if not is_factual(sentence):
            continue
        factual += 1

        sentence_terms = _terms(sentence)
        if not sentence_terms:
            supported += 1
            continue

        best_ratio = 0.0
        best: Evidence | None = None
        for item in evidence:
            overlap = len(sentence_terms & item.terms) / len(sentence_terms)
            if overlap > best_ratio:
                best_ratio, best = overlap, item

        if best is not None and best_ratio >= support_threshold:
            supported += 1
            provenance.setdefault(
                best.reference,
                Provenance(kind=best.kind, reference=best.reference, excerpt=best.text[:200]),
            )
        else:
            unsupported.append(sentence[:300])

    unsupported_numbers: list[str] = []
    unsupported_identifiers: list[str] = []
    if check_numbers:
        for raw in _NUMBER_RE.findall(_without_identifiers(answer)):
            if not (_number_forms(raw) & all_numbers):
                unsupported_numbers.append(raw)
        for token in _IDENTIFIER_RE.findall(answer):
            if token.lower() not in all_identifiers:
                unsupported_identifiers.append(token)

    return VerificationReport(
        total_sentences=len(sentences),
        factual_sentences=factual,
        supported_sentences=supported,
        unsupported=tuple(unsupported),
        unsupported_numbers=tuple(dict.fromkeys(unsupported_numbers)),
        unsupported_identifiers=tuple(dict.fromkeys(unsupported_identifiers)),
        provenance=tuple(provenance.values()),
    )


def strip_unsupported(answer: str, report: VerificationReport) -> str:
    """Remove the sentences the verifier could not support.

    Used by the ``strip`` failure mode. Blunt on purpose: an answer that has had
    sentences removed reads awkwardly, which is a feature — it is visibly a
    degraded answer rather than a confident wrong one.
    """
    if not report.unsupported:
        return answer
    unsupported = {sentence.rstrip(".") for sentence in report.unsupported}
    kept = [
        sentence
        for sentence in split_sentences(answer)
        if sentence.rstrip(".")[:300] not in unsupported
    ]
    return " ".join(kept).strip()


__all__ = [
    "SUPPORT_THRESHOLD",
    "Evidence",
    "VerificationReport",
    "is_factual",
    "split_sentences",
    "strip_unsupported",
    "verify",
]
