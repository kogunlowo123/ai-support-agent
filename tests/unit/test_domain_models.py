"""The transition table is the agent's safety argument. These tests defend it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from support_agent.domain.models import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    AgentState,
    Answer,
    Conversation,
    Intent,
    IntentResult,
    Provenance,
    ProvenanceKind,
    new_id,
)

pytestmark = pytest.mark.unit


class TestTransitionTable:
    def test_every_state_is_declared(self):
        assert set(ALLOWED_TRANSITIONS) == set(AgentState)

    def test_terminal_states_have_no_successors(self):
        for state in TERMINAL_STATES:
            assert ALLOWED_TRANSITIONS[state] == frozenset(), state

    def test_non_terminal_states_have_successors(self):
        for state, successors in ALLOWED_TRANSITIONS.items():
            if state not in TERMINAL_STATES:
                assert successors, f"{state} is not terminal but leads nowhere"

    def test_every_state_can_reach_a_terminal_state(self):
        """No state may trap a run. A run that cannot end is a hung request."""
        for start in AgentState:
            assert self._reaches_terminal(start), f"{start} cannot reach a terminal state"

    def test_escalation_is_reachable_from_every_working_state(self):
        """Handing over to a person is the universal safe exit.

        Every state in which the agent is still deciding something must be able
        to reach ESCALATED. A working state that cannot escalate crashes at
        exactly the moment it decides to be careful, which is how the machine
        raised InvalidTransitionError when composition produced no text.
        """
        working = {
            AgentState.CLASSIFIED,
            AgentState.GATHERING,
            AgentState.DECIDING,
            AgentState.COMPOSING,
            AgentState.VERIFYING,
        }
        for state in working:
            assert AgentState.ESCALATED in ALLOWED_TRANSITIONS[state], state

    def test_failure_is_reachable_from_every_working_state(self):
        for state, successors in ALLOWED_TRANSITIONS.items():
            if state not in TERMINAL_STATES:
                assert AgentState.FAILED in successors, state

    def test_answering_requires_passing_through_verification(self):
        """ANSWERED is reachable only from VERIFYING. Nothing skips the check."""
        predecessors = {
            state
            for state, successors in ALLOWED_TRANSITIONS.items()
            if AgentState.ANSWERED in successors
        }
        assert predecessors == {AgentState.VERIFYING}

    def test_no_state_transitions_to_received(self):
        """RECEIVED is an entry point. Re-entering it would restart a run."""
        for successors in ALLOWED_TRANSITIONS.values():
            assert AgentState.RECEIVED not in successors

    @staticmethod
    def _reaches_terminal(start: AgentState) -> bool:
        seen: set[AgentState] = set()
        frontier = [start]
        while frontier:
            state = frontier.pop()
            if state in TERMINAL_STATES:
                return True
            if state in seen:
                continue
            seen.add(state)
            frontier.extend(ALLOWED_TRANSITIONS[state])
        return False


class TestIntentResult:
    def test_unknown_is_never_confident(self):
        result = IntentResult(intent=Intent.UNKNOWN, confidence=1.0)
        assert result.is_confident is False

    def test_a_clear_winner_is_confident(self):
        result = IntentResult(
            intent=Intent.ORDER_STATUS,
            confidence=0.8,
            alternatives=((Intent.REFUND_REQUEST, 0.2),),
        )
        assert result.is_confident is True

    def test_a_narrow_margin_is_not_confident(self):
        """Two intents scoring almost the same is a message to ask about."""
        result = IntentResult(
            intent=Intent.ORDER_STATUS,
            confidence=0.5,
            alternatives=((Intent.REFUND_REQUEST, 0.45),),
        )
        assert result.is_confident is False

    def test_a_low_score_is_not_confident_even_unopposed(self):
        result = IntentResult(intent=Intent.ORDER_STATUS, confidence=0.2)
        assert result.is_confident is False


class TestAnswer:
    def test_terminal_answers_report_themselves_as_terminal(self):
        answer = Answer(text="done", intent=Intent.RETURN_POLICY, state=AgentState.ANSWERED)
        assert answer.is_terminal is True

    def test_a_mid_run_state_is_not_terminal(self):
        answer = Answer(text="", intent=Intent.RETURN_POLICY, state=AgentState.GATHERING)
        assert answer.is_terminal is False

    def test_answers_are_immutable(self):
        answer = Answer(text="done", intent=Intent.RETURN_POLICY, state=AgentState.ANSWERED)
        with pytest.raises(ValidationError, match=r"frozen|immutable"):
            answer.text = "something else"  # type: ignore[misc]


class TestConversation:
    def test_adding_a_turn_returns_a_new_conversation(self):
        original = Conversation(id=new_id("conv"), tenant_id="acme")
        updated = original.with_turn("customer", "hello")
        assert len(original.turns) == 0
        assert len(updated.turns) == 1
        assert updated.turns[0].role == "customer"

    def test_recent_returns_the_tail_oldest_first(self):
        conversation = Conversation(id=new_id("conv"), tenant_id="acme")
        for index in range(10):
            conversation = conversation.with_turn("customer", f"message {index}")
        recent = conversation.recent(limit=3)
        assert [turn.text for turn in recent] == ["message 7", "message 8", "message 9"]

    def test_recent_with_a_zero_limit_returns_nothing(self):
        conversation = Conversation(id=new_id("conv"), tenant_id="acme").with_turn("customer", "x")
        assert conversation.recent(limit=0) == ()


class TestIdentifiers:
    def test_identifiers_carry_their_prefix(self):
        assert new_id("conv").startswith("conv_")

    def test_identifiers_do_not_repeat(self):
        assert len({new_id("tkt") for _ in range(1000)}) == 1000


class TestProvenance:
    def test_provenance_records_where_a_statement_came_from(self):
        item = Provenance(kind=ProvenanceKind.TOOL_RESULT, reference="lookup_order:ORD-1001")
        assert item.kind is ProvenanceKind.TOOL_RESULT
        assert "ORD-1001" in item.reference
