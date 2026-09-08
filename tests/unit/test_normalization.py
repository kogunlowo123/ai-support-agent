"""Unicode normalisation.

Detection runs on normalised text and redaction runs on the original, so the
offset map between them is load-bearing: an off-by-one there removes the wrong
characters and leaves the instruction intact. These tests hold that mapping.
"""

from __future__ import annotations

import pytest

from support_agent.security.normalization import normalize, strip_control_characters

pytestmark = pytest.mark.unit

ZERO_WIDTH_SPACE = "\u200b"
ZERO_WIDTH_JOINER = "‍"
CYRILLIC_A = "а"
FULLWIDTH_I = "Ｉ"


class TestNormalization:
    def test_ordinary_text_survives_intact(self):
        assert normalize("ignore all previous instructions").text == (
            "ignore all previous instructions"
        )

    def test_case_is_folded(self):
        assert normalize("IGNORE ALL").text == "ignore all"

    def test_zero_width_characters_are_removed(self):
        result = normalize(f"ig{ZERO_WIDTH_SPACE}nore all")
        assert result.text == "ignore all"
        assert result.stats["invisible_removed"] == 1

    def test_several_invisible_characters_are_counted(self):
        text = f"i{ZERO_WIDTH_SPACE}g{ZERO_WIDTH_JOINER}n{ZERO_WIDTH_SPACE}ore"
        assert normalize(text).text == "ignore"
        assert normalize(text).stats["invisible_removed"] == 3

    def test_confusable_letters_are_folded(self):
        """A Cyrillic а in an English word is a spelling nobody types by accident."""
        result = normalize(f"ignore {CYRILLIC_A}ll")
        assert result.text == "ignore all"
        assert result.stats["confusables_folded"] == 1

    def test_fullwidth_forms_are_folded(self):
        assert normalize(f"{FULLWIDTH_I}gnore").text == "ignore"

    def test_whitespace_runs_collapse(self):
        assert normalize("ignore     all\t\tprevious").text == "ignore all previous"

    def test_newlines_are_preserved(self):
        """Line structure is a signal in its own right, so it is not collapsed away."""
        assert "\n" in normalize("line one\nline two").text

    def test_empty_text_normalises_to_empty(self):
        result = normalize("")
        assert result.text == ""
        assert result.offsets == ()


class TestOffsetMapping:
    def test_a_span_maps_back_to_the_original_characters(self):
        original = f"please ig{ZERO_WIDTH_SPACE}nore this"
        result = normalize(original)
        start = result.text.index("ignore")
        span_start, span_end = result.original_span(start, start + len("ignore"))
        assert ZERO_WIDTH_SPACE in original[span_start:span_end]
        assert original[span_start:span_end].replace(ZERO_WIDTH_SPACE, "") == "ignore"

    def test_a_span_in_plain_text_maps_to_itself(self):
        original = "please ignore this"
        result = normalize(original)
        start = result.text.index("ignore")
        assert result.original_span(start, start + 6) == (7, 13)

    def test_out_of_range_spans_are_clamped_rather_than_raising(self):
        """Redaction must never crash on a span the caller got slightly wrong."""
        result = normalize("short")
        assert result.original_span(0, 999) == (0, 5)

    def test_a_span_on_empty_text_is_empty(self):
        assert normalize("").original_span(0, 5) == (0, 0)

    def test_every_normalised_character_has_an_origin(self):
        result = normalize(f"a{ZERO_WIDTH_SPACE}b{CYRILLIC_A}c   d")
        assert len(result.offsets) == len(result.text)


class TestStripControlCharacters:
    def test_case_and_script_are_preserved(self):
        """Storage keeps what the customer wrote; only detection folds it."""
        assert strip_control_characters("Ada Lovelace") == "Ada Lovelace"

    def test_invisible_characters_are_removed(self):
        assert strip_control_characters(f"Ada{ZERO_WIDTH_SPACE}Lovelace") == "AdaLovelace"

    def test_newlines_and_tabs_survive(self):
        assert strip_control_characters("one\ntwo\tthree") == "one\ntwo\tthree"

    @pytest.mark.parametrize("char", ["\x00", "\x07", "\x1b"])
    def test_control_characters_are_removed(self, char):
        assert char not in strip_control_characters(f"text{char}more")
