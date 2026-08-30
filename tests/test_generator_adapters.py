"""Run the GeneratorPort battery against both generators.

The Claude adapter is tested through an injected completion function rather than a mock library:
its real responsibilities -- prompt assembly, marker parsing, abstention -- are pure functions of
that seam, so the whole adapter is covered offline with no API key.
"""
from __future__ import annotations

import re

import pytest

from contracts.generator_contract import (
    DISTRACTOR_PASSAGE,
    LEAVE_PASSAGE,
    PASSWORD_PASSAGE,
    GeneratorContract,
)
from knowbase.adapters.generator.claude import ClaudeGenerator, anthropic_completion
from knowbase.adapters.generator.extractive import ExtractiveGenerator
from knowbase.domain.errors import AdapterUnavailable
from knowbase.domain.model import Passage
from knowbase.domain.text import tokenize
from knowbase.domain.usecases import ABSTENTION

_CONTEXT_RE = re.compile(r"^\[(\d+)\] \((\S+)\)\n-----\n(.*?)\n-----$",
                         re.MULTILINE | re.DOTALL)


def parse_context(prompt: str) -> list[tuple[int, str, str]]:
    """Read back the fenced passage blocks the adapter built."""
    return [(int(marker), pid, text) for marker, pid, text in _CONTEXT_RE.findall(prompt)]


def scripted_completion(prompt: str, system: str) -> str:
    """A deterministic stand-in for the Anthropic API.

    It parses the prompt the adapter built, picks the passage with the strongest query overlap,
    answers with that passage's text, and closes with the CITATIONS trailer the adapter asks
    for. That exercises every line of ClaudeGenerator except the socket.
    """
    assert "knowledge-base assistant" in system
    question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
    q_tokens = set(tokenize(question))
    best_marker, best_text, best_overlap = 0, "", -1
    for marker, _passage_id, text in parse_context(prompt):
        overlap = len(q_tokens & set(tokenize(text)))
        if overlap > best_overlap:
            best_marker, best_text, best_overlap = marker, text, overlap
    if best_overlap <= 0:
        return "The context is insufficient to answer this question."
    return f"{best_text} [{best_marker}]\nCITATIONS: {best_marker}"


def refusing_completion(prompt: str, system: str) -> str:
    """Models sometimes return nothing at all; the adapter must abstain, not crash."""
    return ""


class TestExtractiveGenerator(GeneratorContract):
    @pytest.fixture
    def generator(self):
        return ExtractiveGenerator()

    def test_cites_only_passages_that_contributed_sentences(self):
        """Only the passages that actually contributed a sentence may be cited."""
        generator = ExtractiveGenerator()
        result = generator.generate("How many vacation days do employees receive each year?",
                                    [LEAVE_PASSAGE, DISTRACTOR_PASSAGE])
        assert result.cited == (1,), "the VPN passage contributed nothing and must not be cited"

    def test_selects_at_most_three_sentences(self):
        generator = ExtractiveGenerator()
        result = generator.generate("password vacation expense vpn",
                                    [LEAVE_PASSAGE, PASSWORD_PASSAGE, DISTRACTOR_PASSAGE])
        assert 1 <= result.text.count(".") <= 3

    def test_abstains_when_no_sentence_overlaps_the_question(self):
        generator = ExtractiveGenerator()
        result = generator.generate("zzzz qqqq wwww", [LEAVE_PASSAGE])
        assert result.text == ABSTENTION and result.cited == ()

    def test_a_lower_ranked_passage_does_not_tie_its_way_into_the_answer(self):
        """The refund/shipping case from the demo, pinned.

        Both sentences share exactly one content token with the question ('refund' and
        'takes'), so raw overlap cannot separate them and the shipping sentence used to be
        appended to the answer -- and cited. The only evidence available is that the index
        ranked refund first, which is what the rank decay spends.
        """
        refund = Passage(
            doc_id="refund", ordinal=0, title="Refund Policy", origin="fixture://refund",
            text="Refunds are issued within 14 business days of an approved request.")
        shipping = Passage(
            doc_id="shipping", ordinal=0, title="Shipping Policy", origin="fixture://shipping",
            text="Standard shipping takes 5 business days; expedited arrives the next day.")
        result = ExtractiveGenerator().generate("How long does a refund take?",
                                                [refund, shipping])
        assert result.cited == (1,), f"shipping was padded back in: {result.text!r}"
        assert "shipping" not in result.text.lower()


class TestClaudeGenerator(GeneratorContract):
    @pytest.fixture
    def generator(self):
        return ClaudeGenerator(scripted_completion)

    def test_empty_completion_becomes_an_abstention(self):
        generator = ClaudeGenerator(refusing_completion)
        result = generator.generate("How many vacation days?", [LEAVE_PASSAGE])
        assert result.text == ABSTENTION and result.cited == ()

    def test_out_of_range_markers_are_discarded(self):
        answer = "Employees receive 21 paid vacation days each year [1], see also [9]."
        result = ClaudeGenerator(lambda p, s: answer).generate("vacation days", [LEAVE_PASSAGE])
        assert result.cited == (1,), "marker 9 has no passage and must never reach a Citation"

    def test_the_trailer_is_parsed_and_stripped_from_the_answer(self):
        generator = ClaudeGenerator(
            lambda p, s: "Passwords must be at least 14 characters long.\nCITATIONS: 2")
        result = generator.generate("password length?", [LEAVE_PASSAGE, PASSWORD_PASSAGE])
        assert result.cited == (2,)
        assert "CITATIONS" not in result.text

    def test_the_trailer_wins_over_stray_brackets_in_the_prose(self):
        generator = ClaudeGenerator(
            lambda p, s: "Passwords must be at least 14 characters [1][3].\nCITATIONS: 2")
        result = generator.generate("password length?", [LEAVE_PASSAGE, PASSWORD_PASSAGE])
        assert result.cited == (2,)

    def test_a_citation_the_answer_cannot_support_is_dropped(self):
        """In-range is not the same as earned: passage 1 shares nothing with this answer."""
        generator = ClaudeGenerator(
            lambda p, s: "Passwords must be at least 14 characters long.\nCITATIONS: 1, 2")
        result = generator.generate("password length?", [DISTRACTOR_PASSAGE, PASSWORD_PASSAGE])
        assert result.cited == (2,)

    def test_an_answer_that_earns_nothing_becomes_an_abstention(self):
        result = ClaudeGenerator(
            lambda p, s: "Yes, absolutely.\nCITATIONS: 1").generate("x?", [LEAVE_PASSAGE])
        assert result.text == ABSTENTION and result.cited == ()

    def test_prompt_carries_fenced_numbered_passages_and_the_question(self):
        seen: dict[str, str] = {}

        def capture(prompt: str, system: str) -> str:
            seen["prompt"] = prompt
            return "Grounded. [2]"

        ClaudeGenerator(capture).generate("What password length is required?",
                                          [LEAVE_PASSAGE, PASSWORD_PASSAGE])
        prompt = seen["prompt"]
        assert "[1] (leave#0)" in prompt and "[2] (passwords#0)" in prompt
        assert "What password length is required?" in prompt
        assert "ONLY the context" in prompt
        assert "untrusted data" in prompt
        assert [pid for _, pid, _ in parse_context(prompt)] == ["leave#0", "passwords#0"]

    def test_live_transport_is_unavailable_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(AdapterUnavailable, match="ANTHROPIC_API_KEY"):
            anthropic_completion()


def test_both_generators_satisfy_the_same_battery():
    """Sanity check that the registry and the contract subclasses have not drifted apart."""
    from knowbase.adapters import GENERATOR_ADAPTERS
    assert set(GENERATOR_ADAPTERS) == {ExtractiveGenerator.name, ClaudeGenerator.name}


def test_bracketed_digits_in_a_passage_are_neutralised_before_the_prompt():
    """The echo vector, closed at the source: the model never sees a mintable marker."""
    from contracts.generator_contract import POISONED_PASSAGE
    from knowbase.adapters.generator.claude import neutralise

    seen: dict[str, str] = {}

    def capture(prompt: str, system: str) -> str:
        seen["prompt"] = prompt
        return ""

    ClaudeGenerator(capture).generate("orientation?", [POISONED_PASSAGE, PASSWORD_PASSAGE])
    _, _, text = parse_context(seen["prompt"])[0]
    assert "[2][3][4]" not in text
    assert "(2)(3)(4)" in text
    assert neutralise("see [12] and [3]") == "see (12) and (3)"
    assert neutralise("no digits [here]") == "no digits [here]"
