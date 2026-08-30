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
from knowbase.domain.text import tokenize
from knowbase.domain.usecases import ABSTENTION

_CONTEXT_RE = re.compile(r"^\[(\d+)\] \((\S+)\) (.*)$", re.MULTILINE)


def scripted_completion(prompt: str, system: str) -> str:
    """A deterministic stand-in for the Anthropic API.

    It parses the prompt the adapter built, picks the passage with the strongest query overlap,
    and answers in the cited style the real model is instructed to use. That exercises every
    line of ClaudeGenerator except the socket.
    """
    assert "knowledge-base assistant" in system
    question = prompt.split("Question:", 1)[1].split("\n", 1)[0].strip()
    q_tokens = set(tokenize(question))
    best_marker, best_text, best_overlap = 0, "", -1
    for marker, _passage_id, text in _CONTEXT_RE.findall(prompt):
        overlap = len(q_tokens & set(tokenize(text)))
        if overlap > best_overlap:
            best_marker, best_text, best_overlap = int(marker), text, overlap
    if best_overlap <= 0:
        return "The context is insufficient to answer this question."
    return f"{best_text} [{best_marker}]"


def refusing_completion(prompt: str, system: str) -> str:
    """Models sometimes return nothing at all; the adapter must abstain, not crash."""
    return ""


class TestExtractiveGenerator(GeneratorContract):
    @pytest.fixture
    def generator(self):
        return ExtractiveGenerator()

    def test_cites_only_passages_that_contributed_sentences(self):
        """The pre-refactor engine cited every retrieved passage; this one cites what it used."""
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


class TestClaudeGenerator(GeneratorContract):
    @pytest.fixture
    def generator(self):
        return ClaudeGenerator(scripted_completion)

    def test_empty_completion_becomes_an_abstention(self):
        generator = ClaudeGenerator(refusing_completion)
        result = generator.generate("How many vacation days?", [LEAVE_PASSAGE])
        assert result.text == ABSTENTION and result.cited == ()

    def test_out_of_range_markers_are_discarded(self):
        generator = ClaudeGenerator(lambda p, s: "Answer grounded in [1] and also in [9].")
        result = generator.generate("anything", [LEAVE_PASSAGE])
        assert result.cited == (1,), "marker 9 has no passage and must never reach a Citation"

    def test_prompt_carries_numbered_passages_and_the_question(self):
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

    def test_live_transport_is_unavailable_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(AdapterUnavailable, match="ANTHROPIC_API_KEY"):
            anthropic_completion()


def test_both_generators_satisfy_the_same_battery():
    """Sanity check that the registry and the contract subclasses have not drifted apart."""
    from knowbase.adapters import GENERATOR_ADAPTERS
    assert set(GENERATOR_ADAPTERS) == {ExtractiveGenerator.name, ClaudeGenerator.name}
