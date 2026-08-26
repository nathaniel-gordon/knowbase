"""Contract battery for GeneratorPort.

The central promise is *grounding*: a generator may only cite passages it was handed, and it
must abstain rather than invent when it has none. Because both the extractive adapter and the
Claude adapter are held to this battery, the citation guarantee is a property of the port, not
of one implementation.
"""
from __future__ import annotations

import pytest

from kb.domain.model import Generated, Passage
from kb.domain.usecases import ABSTENTION
from kb.ports import GeneratorPort

LEAVE_PASSAGE = Passage(
    doc_id="leave", ordinal=0, title="Time Off Rules", origin="fixture://leave",
    text=("Employees receive 21 paid vacation days each year and accrue them monthly. "
          "Unused vacation expires on March 31 of the following year."))
PASSWORD_PASSAGE = Passage(
    doc_id="passwords", ordinal=0, title="Password Rules", origin="fixture://passwords",
    text=("Passwords must be at least 14 characters long and rotated every 180 days. "
          "Reused passwords are rejected by the identity provider."))
DISTRACTOR_PASSAGE = Passage(
    doc_id="vpn", ordinal=0, title="Remote Access Rules", origin="fixture://vpn",
    text=("The GlobalConnect VPN client is required for remote access to internal systems. "
          "Set the router MTU to 1400 if the tunnel drops repeatedly."))


class GeneratorContract:
    """Every GeneratorPort adapter must pass all of these.

    Subclasses provide a ``generator`` fixture.
    """

    @pytest.fixture
    def generator(self) -> GeneratorPort:
        raise NotImplementedError("contract subclass must provide a 'generator' fixture")

    def test_satisfies_the_protocol(self, generator):
        assert isinstance(generator, GeneratorPort)
        assert isinstance(generator.name, str) and generator.name == generator.name.lower()

    def test_returns_a_generated_value(self, generator):
        result = generator.generate("How many vacation days?", [LEAVE_PASSAGE])
        assert isinstance(result, Generated)
        assert isinstance(result.text, str) and result.text.strip()
        assert isinstance(result.cited, tuple)

    def test_abstains_without_passages(self, generator):
        result = generator.generate("How many vacation days?", [])
        assert result.text == ABSTENTION
        assert result.cited == ()

    def test_answers_from_the_supplied_passage(self, generator):
        result = generator.generate("How many vacation days do employees get each year?",
                                    [LEAVE_PASSAGE])
        assert "21" in result.text, f"{generator.name} lost the fact: {result.text!r}"
        assert result.cited == (1,)

    def test_citations_stay_within_range(self, generator):
        passages = [LEAVE_PASSAGE, PASSWORD_PASSAGE, DISTRACTOR_PASSAGE]
        result = generator.generate("What password length is required?", passages)
        assert result.cited, "an answered question must cite something"
        assert all(1 <= marker <= len(passages) for marker in result.cited)
        assert len(set(result.cited)) == len(result.cited), "markers must not repeat"
        assert list(result.cited) == sorted(result.cited)

    def test_citations_never_exceed_a_shrinking_passage_list(self, generator):
        result = generator.generate("What password length is required?", [PASSWORD_PASSAGE])
        assert all(marker == 1 for marker in result.cited)

    def test_is_deterministic(self, generator):
        passages = [LEAVE_PASSAGE, PASSWORD_PASSAGE]
        first = generator.generate("How long must a password be?", passages)
        second = generator.generate("How long must a password be?", passages)
        assert first == second

    @pytest.mark.parametrize("question", ["qwertyuiop zxcvbnm asdfgh?",
                                          "What password length is required?"])
    def test_an_answer_is_served_only_when_it_is_attributable(self, generator, question):
        """The port's core invariant: no citations <=> explicit abstention, never a guess."""
        result = generator.generate(question, [DISTRACTOR_PASSAGE, PASSWORD_PASSAGE])
        assert (result.cited == ()) == (result.text == ABSTENTION)
