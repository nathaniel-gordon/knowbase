"""Deterministic extractive generator -- the always-available answer path.

It ranks sentences drawn from the retrieved passages by how much of the *question* they cover
and stitches the best few back together in retrieval order. No model, no network, no
randomness: given the same passages it always produces the same prose, which is what makes the
HTTP eval reproducible.

It reports *which* passages its sentences came from, so the use-case can cite only those
instead of blanket-citing everything retrieved.

Two rules keep the answers honest rather than merely long. A sentence must cover a real
fraction of the question, not just share one word with it, and it must score within reach of
the best sentence found. Without them the generator always took three sentences, so an
expense-policy question came back with a sentence about phishing attached -- and the citation
mechanism then rubber-stamped the IT security policy as a source.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from ...domain.model import Generated, Passage
from ...domain.text import sentences, tokenize
from ...domain.usecases import ABSTENTION

MIN_SENTENCE_CHARS = 25
MAX_SENTENCES = 3

#: Retrieval rank is discounted multiplicatively, not subtracted. On a small corpus an
#: incidental verb ("how long does a refund *take*" against "shipping *takes* 5 days") gives a
#: sentence exactly the same raw overlap as the sentence that actually answers the question,
#: and no lexical rarity signal separates them -- the only real evidence is that the index put
#: the other passage first. Dividing by the rank makes that evidence count: a sentence from the
#: second passage must match the question about twice as well as the leading sentence to
#: survive the relative floor below, instead of trailing it by a fixed 0.15.
RANK_DECAY = 1.0

#: A sentence must cover at least this share of the question's content tokens, so a single
#: incidental word shared with a long question is not enough on its own. Kept deliberately low:
#: the answer sentence often carries the fact while the question's other words live in the
#: document title ("what password length is required by the security policy" against
#: "Passwords must be at least 14 characters").
MIN_QUERY_COVERAGE = 0.20

#: The floor that actually stops padding: a sentence must score within reach of the best one
#: found, so a strong match is not topped up with two weak ones to reach MAX_SENTENCES. This
#: is what keeps the phishing sentence out of an answer about expense reports.
RELATIVE_FLOOR = 0.55

_HEADING_MARK_RE = re.compile(r"^#{1,6}\s*")


class ExtractiveGenerator:
    """Offline sentence-selection generator."""

    name = "extractive"

    def __init__(self, max_sentences: int = MAX_SENTENCES,
                 min_coverage: float = MIN_QUERY_COVERAGE,
                 relative_floor: float = RELATIVE_FLOOR) -> None:
        self._max_sentences = max_sentences
        self._min_coverage = min_coverage
        self._relative_floor = relative_floor

    def generate(self, question: str, passages: Sequence[Passage]) -> Generated:
        if not passages:
            return Generated(text=ABSTENTION, cited=())

        q_tokens = set(tokenize(question))
        if not q_tokens:
            return Generated(text=ABSTENTION, cited=())

        scored = self._score(q_tokens, passages)
        if not scored:
            return Generated(text=ABSTENTION, cited=())

        scored.sort(key=lambda t: (-t[0], t[2]))
        best = scored[0][0]
        keep = [row for row in scored[:self._max_sentences]
                if row[0] >= best * self._relative_floor]
        chosen = sorted(keep, key=lambda t: t[2])
        text = " ".join(s for _, _, _, s in chosen)
        cited = sorted({idx + 1 for _, idx, _, _ in chosen})
        return Generated(text=text, cited=tuple(cited))

    # --------------------------------------------------------------- internals

    def _score(self, q_tokens: set[str],
               passages: Sequence[Passage]) -> list[tuple[float, int, int, str]]:
        """(score, passage index, discovery order, sentence) for every eligible sentence."""
        scored: list[tuple[float, int, int, str]] = []
        seen: set[str] = set()
        seq = 0
        for idx, passage in enumerate(passages):
            for raw in sentences(passage.text):
                sentence = self._strip_heading(raw, passage.title)
                key = sentence.lower()
                if key in seen or len(sentence) < MIN_SENTENCE_CHARS:
                    continue
                seen.add(key)
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap / len(q_tokens) < self._min_coverage:
                    continue
                # Rank decay: a weaker sentence from the top passage still beats an equally
                # matching sentence from a passage the index ranked far lower.
                scored.append((overlap / (1.0 + RANK_DECAY * idx), idx, seq, sentence))
                seq += 1
        return scored

    @staticmethod
    def _strip_heading(sentence: str, title: str) -> str:
        """Remove a leading markdown title from a sentence.

        Chunking joins lines with spaces, so a document's first sentence arrives glued to its
        own heading and answers used to open '# Travel and Expense Policy Expenses must ...'.
        The title is already carried on the passage and printed with the citation, so it is
        cut here rather than quoted back. A sentence that is *only* the heading becomes empty
        and drops out on the length check.
        """
        text = _HEADING_MARK_RE.sub("", sentence.strip())
        if title and text.lower().startswith(title.lower()):
            text = text[len(title):].lstrip(" :-" + "\u2013\u2014")
        return text.strip()
