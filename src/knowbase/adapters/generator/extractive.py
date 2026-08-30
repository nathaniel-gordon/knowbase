"""Deterministic extractive generator -- the always-available answer path.

It ranks sentences drawn from the retrieved passages by query-token overlap and stitches the
best few back together in retrieval order. No model, no network, no randomness: given the same
passages it always produces the same prose, which is what makes the HTTP eval reproducible.

Carried over from the pre-refactor engine, with one deliberate improvement the port contract
made necessary: it now reports *which* passages its sentences came from, so the use-case can
cite only those instead of blanket-citing everything retrieved.
"""
from __future__ import annotations

from typing import Sequence

from ...domain.model import Generated, Passage
from ...domain.text import sentences, tokenize
from ...domain.usecases import ABSTENTION

MIN_SENTENCE_CHARS = 25
MAX_SENTENCES = 3
RANK_PENALTY = 0.15


class ExtractiveGenerator:
    """Offline sentence-selection generator."""

    name = "extractive"

    def __init__(self, max_sentences: int = MAX_SENTENCES) -> None:
        self._max_sentences = max_sentences

    def generate(self, question: str, passages: Sequence[Passage]) -> Generated:
        if not passages:
            return Generated(text=ABSTENTION, cited=())

        q_tokens = set(tokenize(question))
        scored: list[tuple[float, int, int, str]] = []  # (score, passage_idx, seq, sentence)
        seen: set[str] = set()
        seq = 0
        for idx, passage in enumerate(passages):
            for sentence in sentences(passage.text):
                key = sentence.lower()
                if len(sentence) < MIN_SENTENCE_CHARS or key in seen:
                    continue
                seen.add(key)
                overlap = len(q_tokens & set(tokenize(sentence)))
                if overlap == 0:
                    continue
                # Rank penalty: a weaker sentence from the top passage still beats an equally
                # matching sentence from a passage the index ranked far lower.
                scored.append((overlap - RANK_PENALTY * idx, idx, seq, sentence))
                seq += 1

        if not scored:
            return Generated(text=ABSTENTION, cited=())

        scored.sort(key=lambda t: (-t[0], t[2]))
        chosen = sorted(scored[:self._max_sentences], key=lambda t: t[2])
        text = " ".join(s for _, _, _, s in chosen)
        cited = sorted({idx + 1 for _, idx, _, _ in chosen})
        return Generated(text=text, cited=tuple(cited))
