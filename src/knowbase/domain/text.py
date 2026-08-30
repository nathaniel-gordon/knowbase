"""Pure text services owned by the domain: tokenisation, chunking, sentence splitting.

These live in the core rather than in an adapter because they define what a *passage* is,
and every index adapter must agree on that definition for retrieval scores to be comparable
in an A/B run.

Nothing here touches IO, the clock, or randomness, so the whole module is referentially
transparent and testable without any fixture.
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

from .model import Document, Passage

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

DEFAULT_TARGET_WORDS = 120
DEFAULT_OVERLAP = 25

STOPWORDS = frozenset(
    "a an the and or but if then else for to of in on at by with from as is are was were be been "
    "this that these those it its they them their we you your i he she his her not no do does did "
    "can could should would will shall may might must have has had what which who whom how when "
    "where why all each any both more most other some such only own same so than too very".split()
)


def stem(token: str) -> str:
    """Light suffix stripping so 'passwords' matches 'password' and 'policies' matches 'policy'.

    A full Porter stemmer over-stems policy jargon ("retention" -> "retent"), which hurts
    precision on this corpus, so the rule set stays deliberately conservative.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords and 1-character noise, then stem."""
    return [stem(t) for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


def sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation; used by the extractive generator."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def chunk_document(document: Document,
                   target_words: int = DEFAULT_TARGET_WORDS,
                   overlap: int = DEFAULT_OVERLAP) -> list[Passage]:
    """Split a document into overlapping word windows that respect paragraph breaks.

    Overlap matters: a policy sentence that straddles a window boundary would otherwise be
    unretrievable. ``target_words``/``overlap`` default to 120/25.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if not 0 <= overlap < target_words:
        raise ValueError("overlap must be in [0, target_words)")

    words: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(document.text):
        paragraph = paragraph.strip()
        if paragraph:
            words.extend(paragraph.split())
    if not words:
        words = document.text.split()

    passages: list[Passage] = []
    start = 0
    while start < len(words):
        window = words[start:start + target_words]
        passages.append(Passage(doc_id=document.doc_id,
                                ordinal=len(passages),
                                text=" ".join(window),
                                title=document.title or document.doc_id,
                                origin=document.origin))
        if start + target_words >= len(words):
            break
        start += target_words - overlap
    return passages


def title_of(text: str, fallback: str) -> str:
    """First markdown heading (or first line) of a document, used as the citation label."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.lstrip("#").strip() or fallback
    return fallback


def corpus_fingerprint(passages: Sequence[Passage]) -> str:
    """A stable identity for one ingested corpus.

    The answer cache is keyed partly on this. Without it a cached answer computed over corpus
    A could be served while corpus B is loaded -- the reply would carry A's prose and, because
    citations are resolved against today's retrieval, no citations at all. Hashing the ordered
    passage ids *and* their text means both a different corpus and an edited document produce
    a different key.
    """
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(passage.passage_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(passage.text.encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()[:32]
