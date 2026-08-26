"""Claude generator adapter.

The network call is itself injected as a ``CompletionFn``. That extra seam exists so the
adapter's real responsibilities -- prompt assembly, citation-marker parsing, abstention, refusal
handling -- are covered by the same contract battery as every other generator, offline, with no
API key and no mocking framework. ``anthropic_completion()`` builds the live transport and
raises :class:`AdapterUnavailable` when credentials or the SDK are missing, which is the signal
the composition root uses to fall back to the extractive adapter.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Sequence

from ...domain.errors import AdapterUnavailable
from ...domain.model import Generated, Passage
from ...domain.usecases import ABSTENTION

CompletionFn = Callable[[str, str], str]
"""(prompt, system) -> completion text. Returns '' when the model declines to answer."""

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = (
    "You are a precise enterprise knowledge-base assistant. Answer only from the supplied "
    "context passages, cite every claim with its passage number in square brackets, and say "
    "the context is insufficient rather than guessing."
)


def anthropic_completion(model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> CompletionFn:
    """Build a live Claude completion function, or raise ``AdapterUnavailable``.

    Called by the composition root only; failure here is expected and handled, because the
    service must remain fully functional with no credentials at all.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise AdapterUnavailable("ANTHROPIC_API_KEY is not set")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the host environment
        raise AdapterUnavailable("anthropic SDK not installed") from exc

    client = anthropic.Anthropic()

    def complete(prompt: str, system: str) -> str:
        try:
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}])
        except anthropic.APIConnectionError as exc:
            raise AdapterUnavailable(f"connection error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise AdapterUnavailable(f"API error {exc.status_code}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return complete


class ClaudeGenerator:
    """Grounded generation over retrieved passages via an injected completion function."""

    name = "claude"

    def __init__(self, completion: CompletionFn, model: str = DEFAULT_MODEL) -> None:
        self._completion = completion
        self.model = model

    def generate(self, question: str, passages: Sequence[Passage]) -> Generated:
        if not passages:
            return Generated(text=ABSTENTION, cited=())

        context = "\n\n".join(
            f"[{i}] ({p.passage_id}) {p.text}" for i, p in enumerate(passages, start=1))
        prompt = (f"Context passages:\n{context}\n\nQuestion: {question}\n\n"
                  "Answer using ONLY the context above. Cite passage numbers like [1]. "
                  "If the context is insufficient, say so explicitly.")
        text = (self._completion(prompt, SYSTEM_PROMPT) or "").strip()
        if not text:
            # Empty completion means a refusal or an empty response; abstain rather than
            # emit an unsupported answer.
            return Generated(text=ABSTENTION, cited=())

        cited = sorted({int(m) for m in _MARKER_RE.findall(text)
                        if 1 <= int(m) <= len(passages)})
        if not cited:
            # No usable marker means the model either declined or answered from outside the
            # context. Either way the answer is unattributable, so it is not served: the port
            # contract requires that an answer with no citations is an explicit abstention.
            return Generated(text=ABSTENTION, cited=())
        return Generated(text=text, cited=tuple(cited))
