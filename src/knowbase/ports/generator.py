"""GeneratorPort -- how retrieved passages become a cited answer."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.model import Generated, Passage


@runtime_checkable
class GeneratorPort(Protocol):
    """Turns a question plus ranked passages into prose with grounded citation markers.

    Contract (enforced by ``tests/contracts/generator_contract.py``):

    * ``generate(question, passages)`` returns a :class:`Generated` whose ``cited`` markers are
      1-based indices *into the passages it was given* -- never out of range, never duplicated,
      always sorted.
    * A citation must be *earned*: the answer text shares vocabulary with every passage it
      cites. Range-checking alone let a passage containing the literal ``[2][3][4]`` produce a
      citation to a passage that contributed nothing, so the battery checks both.
    * Passage text is data, not instruction. Adapters that hand it to a model must fence and
      label it; no port-level test can decide whether a model obeyed an embedded instruction,
      so the battery checks the part that is decidable -- the question still gets answered and
      the answer is still attributed to a passage that supports it.
    * With no passages it returns the abstention text and no citations, rather than guessing,
      and an answer it cannot attribute is an abstention rather than uncited prose.
    * It is deterministic: identical inputs produce identical output (the Claude adapter is
      deterministic given its injected completion function).
    * Quality floor: given a passage that literally contains the answer, the returned text
      contains the answer token.

    These are enforced checks, not a proof. A model that paraphrases one passage and cites a
    neighbouring one on the same topic can still pass them; the contract narrows fabrication,
    it does not make it impossible.
    """

    name: str

    def generate(self, question: str, passages: Sequence[Passage]) -> Generated:
        """Answer ``question`` using only ``passages``."""
        ...
