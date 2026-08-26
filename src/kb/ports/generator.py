"""GeneratorPort -- how retrieved passages become a cited answer."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..domain.model import Generated, Passage


@runtime_checkable
class GeneratorPort(Protocol):
    """Turns a question plus ranked passages into prose with grounded citation markers.

    Contract (enforced by ``tests/contracts/generator_contract.py``):

    * ``generate(question, passages)`` returns a :class:`Generated` whose ``cited`` markers are
      1-based indices *into the passages it was given* -- never out of range, never duplicated.
      This is what makes citation fabrication structurally impossible.
    * With no passages it returns the abstention text and no citations, rather than guessing.
    * It is deterministic: identical inputs produce identical output (the Claude adapter is
      deterministic given its injected completion function).
    * Quality floor: given a passage that literally contains the answer, the returned text
      contains the answer token.
    """

    name: str

    def generate(self, question: str, passages: Sequence[Passage]) -> Generated:
        """Answer ``question`` using only ``passages``."""
        ...
