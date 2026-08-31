# 0005. Multiplicative rank decay in the extractive generator

Status: accepted (replaces the subtractive penalty)

## Context

The extractive generator scores candidate sentences from the retrieved passages by term
overlap with the question, discounted by the passage's retrieval rank, and every sentence
that makes it into the answer earns a citation to its passage.

The discount was originally subtractive: `score - 0.15 * rank`. On short questions the
overlap scores cluster tightly, and a flat subtraction let a sentence from a lower-ranked,
topically adjacent passage tie with — and be appended after — the right answer. The demo
showed it concretely: the refund question's answer arrived with a shipping-policy sentence
padded on, citing the shipping document.

## Problem

Discount lower-ranked passages in a way that breaks these ties in favour of rank, without
banning lower-ranked passages outright (a genuinely better sentence at rank 2 should still
win over a weak one at rank 1).

## Options considered

1. **Keep subtraction, raise the penalty.** Larger constants push toward "rank 1 always
   wins", and any constant still produces exact ties at some score gap.
2. **Only ever answer from the top-ranked passage.** Removes multi-passage answers, which
   are legitimate when a question genuinely spans documents.
3. **Multiplicative decay: `score / (1 + RANK_DECAY * rank)`.** The discount scales with
   the score, so equal-overlap sentences can no longer tie across ranks, while a
   sufficiently stronger overlap at a lower rank still wins.

## Decision

Option 3 (`RANK_DECAY` in `adapters/generator/extractive.py`, currently 1.0: the top
passage scores at full strength, the second at half, the third at a third).

## Trade-offs and consequences

- Verified by A/B on the demo corpus: the shipping sentence (and its citation) drops out of
  the refund answer; `tests/test_generator_adapters.py::test_a_lower_ranked_passage_does_not_tie_its_way_into_the_answer`
  pins the case, and was checked non-vacuous by restoring the old scoring and watching it
  fail.
- A zero-overlap sentence stays zero at any rank; decay cannot resurrect it.
- The decay base is a constant chosen by inspection on a 14-document corpus, not a tuned
  parameter; a different corpus might want a different value, and nothing here claims
  otherwise.
