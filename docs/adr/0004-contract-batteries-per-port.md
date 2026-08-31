# 0004. One contract battery per port, subclassed by every adapter

Status: accepted

## Context

Six ports, fourteen adapters, and the promise that any adapter of a port is substitutable
for any other. Substitutability is a behavioural claim; a type checker can only verify the
shapes (and with structural Protocols, only loosely).

## Problem

How do we keep fourteen adapter test suites from drifting apart, and how do we make "write a
new adapter" cheap without making it possible to write a hollow one?

## Options considered

1. **Independent test files per adapter.** Each suite tests what its author thought of;
   the sqlite cache gets expiry tests, the memory cache doesn't, and substitutability is
   untested folklore.
2. **A shared battery per port** (`tests/contracts/`), technology-free — it imports no
   concrete adapter. Each adapter's suite subclasses the battery and supplies only a
   construction fixture, then adds tests for behaviour genuinely unique to it (a sqlite
   adapter's restart test, a zip source's bomb cap).

## Decision

Option 2. Batteries also carry **quality floors**, not just structural assertions: the
`IndexPort` battery requires a labelled question to retrieve its document at rank 1 on the
fixture corpus, so an index that returns nothing — or garbage — cannot pass by having the
right shape.

## Trade-offs and consequences

- Every behavioural guarantee is stated once and enforced fourteen times; a new source
  adapter is about six lines of test to get full contract coverage.
- Fixture design becomes load-bearing: the fixture corpus must be lexically unambiguous
  enough that *any* conforming index can meet the rank-1 floor, or the battery turns into a
  benchmark of one implementation. That constraint is documented in `tests/conftest.py`.
- A battery bug is fourteen wrong suites, so changes to `tests/contracts/` deserve more
  suspicion than changes to any single adapter's tests.
