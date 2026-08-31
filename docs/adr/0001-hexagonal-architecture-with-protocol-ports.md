# 0001. Hexagonal architecture with `typing.Protocol` ports

Status: accepted

## Context

The service exists to answer questions over a document corpus and to *compare retrieval
strategies against each other*. The comparison is only meaningful if everything except the
ranking is held constant: same chunking, same corpus, same evaluation harness. A codebase
where retrieval logic, HTTP handling and persistence are interleaved cannot guarantee that.

## Problem

How do we structure the code so that a retrieval strategy, a document source, or an answer
generator can be swapped without touching — or accidentally influencing — anything else?

## Options considered

1. **A conventional layered app** (handlers → services → helpers). Cheap to start, but
   nothing stops `import sklearn` from appearing in a handler, and A/B comparisons would be
   comparing code paths, not strategies.
2. **Abstract base classes** for the seams. Enforces the interface, but forces adapters to
   inherit from the core, which inverts the dependency we want: the core should know nothing
   about adapters, including being their base class.
3. **Hexagonal (ports and adapters) with `typing.Protocol` ports.** The domain depends on
   six structural interfaces; adapters implement them by shape, importing the port only in
   tests. Third-party libraries are confined to `adapters/`.

## Decision

Option 3. Ports are `typing.Protocol` classes in `ports/`; every concrete technology
(numpy, scikit-learn, sqlite3, anthropic) lives under `adapters/`; `app/` is the composition
root that binds names from `AppConfig` to adapter constructors.

The rule is enforced, not documented: `tests/test_domain_purity.py` parses every domain
module with `ast`, checks import roots against an allow-list, and then spawns a clean
interpreter to verify none of the heavy libraries leaked into `sys.modules` after importing
the domain.

## Trade-offs and consequences

- `POST /api/v1/compare` falls out almost for free: `CompareStrategies` sees only
  `IndexPort`, and one ingest feeds every registered index, so chunking is constant by
  construction.
- A new retrieval strategy is an adapter, a registry line, and a contract-test subclass.
- The cost is indirection: reading a request's path through the system crosses app → domain
  → port → adapter, and there are more files than a flat script would need.
- Protocols are structural, so a subtly wrong adapter can satisfy the type checker; the
  contract batteries (ADR 0004) exist to catch what mypy cannot.
