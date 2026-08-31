# Contributing

This is a reference implementation maintained in spare time. Issues and patches are welcome;
response times are best-effort and not promised.

## Setup

```bash
git clone <this repo> && cd knowbase
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

Python 3.11+. The only runtime dependencies are numpy and scikit-learn; everything else is
the standard library. `pip install -e ".[llm]"` adds the optional anthropic client.

## Running things

```bash
python run.py demo                 # end-to-end walkthrough, no install needed
python run.py serve --port 8080    # the service
python -m pytest tests -q          # the suite (~1 min; starts real sockets)
python -m ruff check .
python -m mypy
```

CI runs exactly those three commands plus a docker build, on Python 3.11–3.14. Run them
before opening a PR; a patch that fails any of them locally will fail there too.

## Architecture, in one paragraph

The dependency arrow points inward: `domain/` (use-cases, entities) depends only on the
`typing.Protocol` interfaces in `ports/`; concrete technology (numpy, scikit-learn, sqlite3,
anthropic, the HTTP server) lives in `adapters/` and `app/`. This is enforced by
`tests/test_domain_purity.py`, which parses domain imports with `ast` and then checks a clean
interpreter for leaked modules — so an `import sklearn` in the wrong layer is a red test, not
a review comment. The decisions and their trade-offs are written down in `docs/adr/`.

## Adding an adapter

The most common kind of change:

1. Implement the port's `Protocol` under the matching `adapters/` package.
2. Register it in the adapter registry in `app/composition.py`.
3. Subclass the port's contract battery from `tests/contracts/` and supply the construction
   fixture. Add tests only for behaviour unique to your adapter.
4. If the adapter needs a new third-party package, expect pushback: the stdlib-first stance
   is ADR 0002, and heavyweight dependencies belong behind an optional extra.

## Coding standards

- ruff and mypy `--strict` are the standard; both run clean on `main` and must stay clean.
  Config for both lives in `pyproject.toml`.
- No new claims without a test. If a docstring or the README says the code does something,
  a test somewhere must fail when it stops being true.
- Measured numbers in docs must come from a command that actually ran, quoted with the
  command. Never paste a number you did not produce.
- Don't log user content: question text and document bodies never appear in logs, only
  their sizes. Keep it that way.
- Security-relevant behaviour (path containment, caps, auth) changes only with an
  accompanying regression test in `tests/test_security.py`.

## Commits and PRs

- Small, real commits with plain imperative subjects ("cap zip member size", not
  "Enhance resource management"). A body only when the why is not obvious.
- One topic per PR. Refactors and behaviour changes in separate commits at minimum.
- If you change a port's contract, say so loudly in the PR description — every adapter of
  that port is affected.
