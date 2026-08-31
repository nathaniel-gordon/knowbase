# 0002. Standard-library HTTP server and SQLite, not a framework stack

Status: accepted

## Context

The service needs an HTTP surface (about ten routes, JSON in and out) and durable storage
for two things: an answer cache and the ingested corpus. The obvious contemporary stack is
FastAPI + uvicorn + PostgreSQL.

## Problem

What does the driving adapter and the persistence adapter actually require, and what is the
cheapest technology that meets it honestly?

## Options considered

1. **FastAPI + uvicorn + PostgreSQL.** Idiomatic and well-known, but it adds an ASGI
   server, a validation framework and a database *server* as dependencies for a service
   whose request model is small enough to validate by hand and whose write concurrency is
   one ingest at a time. Tests would need a running database or a second code path (mocks),
   and the second code path is the one that rots.
2. **`http.server.ThreadingHTTPServer` + `sqlite3`, both stdlib.** No dependencies beyond
   numpy/scikit-learn (which retrieval needs anyway). The full stack — socket to SQL — runs
   inside pytest with no services to stand up, so integration and security tests hit a real
   socket and a real database file on every run.

## Decision

Option 2. The HTTP layer is a `BaseHTTPRequestHandler` with an explicit route table and
hand-written validation (`app/validation.py`); persistence is one SQLite file opened through
a shared helper (`adapters/db.py`) with WAL, foreign keys, and bound parameters everywhere.

## Trade-offs and consequences

- Everything the tests exercise is exactly what runs; there is no mock transport and no
  in-memory stand-in for the database (the security suite drives a real socket).
- We own concerns a framework would own: request-size caps, socket timeouts, refusing
  chunked bodies, the JSON error envelope. Each of those is code we had to write and test —
  and each is also visible and pinned by a test, which is the point.
- SQLite means single-writer, single-host. Horizontal scaling is off the table until the
  store adapter is reimplemented against a server database; the `CorpusStorePort` seam is
  where that would happen.
- `ThreadingHTTPServer` is thread-per-connection, which caps concurrency far below what an
  async stack could take. For a corpus-sized internal service that is acceptable and the
  concurrency tests document what is actually safe.
