# knowbase

A retrieval service for a corpus of internal documents, built as a hexagonal (ports and
adapters) application. It runs as a small JSON HTTP service on the standard library, answers
questions with citations that point at the passage they came from, and measures its own
retrieval quality against labelled questions — yours or the two bundled suites.

This is a **reference implementation**, not a deployed product. Nothing is running anywhere,
there is no hosted URL, and it has no users. What it does have is a strictly enforced
architecture and a test suite that checks the claims below rather than asserting them.

---

## Quickstart

```bash
git clone <this repo> && cd knowbase
pip install -r requirements.txt

python run.py demo                    # end-to-end walkthrough, starts and stops its own server
python run.py serve --port 8080       # long-running service on 127.0.0.1:8080
python -m pytest tests -q             # the test suite
```

`run.py` puts `src/` on `sys.path` so a clone runs without installing. After
`pip install -e .` the same entry points are reachable as `python -m knowbase demo|serve` and
as the `knowbase` console script.

A first request:

```bash
curl -sX POST localhost:8080/api/v1/ingest \
     -H 'content-type: application/json' \
     -d '{"source":"filesystem","path":"./output/demo_docs"}'

curl -sX POST localhost:8080/api/v1/ask \
     -H 'content-type: application/json' \
     -d '{"question":"How long do I have to submit an expense report?","k":4}'
```

`/ingest` only reads from directories you allow. Pass `--corpus-root DIR` (repeatable) when
starting the service; with no roots configured it will read from the working directory and
nowhere else.

---

## Architecture

The dependency arrow points inward and is checked, not asserted.

```
              ┌──────────────────────── driving side ────────────────────────┐
              │  HTTP service (/api/v1)      typed client SDK      demo/CLI  │
              └───────────────────────────────┬─────────────────────────────┘
                                              │
                       ┌──────────────────────▼──────────────────────┐
                       │  app/  composition root, config, limits,    │
                       │        validation, path allow-list, auth,   │
                       │        observability, JSON wire mapping     │
                       └──────────────────────┬──────────────────────┘
                                              │  constructs adapters,
                                              │  hands out use-cases
   ┌──────────────────────────────────────────▼──────────────────────────────────────────┐
   │  domain/   IngestCorpus · AskQuestion · EvaluateRetrieval · CompareStrategies        │
   │            frozen entities, pure text services, error taxonomy                       │
   │            imports: __future__ dataclasses typing collections re hashlib math enum   │
   └──────────────────────────────────────────┬──────────────────────────────────────────┘
                                              │  depends only on
                       ┌──────────────────────▼──────────────────────┐
                       │  ports/  six typing.Protocol interfaces      │
                       └──────────────────────┬──────────────────────┘
                                              │  implemented by
   ┌──────────────────────────────────────────▼──────────────────────────────────────────┐
   │  adapters/  index {hybrid_rrf, bm25, tfidf}   source {filesystem, memory, zip}       │
   │             generator {extractive, claude}    cache {memory, sqlite}                 │
   │             store {memory, sqlite}            clock {system, frozen}                 │
   │             numpy / scikit-learn / anthropic / sqlite3 live here and only here       │
   └─────────────────────────────────────────────────────────────────────────────────────┘
```

### The six ports

| Port | What it abstracts | Adapters |
| --- | --- | --- |
| `DocumentSourcePort` | where documents come from | `filesystem`, `memory`, `zip` |
| `IndexPort` | how passages are ranked | `hybrid_rrf`, `bm25`, `tfidf` |
| `GeneratorPort` | how an answer is written and attributed | `extractive`, `claude` |
| `CachePort` | memoisation of generated answers | `memory`, `sqlite` |
| `CorpusStorePort` | durable custody of the ingested corpus | `memory`, `sqlite` |
| `ClockPort` | time, so expiry and latency are testable | `system`, `frozen` |

Every adapter choice is a string in `AppConfig`. `POST /api/v1/config` rebinds the index,
generator or cache on a running service, and nothing inside the hexagon notices.

### Why it is built this way

`POST /api/v1/compare` is the architecture paying for itself. `CompareStrategies` only ever
sees `IndexPort`, and a single ingest feeds every registered index, so chunking is held
constant by construction and an A/B result isolates ranking. Adding a fourth retrieval
strategy is a new adapter, one registry line, and six lines of test.

---

## HTTP API

Operational endpoints sit outside the version prefix, because a probe or a scrape should not
have to follow a version bump.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | what this service is; where the API lives; whether auth is on |
| `GET` | `/health` | liveness, bound adapters, the corpus on hand, cache counters |
| `GET` | `/ready` | `200` once a corpus is queryable, `503` before that |
| `GET` | `/metrics` | per-route counters, latency buckets, cache and corpus gauges |
| `GET` | `/api/v1/adapters` | every adapter registered for every port |
| `POST` | `/api/v1/ingest` | build the passage set and every index from a document source |
| `POST` | `/api/v1/ask` | a cited answer |
| `POST` | `/api/v1/eval` | retrieval metrics over a labelled suite, bundled or your own |
| `POST` | `/api/v1/compare` | the same suite through several index adapters, with a winner |
| `POST` | `/api/v1/config` | rebind index / generator / cache at runtime |

### Errors

Every failure is `{"error": {"code", "message", "details", "request_id"}}`. The `code` is
stable and meant to be switched on; the message is prose. Python exception types never appear.

| Code | Status | Meaning |
| --- | --- | --- |
| `invalid_request` | 400 | wrong type, missing field, out of range, unknown field |
| `empty_corpus` / `index_build_failed` | 400 | the documents cannot be indexed |
| `source_unavailable` | 400 | the source cannot be read |
| `unknown_adapter` | 400 | no such adapter or suite |
| `unauthenticated` | 401 | missing or wrong bearer token |
| `path_not_allowed` | 403 | the path resolves outside every configured corpus root |
| `corpus_not_ingested` | 409 | `/ask` before anything was ingested |
| `resource_limit_exceeded` | 413 | a cap was hit (body size, file size, zip member, …) |
| `adapter_unavailable` / `corpus_store_unavailable` | 503 | the service could not do its job |
| `internal_error` | 500 | anything unforeseen; the traceback is in the log under the same `request_id` |

### Evaluation

`/eval` and `/compare` accept `cases`, a list of `{question, expected_doc_id}`, so the harness
can be pointed at your own labelled data rather than only the two bundled suites. Per-question
rows are paginated (`limit` / `offset`), and the summary metrics always cover every case, not
just the page.

### Limits

Every bound a request can hit lives in `app/limits.py` and is configuration, not a constant
buried in a handler: body size, question length, `k`, inline document count and bytes, files
and bytes read from a directory, zip members and decompressed bytes, eval cases, page size,
and the per-request socket timeout.

---

## Persistence

One SQLite file holds both the answer cache and the corpus store, behind a forward-only
migration runner (`adapters/db.py`): a `schema_version` table, ordered migrations applied one
transaction each, and a refusal to open a database written by a newer build. Foreign keys are
on, the corpus snapshot is a singleton row whose passages cascade with it, and every statement
binds its parameters. Values are stored as JSON and plain text — nothing is pickled, so a
database file cannot execute anything when it is read back.

```bash
python run.py serve --store sqlite --cache sqlite --sqlite-path ./var/knowbase.sqlite3
```

With `--store sqlite`, a restart rebuilds the indexes from the stored passages, in the same
order, so citations issued before the restart still resolve. The default (`memory`) keeps
nothing, and `/health` says which one is bound.

---

## Security

- **Corpus roots.** `/ingest` takes a path from the caller. Every path is resolved (collapsing
  `..`, following symlinks, normalising drive letters and UNC prefixes) and then required to
  sit inside an allow-listed root. Containment is re-checked per file, because a symlink inside
  a root can point anywhere. Rejections do not echo the configured roots.
- **Authentication.** Set `KB_API_TOKEN` and every endpoint except `/health` and `/ready`
  requires `Authorization: Bearer …`, compared with `hmac.compare_digest`. Binding any address
  other than loopback without a token is refused before the socket opens. The token is read
  from the environment only — never from a config file, a flag, a response, or a log line.
- **Resource caps.** Zip members are streamed under a byte cap rather than `read()` whole (a
  bomb lies in its header); the filesystem walk is bounded in file count, file size and entries
  scanned; `k` and the request body are capped; `Transfer-Encoding: chunked` is refused rather
  than silently read as an empty body.
- **Prompt injection.** Passage text reaching the Claude adapter is fenced, labelled untrusted,
  and has its own bracketed digits rewritten, so a document containing `[2][3][4]` cannot mint
  citations by being echoed back. Citations come from a parseable trailer and must share
  vocabulary with the answer before they are emitted.
- **No secrets in the repo.** `.env.example` lists the variables; `.env` is gitignored.
- **Logs.** One JSON line per request with a correlation id. Question text and document bodies
  are never logged — only their size.

---

## Observability

```json
{"ts":"2026-08-31T09:14:02.117Z","level":"info","logger":"knowbase","event":"request",
 "request_id":"9f2c1ab77d0e4a51","method":"POST","route":"/api/v1/ask","status":200,
 "duration_ms":7.412,"question_chars":47,"index":"hybrid_rrf","generator":"extractive",
 "hits":4,"citations":1,"cache":"miss"}
```

The same `request_id` is returned in the `X-Request-Id` header and inside every error body, so
a user's screenshot leads to the line that produced it. `/metrics` reports per-route request
and error counts, status classes, latency histogram buckets, cache hit rate, and corpus size.

---

## Testing

The suite is the reason to trust anything above.

- **Contract batteries** (`tests/contracts/`) are technology-free: they import no concrete
  adapter. Every adapter of every port subclasses its battery and supplies only a construction
  fixture, so a new source adapter is about six lines of test. Batteries carry quality floors,
  not just structural assertions — the `IndexPort` battery requires that a labelled question
  retrieve its document at rank 1, so a "return nothing" index cannot pass.
- **A fitness function** (`tests/test_domain_purity.py`) parses every domain module with `ast`,
  checks import roots against an allow-list, checks relative-import targets, asserts every port
  is a `typing.Protocol`, and then re-verifies at runtime by spawning a clean interpreter and
  reporting whether numpy/sklearn/scipy/pandas/anthropic/sqlite3 leaked into `sys.modules`.
- **Adversarial tests** (`tests/test_security.py`) drive path traversal, drive roots, type
  confusion, chunked bodies, a lying `Content-Length`, binary and RTL documents, a
  stopword-only corpus, and every cap — over a real socket.
- **Concurrency tests** (`tests/test_concurrency.py`) hammer `search` while `build` runs
  repeatedly, and hammer both cache adapters from eight threads. These found a real defect: a
  SQLite connection shared between threads is not made safe by `check_same_thread=False`, and
  a just-written row could come back missing.

---

## Measured results

Every number here comes from a command that ran in this repository. Nothing is quoted from
anywhere else, and there is no historical baseline to compare against — this repo has one
lineage and these are its current numbers.

```
$ python -m pytest tests -q
408 passed, 1 skipped in 51.05s
```

The skip is the symlink-containment test, which needs a platform/user that can create symlinks.

```
$ python run.py demo
   core suite   hybrid_rrf  n=12  hit@1=1.000  hit@5=1.000  mrr=1.000
   hard suite   bm25        n=12  hit@1=0.750  hit@5=0.750  mrr=0.750
                hybrid_rrf  n=12  hit@1=0.667  hit@5=0.750  mrr=0.708
                tfidf       n=12  hit@1=0.667  hit@5=0.750  mrr=0.708
   -> on the hard suite bm25 wins by +0.042 MRR
```

Both suites are 12 labelled questions over the 14 bundled demo documents. The "core" suite uses
the vocabulary of the documents; the "hard" suite paraphrases it. Fusion loses to plain BM25 on
the hard suite here — that is what the A/B endpoint is for, and the honest reading is that on a
corpus this small the fusion has nothing to add.

```
$ python -m ruff check .
All checks passed!

$ python -m mypy
Success: no issues found in 49 source files
```

---

## Limitations

Stated plainly, because most of these are the difference between this and something you would
put in front of users.

- **Retrieval is lexical only.** BM25 and TF-IDF, no embeddings, no semantic matching. A
  paraphrase that shares no vocabulary with its document is simply missed: 3 of the 12
  hard-suite questions are never retrieved at any rank, and the suite pins that number.
- **Every ingest is a full rebuild.** All three indexes are refitted from scratch. There is no
  incremental update and no way to add a single document.
- **The demo corpus is 14 short documents.** The measured numbers describe that corpus. They
  are not a benchmark and they will not transfer to yours.
- **The Claude adapter has never been exercised against the live API here.** It is fully
  tested through an injected completion function, offline. Without `ANTHROPIC_API_KEY` the
  composition root binds the extractive generator and `/health` says so.
- **Citation forgery is narrowed, not eliminated.** A model that paraphrases one passage and
  cites a neighbouring one on the same topic can still pass the checks.
- **Authentication is a single shared bearer token.** No users, no roles, no rotation, no rate
  limiting.
- **No deployment exists.** No container image, no CI, no hosted instance, no uptime to report.
- **`/config` mutates process-wide state.** It is a demonstration of the ports seam, not a
  multi-tenant control plane.

---

## Repository layout

```
run.py                        bootstrap launcher (adds src/ to sys.path)
src/knowbase/domain/          entities, pure text services, four use-cases, error taxonomy
src/knowbase/ports/           six typing.Protocol ports
src/knowbase/adapters/        fourteen adapters + the sqlite schema and migration runner
src/knowbase/app/             composition root, HTTP service, client SDK, validation,
                              limits, path allow-list, auth, observability, demo
tests/contracts/              technology-free contract batteries, one per port
tests/                        adapter suites, HTTP integration, security, concurrency,
                              migrations, evaluation, and the domain-purity fitness function
```

## License

MIT — see [LICENSE](LICENSE).

---

## Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
