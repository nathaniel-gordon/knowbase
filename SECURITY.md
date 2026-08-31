# Security policy

## Supported versions

Only the tip of `main`. There are no maintained release branches; 0.1.x is the only line.

## Reporting a vulnerability

Email nathanielgordon346@gmail.com with a description and, if you can, a reproduction.
Please do not open a public issue for anything exploitable. This is a spare-time project:
you will get an acknowledgement, but there is no SLA and no bounty.

## The security model, so reports can be judged against it

What the service tries to guarantee (each of these has tests in `tests/test_security.py`):

- `/ingest` reads only inside the configured corpus roots; every path is resolved before
  the check and containment is re-checked per file because symlinks inside a root can point
  out of it.
- Binding any non-loopback address without `KB_API_TOKEN` is refused at start-up; the token
  is compared with `hmac.compare_digest` and read from the environment only.
- Request bodies, file sizes, zip member counts and decompressed bytes, eval-case counts,
  question length and `k` are all capped; `Transfer-Encoding: chunked` is refused.
- SQLite statements bind every parameter; stored values are JSON and text, never pickled.
- Passage text sent to the Claude adapter is fenced and labelled untrusted, and bracketed
  digits inside it are rewritten so a document cannot mint citations by echoing markers.
- Logs carry sizes and ids, never question text, document bodies, or the token.

Known, documented weaknesses that are *not* news (see README Limitations): the single
shared bearer token (no users, roles, rotation, or rate limiting), no TLS (a reverse proxy
is assumed), and citation forgery being narrowed rather than eliminated. Reports that
sharpen these into something concretely exploitable are still very welcome.
