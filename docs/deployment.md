# Deployment

**No deployment of knowbase currently exists.** There is no hosted instance, no registry
image, and no uptime history. This document is the procedure you would follow to stand one
up, and every command in it has been run against the code in this repository — the container
was built and probed on the machine this was written on, not on a server.

## Target: one Docker host

The target is a single Linux host running Docker — a VM you already have, behind whatever
TLS-terminating reverse proxy your network uses.

Why that and not Kubernetes or a PaaS:

- The service is **one process**. There is no worker pool, no queue, no sidecar. An
  orchestrator would be managing a set of one.
- State is **one SQLite file**. That rules out horizontal scaling anyway (two replicas would
  race on the file), so the unit of deployment is one container plus one volume, which is
  exactly what a single host does well.
- The failure story is honest at this size: if the host dies you restore the volume from a
  copy and start the container elsewhere. Pretending to more availability than that would
  need a different persistence layer first (see Limitations in the README).

There is no `docker-compose.yml` for the same reason: compose earns its keep coordinating
several containers, and there is only one.

## Prerequisites

- Docker on the host (the image was built and probed with Docker 29 locally).
- A reverse proxy for TLS. The service speaks plain HTTP and does not do TLS itself.
- A directory on the host for the volume, e.g. `/srv/knowbase/data`, and one for the corpus
  documents, e.g. `/srv/knowbase/data/corpus`.

## Environment variables

| Variable | Required | Effect |
| --- | --- | --- |
| `KB_API_TOKEN` | in a container, yes | Bearer token for every endpoint except `/health` and `/ready`. Minimum 16 characters. The service refuses to bind `0.0.0.0` without it, so the container will not start without it. |
| `ANTHROPIC_API_KEY` | no | Enables the Claude generator adapter. Without it the composition root binds the extractive generator and `/health` says so; the service is fully functional offline. |
| `CLAUDE_MODEL` | no | Overrides the model the Claude adapter asks for. |

Generate a token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Pass secrets with `-e`/`--env-file` at run time. They are not baked into the image, and the
`.dockerignore` is an allow-list precisely so a local `.env` can never end up in a layer.

## Migrations

There is no separate migration step. `adapters/db.py` opens the SQLite file, reads
`schema_version`, and applies any pending forward migrations, one transaction each, before
the first request is served. Two consequences matter operationally:

- A new build against an old volume upgrades the file on start-up.
- An **old** build against a **newer** file refuses to open it. That is what makes rollback
  safe rather than corrupting (see Rollback).

## Build

```bash
git clone <this repo> && cd knowbase
docker build -t knowbase:$(git rev-parse --short HEAD) .
```

Tag with the commit, not `latest`, so a rollback is a tag you can name. Move the image to the
host with your registry, or with `docker save | ssh ... docker load` if you have none.

## Deploy

```bash
docker run -d --name knowbase \
  --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -e KB_API_TOKEN="$KB_API_TOKEN" \
  -v /srv/knowbase/data:/data \
  knowbase:<tag>
```

Publish on loopback and let the reverse proxy own the public interface. Then ingest:

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/ingest \
  -H "authorization: Bearer $KB_API_TOKEN" \
  -H "content-type: application/json" \
  -d '{"source":"filesystem","path":"/data/corpus"}'
```

The container's corpus allow-list is `/data/corpus` (set in the Dockerfile `CMD`); paths
outside it are rejected with `path_not_allowed`. With `--store sqlite` (also in the `CMD`)
the ingested corpus survives a restart and the indexes are rebuilt from it on start-up.

## Health checks

- `GET /health` — liveness plus the bound adapters and corpus state. Open route, no token.
  This is also the image's `HEALTHCHECK`, so `docker ps` shows `healthy`/`unhealthy`.
- `GET /ready` — `200` once a corpus is queryable, `503` before the first ingest. Point the
  proxy's upstream check here so traffic is not routed to an empty service.
- `GET /metrics` — per-route counters and latency histograms (JSON; token required).

## Upgrade

```bash
docker build -t knowbase:<newtag> .        # on the build machine
docker stop knowbase && docker rm knowbase
docker run ... knowbase:<newtag>           # same volume, same flags
curl -s http://127.0.0.1:8080/health
```

Stop-then-start, not blue/green: two containers must never share the SQLite file. The gap is
the seconds between stop and healthy; if that is unacceptable, this architecture is the wrong
fit and the persistence layer needs replacing first.

## Rollback

1. `docker stop knowbase && docker rm knowbase`
2. `docker run ...` with the previous image tag.
3. If the new build had migrated the schema, the old build will refuse to open the file and
   the container will exit — this is deliberate. Restore the volume from the copy taken
   before the upgrade, then start the old tag again.

Which is why the upgrade procedure implies: **copy `/srv/knowbase/data/knowbase.sqlite3`
before upgrading.** SQLite makes that a file copy while the container is stopped.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Container exits immediately, log ends `refusing to bind 0.0.0.0 without authentication` | `KB_API_TOKEN` not set. |
| Container exits, log ends `KB_API_TOKEN must be at least 16 characters` | Token too short. |
| Container exits mentioning `schema_version` | Old build, newer database file — see Rollback. |
| `/ready` returns 503 | No corpus ingested yet, or the store is `memory` and the container restarted. |
| `path_not_allowed` on ingest | The path is outside `/data/corpus` as seen *inside* the container. |
| `401 unauthenticated` | Missing or wrong `Authorization: Bearer` header; compare against the token the container was actually started with (`docker inspect`). |
| Slow answers after restart | First requests rebuild caches; check `/metrics` — the ingest route's histogram shows the rebuild cost. |

Every request logs one JSON line on the container's stderr with a `request_id` that also
appears in the `X-Request-Id` response header and in error bodies, so `docker logs knowbase`
plus the id from a failing client gets you to the exact line.
