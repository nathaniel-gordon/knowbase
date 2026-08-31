# 0003. Forward-only migrations, applied at open

Status: accepted

## Context

One SQLite file holds the answer cache and the corpus store. Its schema will change across
versions, and a service instance can meet a file written by an older or a newer build.

## Problem

How are schema changes applied, and what happens when the code and the file disagree about
the schema version?

## Options considered

1. **A migration tool (alembic-style) run as a deploy step.** Real leverage for a server
   database with many environments; for one embedded file it adds an external tool and a
   step an operator can forget, and the failure mode of forgetting is a crash at first
   query.
2. **Ordered forward migrations applied automatically when the file is opened, and a hard
   refusal to open a file whose version is newer than the build knows.** Rollback is not a
   down-migration; it is restoring the file from the copy taken before the upgrade.

## Decision

Option 2, implemented in `adapters/db.py`: a `schema_version` table records every applied
migration, each migration runs in its own transaction, and opening a database from a newer
build raises instead of guessing.

## Trade-offs and consequences

- Deploys have no separate migration step to forget; a new build against an old volume
  upgrades itself before serving.
- Down-migrations are never written, which means never testing rarely-run destructive code.
  The price is operational: rolling back after a migration requires the pre-upgrade copy of
  the file, and docs/deployment.md makes that copy part of the upgrade procedure.
- The version check turns "old code silently misreads new schema" — the corrupting case —
  into a refusal to start, which an operator can see and act on.
- A crash mid-migration leaves the file at the last completed version, because each
  migration is one transaction.
