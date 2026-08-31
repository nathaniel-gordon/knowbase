---
name: Bug report
about: Something the code does that its docs or tests say it should not
labels: bug
---

**What happened**

**What you expected**

**Reproduction**

Ideally a failing test, otherwise the exact commands. If the service was involved, include
the `request_id` from the response header or error body and the matching JSON log line —
that is what it exists for.

```
python/OS versions:
command:
output:
```

**Adapters in play**

`/health` reports the bound wiring; paste the `wiring` object if you can.
