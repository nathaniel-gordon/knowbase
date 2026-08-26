# Code Review Guidelines
Every change requires two approvals, one of which must come from a code owner of the touched
directory. Reviews should complete within one business day; authors may escalate stale reviews
in the team channel after 24 hours. Keep pull requests under 400 changed lines where practical;
larger changes should be split or accompanied by a design doc. Reviewers check correctness,
tests, security implications, and observability, not style, which is enforced by linters.
Nit-level comments must be prefixed "nit:" and are non-blocking. Self-merges are prohibited
except for documentation-only changes with a green pipeline.
