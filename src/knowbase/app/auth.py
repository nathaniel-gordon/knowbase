"""Bearer-token authentication for the HTTP service.

Two endpoints mutate global state -- ``/ingest`` replaces the corpus every other client reads
from, ``/config`` rebinds ports process-wide -- and neither used to require anything at all.
The default bind is loopback, which is a real mitigation, but ``--host 0.0.0.0`` was offered
with nothing to fall back on.

The rule this module enforces:

* the token is read from the ``KB_API_TOKEN`` environment variable, never from a config file
  and never from a command-line flag, so it cannot be committed or captured in a shell history
* comparison is constant time (``hmac.compare_digest``), so a wrong token leaks no timing
* ``/health`` and ``/ready`` stay open, because a probe cannot hold a credential
* binding to anything other than loopback without a token is refused at start-up, rather than
  serving an unauthenticated write endpoint to a network
"""
from __future__ import annotations

import hmac
import ipaddress
import os
from dataclasses import dataclass

from ..domain.errors import KbError

TOKEN_ENV_VAR = "KB_API_TOKEN"
MIN_TOKEN_CHARS = 16


class Unauthenticated(KbError):
    """A request to a guarded endpoint arrived without a usable bearer token."""

    code = "unauthenticated"


def is_loopback(host: str) -> bool:
    """Whether binding to ``host`` exposes the service only to this machine."""
    if host in ("", "localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class Authenticator:
    """Decides whether a request may proceed. An empty token means "no auth configured"."""

    token: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @classmethod
    def from_environment(cls, token: str | None = None) -> "Authenticator":
        """Read the token from the environment unless one is passed in (tests do that)."""
        raw = (token if token is not None else os.environ.get(TOKEN_ENV_VAR, "")).strip()
        if raw and len(raw) < MIN_TOKEN_CHARS:
            raise ValueError(
                f"{TOKEN_ENV_VAR} must be at least {MIN_TOKEN_CHARS} characters")
        return cls(token=raw)

    def check_bind(self, host: str) -> None:
        """Refuse to start a token-less service on a non-loopback address."""
        if not self.enabled and not is_loopback(host):
            raise ValueError(
                f"refusing to bind {host} without authentication: /ingest and /config mutate "
                f"state for every client, so set {TOKEN_ENV_VAR} or bind 127.0.0.1")

    def authorise(self, header: str | None) -> None:
        """Raise :class:`Unauthenticated` unless ``header`` carries the configured token."""
        if not self.enabled:
            return
        scheme, _, presented = (header or "").partition(" ")
        if scheme.lower() != "bearer" or not presented:
            raise Unauthenticated("this endpoint requires a bearer token")
        if not hmac.compare_digest(presented.strip(), self.token):
            raise Unauthenticated("the bearer token is not valid")
