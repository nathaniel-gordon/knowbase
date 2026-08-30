"""Every bound the service imposes on client-supplied work, in one place.

These are not tuning knobs. Each one closes a specific way an unauthenticated caller could
make the process do unbounded work: walk an arbitrary tree, decompress an arbitrary archive,
ask for a million hits, or post a body the handler then holds in memory.

The defaults are sized for a knowledge base of policy documents, which is what this service
is for. They are configuration (``AppConfig.limits``), so a deployment can raise them
deliberately rather than discovering there was never a ceiling.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

KIB = 1024
MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Limits:
    """Resource ceilings applied to requests and to the ingest path."""

    # --- HTTP transport
    max_body_bytes: int = 4 * MIB
    request_timeout_s: float = 30.0
    max_page_size: int = 200
    default_page_size: int = 50

    # --- /ask
    max_question_chars: int = 2_000
    max_k: int = 50

    # --- inline documents
    max_documents: int = 500
    max_document_bytes: int = 1 * MIB
    max_inline_total_bytes: int = 8 * MIB
    max_doc_id_chars: int = 200

    # --- filesystem source
    max_files: int = 2_000
    max_file_bytes: int = 4 * MIB
    max_scanned_entries: int = 50_000

    # --- zip source
    max_zip_members: int = 2_000
    max_zip_member_bytes: int = 4 * MIB
    max_zip_total_bytes: int = 64 * MIB

    # --- eval
    max_eval_cases: int = 500

    def with_(self, **changes: Any) -> Limits:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__slots__}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Limits:
        unknown = sorted(set(data) - set(cls.__slots__))
        if unknown:
            raise ValueError(f"unknown limit(s): {unknown}")
        return cls(**dict(data))


DEFAULT_LIMITS = Limits()
