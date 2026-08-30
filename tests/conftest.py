"""Shared fixtures.

The fixture corpus is deliberately tiny and lexically unambiguous: contract tests must be able
to assert a *quality floor* ("the labelled document comes back at rank 1") that any conforming
index adapter can meet, without turning the battery into a benchmark of one implementation.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"
for entry in (str(SRC), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from contracts.corpus_fixture import FIXTURE_DOCS  # noqa: E402
from knowbase.adapters.clock.system import FrozenClock  # noqa: E402
from knowbase.app.corpus import write_demo_corpus  # noqa: E402
from knowbase.domain.model import Document  # noqa: E402
from knowbase.domain.text import chunk_document  # noqa: E402


@pytest.fixture(scope="session")
def fixture_documents() -> list[Document]:
    """The fixture corpus as domain documents, in the order every source adapter must yield."""
    return [Document(doc_id=doc_id,
                     title=text.splitlines()[0].lstrip("# ").strip(),
                     text=text,
                     origin=f"fixture://{doc_id}")
            for doc_id, text in sorted(FIXTURE_DOCS.items())]


@pytest.fixture(scope="session")
def fixture_passages(fixture_documents) -> list:
    """Chunked fixture corpus -- shared by every index contract subclass."""
    passages: list = []
    for document in fixture_documents:
        passages.extend(chunk_document(document))
    return passages


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """The fixture corpus written to disk for the filesystem source adapter."""
    folder = tmp_path_factory.mktemp("fixture_docs")
    for doc_id, text in FIXTURE_DOCS.items():
        (folder / f"{doc_id}.md").write_text(text, encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def fixture_zip(tmp_path_factory, fixture_dir) -> Path:
    """The fixture corpus packed into an archive for the zip source adapter."""
    archive = tmp_path_factory.mktemp("fixture_zip") / "kb.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(fixture_dir.glob("*.md")):
            zf.write(path, arcname=f"export/{path.name}")
    return archive


@pytest.fixture(scope="session")
def corpus_roots(tmp_path_factory) -> tuple[str, ...]:
    """Allow-list for ``/ingest`` in tests.

    Everything pytest writes lives under one base temp directory, so allow-listing that
    directory lets the HTTP tests ingest their fixtures while every path outside it is still
    refused -- which is the behaviour ``tests/test_security.py`` then pins.
    """
    return (str(tmp_path_factory.getbasetemp()),)


@pytest.fixture(scope="session")
def demo_corpus_dir(tmp_path_factory) -> Path:
    """The full bundled 14-document knowledge base -- used for the preserved-quality tests."""
    folder = tmp_path_factory.mktemp("demo_docs")
    write_demo_corpus(folder)
    return folder


@pytest.fixture
def frozen_clock() -> FrozenClock:
    """A fresh deterministic clock; one second per read keeps TTL arithmetic obvious."""
    return FrozenClock(start=1_700_000_000.0, step_s=1.0)
