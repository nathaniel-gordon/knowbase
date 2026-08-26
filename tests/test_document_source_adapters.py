"""Run the DocumentSourcePort battery against all three adapters.

Each class below contributes only construction. Every assertion comes from the shared battery,
so a new source adapter becomes trustworthy by adding six lines here.
"""
from __future__ import annotations

import pytest

from contracts.corpus_fixture import FIXTURE_DOCS
from contracts.document_source_contract import DocumentSourceContract
from kb.adapters.source.filesystem import FilesystemDocumentSource
from kb.adapters.source.memory import InMemoryDocumentSource
from kb.adapters.source.zip_archive import ZipDocumentSource


class TestFilesystemDocumentSource(DocumentSourceContract):
    @pytest.fixture
    def source(self, fixture_dir):
        return FilesystemDocumentSource(fixture_dir)

    @pytest.fixture
    def broken_source(self, tmp_path):
        return FilesystemDocumentSource(tmp_path / "does-not-exist")

    def test_empty_folder_is_also_unavailable(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(Exception, match="no .* files under"):
            list(FilesystemDocumentSource(tmp_path / "empty").documents())

    def test_origin_is_the_file_path(self, source, fixture_dir):
        for document in source.documents():
            assert document.origin.endswith(f"{document.doc_id}.md")
            assert str(fixture_dir) in document.origin


class TestInMemoryDocumentSource(DocumentSourceContract):
    @pytest.fixture
    def source(self):
        return InMemoryDocumentSource(FIXTURE_DOCS, label="contract")

    @pytest.fixture
    def broken_source(self):
        return InMemoryDocumentSource({})

    def test_blank_documents_are_skipped(self):
        source = InMemoryDocumentSource({**FIXTURE_DOCS, "blank": "   \n  "})
        assert "blank" not in {d.doc_id for d in source.documents()}


class TestZipDocumentSource(DocumentSourceContract):
    @pytest.fixture
    def source(self, fixture_zip):
        return ZipDocumentSource(fixture_zip)

    @pytest.fixture
    def broken_source(self, tmp_path):
        return ZipDocumentSource(tmp_path / "missing.zip")

    def test_corrupt_archive_is_unavailable(self, tmp_path):
        bad = tmp_path / "corrupt.zip"
        bad.write_bytes(b"not a zip file at all")
        with pytest.raises(Exception, match="corrupt archive"):
            list(ZipDocumentSource(bad).documents())

    def test_origin_points_inside_the_archive(self, source):
        for document in source.documents():
            assert "!export/" in document.origin


def test_every_registered_source_adapter_has_a_contract_subclass():
    """Guard against an adapter being added to the registry but never contract-tested."""
    from kb.adapters import SOURCE_ADAPTERS
    covered = {cls.__name__ for cls in (FilesystemDocumentSource, InMemoryDocumentSource,
                                        ZipDocumentSource)}
    assert {cls.__name__ for cls in SOURCE_ADAPTERS.values()} == covered
