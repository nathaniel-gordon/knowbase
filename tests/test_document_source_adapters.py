"""Run the DocumentSourcePort battery against all three adapters.

Each class below contributes only construction. Every assertion comes from the shared battery,
so a new source adapter becomes trustworthy by adding six lines here.
"""
from __future__ import annotations

import zipfile

import pytest

from contracts.corpus_fixture import FIXTURE_DOCS
from contracts.document_source_contract import DocumentSourceContract
from knowbase.adapters.source.filesystem import FilesystemDocumentSource
from knowbase.adapters.source.memory import InMemoryDocumentSource
from knowbase.adapters.source.zip_archive import ZipDocumentSource
from knowbase.domain.errors import DocumentSourceUnavailable, ResourceLimitExceeded


class TestFilesystemDocumentSource(DocumentSourceContract):
    @pytest.fixture
    def source(self, fixture_dir):
        return FilesystemDocumentSource(fixture_dir)

    @pytest.fixture
    def broken_source(self, tmp_path):
        return FilesystemDocumentSource(tmp_path / "does-not-exist")

    def test_empty_folder_is_also_unavailable(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(DocumentSourceUnavailable, match=r"no .* files under"):
            list(FilesystemDocumentSource(tmp_path / "empty").documents())

    def test_origin_is_relative_to_the_ingest_root(self, source, fixture_dir):
        """Citations must locate the file without printing the server's directory layout."""
        for document in source.documents():
            assert document.origin == f"file://{document.doc_id}.md"
            assert str(fixture_dir) not in document.origin

    def test_nested_folders_keep_their_relative_prefix(self, tmp_path):
        nested = tmp_path / "root" / "policies"
        nested.mkdir(parents=True)
        (nested / "vpn.md").write_text("# Remote Access Rules\nUse the VPN.", encoding="utf-8")
        produced = list(FilesystemDocumentSource(tmp_path / "root").documents())
        assert [d.origin for d in produced] == ["file://policies/vpn.md"]


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
    from knowbase.adapters import SOURCE_ADAPTERS
    covered = {cls.__name__ for cls in (FilesystemDocumentSource, InMemoryDocumentSource,
                                        ZipDocumentSource)}
    assert {cls.__name__ for cls in SOURCE_ADAPTERS.values()} == covered


# ------------------------------------------------------------- resource limits

def _write(folder, name, text="# Title\nSome body text that is long enough to keep."):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_filesystem_source_caps_the_file_count(tmp_path):
    for i in range(4):
        _write(tmp_path, f"doc{i}.md")
    with pytest.raises(ResourceLimitExceeded, match="matching documents"):
        list(FilesystemDocumentSource(tmp_path, max_files=2).documents())


def test_filesystem_source_caps_a_single_file(tmp_path):
    _write(tmp_path, "big.md", "# Big\n" + "word " * 5_000)
    with pytest.raises(ResourceLimitExceeded, match="per-file limit"):
        list(FilesystemDocumentSource(tmp_path, max_file_bytes=64).documents())


def test_filesystem_source_caps_how_much_of_the_tree_it_walks(tmp_path):
    for i in range(10):
        _write(tmp_path, f"sub{i}/doc.md")
    with pytest.raises(ResourceLimitExceeded, match="entries"):
        list(FilesystemDocumentSource(tmp_path, max_scanned_entries=3).documents())


def test_filesystem_source_reports_no_host_paths_when_the_root_is_missing(tmp_path):
    missing = tmp_path / "secret-project" / "nope"
    with pytest.raises(Exception) as excinfo:
        list(FilesystemDocumentSource(missing).documents())
    assert str(tmp_path) not in str(excinfo.value)


def test_zip_source_refuses_a_member_that_expands_past_the_cap(tmp_path):
    """A zip bomb lies in its header, so the cap is applied to bytes actually read."""
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("export/huge.md", "# Huge\n" + ("a" * 4_000_000))
    assert archive.stat().st_size < 100_000, "the archive itself is small; the member is not"
    with pytest.raises(ResourceLimitExceeded, match="decompression limit"):
        list(ZipDocumentSource(archive, max_member_bytes=32_768).documents())


def test_zip_source_caps_the_member_count(tmp_path):
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(5):
            zf.writestr(f"export/doc{i}.md", "# Doc\nbody text here")
    with pytest.raises(ResourceLimitExceeded, match="matching members"):
        list(ZipDocumentSource(archive, max_members=2).documents())


def test_zip_source_spends_a_shared_total_budget(tmp_path):
    archive = tmp_path / "total.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(3):
            zf.writestr(f"export/doc{i}.md", "# Doc\n" + "x" * 2_000)
    with pytest.raises(ResourceLimitExceeded):
        list(ZipDocumentSource(archive, max_total_bytes=3_000).documents())
