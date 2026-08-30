"""Contract battery for DocumentSourcePort."""
from __future__ import annotations

import pytest

from knowbase.domain.errors import DocumentSourceUnavailable
from knowbase.domain.model import Document
from knowbase.ports import DocumentSourcePort


class DocumentSourceContract:
    """Every DocumentSourcePort adapter must pass all of these.

    Subclasses provide two fixtures:

    ``source``          an adapter loaded with the shared fixture corpus
    ``broken_source``   the same adapter pointed at a backing store that cannot be read
    """

    @pytest.fixture
    def source(self) -> DocumentSourcePort:
        raise NotImplementedError("contract subclass must provide a 'source' fixture")

    @pytest.fixture
    def broken_source(self) -> DocumentSourcePort:
        raise NotImplementedError("contract subclass must provide a 'broken_source' fixture")

    def test_satisfies_the_protocol(self, source):
        assert isinstance(source, DocumentSourcePort)
        assert isinstance(source.name, str) and source.name == source.name.lower()

    def test_yields_the_whole_fixture_corpus(self, source, fixture_documents):
        produced = list(source.documents())
        assert [d.doc_id for d in produced] == [d.doc_id for d in fixture_documents]

    def test_documents_are_well_formed(self, source):
        for document in source.documents():
            assert isinstance(document, Document)
            assert document.doc_id.strip()
            assert document.title.strip()
            assert len(document.text.strip()) > 20
            assert document.origin, "origin must let an operator find the document again"

    def test_document_ids_are_unique(self, source):
        ids = [d.doc_id for d in source.documents()]
        assert len(ids) == len(set(ids))

    def test_iteration_is_repeatable_and_ordered(self, source):
        first = [(d.doc_id, d.text) for d in source.documents()]
        second = [(d.doc_id, d.text) for d in source.documents()]
        assert first == second, "passage ordinals (and therefore citations) depend on this"
        assert [i for i, _ in first] == sorted(i for i, _ in first)

    def test_titles_come_from_the_document_heading(self, source):
        titles = {d.doc_id: d.title for d in source.documents()}
        assert titles["passwords"] == "Password Rules"
        assert titles["vpn"] == "Remote Access Rules"

    def test_describe_locates_the_backing_store(self, source):
        described = source.describe()
        assert described.startswith(source.name)
        assert len(described) > len(source.name) + 1

    def test_unreadable_store_raises_the_domain_error(self, broken_source):
        with pytest.raises(DocumentSourceUnavailable):
            list(broken_source.documents())
