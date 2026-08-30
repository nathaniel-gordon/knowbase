"""Unit tests for the request-validation layer, the path allow-list, and the limits object.

These three modules sit between the socket and the hexagon. Everything they reject here is a
400/403/413 the HTTP tests would otherwise have to discover one endpoint at a time, and three
of the cases below are the type-confusion crashes the old handler turned into 500s.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from knowbase.app.limits import DEFAULT_LIMITS, Limits
from knowbase.app.paths import CorpusRoots
from knowbase.app.validation import Page, RequestValidator
from knowbase.domain.errors import PathNotAllowed, ValidationError


@pytest.fixture
def validator() -> RequestValidator:
    return RequestValidator(DEFAULT_LIMITS)


# ------------------------------------------------------------------------ /ask

def test_ask_requires_a_question(validator):
    with pytest.raises(ValidationError, match="'question' is required"):
        validator.ask({})


@pytest.mark.parametrize("question", ["", "   ", "\n\t "])
def test_blank_questions_are_rejected(validator, question):
    with pytest.raises(ValidationError, match="must not be blank"):
        validator.ask({"question": question})


def test_question_length_is_capped(validator):
    with pytest.raises(ValidationError) as excinfo:
        validator.ask({"question": "x" * (DEFAULT_LIMITS.max_question_chars + 1)})
    assert excinfo.value.details["limit"] == DEFAULT_LIMITS.max_question_chars


@pytest.mark.parametrize("k", [[], {}, "3", 3.5])
def test_non_integer_k_is_a_client_error_not_a_crash(validator, k):
    """``{"k": []}`` used to reach ``int()`` and come back as a 500 carrying a TypeError."""
    with pytest.raises(ValidationError, match="'k' must be an integer"):
        validator.ask({"question": "hi", "k": k})


def test_booleans_are_not_accepted_as_integers(validator):
    with pytest.raises(ValidationError, match="'k' must be an integer"):
        validator.ask({"question": "hi", "k": True})


@pytest.mark.parametrize("k", [0, -1, 10 ** 9])
def test_k_is_bounded_on_both_sides(validator, k):
    with pytest.raises(ValidationError, match="must be between"):
        validator.ask({"question": "hi", "k": k})


def test_ask_accepts_a_valid_body(validator):
    request = validator.ask({"question": "how long is the retention period?", "k": 3})
    assert request.k == 3


def test_unknown_fields_are_rejected(validator):
    with pytest.raises(ValidationError, match="unknown field"):
        validator.ask({"question": "hi", "kk": 3})


# --------------------------------------------------------------------- /ingest

def test_ingest_requires_a_path_for_file_sources(validator):
    with pytest.raises(ValidationError, match="'path' is required"):
        validator.ingest({"source": "filesystem"})


def test_exts_must_hold_strings(validator):
    """``{"exts": [1, 2]}`` used to reach ``str.startswith`` and come back as a 500."""
    with pytest.raises(ValidationError, match=r"'exts\[0\]' must be a non-empty string"):
        validator.ingest({"source": "filesystem", "path": ".", "exts": [1, 2]})


def test_documents_must_be_an_object(validator):
    with pytest.raises(ValidationError, match="must be an object"):
        validator.ingest({"source": "memory", "documents": ["a", "b"]})


def test_inline_document_count_is_capped(validator):
    tiny = Limits(max_documents=2)
    with pytest.raises(ValidationError, match="more than 2 entries"):
        RequestValidator(tiny).ingest(
            {"source": "memory", "documents": {"a": "x", "b": "y", "c": "z"}})


def test_inline_total_bytes_are_capped(validator):
    tiny = Limits(max_inline_total_bytes=16)
    with pytest.raises(ValidationError, match="exceed 16 bytes"):
        RequestValidator(tiny).ingest(
            {"source": "memory", "documents": {"a": "x" * 10, "b": "y" * 10}})


def test_memory_ingest_preserves_document_order(validator):
    request = validator.ingest({"source": "memory", "documents": {"b": "two", "a": "one"}})
    assert request.documents == (("b", "two"), ("a", "one"))


# ------------------------------------------------------------- /eval, /compare

def test_compare_indexes_must_be_a_list(validator):
    """``{"indexes": 5}`` used to reach ``list()`` and come back as a 500."""
    with pytest.raises(ValidationError, match="'indexes' must be a list of strings"):
        validator.compare({"indexes": 5})


def test_compare_needs_two_indexes_to_compare(validator):
    with pytest.raises(ValidationError, match="at least two"):
        validator.compare({"indexes": ["bm25"]})


def test_user_supplied_cases_are_parsed(validator):
    request = validator.evaluate({"cases": [{"question": "q?", "expected_doc_id": "d"}]})
    assert request.cases is not None
    assert request.cases[0].expected_doc_id == "d"


def test_cases_reject_unknown_keys(validator):
    with pytest.raises(ValidationError, match="unknown field"):
        validator.evaluate({"cases": [{"question": "q?", "expected_doc_id": "d", "gold": 1}]})


def test_cases_reject_missing_labels(validator):
    with pytest.raises(ValidationError, match="expected_doc_id"):
        validator.evaluate({"cases": [{"question": "q?"}]})


def test_case_count_is_capped(validator):
    tiny = RequestValidator(Limits(max_eval_cases=1))
    with pytest.raises(ValidationError, match="more than 1 items"):
        tiny.evaluate({"cases": [{"question": "a?", "expected_doc_id": "d"},
                                 {"question": "b?", "expected_doc_id": "d"}]})


def test_config_needs_at_least_one_port(validator):
    with pytest.raises(ValidationError, match="at least one port"):
        validator.configure({})


# ------------------------------------------------------------------ pagination

def test_page_defaults_come_from_limits(validator):
    page = validator.page({})
    assert (page.limit, page.offset) == (DEFAULT_LIMITS.default_page_size, 0)


def test_page_size_is_capped(validator):
    with pytest.raises(ValidationError, match="must be between"):
        validator.page({"limit": DEFAULT_LIMITS.max_page_size + 1})


def test_page_slices_and_reports_the_total():
    window, envelope = Page(limit=2, offset=1).apply(["a", "b", "c", "d"])
    assert window == ["b", "c"]
    assert envelope == {"limit": 2, "offset": 1, "total": 4, "returned": 2}


def test_page_past_the_end_is_empty_not_an_error():
    window, envelope = Page(limit=5, offset=99).apply(["a"])
    assert window == [] and envelope["total"] == 1


# ------------------------------------------------------------ path allow-list

def test_a_path_inside_a_root_resolves(tmp_path):
    (tmp_path / "docs").mkdir()
    roots = CorpusRoots([tmp_path])
    assert roots.resolve(tmp_path / "docs") == (tmp_path / "docs").resolve()


def test_traversal_out_of_the_root_is_refused(tmp_path):
    inside = tmp_path / "corpus"
    inside.mkdir()
    roots = CorpusRoots([inside])
    with pytest.raises(PathNotAllowed):
        roots.resolve(inside / ".." / ".." / "etc")


def test_an_absolute_path_elsewhere_is_refused(tmp_path):
    roots = CorpusRoots([tmp_path / "corpus"])
    with pytest.raises(PathNotAllowed):
        roots.resolve(Path.home())


def test_a_sibling_with_the_root_as_a_string_prefix_is_refused(tmp_path):
    """``/corpus-evil`` starts with ``/corpus``; only path containment may decide this."""
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus-evil").mkdir()
    roots = CorpusRoots([tmp_path / "corpus"])
    with pytest.raises(PathNotAllowed):
        roots.resolve(tmp_path / "corpus-evil")


def test_rejection_never_echoes_the_configured_roots(tmp_path):
    roots = CorpusRoots([tmp_path / "corpus"])
    with pytest.raises(PathNotAllowed) as excinfo:
        roots.resolve("/etc/passwd")
    rendered = f"{excinfo.value.message} {excinfo.value.details}"
    assert str(tmp_path) not in rendered


def test_a_null_byte_in_a_path_is_a_validation_error(tmp_path):
    with pytest.raises(ValidationError, match="null byte"):
        CorpusRoots([tmp_path]).resolve("docs\x00.md")


def test_no_configured_roots_means_the_working_directory():
    assert CorpusRoots().roots == (Path.cwd().resolve(),)


# ---------------------------------------------------------------------- limits

def test_limits_round_trip_through_a_mapping():
    limits = Limits(max_k=7, max_files=9)
    assert Limits.from_mapping(limits.to_dict()) == limits


def test_unknown_limits_are_rejected():
    with pytest.raises(ValueError, match="unknown limit"):
        Limits.from_mapping({"max_everything": 1})
