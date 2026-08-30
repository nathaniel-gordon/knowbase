"""The evaluation harness pointed at data the caller supplies.

``Container.evaluate`` always took a ``cases`` argument and no caller ever passed one, so every
evaluation was restricted to the two bundled suites. A retrieval harness you cannot aim at your
own labelled questions is a demo, not a harness, which is what these tests are here to keep
honest.
"""
from __future__ import annotations

import pytest

from knowbase.app.client import ServiceError, running_service
from knowbase.app.config import AppConfig

CORPUS = {
    "leave": "# Time Off\nEmployees receive 21 paid vacation days each year.",
    "passwords": "# Password Rules\nPasswords must be at least 14 characters long.",
    "vpn": "# Remote Access\nThe GlobalConnect VPN client is required for remote access.",
}

CASES = [
    {"question": "how many vacation days do employees get", "expected_doc_id": "leave"},
    {"question": "minimum password length", "expected_doc_id": "passwords"},
    {"question": "which vpn client is required", "expected_doc_id": "vpn"},
]


@pytest.fixture
def client(corpus_roots):
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config) as live:
        live.ingest_documents(CORPUS)
        yield live


def test_a_caller_can_score_their_own_labelled_questions(client):
    report = client.evaluate(cases=CASES, k=3)
    assert report.n == 3
    assert report.suite == "custom"
    assert report.hit_at_1 == 1.0 and report.mrr == 1.0


def test_the_rows_explain_each_case(client):
    report = client.evaluate(cases=CASES, k=3)
    assert [row["expected_doc_id"] for row in report.rows] == ["leave", "passwords", "vpn"]
    assert all(row["rank"] == 1 for row in report.rows)


def test_a_case_the_index_misses_is_reported_as_rank_zero(client):
    miss = [{"question": "zzzz qqqq wwww", "expected_doc_id": "leave"}]
    report = client.evaluate(cases=miss, k=3)
    assert report.rows[0]["rank"] == 0
    assert report.hit_at_1 == 0.0 and report.mrr == 0.0


def test_custom_cases_work_through_compare_as_well(client):
    result = client.compare(["bm25", "tfidf"], cases=CASES, k=3)
    assert {r.index for r in result.reports} == {"bm25", "tfidf"}
    assert all(r.suite == "custom" and r.n == 3 for r in result.reports)


def test_supplied_cases_win_over_the_named_suite(client):
    report = client.evaluate(suite="core", cases=CASES, k=3)
    assert report.n == 3, "the bundled 12-question suite was not what the caller asked for"


def test_a_malformed_case_is_a_400_naming_the_position(client):
    with pytest.raises(ServiceError) as excinfo:
        client.evaluate(cases=[{"question": "ok?", "expected_doc_id": "leave"},
                               {"question": "missing its label"}])
    assert excinfo.value.status == 400
    assert excinfo.value.details["position"] == 1


def test_an_empty_case_list_is_a_400(client):
    with pytest.raises(ServiceError) as excinfo:
        client.evaluate(cases=[])
    assert excinfo.value.status == 400


# ------------------------------------------------------------------ pagination

@pytest.fixture
def many_cases() -> list[dict[str, str]]:
    return [{"question": f"vacation days question {i}", "expected_doc_id": "leave"}
            for i in range(120)]


def test_rows_are_paginated_with_an_envelope(client, many_cases):
    report = client.evaluate(cases=many_cases, k=3, limit=25, offset=0)
    assert len(report.rows) == 25
    assert report.page == {"limit": 25, "offset": 0, "total": 120, "returned": 25}


def test_a_later_page_continues_where_the_first_stopped(client, many_cases):
    first = client.evaluate(cases=many_cases, k=3, limit=25, offset=0)
    second = client.evaluate(cases=many_cases, k=3, limit=25, offset=25)
    assert first.rows[0] != second.rows[0]
    assert second.page["offset"] == 25
    assert second.rows[0]["question"] == many_cases[25]["question"]


def test_a_page_past_the_end_is_empty_not_an_error(client, many_cases):
    report = client.evaluate(cases=many_cases, k=3, limit=25, offset=500)
    assert report.rows == () and report.page["total"] == 120


def test_the_summary_metrics_cover_every_case_not_just_the_page(client, many_cases):
    page = client.evaluate(cases=many_cases, k=3, limit=5)
    whole = client.evaluate(cases=many_cases, k=3, rows=False)
    assert page.n == whole.n == 120
    assert page.mrr == whole.mrr


def test_an_oversized_page_is_refused(client, many_cases):
    with pytest.raises(ServiceError) as excinfo:
        client.evaluate(cases=many_cases, limit=10_000)
    assert excinfo.value.status == 400
    assert excinfo.value.details["field"] == "limit"


def test_rows_can_be_switched_off_entirely(client, many_cases):
    report = client.evaluate(cases=many_cases, k=3, rows=False)
    assert report.rows == () and report.page == {}
