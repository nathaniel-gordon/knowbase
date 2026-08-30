"""HTTP integration tests: the driving adapter, end to end, over a real socket.

Every test here starts a server on an ephemeral 127.0.0.1 port in a background thread and stops
it again, so the suite never leaves a listener behind and never needs a fixed port.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from knowbase.app.client import ServiceError, running_service
from knowbase.app.config import AppConfig


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A live service with the demo corpus ingested, shared by the read-only tests."""
    corpus = tmp_path_factory.mktemp("http_docs")
    from knowbase.app.corpus import write_demo_corpus
    write_demo_corpus(corpus)
    config = AppConfig(clock="frozen", cache="memory", generator="extractive")
    with running_service(config) as live:
        live.ingest_path(str(corpus))
        yield live


@pytest.fixture
def fresh_client():
    """An isolated service with nothing ingested (for error-path tests)."""
    with running_service(AppConfig(clock="frozen", generator="extractive")) as live:
        yield live


# ------------------------------------------------------------------- lifecycle

def test_server_binds_an_ephemeral_loopback_port(fresh_client):
    host, _, port = fresh_client.base_url.removeprefix("http://").partition(":")
    assert host == "127.0.0.1"
    assert 1024 < int(port) < 65536


def test_health_reports_the_bound_adapters(client):
    health = client.health()
    assert health["status"] == "ok"
    wiring = health["wiring"]
    assert wiring["index"] == "hybrid_rrf"
    assert wiring["generator"] == "extractive"
    assert wiring["clock"] == "frozen"
    assert set(wiring["index_pool"]) == {"hybrid_rrf", "bm25", "tfidf"}
    assert health["indexed_passages"] == 14
    assert health["corpus"]["documents"] == 14


def test_adapters_endpoint_lists_every_port(client):
    ports = client.adapters()["ports"]
    assert ports["DocumentSourcePort"] == ["filesystem", "memory", "zip"]
    assert ports["IndexPort"] == ["bm25", "hybrid_rrf", "tfidf"]
    assert ports["GeneratorPort"] == ["claude", "extractive"]
    assert ports["CachePort"] == ["memory", "sqlite"]
    assert ports["ClockPort"] == ["frozen", "system"]


# --------------------------------------------------------------------- ingest

def test_ingest_from_a_folder(fresh_client, demo_corpus_dir):
    report = fresh_client.ingest_path(str(demo_corpus_dir))
    assert report.documents == 14 and report.passages == 14
    assert report.source == "filesystem"
    assert set(report.index.split("+")) == {"bm25", "hybrid_rrf", "tfidf"}


def test_ingest_from_inline_documents(fresh_client):
    report = fresh_client.ingest_documents({
        "refunds": "# Refunds\nRefunds are issued within 14 business days of approval.",
        "shipping": "# Shipping\nStandard shipping takes 5 business days.",
    })
    assert (report.source, report.documents, report.passages) == ("memory", 2, 2)
    answer = fresh_client.ask("How long do refunds take?", k=2)
    assert "14 business days" in answer.answer
    assert answer.citations[0].passage_id == "refunds#0"


def test_ingest_from_a_zip_export(fresh_client, fixture_zip):
    report = fresh_client.ingest_path(str(fixture_zip), source="zip")
    assert (report.source, report.documents) == ("zip", 4)
    assert fresh_client.health()["wiring"]["source"].startswith("zip:")


def test_reingest_invalidates_cached_answers(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    fresh_client.ask("How many PTO days do employees get per year?")
    assert fresh_client.health()["cache"]["entries"] == 1
    fresh_client.ingest_documents({"only": "# Only\nThis is the only document now."})
    assert fresh_client.health()["cache"]["entries"] == 0, (
        "answers grounded in the old corpus must not survive a re-ingest")


# ------------------------------------------------------------------------ ask

def test_ask_returns_a_cited_answer(client):
    answer = client.ask("How many PTO days do employees get per year?")
    assert "21" in answer.answer or "1.75" in answer.answer
    assert answer.index == "hybrid_rrf" and answer.generator == "extractive"
    assert answer.citations and answer.retrieved
    retrieved = {h.passage_id for h in answer.retrieved}
    assert {c.passage_id for c in answer.citations} <= retrieved
    assert [h.rank for h in answer.retrieved] == list(range(1, len(answer.retrieved) + 1))


def test_ask_honours_k(client):
    assert len(client.ask("security policy requirements", k=2).retrieved) <= 2


def test_second_identical_ask_is_cached(client):
    question = "When does the weekly release train cut?"
    client.ask(question)
    assert client.ask(question).cached is True


# ----------------------------------------------------------------- eval / A/B

def test_eval_preserves_the_pre_refactor_quality(client):
    report = client.evaluate(suite="core", k=5, rows=False)
    assert (report.index, report.n) == ("hybrid_rrf", 12)
    assert report.hit_at_1 == 1.0 and report.hit_at_k == 1.0 and report.mrr == 1.0


def test_eval_returns_per_question_rows(client):
    report = client.evaluate(suite="core", k=5, rows=True)
    assert len(report.rows) == 12
    assert all(row["rank"] == 1 for row in report.rows)


def test_compare_runs_one_battery_through_several_adapters(client):
    result = client.compare(["hybrid_rrf", "bm25", "tfidf"], suite="hard", k=5)
    assert {r.index for r in result.reports} == {"hybrid_rrf", "bm25", "tfidf"}
    assert all(r.n == 12 and r.suite == "hard" for r in result.reports)
    assert [r.mrr for r in result.reports] == sorted((r.mrr for r in result.reports),
                                                     reverse=True)
    assert result.winner != "tie" and result.margin > 0


def test_compare_on_the_saturated_core_suite_reports_a_tie(client):
    result = client.compare(["hybrid_rrf", "bm25"], suite="core", k=5)
    assert result.winner == "tie" and result.margin == 0.0


# ----------------------------------------------------------- runtime rebinding

def test_index_can_be_hot_swapped_without_reingesting(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    before = fresh_client.evaluate(suite="hard", k=5, rows=False)
    assert before.index == "hybrid_rrf"

    changed = fresh_client.configure(index="bm25")
    assert changed["changed"]["index"] == {"from": "hybrid_rrf", "to": "bm25"}
    assert changed["wiring"]["index"] == "bm25"

    after = fresh_client.evaluate(suite="hard", k=5, rows=False)
    assert after.index == "bm25"
    assert after.mrr != before.mrr, "swapping the strategy must change the measured behaviour"
    assert fresh_client.health()["indexed_passages"] == 14, "no re-ingest was needed"

    fresh_client.configure(index="hybrid_rrf")
    assert fresh_client.evaluate(suite="hard", k=5, rows=False).mrr == before.mrr


def test_generator_and_cache_can_be_hot_swapped(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    result = fresh_client.configure(generator="extractive", cache="sqlite")
    assert result["wiring"]["cache"] == "sqlite"
    answer = fresh_client.ask("What password length is required by the security policy?")
    assert "14 characters" in answer.answer
    assert fresh_client.health()["cache"]["entries"] == 1


# ------------------------------------------------------------------ error paths

def test_unknown_route_is_404(fresh_client):
    with pytest.raises(ServiceError) as excinfo:
        fresh_client._request("GET", "/nope")
    assert excinfo.value.status == 404
    assert "routes" in excinfo.value.payload


def test_asking_before_ingest_is_a_client_error(fresh_client):
    with pytest.raises(ServiceError) as excinfo:
        fresh_client.ask("anything at all")
    assert excinfo.value.status == 400
    assert excinfo.value.payload["kind"] == "IndexNotBuiltError"


def test_blank_question_is_rejected(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    with pytest.raises(ServiceError) as excinfo:
        fresh_client.ask("   ")
    assert excinfo.value.status == 400


def test_unknown_adapter_is_rejected_with_the_available_list(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    with pytest.raises(ServiceError) as excinfo:
        fresh_client.configure(index="faiss")
    assert excinfo.value.status == 400
    assert excinfo.value.payload["kind"] == "unknown_adapter"
    assert "hybrid_rrf" in excinfo.value.payload["error"]


def test_unknown_suite_is_rejected(fresh_client, demo_corpus_dir):
    fresh_client.ingest_path(str(demo_corpus_dir))
    with pytest.raises(ServiceError) as excinfo:
        fresh_client.evaluate(suite="does-not-exist")
    assert excinfo.value.status == 400


def test_missing_source_path_is_rejected(fresh_client):
    with pytest.raises(ServiceError) as excinfo:
        fresh_client.ingest_path("/no/such/folder/anywhere")
    assert excinfo.value.status == 400
    assert excinfo.value.payload["kind"] == "DocumentSourceUnavailable"


def test_malformed_json_body_is_rejected(fresh_client):
    request = urllib.request.Request(f"{fresh_client.base_url}/ask", data=b"{not json",
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 400
    assert json.loads(excinfo.value.read())["kind"] == "bad_request"
