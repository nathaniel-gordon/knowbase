"""Operational surface of the service: versioning, probes, metrics, logs, auth.

None of this is knowledge-base behaviour, which is exactly why it is tested apart from the
retrieval suites. It is the part an operator interacts with when something is wrong.
"""
from __future__ import annotations

import io
import json
import logging
import urllib.error
import urllib.request

import pytest

from knowbase.app.auth import TOKEN_ENV_VAR, Authenticator, Unauthenticated, is_loopback
from knowbase.app.client import ServiceError, running_service
from knowbase.app.config import AppConfig
from knowbase.app.observability import (
    JsonFormatter,
    Metrics,
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
)
from knowbase.app.service import API_PREFIX, create_server

TOKEN = "a-token-long-enough-to-be-real"
DOCS = {"pto": "# Time Off\nEmployees receive 21 paid vacation days each year."}


@pytest.fixture
def service(corpus_roots):
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config) as live:
        yield live


# ------------------------------------------------------------------ versioning

def test_the_index_advertises_the_api_and_the_probes(service):
    index = service.index()
    assert index["api"] == API_PREFIX
    assert f"{API_PREFIX}/ask" in index["endpoints"]
    assert index["operational"] == ["/health", "/ready", "/metrics"]
    assert index["authentication"] == "none"


def test_the_unversioned_paths_are_gone(service):
    """The API lives under /api/v1; only the probes sit outside it."""
    for path in ("/ask", "/ingest", "/adapters"):
        with pytest.raises(ServiceError) as excinfo:
            service._request("GET", path)
        assert excinfo.value.status == 404


def test_a_404_names_the_routes_that_do_exist(service):
    with pytest.raises(ServiceError) as excinfo:
        service._request("GET", f"{API_PREFIX}/nope")
    assert excinfo.value.code == "no_such_endpoint"
    assert "/health" in excinfo.value.details["routes"]


# ----------------------------------------------------------------- health/ready

def test_ready_is_false_until_a_corpus_exists(service):
    assert service.ready() is False
    service.ingest_documents(DOCS)
    assert service.ready() is True


def test_ready_answers_503_before_the_first_ingest(service):
    request = urllib.request.Request(f"{service.base_url}/ready")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 503
    body = json.loads(excinfo.value.read())
    assert body["ready"] is False and body["reason"]


def test_health_is_ok_even_with_no_corpus(service):
    """Liveness and readiness are different questions; /health must not answer the wrong one."""
    assert service.health()["status"] == "ok"
    assert service.health()["corpus"] is None


def test_health_names_the_corpus_store(service):
    assert service.health()["wiring"]["store"] == "memory"


# --------------------------------------------------------------------- metrics

def test_metrics_count_requests_per_route(service):
    service.ingest_documents(DOCS)
    service.ask("How many vacation days?")
    service.ask("How many vacation days?")

    metrics = service.metrics()
    ask = metrics["routes"][f"{API_PREFIX}/ask"]
    assert ask["requests"] == 2
    assert ask["by_status"] == {"2xx": 2}
    assert ask["avg_ms"] > 0
    assert metrics["totals"]["requests"] >= 3


def test_metrics_count_failures_separately(service):
    with pytest.raises(ServiceError):
        service.ask("")
    route = service.metrics()["routes"][f"{API_PREFIX}/ask"]
    assert route["errors"] == 1 and route["by_status"] == {"4xx": 1}


def test_metrics_report_cache_and_corpus_gauges(service):
    service.ingest_documents(DOCS)
    service.ask("How many vacation days?")
    service.ask("How many vacation days?")
    metrics = service.metrics()
    assert metrics["cache"]["hits"] == 1
    assert 0.0 < metrics["cache"]["hit_rate"] <= 1.0
    assert metrics["corpus"]["passages"] == 1
    assert metrics["corpus"]["fingerprint"]


def test_latency_buckets_are_monotonic_and_summed():
    metrics = Metrics()
    for ms in (0.5, 3, 3, 40, 9000):
        metrics.observe("/x", 200, ms)
    route = metrics.snapshot()["routes"]["/x"]
    assert route["requests"] == 5
    assert sum(route["buckets_ms"].values()) == 4, "the 9000ms sample lands in the overflow"
    assert route["over_bucket_ms"] == 1
    assert route["max_ms"] == 9000


def test_a_quantile_over_the_largest_bucket_says_so():
    metrics = Metrics()
    metrics.observe("/x", 200, 60_000)
    assert metrics.snapshot()["routes"]["/x"]["p95_ms_upper_bound"] == -1.0


# --------------------------------------------------------------------- logging

def _capture() -> tuple[io.StringIO, logging.Logger]:
    stream = io.StringIO()
    return stream, configure_logging("INFO", stream)


def test_every_request_emits_one_json_line_carrying_its_id(corpus_roots):
    stream, _ = _capture()
    try:
        config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
        with running_service(config) as client:
            client.ingest_documents(DOCS)
            client.ask("How many vacation days?")
        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finally:
        configure_logging("INFO")

    asks = [line for line in lines if line.get("route") == f"{API_PREFIX}/ask"]
    assert len(asks) == 1
    entry = asks[0]
    assert entry["event"] == "request"
    assert entry["status"] == 200
    assert entry["method"] == "POST"
    assert entry["duration_ms"] >= 0
    assert entry["cache"] == "miss"
    assert entry["citations"] >= 1
    assert len(entry["request_id"]) == 16


def test_the_question_text_never_reaches_the_log(corpus_roots):
    """Questions are user content; the log records the shape of one, not its contents."""
    secret = "what is my colleague's home address"
    stream, _ = _capture()
    try:
        config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
        with running_service(config) as client:
            client.ingest_documents(DOCS)
            client.ask(secret)
        rendered = stream.getvalue()
    finally:
        configure_logging("INFO")
    assert secret not in rendered
    assert f'"question_chars": {len(secret)}' in rendered


def test_a_failing_request_logs_the_id_the_client_was_given(corpus_roots):
    stream, _ = _capture()
    try:
        config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
        with running_service(config) as client:
            with pytest.raises(ServiceError) as excinfo:
                client.ask("")
            request_id = excinfo.value.request_id
        lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    finally:
        configure_logging("INFO")

    assert request_id, "the error body must carry the correlation id"
    matching = [line for line in lines if line.get("request_id") == request_id]
    assert len(matching) == 1
    assert matching[0]["status"] == 400


def test_the_request_id_is_also_a_response_header(service):
    request = urllib.request.Request(f"{service.base_url}/health")
    with urllib.request.urlopen(request, timeout=10) as response:
        assert len(response.headers["X-Request-Id"]) == 16


def test_request_ids_are_unique():
    assert len({new_request_id() for _ in range(500)}) == 500


def test_the_formatter_promotes_extras_to_top_level_fields():
    record = logging.LogRecord("knowbase", logging.INFO, "f", 1, "ingested", (), None)
    record.documents = 14
    token = request_id_var.set("abc123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["event"] == "ingested"
    assert payload["documents"] == 14
    assert payload["request_id"] == "abc123"
    assert payload["level"] == "info"


def test_configuring_logging_twice_does_not_double_the_output():
    stream, _ = _capture()
    try:
        configure_logging("INFO", stream)
        get_logger().info("once")
        assert stream.getvalue().count('"event": "once"') == 1
    finally:
        configure_logging("INFO")


# ------------------------------------------------------------------------ auth

def test_no_token_configured_means_no_authentication():
    auth = Authenticator.from_environment("")
    assert auth.enabled is False
    auth.authorise(None)  # does not raise


def test_a_token_is_required_once_configured():
    auth = Authenticator(token=TOKEN)
    auth.authorise(f"Bearer {TOKEN}")
    for header in (None, "", "Bearer", "Basic abc", f"Bearer {TOKEN}x", "Bearer wrong"):
        with pytest.raises(Unauthenticated):
            auth.authorise(header)


def test_a_short_token_is_rejected_at_startup():
    with pytest.raises(ValueError, match="at least"):
        Authenticator.from_environment("tiny")


def test_the_token_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
    assert Authenticator.from_environment().token == TOKEN
    monkeypatch.delenv(TOKEN_ENV_VAR)
    assert Authenticator.from_environment().enabled is False


@pytest.mark.parametrize("host,loopback", [("127.0.0.1", True), ("localhost", True),
                                           ("::1", True), ("0.0.0.0", False),
                                           ("10.0.0.5", False), ("example.com", False)])
def test_loopback_detection(host, loopback):
    assert is_loopback(host) is loopback


def test_binding_a_public_address_without_a_token_is_refused(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="refusing to bind"):
        create_server(AppConfig(clock="frozen"), host="0.0.0.0", port=0)


def test_binding_a_public_address_with_a_token_is_allowed(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
    Authenticator.from_environment().check_bind("0.0.0.0")


def test_guarded_endpoints_reject_an_anonymous_caller(corpus_roots):
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config, token=TOKEN) as client:
        client.ingest_documents(DOCS)          # the SDK carries the token
        anonymous = type(client)(client.base_url)
        for call in (lambda: anonymous.ask("anything"),
                     lambda: anonymous.ingest_documents(DOCS),
                     lambda: anonymous.adapters(),
                     lambda: anonymous.configure(index="bm25")):
            with pytest.raises(ServiceError) as excinfo:
                call()
            assert excinfo.value.status == 401
            assert excinfo.value.code == "unauthenticated"


def test_probes_stay_open_when_auth_is_on(corpus_roots):
    """A load balancer cannot present a bearer token, so /health and /ready must not need one."""
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config, token=TOKEN) as client:
        anonymous = type(client)(client.base_url)
        assert anonymous.health()["status"] == "ok"
        assert anonymous.ready() is False
        assert anonymous.index()["authentication"] == "bearer"


def test_a_wrong_token_is_rejected(corpus_roots):
    config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
    with running_service(config, token=TOKEN) as client:
        wrong = type(client)(client.base_url, token="a-different-token-entirely")
        with pytest.raises(ServiceError) as excinfo:
            wrong.adapters()
        assert excinfo.value.status == 401


def test_the_token_never_appears_in_a_response_or_a_log(corpus_roots):
    stream, _ = _capture()
    try:
        config = AppConfig(clock="frozen", generator="extractive", corpus_roots=corpus_roots)
        with running_service(config, token=TOKEN) as client:
            rendered = json.dumps(client.index()) + json.dumps(client.health())
            client.ingest_documents(DOCS)
        rendered += stream.getvalue()
    finally:
        configure_logging("INFO")
    assert TOKEN not in rendered
