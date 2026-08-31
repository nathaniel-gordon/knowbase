"""Adversarial input, end to end over the socket.

Everything here is something a caller can actually send. The service is unauthenticated by
default on loopback, so "a caller" includes anything running on the same machine, and
``/ingest`` reads whatever path it is handed -- which is why containment is the first section.
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
import zipfile

import pytest

from knowbase.app.client import ServiceError, running_service
from knowbase.app.config import AppConfig
from knowbase.app.limits import Limits

SECRET = "AWS_SECRET_ACCESS_KEY is hunter2trustno1 for prod."


@pytest.fixture
def corpus_dir(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "handbook.md").write_text(
        "# Handbook\nEmployees receive 21 paid vacation days each year.", encoding="utf-8")
    return root


@pytest.fixture
def secret_file(tmp_path):
    """A file the server can read but that sits outside every allowed root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    path = outside / "credentials.md"
    path.write_text(f"# Credentials\n{SECRET}", encoding="utf-8")
    return path


@pytest.fixture
def client(corpus_dir):
    config = AppConfig(clock="frozen", generator="extractive",
                       corpus_roots=(str(corpus_dir),))
    with running_service(config) as live:
        yield live


# ------------------------------------------------------- arbitrary file read

def test_a_path_outside_the_roots_cannot_be_ingested(client, secret_file):
    """The reported arbitrary-read primitive: ingest anything, then quote it back via /ask."""
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_path(str(secret_file.parent))
    assert excinfo.value.status == 403
    assert excinfo.value.code == "path_not_allowed"


def test_the_secret_is_not_reachable_through_ask(client, corpus_dir, secret_file):
    client.ingest_path(str(corpus_dir))
    with pytest.raises(ServiceError):
        client.ingest_path(str(secret_file.parent))
    answer = client.ask("what is the aws secret access key")
    assert "hunter2trustno1" not in answer.answer
    assert "credentials" not in json.dumps([c.__dict__ for c in answer.citations])


@pytest.mark.parametrize("suffix", ["/..", "/../..", "/../outside", "/./../outside"])
def test_traversal_out_of_an_allowed_root_is_refused(client, corpus_dir, suffix):
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_path(str(corpus_dir) + suffix)
    assert excinfo.value.status == 403


def test_a_windows_drive_root_is_refused(client):
    for candidate in ("C:\\", "C:\\Windows", "/etc", "//server/share"):
        with pytest.raises(ServiceError) as excinfo:
            client.ingest_path(candidate)
        assert excinfo.value.status == 403


def test_a_zip_outside_the_roots_is_refused_too(client, tmp_path):
    archive = tmp_path / "outside.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("export/leak.md", f"# Leak\n{SECRET}")
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_path(str(archive), source="zip")
    assert excinfo.value.status == 403


def test_a_symlink_pointing_out_of_the_root_is_skipped(client, corpus_dir, secret_file):
    """Containment is re-checked per file, because a link inside a root can point anywhere."""
    link = corpus_dir / "linked.md"
    try:
        link.symlink_to(secret_file)
    except (OSError, NotImplementedError):
        pytest.skip("this platform/user cannot create symlinks")
    client.ingest_path(str(corpus_dir))
    answer = client.ask("what is the aws secret access key")
    assert "hunter2trustno1" not in answer.answer


def test_a_rejection_does_not_disclose_the_configured_roots(client):
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_path("/etc/shadow")
    rendered = excinfo.value.message + json.dumps(excinfo.value.details)
    assert "corpus" not in rendered.lower() or "corpus root" in rendered.lower()
    assert "Temp" not in rendered


# --------------------------------------------------------------- type confusion

@pytest.mark.parametrize("body,expected_field", [
    ({"question": "hi", "k": []}, "k"),
    ({"question": "hi", "k": {}}, "k"),
    ({"question": 5}, "question"),
    ({"question": None}, "question"),
])
def test_type_confusion_is_a_400_not_a_500(client, body, expected_field):
    """These three shapes used to come back as 500s carrying a Python TypeError."""
    with pytest.raises(ServiceError) as excinfo:
        client._api("POST", "ask", body)
    assert excinfo.value.status == 400
    assert excinfo.value.code == "invalid_request"
    assert excinfo.value.details.get("field") == expected_field


def test_a_non_list_indexes_field_is_a_400(client):
    with pytest.raises(ServiceError) as excinfo:
        client._api("POST", "compare", {"indexes": 5})
    assert excinfo.value.status == 400


def test_non_string_exts_are_a_400(client):
    with pytest.raises(ServiceError) as excinfo:
        client._api("POST", "ingest", {"source": "filesystem", "path": ".", "exts": [1, 2]})
    assert excinfo.value.status == 400


def test_no_python_exception_name_ever_reaches_the_client(client):
    for body in ({"question": []}, {"k": 1}, {"question": "x", "extra": 1}):
        with pytest.raises(ServiceError) as excinfo:
            client._api("POST", "ask", body)
        rendered = json.dumps(excinfo.value.payload)
        for leak in ("TypeError", "AttributeError", "Traceback", "File \""):
            assert leak not in rendered


def test_a_stopword_only_corpus_fails_without_sklearns_wording(client):
    """The index adapters must translate the driver's 'empty vocabulary' into a domain error."""
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_documents({"noise": "the and or but if then else for to of in on at by"})
    assert excinfo.value.status == 400
    assert excinfo.value.code == "index_build_failed"
    assert "vocabulary" not in excinfo.value.message.lower() or "sklearn" not in str(
        excinfo.value.payload)
    assert "stop words" not in json.dumps(excinfo.value.payload)


# --------------------------------------------------------------------- limits

def test_k_is_capped(client, corpus_dir):
    client.ingest_path(str(corpus_dir))
    with pytest.raises(ServiceError) as excinfo:
        client.ask("vacation days", k=1_000_000_000)
    assert excinfo.value.status == 400
    assert excinfo.value.details["maximum"] == Limits().max_k


def test_an_oversized_body_is_refused_before_it_is_parsed():
    """Refused on the declared length alone: the megabytes are never read into memory."""
    limits = Limits(max_body_bytes=4096)
    with running_service(AppConfig(clock="frozen", generator="extractive",
                                   limits=limits)) as small:
        host, _, port = small.base_url.removeprefix("http://").partition(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=10)
        try:
            conn.putrequest("POST", "/api/v1/ask")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", "50000000")
            conn.endheaders()
            response = conn.getresponse()
            status, body = response.status, json.loads(response.read())
            closed = response.headers.get("Connection")
        finally:
            conn.close()
    assert status == 413
    assert body["error"]["code"] == "resource_limit_exceeded"
    assert closed == "close", "an unread body must not be parsed as the next request"


def test_a_chunked_body_is_refused_rather_than_read_as_empty(client):
    """It used to be silently processed as {}, which is a request nobody sent.

    Deliberately sends no chunk data: the refusal is on the Transfer-Encoding header alone,
    and writing chunks after it races the server's close (the RST can discard the buffered
    400 on Windows). The Connection: close assertion is what proves unsent chunks could
    never be parsed as a next request.
    """
    host, _, port = client.base_url.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        conn.putrequest("POST", "/api/v1/ask")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Transfer-Encoding", "chunked")
        conn.endheaders()
        response = conn.getresponse()
        body = json.loads(response.read())
        closed = response.headers.get("Connection")
    finally:
        conn.close()
    assert response.status == 400
    assert body["error"]["code"] == "invalid_request"
    assert closed == "close", "whatever the client was about to send must not be re-parsed"


def test_a_lying_content_length_is_refused(client):
    host, _, port = client.base_url.removeprefix("http://").partition(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    try:
        conn.putrequest("POST", "/api/v1/ask")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        response = conn.getresponse()
        status = response.status
        response.read()
    finally:
        conn.close()
    assert status == 400


def test_too_many_inline_documents_are_refused(client):
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_documents({f"doc{i}": "body text goes here" for i in range(600)})
    assert excinfo.value.status == 400
    assert excinfo.value.details["limit"] == Limits().max_documents


def test_a_huge_inline_document_is_refused(client):
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_documents({"big": "word " * 400_000})
    assert excinfo.value.status == 400


# ------------------------------------------------------------- odd but legal input

def test_a_document_of_pure_whitespace_is_an_empty_corpus(client):
    with pytest.raises(ServiceError) as excinfo:
        client.ingest_documents({"blank": "   \n\t  "})
    assert excinfo.value.status == 400
    assert excinfo.value.code in {"empty_corpus", "source_unavailable"}


def test_binary_content_decoded_with_replacement_does_not_crash(client, corpus_dir):
    (corpus_dir / "binary.md").write_bytes(b"# Notes\n\xff\xfe\x00 pension policy \xc3\x28")
    report = client.ingest_path(str(corpus_dir))
    assert report.documents == 2
    assert client.ask("pension policy").answer


def test_unicode_and_rtl_text_round_trip(client):
    client.ingest_documents({
        "fr": "# Congés\nLes employés reçoivent 21 jours de congés payés chaque année.",
        "ar": "# سياسة\nيحصل الموظفون على إجازة سنوية مدتها 21 يوما.",
    })
    answer = client.ask("combien de jours de congés payés")
    assert "21" in answer.answer
    assert answer.citations


def test_a_question_of_only_punctuation_abstains_rather_than_failing(client, corpus_dir):
    client.ingest_path(str(corpus_dir))
    answer = client.ask("?!?!?! ... ---")
    assert answer.citations == ()
    assert "No relevant information" in answer.answer


def test_a_document_that_is_one_long_line_still_chunks(client):
    body = "This policy sentence is repeated many times over. " * 2_000
    report = client.ingest_documents({"long": f"# Long\n{body}"})
    assert report.passages > 1
    assert client.ask("policy sentence repeated").citations


def test_a_document_id_with_path_characters_cannot_escape_anything(client):
    report = client.ingest_documents({"../../etc/passwd": "# Odd\nThis is a normal document."})
    assert report.documents == 1
    answer = client.ask("normal document")
    assert answer.citations[0].passage_id == "../../etc/passwd#0"  # an id, not a path


def test_a_json_array_body_is_refused(client):
    request = urllib.request.Request(f"{client.base_url}/api/v1/ask", data=b"[1,2,3]",
                                     method="POST",
                                     headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 400
