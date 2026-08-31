"""The two things ``python -m knowbase`` can do, exercised for real.

The demo is the repo's front door and had no test at all: a refactor could break it and
nothing would go red until someone ran it by hand. The test below runs the whole thing --
its own server, ingest, eval, compare, hot-swap -- and checks the printed transcript for
the facts it claims. The serve path is checked up to the blocking call by swapping
``serve_forever`` for a recorder, because a test must not adopt a foreground server.
"""
from __future__ import annotations

import pytest

from knowbase import __main__ as entry
from knowbase.app.config import AppConfig
from knowbase.app.demo import main as demo_main


class TestDispatch:
    def test_no_command_prints_usage_and_exits_2(self, capsys):
        assert entry.main([]) == 2
        assert "usage" in capsys.readouterr().err

    def test_an_unknown_command_is_not_guessed_at(self, capsys):
        assert entry.main(["srve"]) == 2
        assert "usage" in capsys.readouterr().err


class TestServeFlags:
    """Flag parsing is the only logic ``serve`` owns; the server itself is tested elsewhere."""

    @pytest.fixture
    def recorded(self, monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(entry, "serve_forever",
                            lambda config, host, port, log_level: calls.append(
                                (config, host, port, log_level)))
        return calls

    def test_flags_reach_the_config(self, recorded, tmp_path):
        root_a, root_b = tmp_path / "a", tmp_path / "b"
        entry.main(["serve", "--store", "sqlite", "--cache", "sqlite",
                    "--sqlite-path", str(tmp_path / "kb.sqlite3"),
                    "--corpus-root", str(root_a), "--corpus-root", str(root_b),
                    "--port", "8123", "--log-level", "warning"])
        config, host, port, log_level = recorded[0]
        assert (config.store, config.cache) == ("sqlite", "sqlite")
        assert config.corpus_roots == (str(root_a), str(root_b))
        assert (host, port, log_level) == ("127.0.0.1", 8123, "warning")

    def test_a_config_file_overrides_the_flags(self, recorded, tmp_path):
        path = tmp_path / "app.json"
        path.write_text('{"index": "bm25", "cache_ttl_s": 30}', encoding="utf-8")
        entry.main(["serve", "--config", str(path), "--index", "tfidf"])
        config = recorded[0][0]
        assert config.index == "bm25"
        assert config.cache_ttl_s == 30

    def test_defaults_match_the_documented_wiring(self, recorded):
        entry.main(["serve"])
        config = recorded[0][0]
        assert config == AppConfig(index="hybrid_rrf", generator="claude", cache="memory",
                                   store="memory", sqlite_path=":memory:", corpus_roots=())


def test_the_demo_runs_end_to_end_and_prints_what_it_measured(tmp_path, capsys):
    """The full walkthrough: server up, ingest, ask, eval, compare, swap, clean stop."""
    assert demo_main(["--output", str(tmp_path)]) == 0
    transcript = capsys.readouterr().out
    assert "POST /ingest" in transcript
    assert "documents=14" in transcript, "the bundled corpus is 14 documents"
    assert "cached=True" in transcript, "the repeated question must hit the sqlite cache"
    assert "changed: {'index': {'from': 'hybrid_rrf', 'to': 'bm25'}}" in transcript, \
        "the hot swap must report itself"
    assert "service stopped cleanly." in transcript
    assert (tmp_path / "knowbase.sqlite3").exists(), "the demo cache is real, on disk"
    # every metric line is computed this run; the transcript must never claim otherwise
    assert "hit@1" in transcript and "mrr" in transcript
