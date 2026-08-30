"""End-to-end demo: start the service, drive it through the SDK, shut it down.

Every number printed below is computed at runtime by the running service over the bundled
corpus. Nothing is hardcoded and nothing is compared against a figure this run did not
produce.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .client import KbClient, running_service
from .config import AppConfig
from .corpus import write_demo_corpus

DEMO_QUESTIONS = [
    "How long do I have to submit an expense report?",
    "What happens when the error budget burns during a canary?",
    "How do I get access to the production subnet?",
]


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run_demo(output_dir: Path, deterministic: bool = True) -> int:
    """Drive the full hexagon over HTTP. Returns a process exit code."""
    docs_dir = output_dir / "demo_docs"
    n_docs = write_demo_corpus(docs_dir)

    config = AppConfig(clock="frozen" if deterministic else "system",
                       cache="sqlite",
                       sqlite_path=str(output_dir / "answers.sqlite3"),
                       source="filesystem",
                       source_path=str(docs_dir))

    with running_service(config) as client:
        _rule("1. Service up -- composition root wiring")
        print(f"   endpoint          {client.base_url}")
        health = client.health()
        for port, bound in health["wiring"].items():
            print(f"   {port:<18}{bound}")
        catalogue = client.adapters()["ports"]
        print("\n   adapters registered per port:")
        for port, names in catalogue.items():
            print(f"   {port:<22}{', '.join(names)}")

        _rule("2. POST /ingest -- filesystem source -> every index adapter")
        ingest = client.ingest_path(str(docs_dir))
        print(f"   wrote {n_docs} demo documents to {docs_dir}")
        print(f"   documents={ingest.documents} passages={ingest.passages} "
              f"indexes=[{ingest.index}] elapsed={ingest.elapsed_ms:.1f}ms")

        _rule("3. POST /eval -- retrieval quality of the active index (core suite)")
        core = client.evaluate(suite="core", k=5, rows=False)
        print(f"   {core.summary()}")

        _rule("4. POST /ask -- cited answers")
        for question in DEMO_QUESTIONS:
            print(client.ask(question).render())
            print("-" * 78)
        repeat = client.ask(DEMO_QUESTIONS[0])
        print(f"   repeat of question 1 -> cached={repeat.cached} "
              f"(sqlite cache, {client.health()['cache']['entries']} entries)")

        _rule("5. POST /compare -- A/B the same eval battery through three index adapters")
        for suite in ("core", "hard"):
            comparison = client.compare(["hybrid_rrf", "bm25", "tfidf"], suite=suite, k=5)
            print(f"   suite={suite}")
            for line in comparison.table().splitlines():
                print(f"     {line}")

        _rule("6. POST /config -- hot-swap the IndexPort with zero domain changes")
        swapped = client.configure(index="bm25")
        print(f"   changed: {swapped['changed']}")
        bm25_core = client.evaluate(suite="core", k=5, rows=False)
        bm25_hard = client.evaluate(suite="hard", k=5, rows=False)
        print(f"   {bm25_core.summary()}")
        print(f"   {bm25_hard.summary()}")
        client.configure(index="hybrid_rrf")
        back_core = client.evaluate(suite="core", k=5, rows=False)
        back_hard = client.evaluate(suite="hard", k=5, rows=False)
        print("   swapped back to hybrid_rrf:")
        print(f"   {back_core.summary()}")
        print(f"   {back_hard.summary()}")

        _rule("7. Hot-swap the DocumentSourcePort -- inline documents, same use-cases")
        inline = client.ingest_documents({
            "refund-policy": "# Refund Policy\nRefunds are issued within 14 business days of an "
                             "approved request, to the original payment method only.",
            "shipping-policy": "# Shipping Policy\nStandard shipping takes 5 business days; "
                               "expedited shipping arrives the next business day.",
        }, label="demo-inline")
        print(f"   source={inline.source} documents={inline.documents} "
              f"passages={inline.passages}")
        print(client.ask("How long does a refund take?", k=2).render())

        _rule("8. Verdict")
        _summarise(client, core, back_core, back_hard)

    print("\nservice stopped cleanly.")
    return 0


def _summarise(client: KbClient, core_before, core_after, hard_after) -> None:
    """Print what this run measured, before and after the index was swapped and swapped back."""
    print(f"   core suite  hit@1 {core_before.hit_at_1:.3f} -> {core_after.hit_at_1:.3f}, "
          f"mrr {core_before.mrr:.3f} -> {core_after.mrr:.3f}  (hybrid_rrf, both times)")
    print(f"   hard suite  hit@1 {hard_after.hit_at_1:.3f} mrr {hard_after.mrr:.3f} "
          f"(paraphrase battery)")
    cache = client.health()["cache"]
    print(f"   answer cache hits={cache['hits']} misses={cache['misses']} "
          f"entries={cache['entries']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowbase demo", description=__doc__)
    parser.add_argument("--output", default=str(Path.cwd() / "output"),
                        help="where the demo corpus and sqlite cache are written")
    parser.add_argument("--live-clock", action="store_true",
                        help="use the system clock instead of the deterministic frozen clock")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    return run_demo(output, deterministic=not args.live_clock)
