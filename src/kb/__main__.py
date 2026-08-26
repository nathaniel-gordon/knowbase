"""``python -m kb <demo|serve>`` -- the module entry point.

``serve`` is the operational entry point (a long-running JSON service); ``demo`` is the
self-contained walkthrough that starts and stops its own server.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .app.config import AppConfig
from .app.demo import main as demo_main
from .app.service import serve_forever

USAGE = "usage: python -m kb {demo|serve} [options]   (try 'demo --help' or 'serve --help')"


def _serve(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="kb serve",
                                     description="Run the knowledge-base JSON service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = ephemeral port chosen by the OS")
    parser.add_argument("--index", default="hybrid_rrf", help="IndexPort adapter to bind")
    parser.add_argument("--generator", default="claude",
                        help="GeneratorPort adapter to bind ('claude' degrades to 'extractive')")
    parser.add_argument("--cache", default="memory", help="CachePort adapter to bind")
    parser.add_argument("--config", help="path to a JSON AppConfig file (overrides the flags)")
    args = parser.parse_args(list(argv))
    config = (AppConfig.from_json_file(args.config) if args.config
              else AppConfig(index=args.index, generator=args.generator, cache=args.cache))
    serve_forever(config, host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else ""
    if command == "demo":
        return demo_main(argv[1:])
    if command == "serve":
        return _serve(argv[1:])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
