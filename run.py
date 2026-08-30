"""Bootstrap launcher for the src/ layout.

The package lives under ``src/`` and is normally installed (``pip install -e .``). This shim
puts ``src`` on ``sys.path`` so the service and the demo also run straight from a clone::

    python run.py demo
    python run.py serve --port 8080 --index bm25

It contains no application logic; everything is delegated to ``knowbase.__main__``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from knowbase.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
