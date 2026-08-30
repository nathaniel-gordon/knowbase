"""The architectural fitness function.

Hexagonal architecture is a claim about *dependency direction*, and a claim that is not checked
rots. These tests read the domain's own source with ``ast`` and then re-verify at runtime in a
clean interpreter, so a stray ``import numpy`` in a use-case fails the build rather than
quietly turning the core into a shell around scikit-learn.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
DOMAIN = SRC / "knowbase" / "domain"
PORTS = SRC / "knowbase" / "ports"

#: Everything the core is permitted to name. Pure computation only: no IO, no clock, no
#: randomness, no third-party package, and above all nothing from ``knowbase.adapters`` or ``knowbase.app``.
DOMAIN_ALLOWED_ROOTS = frozenset({
    "__future__", "dataclasses", "typing", "collections", "re", "hashlib", "math", "enum",
})

FORBIDDEN_IN_CORE = ("numpy", "sklearn", "scipy", "pandas", "anthropic", "sqlite3", "pathlib",
                     "os", "sys", "time", "datetime", "random", "json", "http", "urllib",
                     "socket", "threading", "pickle", "logging")


def _module_files(package: Path) -> list[Path]:
    return sorted(package.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Absolute import roots used by a module; relative imports are internal and ignored."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _relative_targets(path: Path) -> set[str]:
    """Packages reached by relative imports, e.g. ``from ..ports import`` -> {'ports'}."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            targets.add((node.module or "").split(".")[0])
    return targets


def test_the_domain_package_is_not_empty():
    modules = _module_files(DOMAIN)
    assert {p.name for p in modules} >= {"model.py", "text.py", "usecases.py", "errors.py"}


@pytest.mark.parametrize("module", _module_files(DOMAIN), ids=lambda p: p.name)
def test_domain_module_imports_only_pure_stdlib(module: Path):
    offenders = _imported_roots(module) - DOMAIN_ALLOWED_ROOTS
    assert not offenders, (
        f"{module.name} reaches outside the hexagon: {sorted(offenders)}. "
        f"Only {sorted(DOMAIN_ALLOWED_ROOTS)} are allowed in the core.")


@pytest.mark.parametrize("module", _module_files(DOMAIN), ids=lambda p: p.name)
def test_domain_module_never_reaches_an_adapter(module: Path):
    """Relative imports may only go inward: siblings in ``domain`` or up to ``ports``."""
    illegal = _relative_targets(module) - {"", "model", "text", "usecases", "errors", "ports"}
    assert not illegal, f"{module.name} relatively imports {sorted(illegal)}"


@pytest.mark.parametrize("module", _module_files(DOMAIN), ids=lambda p: p.name)
def test_domain_source_never_mentions_a_driver(module: Path):
    source = module.read_text(encoding="utf-8")
    for token in FORBIDDEN_IN_CORE:
        assert f"import {token}" not in source, f"{module.name} imports {token}"


@pytest.mark.parametrize("module", _module_files(PORTS), ids=lambda p: p.name)
def test_ports_declare_protocols_and_import_no_adapter(module: Path):
    offenders = _imported_roots(module) - {"__future__", "typing", "dataclasses"}
    assert not offenders, f"{module.name} imports {sorted(offenders)}"
    assert _relative_targets(module) <= {"", "domain", "cache", "clock", "document_source",
                                         "generator", "index"}


def test_ports_are_protocols_not_base_classes():
    """Structural typing is what lets adapters conform without inheriting from the core."""
    import knowbase.ports as ports
    from typing import Protocol
    for name in ports.__all__:
        port = getattr(ports, name)
        assert Protocol in port.__mro__, f"{name} must be a typing.Protocol"


def test_importing_the_domain_pulls_in_no_scientific_stack():
    """Runtime proof, in a clean interpreter: the core does not drag numpy/sklearn along."""
    probe = (
        "import sys;"
        f"sys.path.insert(0, r'{SRC}');"
        "import knowbase.domain.usecases, knowbase.domain.text, knowbase.domain.model, knowbase.ports;"
        "leaked=[m for m in ('numpy','sklearn','scipy','pandas','anthropic','sqlite3')"
        " if m in sys.modules];"
        "print('LEAKED:'+','.join(leaked) if leaked else 'PURE')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PURE", result.stdout


def test_adapters_are_allowed_to_depend_on_the_domain():
    """The arrow points inward -- confirm it actually does, so the test above means something."""
    hybrid = SRC / "knowbase" / "adapters" / "index" / "hybrid_rrf.py"
    assert "numpy" not in hybrid.read_text(encoding="utf-8")  # it is in _lexical.py
    lexical = SRC / "knowbase" / "adapters" / "index" / "_lexical.py"
    assert "import numpy" in lexical.read_text(encoding="utf-8")
    assert _relative_targets(hybrid) >= {"domain"}
