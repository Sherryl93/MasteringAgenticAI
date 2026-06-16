"""Static test-coverage analysis (no code execution).

Rather than running an untrusted repo's `pytest` (which requires installing
its dependencies, usually fails, and executes arbitrary code on the host),
we analyze the *structure* of the codebase via Python's `ast` module:

  - which files are tests vs. source
  - how many test functions / assertions exist
  - which source modules have no corresponding tests
  - a heuristic test-to-source ratio

This is safe, reproducible, and works on any repo without its dependencies.
The trade-off — no runtime line coverage — is documented in the README.
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..repo_loader import read_file_snippet
from ..state import ToolResult


def _is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name.startswith("test_") or name.endswith("_test.py") or "/tests/" in (
        "/" + rel_path.replace("\\", "/")
    )


def _module_key(rel_path: str) -> str:
    """A normalized stem used to match source files to their tests."""
    stem = Path(rel_path).stem
    for prefix in ("test_",):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    if stem.endswith("_test"):
        stem = stem[: -len("_test")]
    return stem


def _count_tests_and_asserts(source: str) -> tuple[int, int]:
    """Return (#test functions, #assert statements) via AST. Never raises."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0, 0
    test_funcs = 0
    asserts = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test"
        ):
            test_funcs += 1
        if isinstance(node, ast.Assert):
            asserts += 1
    return test_funcs, asserts


def analyze_tests(repo_dir: str, python_files: list[str]) -> ToolResult:
    """Compute static test-coverage signals for the scanned Python files."""
    test_files: list[str] = []
    source_files: list[str] = []
    for f in python_files:
        (test_files if _is_test_file(f) else source_files).append(f)

    tested_keys: set[str] = set()
    total_test_funcs = 0
    total_asserts = 0
    for tf in test_files:
        src = read_file_snippet(repo_dir, tf)
        funcs, asserts = _count_tests_and_asserts(src)
        total_test_funcs += funcs
        total_asserts += asserts
        tested_keys.add(_module_key(tf))

    untested = sorted(
        f for f in source_files if _module_key(f) not in tested_keys
    )

    ratio = (len(test_files) / len(source_files)) if source_files else 0.0
    data = {
        "n_source_files": len(source_files),
        "n_test_files": len(test_files),
        "test_to_source_ratio": round(ratio, 3),
        "total_test_functions": total_test_funcs,
        "total_assertions": total_asserts,
        "untested_modules": untested,
        "test_files": test_files,
    }
    summary = (
        f"static tests: {len(test_files)} test file(s), "
        f"{total_test_funcs} test fn(s), {len(untested)} source module(s) "
        f"without an obvious test."
    )
    return ToolResult(name="test_analysis", ok=True, summary=summary, raw="", data=data)
