"""Shared helper for running analyzer CLIs safely.

All analyzer invocations go through `run_cli`, which enforces shell=False,
a timeout, and never raises — a failed tool degrades into a structured
"ok=False" result so the pipeline can continue gracefully.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CliOutput:
    ok: bool
    returncode: int
    stdout: str
    stderr: str


def run_cli(args: list[str], cwd: str | None = None, timeout: int = 120) -> CliOutput:
    """Run a CLI as `python -m <tool> ...` style list. Never raises.

    `ok` means the process *ran to completion*, not that it found nothing —
    linters return non-zero when they report issues, which is expected.
    """
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,  # never interpret args through a shell
        )
        return CliOutput(
            ok=True,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except FileNotFoundError as e:
        return CliOutput(False, -1, "", f"tool not found: {e}")
    except subprocess.TimeoutExpired:
        return CliOutput(False, -1, "", f"timed out after {timeout}s")
    except Exception as e:  # pragma: no cover - defensive
        return CliOutput(False, -1, "", f"error: {e}")


def module_cmd(module: str, *extra: str) -> list[str]:
    """Build a `python -m module ...` command using the current interpreter."""
    return [sys.executable, "-m", module, *extra]
