"""Clone a public GitHub repo (read-only) and scan for Python files.

ReviewPilot never writes to, pushes to, or modifies a target repo. It only
clones into a local temp directory and reads files. Clones are shallow
(`depth=1`) for speed.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings

_GITHUB_URL_RE = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/?$")


def validate_repo_url(url: str) -> tuple[bool, str]:
    """Validate a GitHub HTTPS URL. Returns (ok, message)."""
    url = (url or "").strip()
    if not url:
        return False, "Repository URL is empty."
    if not _GITHUB_URL_RE.match(url.rstrip("/") + ("" if url.endswith("/") else "")):
        # Be lenient: accept with or without trailing slash / .git
        cleaned = url[:-4] if url.endswith(".git") else url
        if not _GITHUB_URL_RE.match(cleaned.rstrip("/")):
            return False, "Expected a URL like https://github.com/owner/repo"
    return True, "ok"


def _repo_slug(url: str) -> str:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    return f"{name}-{h}"


def clone_repo(
    url: str, settings: Optional[Settings] = None
) -> tuple[Optional[str], str]:
    """Shallow-clone `url` into the configured clone dir.

    Returns (local_path, message). local_path is None on failure.
    """
    settings = settings or get_settings()
    ok, msg = validate_repo_url(url)
    if not ok:
        return None, msg

    from git import Repo  # imported here so config import never needs GitPython
    from git.exc import GitCommandError

    base = Path(settings.clone_dir)
    base.mkdir(parents=True, exist_ok=True)
    dest = base / _repo_slug(url)

    # Return an ABSOLUTE path so downstream path normalization can reliably
    # strip the clone prefix from tools (like ruff) that emit absolute paths.
    if dest.exists() and (dest / ".git").exists():
        return str(dest.resolve()), f"Using cached clone at {dest}"

    try:
        Repo.clone_from(url, str(dest), depth=1)
        return str(dest.resolve()), f"Cloned into {dest}"
    except GitCommandError as e:
        return None, f"git clone failed: {e.stderr or e}".strip()
    except Exception as e:  # pragma: no cover - defensive
        return None, f"Clone error: {e}"


def scan_python_files(
    repo_dir: str, target_path: str, settings: Optional[Settings] = None
) -> tuple[list[str], str]:
    """List Python files under repo_dir/target_path, capped by MAX_FILES.

    Returns (relative_paths, message). Paths are relative to repo_dir so they
    read cleanly in the report. Skips vendored/cache directories.
    """
    settings = settings or get_settings()
    root = Path(repo_dir)
    search_root = root / target_path if target_path else root
    if not search_root.exists():
        return [], f"Target path '{target_path}' not found in repo."

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".tmp_repos",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
    }

    found: list[str] = []
    for path in sorted(search_root.rglob("*.py")):
        rel = path.relative_to(root)
        # Only inspect path parts *inside* the repo, so the clone dir name
        # (e.g. ".tmp_repos") can't accidentally match a skip entry.
        if any(part in skip_dirs for part in rel.parts):
            continue
        found.append(str(rel).replace(os.sep, "/"))
        if len(found) >= settings.max_files_to_review:
            break

    if not found:
        return [], f"No Python files found under '{target_path or '.'}'."
    return found, f"Found {len(found)} Python file(s) (cap {settings.max_files_to_review})."


def read_file_snippet(
    repo_dir: str, rel_path: str, settings: Optional[Settings] = None
) -> str:
    """Read a file, truncated to MAX_FILE_CHARS. Never raises."""
    settings = settings or get_settings()
    try:
        text = Path(repo_dir, rel_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"<<could not read {rel_path}: {e}>>"
    if len(text) > settings.max_file_chars:
        return text[: settings.max_file_chars] + "\n<<truncated>>"
    return text
