"""Central configuration for ReviewPilot.

Loads environment variables, exposes a typed Settings object, and provides
startup validation for the two things most likely to silently break a demo:
the Nebius model IDs and the presence of `git` on PATH.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings, sourced from the environment with safe defaults."""

    nebius_api_key: str = field(default_factory=lambda: os.getenv("NEBIUS_API_KEY", ""))
    nebius_base_url: str = field(
        default_factory=lambda: os.getenv(
            "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-235B-A22B")
    )
    fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "NEBIUS_FALLBACK_MODEL", "deepseek-ai/DeepSeek-V3-0324"
        )
    )
    fast_model: str = field(
        default_factory=lambda: os.getenv("NEBIUS_FAST_MODEL", "openai/gpt-oss-120b")
    )

    default_repo_url: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_REPO_URL", "https://github.com/Sherryl93/MasteringAgenticAI"
        )
    )
    default_target_path: str = field(
        default_factory=lambda: os.getenv("DEFAULT_TARGET_PATH", "Week02_RAGApplication")
    )

    max_files_to_review: int = field(
        default_factory=lambda: _int_env("MAX_FILES_TO_REVIEW", 30)
    )
    max_file_chars: int = field(default_factory=lambda: _int_env("MAX_FILE_CHARS", 12000))
    clone_dir: str = field(default_factory=lambda: os.getenv("CLONE_DIR", ".tmp_repos"))

    llm_timeout_seconds: int = field(
        default_factory=lambda: _int_env("LLM_TIMEOUT_SECONDS", 90)
    )
    llm_max_retries: int = field(default_factory=lambda: _int_env("LLM_MAX_RETRIES", 2))

    @property
    def has_api_key(self) -> bool:
        return bool(self.nebius_api_key and self.nebius_api_key != "your_nebius_token_here")


@dataclass
class EnvCheck:
    """Result of a startup environment validation."""

    ok: bool
    git_available: bool
    api_key_present: bool
    models_validated: Optional[bool]  # None => not checked (no key / offline)
    messages: list[str] = field(default_factory=list)


def get_settings() -> Settings:
    """Build a fresh Settings object from the current environment."""
    return Settings()


def git_available() -> bool:
    """True if a usable `git` executable is on PATH."""
    if shutil.which("git") is None:
        return False
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
            shell=False,
        )
        return True
    except Exception:
        return False


def list_catalog_models(settings: Settings) -> Optional[set[str]]:
    """Return the set of model IDs in the Nebius catalog, or None if the check
    could not run (no key / network error). Best-effort; never raises."""
    if not settings.has_api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.nebius_api_key, base_url=settings.nebius_base_url
        )
        return {m.id for m in client.models.list().data}
    except Exception:
        return None


def validate_models(settings: Settings) -> Optional[bool]:
    """Whether the PRIMARY model is in the catalog (what actually determines
    if reviews use the LLM path). Fallbacks are reported separately so a wrong
    fallback ID never makes the UI imply the model is broken.

    Returns True/False, or None if the check could not run.
    """
    catalog = list_catalog_models(settings)
    if catalog is None:
        return None
    return settings.model in catalog


def check_environment(settings: Optional[Settings] = None) -> EnvCheck:
    """Run all startup checks and return a structured result.

    Designed to be loud-but-non-fatal: the UI surfaces the messages so the
    user can fix issues, but a missing key or offline model check does not
    crash the app (the pipeline degrades to a tool-only report).
    """
    settings = settings or get_settings()
    messages: list[str] = []

    git_ok = git_available()
    if not git_ok:
        messages.append(
            "git is not on PATH. Install Git and restart so repos can be cloned."
        )

    key_ok = settings.has_api_key
    if not key_ok:
        messages.append(
            "NEBIUS_API_KEY is not set. LLM agents will be skipped and a "
            "tool-only report will be produced."
        )

    catalog = list_catalog_models(settings)
    if catalog is None:
        models_ok = None
        if key_ok:
            messages.append(
                "Could not validate model IDs against Nebius (offline or API "
                "error). Proceeding; calls fall back if a model is unavailable."
            )
    else:
        models_ok = settings.model in catalog  # panel reflects the PRIMARY model
        if not models_ok:
            messages.append(
                f"Primary model '{settings.model}' is not in the Nebius catalog; "
                f"reviews will use a fallback model."
            )
        # Fallbacks are informational only — a wrong fallback must not make the
        # panel imply the (working) primary model is broken.
        missing_fallbacks = [
            m for m in (settings.fallback_model, settings.fast_model)
            if m and m not in catalog
        ]
        if missing_fallbacks:
            messages.append(
                "Note: fallback model(s) not in catalog: "
                + ", ".join(missing_fallbacks)
                + ". The primary model works; these fallbacks would be skipped."
            )

    ok = git_ok  # git is the only hard requirement to run the pipeline at all
    return EnvCheck(
        ok=ok,
        git_available=git_ok,
        api_key_present=key_ok,
        models_validated=models_ok,
        messages=messages,
    )
