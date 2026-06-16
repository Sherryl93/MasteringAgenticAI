"""Nebius LLM client with timeout, retry, model fallback, and JSON parsing.

The public surface is intentionally small:

    client = LLMClient(settings)
    findings = client.structured_findings(system, user)  # -> list[Finding] | None

`structured_findings` returns None when *all* models fail, which the agents
interpret as "degrade to tool-only" rather than raising.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from .config import Settings, get_settings
from .state import Finding, FindingList


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = None  # lazily constructed so importing never fails
        self.last_model: Optional[str] = None  # model that produced the last result

    # ----------------------------------------------------------------- internals
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.nebius_api_key,
                base_url=self.settings.nebius_base_url,
                timeout=self.settings.llm_timeout_seconds,
            )
        return self._client

    def _models_in_order(self) -> list[str]:
        # Primary, then fallback, then fast (deduplicated, order-preserving).
        seen: set[str] = set()
        out: list[str] = []
        for m in (self.settings.model, self.settings.fallback_model, self.settings.fast_model):
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    def _raw_complete(self, model: str, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------- public
    def complete(self, system: str, user: str) -> Optional[str]:
        """Return raw text, trying each model with retries. None if all fail."""
        if not self.settings.has_api_key:
            return None
        for model in self._models_in_order():
            for attempt in range(self.settings.llm_max_retries + 1):
                try:
                    text = self._raw_complete(model, system, user)
                    self.last_model = model  # record which model actually answered
                    return text
                except Exception:
                    if attempt < self.settings.llm_max_retries:
                        time.sleep(1.5 * (attempt + 1))  # simple backoff
                    continue
        return None

    def structured_findings(
        self, system: str, user: str, agent: str
    ) -> Optional[list[Finding]]:
        """Get findings as validated `Finding` objects, or None if LLMs fail."""
        text = self.complete(system, user)
        if text is None:
            return None
        parsed = _parse_findings(text)
        if parsed is None:
            return None
        # Stamp the producing agent in case the model omitted/!matched it.
        for f in parsed:
            f.agent = agent  # type: ignore[assignment]
        return parsed


def _extract_json(text: str) -> Optional[str]:
    """Best-effort: pull the first JSON object out of a model response."""
    text = text.strip()
    # Strip ```json fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    if text.startswith("{"):
        return text
    brace = text.find("{")
    if brace != -1:
        return text[brace:]
    return None


def _parse_findings(text: str) -> Optional[list[Finding]]:
    """Parse model output into Findings, tolerating minor format drift."""
    candidate = _extract_json(text)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    try:
        if isinstance(data, dict) and "findings" in data:
            return FindingList(**data).findings
        if isinstance(data, list):
            return FindingList(findings=data).findings
        # A single finding object.
        if isinstance(data, dict):
            return [Finding(**data)]
    except Exception:
        return None
    return None
