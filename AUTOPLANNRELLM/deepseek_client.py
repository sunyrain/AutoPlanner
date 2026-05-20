"""Small DeepSeek JSON client used by AUTOPLANNRELLM.

The client uses DeepSeek's OpenAI-compatible chat-completions endpoint. It is
deliberately minimal so the planner does not gain a hard dependency on an SDK.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
PLACEHOLDER_DEEPSEEK_KEY = "replace_with_your_deepseek_key"


class DeepSeekJSONClient:
    """Return JSON objects from DeepSeek, with optional file-backed caching."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.api_key = normalize_deepseek_key_value(api_key or os.environ.get("DEEPSEEK_API_KEY") or "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_URL
        self.timeout_s = float(timeout_s or os.environ.get("AUTOPLANNRELLM_DEEPSEEK_TIMEOUT_S") or 30.0)
        raw_cache = cache_path or os.environ.get("AUTOPLANNRELLM_CACHE")
        self.cache_path = Path(raw_cache) if raw_cache else None
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def request_json(
        self,
        *,
        task: str,
        system: str,
        user_payload: dict[str, Any],
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        mock = os.environ.get(f"AUTOPLANNRELLM_MOCK_{task.upper()}_JSON")
        if mock:
            return _extract_json_object(mock)
        cache_key = self._cache_key(task=task, system=system, user_payload=user_payload)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return dict(cached.get("response") or {})
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        if is_placeholder_deepseek_key(self.api_key):
            raise RuntimeError("DEEPSEEK_API_KEY still contains the dotenv placeholder")
        payload = {
            "model": self.model,
            "thinking": {"type": "disabled"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        response = _extract_json_object(content)
        self._cache[cache_key] = {
            "task": task,
            "model": self.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": response,
        }
        self._append_cache(cache_key, self._cache[cache_key])
        return dict(response)

    def _cache_key(self, *, task: str, system: str, user_payload: dict[str, Any]) -> str:
        payload = {
            "task": task,
            "model": self.model,
            "system": system,
            "user": user_payload,
        }
        return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("cache_key") or "")
                if key:
                    self._cache[key] = dict(row.get("value") or {})
        except (OSError, json.JSONDecodeError):
            self._cache = {}

    def _append_cache(self, cache_key: str, value: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"cache_key": cache_key, "value": value}, ensure_ascii=False) + "\n")


def normalize_deepseek_key_value(value: str | None) -> str:
    normalized = str(value or "").strip()
    for quote in ('"', "'"):
        if normalized.endswith(quote):
            normalized = normalized[:-1]
        if normalized.startswith(quote):
            normalized = normalized[1:]
    return normalized.strip()


def is_placeholder_deepseek_key(value: str | None) -> bool:
    return normalize_deepseek_key_value(value) == PLACEHOLDER_DEEPSEEK_KEY


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start:end + 1])
    if isinstance(obj, dict):
        return obj
    raise ValueError("DeepSeek response did not contain a JSON object")
