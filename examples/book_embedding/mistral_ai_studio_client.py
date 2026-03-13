"""OpenRouter chat completions caller with model failover support."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

_MIN_REQUEST_INTERVAL_SECONDS = 2.0
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0
_consecutive_429_count = 0
_model_cooldown_until: dict[str, float] = {}

_DEFAULT_FALLBACK_MODELS = [
    "cerebras/gpt-oss-120b",
    "groq/gpt-oss-120b",
    "z-ai/glm-4.7-flash",
    "google/gemini-3.1-flash-lite",
    "openrouter/hunter-alpha",
]


def _wait_for_rate_limit() -> None:
    """Ensure at least 2 seconds between API requests."""

    global _last_request_at

    with _rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _last_request_at = time.monotonic()


def _record_429_and_get_backoff_seconds() -> int:
    """Track consecutive 429 responses and return backoff seconds."""

    global _consecutive_429_count

    with _rate_limit_lock:
        _consecutive_429_count += 1
        if _consecutive_429_count <= 1:
            return 10
        if _consecutive_429_count == 2:
            return 30
        return 60


def _reset_429_backoff() -> None:
    """Reset consecutive 429 counter after a successful request."""

    global _consecutive_429_count

    with _rate_limit_lock:
        _consecutive_429_count = 0


def _parse_csv_env(var_name: str) -> list[str]:
    raw = os.getenv(var_name, "")
    return [v.strip() for v in raw.split(",") if v.strip()]


def _parse_bool_env(var_name: str, default: bool) -> bool:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_provider_preferences() -> dict[str, object]:
    """Build optional OpenRouter provider routing preferences from env vars."""

    provider: dict[str, object] = {}

    order = _parse_csv_env("OPENROUTER_PROVIDER_ORDER")
    if order:
        provider["order"] = order

    if os.getenv("OPENROUTER_ALLOW_FALLBACKS") is not None:
        provider["allow_fallbacks"] = _parse_bool_env("OPENROUTER_ALLOW_FALLBACKS", True)

    sort = os.getenv("OPENROUTER_PROVIDER_SORT", "").strip()
    if sort:
        provider["sort"] = sort

    partition = os.getenv("OPENROUTER_PARTITION", "").strip()
    if partition:
        provider["partition"] = partition

    return provider


def _parse_model_candidates(primary_model: Optional[str]) -> list[str]:
    """Return ordered model candidates used for fallback routing."""

    parsed = _parse_csv_env("OPENROUTER_MODEL_CANDIDATES")

    if not parsed:
        parsed = list(_DEFAULT_FALLBACK_MODELS)

    if primary_model and primary_model not in parsed:
        return [primary_model, *parsed]

    return parsed


def _next_available_models(model_candidates: list[str]) -> list[str]:
    """Prioritize models that are not in cooldown, preserving order."""

    now = time.monotonic()
    available: list[str] = []
    cooling: list[str] = []

    with _rate_limit_lock:
        for model in model_candidates:
            cooldown_until = _model_cooldown_until.get(model, 0.0)
            if cooldown_until <= now:
                available.append(model)
            else:
                cooling.append(model)

    return [*available, *cooling]


def _mark_model_in_cooldown(model: str, seconds: int) -> None:
    """Mark a model unavailable until cooldown elapses."""

    with _rate_limit_lock:
        _model_cooldown_until[model] = time.monotonic() + seconds


def generate_code_suggestion(
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: str = "cerebras/gpt-oss-120b",
    temperature: float = 0.7,
    max_tokens: int = 12_000,
) -> str:
    """Generate a response via OpenRouter, with best-effort fallback on 429/5xx."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    provider_preferences = _build_provider_preferences()
    model_candidates = _next_available_models(_parse_model_candidates(model))
    if not model_candidates:
        raise RuntimeError("No model candidates configured")

    response_json: dict[str, object] | None = None
    last_error: Exception | None = None

    for model_name in model_candidates:
        payload: dict[str, object] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if provider_preferences:
            payload["provider"] = provider_preferences

        _wait_for_rate_limit()

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            _reset_429_backoff()
            break
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504}:
                backoff_seconds = _record_429_and_get_backoff_seconds()
                _mark_model_in_cooldown(model_name, backoff_seconds)
                last_error = exc
                continue
            raise

    if response_json is None:
        if last_error is not None:
            raise RuntimeError("All fallback models failed") from last_error
        raise RuntimeError("No response from OpenRouter")

    choices = response_json.get("choices", [])
    if not choices:
        return ""

    return choices[0].get("message", {}).get("content", "") or ""
