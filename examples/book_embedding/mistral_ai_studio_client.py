"""Mistral AI Studio API caller for book embedding experiments."""

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


def generate_code_suggestion(
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: str = "mistral-large-latest",
    temperature: float = 0.7,
    max_tokens: int = 12_000,
) -> str:
    """Generate a response using the Mistral AI Studio chat completions API."""

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    base_url = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

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

    while True:
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            _reset_429_backoff()
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise

            backoff_seconds = _record_429_and_get_backoff_seconds()
            time.sleep(backoff_seconds)
            _wait_for_rate_limit()

    choices = response_json.get("choices", [])
    if not choices:
        return ""

    return choices[0].get("message", {}).get("content", "") or ""
