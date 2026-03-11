"""Mistral AI Studio API caller for book embedding experiments."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from typing import Optional

_MIN_REQUEST_INTERVAL_SECONDS = 1.0
_rate_limit_lock = threading.Lock()
_last_request_at = 0.0


def _wait_for_rate_limit() -> None:
    """Ensure at least 1 second between API requests (RPS <= 1)."""

    global _last_request_at

    with _rate_limit_lock:
        now = time.monotonic()
        elapsed = now - _last_request_at
        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _last_request_at = time.monotonic()


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

    with urllib.request.urlopen(request, timeout=120) as response:
        response_json = json.loads(response.read().decode("utf-8"))

    choices = response_json.get("choices", [])
    if not choices:
        return ""

    return choices[0].get("message", {}).get("content", "") or ""
