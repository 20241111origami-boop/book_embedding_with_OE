"""Cerebras Cloud SDK based model caller for book embedding experiments."""

from __future__ import annotations

import os
from typing import Optional

from cerebras.cloud.sdk import Cerebras


def generate_code_suggestion(
    user_prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: str = "gpt-oss-120b",
    temperature: float = 0.7,
    max_tokens: int = 12_000,
) -> str:
    """Generate a response using cerebras.cloud.sdk directly.

    This avoids the OpenAI-compatible bridge and invokes Cerebras' SDK client.
    """

    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set")

    client = Cerebras(api_key=api_key)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""
