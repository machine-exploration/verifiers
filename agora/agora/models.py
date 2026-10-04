"""Model clients. Any OpenAI-compatible endpoint works (vLLM, OpenRouter, ...)."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request


class OpenAICompatible:
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)["choices"][0]["message"]["content"]


class FakeModel:
    """Offline stand-in: a deterministic price from the request hash. Counts its calls,
    so a replay can be checked to call no model at all."""

    calls = 0

    def __init__(self, low: float = 3, high: float = 18):
        self.low, self.high = low, high

    def complete(self, messages: list[dict]) -> str:
        FakeModel.calls += 1
        h = int(hashlib.sha256(json.dumps(messages).encode()).hexdigest(), 16)
        price = self.low + h % int(self.high - self.low + 1)
        return f'Thinking... {{"action": "set_price", "args": {{"price": {price}}}}}'
