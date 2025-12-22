from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """
    Minimal OpenAI-compatible chat client.
    Expects:
      - API_KEY in env or provided
      - BASE_URL in env or provided
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = base_url or os.environ.get("BASE_URL")
        if not self.api_key:
            raise LLMClientError("Missing API key (API_KEY).")
        if not self.base_url:
            raise LLMClientError("Missing base URL (BASE_URL).")

    def run(self, prompt: str, timeout: int = 120) -> str:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
        if resp.status_code != 200:
            raise LLMClientError(f"LLM call failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - safety
            raise LLMClientError(f"Unexpected response format: {data}") from exc

