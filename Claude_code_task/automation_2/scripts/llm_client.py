from __future__ import annotations

import json
import os
import re
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

    @staticmethod
    def _clean_response(content: str) -> str:
        """
        Lightweight preprocessing to strip reasoning tags from LLM output.
        
        Removes:
        - <think>...</think> tags and their content
        - <redacted_reasoning>...</redacted_reasoning> tags
        - Any other common reasoning wrapper tags
        
        This ensures clean output even if the model ignores prompt instructions.
        """
        # Remove all reasoning/thinking tags (case-insensitive, multiline)
        patterns = [
            r'<think>.*?</think>',
            r'<redacted_reasoning>.*?</redacted_reasoning>',
            r'<reasoning>.*?</reasoning>',
            r'<thought>.*?</thought>',
        ]
        
        cleaned = content
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up excessive whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

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
            raw_content = data["choices"][0]["message"]["content"]
            # Apply lightweight preprocessing to strip reasoning tags
            return self._clean_response(raw_content)
        except Exception as exc:  # pragma: no cover - safety
            raise LLMClientError(f"Unexpected response format: {data}") from exc
