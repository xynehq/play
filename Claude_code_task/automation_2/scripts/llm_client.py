from __future__ import annotations

import json
import os
import re
import time
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
        default_timeout: int = 2400,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = base_url or os.environ.get("BASE_URL")
        self.default_timeout = default_timeout
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

    def run(self, prompt: str, timeout: Optional[int] = None, max_retries: int = 3) -> str:
        """
        Run an LLM completion request with automatic retry for transient failures.
        
        Args:
            prompt: The prompt to send to the LLM
            timeout: Request timeout in seconds. If None, uses default_timeout from initialization.
            max_retries: Maximum number of retry attempts for transient failures (default: 3)
        
        Returns:
            The cleaned LLM response
            
        Raises:
            LLMClientError: If the request fails after all retries
        """
        # Use provided timeout or fall back to instance default
        actual_timeout = timeout if timeout is not None else self.default_timeout
        
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
        
        # Retry logic for transient failures
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=actual_timeout)
                
                # Check for transient error status codes
                if resp.status_code in [502, 503, 504]:
                    # Server error - retry with exponential backoff
                    wait_time = 2 ** attempt  # 1s, 2s, 4s, 8s...
                    print(f"  [LLM Client] Transient error {resp.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    last_error = f"LLM call failed ({resp.status_code}): {resp.text}"
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                        raise LLMClientError(last_error)
                
                # Non-transient error - fail immediately
                if resp.status_code != 200:
                    raise LLMClientError(f"LLM call failed ({resp.status_code}): {resp.text}")
                
                # Success - parse response
                data = resp.json()
                try:
                    raw_content = data["choices"][0]["message"]["content"]
                    # Apply lightweight preprocessing to strip reasoning tags
                    return self._clean_response(raw_content)
                except Exception as exc:  # pragma: no cover - safety
                    raise LLMClientError(f"Unexpected response format: {data}") from exc
                    
            except requests.exceptions.Timeout:
                # Timeout - retry
                wait_time = 2 ** attempt
                print(f"  [LLM Client] Request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                last_error = f"LLM call timed out after {actual_timeout}s"
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMClientError(last_error)
                    
            except requests.exceptions.ConnectionError as e:
                # Connection error - retry
                wait_time = 2 ** attempt
                print(f"  [LLM Client] Connection error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                last_error = f"LLM call failed (connection error): {str(e)}"
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMClientError(last_error)
        
        # Should not reach here, but just in case
        raise LLMClientError(last_error or "LLM call failed after all retries")
