import requests
import json
from typing import List, Dict, Optional

class MultiEndpointChat:
    """Try OpenAI-compatible first (/v1/chat/completions), then Ollama (/api/chat)."""
    def __init__(self, primary: str, secondary: Optional[str] = None, routing: str = "fallback"):
        self.primary = primary.rstrip("/")
        self.secondary = (secondary or "").rstrip("/") or None
        self.routing = routing
        self._toggle = False

    def _order(self):
        if self.routing == "round_robin" and self.secondary:
            self._toggle = not self._toggle
            return [self.secondary, self.primary] if self._toggle else [self.primary, self.secondary]
        return [self.primary, self.secondary]

    def _call_openai(self, base, model, messages, max_tokens=None, temperature=None):
        url = base + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        js = r.json()
        return js["choices"][0]["message"]["content"]
    
    def _call_ollama(self, base, model, messages, max_tokens=None, temperature=None):
        url = base + "/api/chat"
        payload = {"model": model, "messages": messages}
        if max_tokens is not None or temperature is not None:
            payload["options"] = {}
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
            if temperature is not None:
                payload["options"]["temperature"] = temperature
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        
        # Handle streaming NDJSON response
        response_text = r.text.strip()
        if not response_text:
            raise RuntimeError("Empty response from Ollama endpoint")
        
        # Parse NDJSON (newline-delimited JSON)
        lines = response_text.split('\n')
        content_parts = []
        final_message = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
                # Collect content from streaming chunks
                if "message" in js and "content" in js["message"]:
                    content_parts.append(js["message"]["content"])
                # Check if this is the final message
                if js.get("done", False):
                    final_message = js
                # Some proxies return OpenAI-like format
                elif "choices" in js:
                    return js["choices"][0]["message"]["content"]
            except json.JSONDecodeError:
                continue
        
        # Return concatenated content from all chunks
        if content_parts:
            return "".join(content_parts)
        
        # Fallback: try to parse as single JSON (non-streaming)
        try:
            js = json.loads(response_text)
            if "message" in js and "content" in js["message"]:
                return js["message"]["content"]
            if "choices" in js:
                return js["choices"][0]["message"]["content"]
        except json.JSONDecodeError:
            pass
            
        raise RuntimeError("Unexpected Ollama response format")

    def chat(self, model, messages, max_tokens=None, temperature=None):
        last_err = None
        for base in filter(None, self._order()):
            # Try OpenAI-style first, then Ollama-style
            for fn in (self._call_openai, self._call_ollama):
                try:
                    return fn(base, model, messages, max_tokens, temperature)
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"All endpoints failed: {last_err}")
