from typing import List, Dict
import re

def naive_token_count(text: str) -> int:
    # Approximate token count: spaces and punctuation
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

def split_into_chunks(text: str, target_tokens=380, overlap_tokens=60) -> List[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    i = 0
    while i < len(words):
        window = words[i:i+target_tokens]
        if not window: break
        chunks.append(" ".join(window))
        i += max(1, target_tokens - overlap_tokens)
    return chunks
