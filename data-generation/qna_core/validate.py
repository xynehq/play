import json, re
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

_model = None
def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def parse_first_json(s: str):
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else None

def lexical_overlap(ans_text: str, cited_text: str) -> float:
    a = re.sub(r"\s+"," ", ans_text).lower()
    c = re.sub(r"\s+"," ", cited_text).lower()
    inter = sum(1 for ch in a if ch in c)
    return inter / max(len(a),1)

def semantic_sim(ans_text: str, cited_text: str) -> float:
    m = _embedder()
    ea = m.encode(ans_text, normalize_embeddings=True)
    ec = m.encode(cited_text, normalize_embeddings=True)
    return float(util.dot_score(ea, ec))
