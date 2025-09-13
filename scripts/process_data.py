import argparse, json, os, random, sys, csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

# ---------- utils: config merge ----------
def load_config(path: str) -> Dict[str, Any]:
    """
    Loads config_run.yaml and optionally merges include: config_base.yaml.
    run overrides base.
    """
    with open(path, "r") as f:
        run_cfg = yaml.safe_load(f)

    base_path = run_cfg.get("include")
    if base_path:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge(base_cfg, run_cfg)
    else:
        cfg = run_cfg
    return cfg

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# ---------- io helpers ----------
def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

# ---------- normalization ----------
def normalize_item(obj: Any, default_system: str) -> Dict[str, str]:
    """
    Convert various raw shapes into {system,user,assistant}.
    Customize this mapping as you add sources.
    Supported quick-cases:
      - {"question": "...", "answer":"..."}
      - {"query": "...", "answer":"..."}  # Added support for query/answer format
      - {"user":"...", "assistant":"..."}  (system optional)
      - ["what is animal?", {"answer": "..."}]  # pairwise toy case
      - {"prompt":"...", "response":"..."}
    """
    if isinstance(obj, dict):
        # common field names - added 'query' support
        user = obj.get("user") or obj.get("question") or obj.get("query") or obj.get("prompt") or obj.get("input") or obj.get("instruction")
        assistant = obj.get("assistant") or obj.get("answer") or obj.get("response") or obj.get("target") or obj.get("output")
        system = obj.get("system") or default_system
        if user and assistant:
            return {"system": system.strip(), "user": user.strip(), "assistant": assistant.strip()}
        else:
            raise ValueError("Unrecognized dict schema: missing user/assistant fields")

    elif isinstance(obj, list) and len(obj) >= 2:
        # toy: ["q?", {"answer":"..."}]
        user = obj[0] if isinstance(obj[0], str) else None
        ans = obj[1]
        assistant = ans.get("answer") if isinstance(ans, dict) else None
        if user and assistant:
            return {"system": default_system, "user": user.strip(), "assistant": assistant.strip()}
        else:
            raise ValueError("Unrecognized list schema")
    else:
        raise ValueError(f"Unsupported raw item type: {type(obj)}")

def smart_chunk_context(context_content: str, query: str, max_context_tokens: int = 2500) -> str:
    """
    Smart chunking: include most relevant sections for each query.
    Prioritizes documents that contain query keywords.
    """
    # Rough token estimation (4 chars per token)
    estimated_tokens = len(context_content) // 4
    
    if estimated_tokens <= max_context_tokens:
        return context_content  # Use full context if it fits
    
    print(f"[smart_chunk] Context too long ({estimated_tokens} tokens), chunking to {max_context_tokens} tokens")
    
    # Split by document sections
    if "=== Document" in context_content:
        docs = context_content.split("=== Document")
        header = docs[0]  # Keep header
        docs = docs[1:]   # Actual documents
    else:
        # Fallback: split by paragraphs
        docs = context_content.split("\n\n")
        header = ""
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Score documents by relevance
    scored_docs = []
    for i, doc in enumerate(docs):
        doc_words = set(doc.lower().split())
        # Score based on keyword overlap
        score = len(query_words.intersection(doc_words))
        # Boost score for exact phrase matches
        if any(word in doc.lower() for word in query_words if len(word) > 3):
            score += 2
        
        doc_with_header = f"=== Document {i+1}{doc}" if header else doc
        scored_docs.append((score, doc_with_header))
    
    # Sort by relevance (highest score first)
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # Build result within token limit
    result = header if header else ""
    current_tokens = len(result) // 4
    
    for score, doc in scored_docs:
        doc_tokens = len(doc) // 4
        if current_tokens + doc_tokens <= max_context_tokens:
            result += doc + "\n"
            current_tokens += doc_tokens
        else:
            # Include partial doc if space allows
            remaining_chars = (max_context_tokens - current_tokens) * 4
            if remaining_chars > 500:  # Only if meaningful chunk
                result += doc[:remaining_chars] + "...\n"
            break
    
    final_tokens = len(result) // 4
    print(f"[smart_chunk] Chunked to {final_tokens} tokens (included {len([s for s, _ in scored_docs if s > 0])} relevant docs)")
    return result.strip()

def sanitize(row: Dict[str, str], max_user_len=4096, max_assistant_len=4096) -> Dict[str, str]:
    row["user"] = row["user"].strip()[:max_user_len]
    row["assistant"] = row["assistant"].strip()[:max_assistant_len]
    row["system"] = (row.get("system") or "").strip()
    return row

# ---------- split ----------
def split_rows(rows: List[Dict[str,str]], train_ratio: float, val_ratio: float, test_ratio: float
               ) -> Tuple[List[Dict[str,str]], List[Dict[str,str]], List[Dict[str,str]]]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "split ratios must sum to 1.0"
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train = rows[:n_train]
    val = rows[n_train:n_train+n_val]
    test = rows[n_train+n_val:]
    return train, val, test

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_run.yaml")
    ap.add_argument("--raw_path", default="data/raw/raw.json", help="Raw input file (json/jsonl/csv)")
    ap.add_argument("--system_prompt", default="You are a helpful domain assistant.",
                    help="Default system instruction if raw rows don't include one.")
    ap.add_argument("--context_docs", help="Path to file containing document context (txt file with all 4 docs)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # read raw
    raw_p = Path(args.raw_path)
    if not raw_p.exists():
        # Fallback to .jsonl if .json is not found
        if raw_p.suffix == ".json":
            raw_p = raw_p.with_suffix(".jsonl")
        
        if not raw_p.exists():
            print(f"[process_data] raw file not found: {args.raw_path} or {raw_p}", file=sys.stderr)
            sys.exit(1)

    raw_items: List[Any] = []
    if raw_p.suffix == ".json":
        loaded = read_json(raw_p)
        # accept either a list of items or a dict with 'data'
        if isinstance(loaded, dict) and "data" in loaded:
            raw_items = loaded["data"]
        else:
            raw_items = loaded
    elif raw_p.suffix == ".jsonl":
        raw_items = list(iter_jsonl(raw_p))
    elif raw_p.suffix == ".csv":
        raw_items = read_csv(raw_p)
    else:
        print(f"[process_data] unsupported raw format: {raw_p.suffix}", file=sys.stderr)
        sys.exit(1)

    # Load context docs and system prompt from config if available
    context_content = ""
    system_prompt = args.system_prompt
    context_sample_ratio = 0.0
    
    # Check for production_rag config section
    if "production_rag" in cfg and cfg["production_rag"].get("enable_context_injection", False):
        # Load system prompt from file
        system_prompt_path = cfg["production_rag"].get("system_prompt_path")
        if system_prompt_path:
            prompt_path = Path(system_prompt_path)
            if prompt_path.exists():
                system_prompt = prompt_path.read_text(encoding='utf-8').strip()
                print(f"[process_data] Loaded system prompt from: {system_prompt_path}")
            else:
                print(f"[process_data] Warning: System prompt file not found: {system_prompt_path}")
        
        # Load context docs from file
        context_docs_path = cfg["production_rag"].get("context_docs_path")
        if context_docs_path:
            context_path = Path(context_docs_path)
            if context_path.exists():
                context_content = context_path.read_text(encoding='utf-8').strip()
                print(f"[process_data] Loaded context docs from: {context_docs_path} ({len(context_content)} characters)")
            else:
                print(f"[process_data] Warning: Context docs file not found: {context_docs_path}")
        
        # Get context sample ratio
        context_sample_ratio = float(cfg["production_rag"].get("context_sample_ratio", 0.2))
        print(f"[process_data] Context injection ratio: {context_sample_ratio:.1%}")
    
    # Override with command line args if provided
    if args.context_docs:
        context_path = Path(args.context_docs)
        if context_path.exists():
            context_content = context_path.read_text(encoding='utf-8').strip()
            context_sample_ratio = 1.0  # Apply to all if CLI override
            print(f"[process_data] Loaded context docs from CLI: {len(context_content)} characters")
        else:
            print(f"[process_data] Warning: Context docs file not found: {args.context_docs}")

    # normalize with hybrid context injection
    norm: List[Dict[str,str]] = []
    context_count = 0
    normal_count = 0
    
    for i, obj in enumerate(raw_items):
        try:
            row = normalize_item(obj, system_prompt)
            
            # Randomly decide if this sample gets context (based on ratio)
            should_add_context = (context_content and 
                                random.random() < context_sample_ratio)
            
            if should_add_context:
                original_query = row["user"]
                # Use smart chunking to fit within token limits
                chunked_context = smart_chunk_context(context_content, original_query, max_context_tokens=2500)
                # Format like production RAG: [CONTEXT] + [QUESTION]
                row["user"] = f"[CONTEXT]\n{chunked_context}\n\n[QUESTION]\n{original_query}"
                context_count += 1
                norm.append(sanitize(row, max_user_len=8192))  # Increased for context
            else:
                normal_count += 1
                norm.append(sanitize(row, max_user_len=4096))  # Normal length
                
        except Exception as e:
            # skip bad rows but log a hint
            print(f"[process_data] skip row due to: {e}")
    
    print(f"[process_data] Hybrid processing: {context_count} with context, {normal_count} without context")

    if not norm:
        print("[process_data] no valid rows after normalization.", file=sys.stderr)
        sys.exit(1)

    random.seed(cfg.get("seed", 42))
    random.shuffle(norm)

    # split from config
    split_cfg = cfg.get("data", {}).get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio   = float(split_cfg.get("val_ratio", 0.1))
    test_ratio  = float(split_cfg.get("test_ratio", 0.1))

    train_rows, val_rows, test_rows = split_rows(norm, train_ratio, val_ratio, test_ratio)

    # write to data/processed/
    out_train = Path(cfg["data"]["train_path"])
    out_val   = Path(cfg["data"]["val_path"])
    out_test  = Path(cfg["data"]["test_path"])

    write_jsonl(out_train, train_rows)
    write_jsonl(out_val,   val_rows)
    write_jsonl(out_test,  test_rows)

    # Gives tiny report
    def avg_len(key, rows): 
        return round(sum(len(r.get(key,"")) for r in rows)/max(1,len(rows)), 1)

    print(f"[process_data] saved: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    print(f"[process_data] avg chars — user: {avg_len('user', norm)}, assistant: {avg_len('assistant', norm)}")
    print("[process_data] sample:")
    print(json.dumps(train_rows[0], ensure_ascii=False)[:300])

if __name__ == "__main__":
    main()
