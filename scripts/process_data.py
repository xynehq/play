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
      - {"user":"...", "assistant":"..."}  (system optional)
      - ["what is animal?", {"answer": "..."}]  # pairwise toy case
      - {"prompt":"...", "response":"..."}
    """
    if isinstance(obj, dict):
        # common field names
        user = obj.get("user") or obj.get("question") or obj.get("prompt") or obj.get("input")
        assistant = obj.get("assistant") or obj.get("answer") or obj.get("response") or obj.get("target")
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

    # normalize
    norm: List[Dict[str,str]] = []
    for obj in raw_items:
        try:
            row = normalize_item(obj, args.system_prompt)
            norm.append(sanitize(row))
        except Exception as e:
            # skip bad rows but log a hint
            print(f"[process_data] skip row due to: {e}")

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
