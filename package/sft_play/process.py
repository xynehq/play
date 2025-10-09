"""
Data processing module - converts raw data to training format.
Supports JSON, JSONL, and CSV formats.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Any:
    """Read JSON file."""
    return json.loads(path.read_text())


def iter_jsonl(path: Path):
    """Iterate over JSONL file."""
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    """Read CSV file."""
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def normalize_item(obj: Any, default_system: str) -> Dict[str, str]:
    """
    Convert various raw shapes into {system, user, assistant}.

    Supported formats:
      - {"question": "...", "answer": "..."}
      - {"user": "...", "assistant": "..."}
      - {"prompt": "...", "response": "..."}
      - {"instruction": "...", "output": "..."}
    """
    if isinstance(obj, dict):
        # Find user message (various field names)
        user = (obj.get("user") or obj.get("question") or obj.get("prompt") or
                obj.get("input") or obj.get("instruction") or obj.get("query"))

        # Find assistant message (various field names)
        assistant = (obj.get("assistant") or obj.get("answer") or
                     obj.get("response") or obj.get("target") or obj.get("output"))

        # Find system message (optional)
        system = obj.get("system") or default_system

        if user and assistant:
            return {
                "system": system.strip(),
                "user": user.strip(),
                "assistant": assistant.strip()
            }
        else:
            raise ValueError("Missing user/assistant fields in item")

    elif isinstance(obj, list) and len(obj) >= 2:
        # Handle list format: ["question", {"answer": "..."}]
        user = obj[0] if isinstance(obj[0], str) else None
        ans = obj[1]
        assistant = ans.get("answer") if isinstance(ans, dict) else None

        if user and assistant:
            return {
                "system": default_system,
                "user": user.strip(),
                "assistant": assistant.strip()
            }
        else:
            raise ValueError("Unrecognized list format")
    else:
        raise ValueError(f"Unsupported item type: {type(obj)}")


def sanitize(
    row: Dict[str, str],
    max_user_len: int = 4096,
    max_assistant_len: int = 4096
) -> Dict[str, str]:
    """Sanitize and truncate row fields."""
    row["user"] = row["user"].strip()[:max_user_len]
    row["assistant"] = row["assistant"].strip()[:max_assistant_len]
    row["system"] = (row.get("system") or "").strip()
    return row


def split_rows(
    rows: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split rows into train/val/test sets."""
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    return train, val, test


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    """Write rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main data processing entry point."""
    parser = argparse.ArgumentParser(
        description="Process raw data for training"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data file (JSON, JSONL, or CSV)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Default system prompt",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args(argv)

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Read raw data based on file extension
    raw_items: List[Any] = []

    print(f"Reading data from: {input_path}")

    if input_path.suffix == ".json":
        loaded = read_json(input_path)
        # Accept either a list or a dict with 'data' key
        if isinstance(loaded, dict) and "data" in loaded:
            raw_items = loaded["data"]
        elif isinstance(loaded, list):
            raw_items = loaded
        else:
            print("Error: JSON must be a list or dict with 'data' key", file=sys.stderr)
            return 1

    elif input_path.suffix == ".jsonl":
        raw_items = list(iter_jsonl(input_path))

    elif input_path.suffix == ".csv":
        raw_items = read_csv(input_path)

    else:
        print(f"Error: Unsupported format: {input_path.suffix}", file=sys.stderr)
        print("Supported formats: .json, .jsonl, .csv", file=sys.stderr)
        return 1

    if not raw_items:
        print("Error: No data found in input file", file=sys.stderr)
        return 1

    print(f"Loaded {len(raw_items)} raw items")

    # Normalize data
    norm: List[Dict[str, str]] = []
    skipped = 0

    for obj in raw_items:
        try:
            row = normalize_item(obj, args.system_prompt)
            norm.append(sanitize(row))
        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Show first 5 errors
                print(f"Warning: Skipped row due to: {e}")

    if skipped > 5:
        print(f"Warning: Skipped {skipped} total rows")

    if not norm:
        print("Error: No valid rows after normalization", file=sys.stderr)
        return 1

    print(f"Normalized {len(norm)} items")

    # Shuffle
    random.seed(args.seed)
    random.shuffle(norm)

    # Split data
    train_rows, val_rows, test_rows = split_rows(
        norm, args.train_ratio, args.val_ratio, args.test_ratio
    )

    # Write output files
    output_dir = Path(args.output)
    out_train = output_dir / "train.jsonl"
    out_val = output_dir / "val.jsonl"
    out_test = output_dir / "test.jsonl"

    write_jsonl(out_train, train_rows)
    write_jsonl(out_val, val_rows)
    write_jsonl(out_test, test_rows)

    # Print statistics
    def avg_len(key, rows):
        return round(sum(len(r.get(key, "")) for r in rows) / max(1, len(rows)), 1)

    print()
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Train set: {len(train_rows)} samples")
    print(f"Val set:   {len(val_rows)} samples")
    print(f"Test set:  {len(test_rows)} samples")
    print()
    print(f"Average user chars:      {avg_len('user', norm)}")
    print(f"Average assistant chars: {avg_len('assistant', norm)}")
    print()
    print(f"Output directory: {output_dir}")
    print(f"  - {out_train}")
    print(f"  - {out_val}")
    print(f"  - {out_test}")
    print()
    print("Sample item:")
    print(json.dumps(train_rows[0], ensure_ascii=False, indent=2)[:300])
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
