import argparse, json
from pathlib import Path

def iter_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)  # reserved: for symmetry; not used here yet
    ap.add_argument("--style", required=True, help="Instruction to inject into system prompt")
    ap.add_argument("--in", dest="inp", required=True, help="Input processed JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--mode", choices=["prepend","replace"], default="prepend",
                    help="prepend (default) or replace the existing system")
    args = ap.parse_args()

    styled = []
    for row in iter_jsonl(Path(args.inp)):
        old_sys = (row.get("system") or "").strip()
        if args.mode == "replace":
            new_sys = args.style.strip()
        else:  # prepend
            new_sys = args.style.strip() if not old_sys else f"{args.style.strip()}\n\n{old_sys}"
        styled.append({"system": new_sys, "user": row["user"], "assistant": row["assistant"]})

    out_p = Path(args.out)
    write_jsonl(out_p, styled)
    print(f"[style_prompt] wrote {len(styled)} rows -> {out_p}")
    print(f"[style_prompt] system prompt style: {args.style.strip()}")