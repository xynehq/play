import argparse, json
from pathlib import Path
from typing import Dict, Any
import yaml
from jinja2 import Template

def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        run_cfg = yaml.safe_load(f)
    base = run_cfg.get("include")
    if base:
        with open(base, "r") as f:
            base_cfg = yaml.safe_load(f)
        return deep_merge(base_cfg, run_cfg)
    return run_cfg

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config_run.yaml")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (processed chat)")
    ap.add_argument("--out", required=True, help="Output JSONL (rendered seq2seq)")
    ap.add_argument("--template", default=None, help="Optional override to template path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tmpl_path = Path(args.template or cfg["data"]["template_path"])
    template = Template(tmpl_path.read_text())

    rendered = []
    for row in iter_jsonl(Path(args.inp)):
        sys_txt = (row.get("system") or "").strip()
        user_txt = row["user"].strip()
        asst_txt = row["assistant"].strip()
        inp_txt = template.render(system=sys_txt, user=user_txt).strip()
        rendered.append({"input": inp_txt, "target": asst_txt})

    out_p = Path(args.out)
    write_jsonl(out_p, rendered)
    print(f"[render_template] rendered {len(rendered)} rows -> {out_p}")
    # small preview
    if rendered:
        print("[render_template] preview input:")
        print(rendered[0]["input"][:300])

if __name__ == "__main__":
    main()
