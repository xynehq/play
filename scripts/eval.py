import argparse, json, re, math, os
from pathlib import Path
from typing import Dict, Any, List

import torch, yaml
from jinja2 import Template
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM)
from peft import PeftModel
from evaluate import load as load_metric
from scripts.utils.model_store import prepare_local_model_dir

# ---------- config loader ----------
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str) -> Dict[str, Any]:
    run_cfg = yaml.safe_load(open(path))
    base = run_cfg.get("include")
    if base:
        base_cfg = yaml.safe_load(open(base))
        return deep_merge(base_cfg, run_cfg)
    return run_cfg

# ---------- data io ----------
def iter_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- metrics ----------
rouge = load_metric("rouge")
sari = load_metric("sari")

def exact_match(a: str, b: str) -> int:
    return int(a.strip() == b.strip())

def schema_ok(txt: str) -> bool:
    """
    Optional: check for your schema/headings; edit to your task.
    Example: ensure it contains "Role:" and "Output:" once.
    """
    # Return True if not using schema checks
    return True

# ---------- generate ----------
@torch.inference_mode()
def generate_batch(model, tok, texts: List[str], max_new_tokens=200, temperature=0.2, top_p=0.9, model_type="causal"):
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    if model_type == "seq2seq":
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature>0), temperature=temperature, top_p=top_p)
    else:
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature>0), temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out_ids, skip_special_tokens=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    ap.add_argument("--split", default="val", choices=["val","test"], help="which split to evaluate")
    ap.add_argument("--adapters", default="adapters/last", help="path to LoRA adapters (if used)")
    ap.add_argument("--samples_out", default="outputs/samples.jsonl")
    ap.add_argument("--metrics_out", default="outputs/metrics.json")
    ap.add_argument("--limit", type=int, default=512, help="evaluate at most N examples for speed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)

    model_name = local_model_dir   # from now on, load FROM DISK
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))
    model_type = cfg["model"]["type"]  # "causal" | "seq2seq"
    tmpl = Template(Path(cfg["data"]["template_path"]).read_text())

    # load tokenizer & model
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code)

    # attach adapters if present
    adapters_path = Path(args.adapters)
    if adapters_path.exists():
        try:
            model = PeftModel.from_pretrained(model, str(adapters_path))
            print(f"[eval] loaded adapters from {adapters_path}")
        except Exception as e:
            print(f"[eval] adapters not loaded ({e}); evaluating base model.")

    model.eval()

    # load data
    split_path = cfg["data"]["val_path"] if args.split=="val" else cfg["data"]["test_path"]
    rows = list(iter_jsonl(Path(split_path)))
    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]

    # render inputs
    inputs = [tmpl.render(system=(r.get("system") or "").strip(), user=r["user"].strip()).strip() for r in rows]
    refs   = [r["assistant"].strip() for r in rows]

    # generate predictions
    gen_cfg = cfg.get("gen", {})
    preds = generate_batch(model, tok, inputs,
                           max_new_tokens=int(gen_cfg.get("max_new_tokens", 200)),
                           temperature=float(gen_cfg.get("temperature", 0.2)),
                           top_p=float(gen_cfg.get("top_p", 0.9)),
                           model_type=model_type)

    # compute metrics
    r = rouge.compute(predictions=preds, references=refs)
    # SARI expects (source, prediction, reference); use input as "source" for rewrite-ish tasks
    sari_scores = sari.compute(sources=inputs, predictions=preds, references=[[t] for t in refs])

    em = sum(exact_match(p, t) for p,t in zip(preds, refs)) / max(1,len(refs))
    sc = sum(1 if schema_ok(p) else 0 for p in preds) / max(1,len(preds))

    metrics = {
        "count": len(rows),
        "rougeL": r["rougeL"],
        "sari": sari_scores["sari"],
        "exact_match": round(float(em), 4),
        "schema_compliance": round(float(sc), 4),
    }

    # write outputs
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] metrics -> {args.metrics_out}")
    print(json.dumps(metrics, indent=2))

    with open(args.samples_out, "w") as f:
        for i, (inp, pred, ref) in enumerate(zip(inputs, preds, refs)):
            f.write(json.dumps({"input": inp, "pred": pred, "ref": ref}, ensure_ascii=False) + "\n")
    print(f"[eval] samples -> {args.samples_out}")

if __name__ == "__main__":
    main()
