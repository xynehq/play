import argparse, json
from pathlib import Path
from typing import Dict, Any

import torch, yaml
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer)
from peft import PeftModel

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    ap.add_argument("--adapters", default="adapters/last", help="LoRA adapters directory")
    ap.add_argument("--out", default="outputs/merged_fp16", help="Output model dir")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16"], help="Save dtype for merged base")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    model_type = cfg["model"]["type"]  # "causal" | "seq2seq"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Load base in full precision (fp16/bf16), NOT 4-bit
    if model_type == "seq2seq":
        base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
    else:
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

    # Attach adapters
    adapters_dir = Path(args.adapters)
    if not adapters_dir.exists():
        raise FileNotFoundError(f"Adapters not found at {adapters_dir}")
    model = PeftModel.from_pretrained(base, str(adapters_dir))
    print(f"[merge] Loaded adapters from {adapters_dir}")

    # Merge and unload PEFT layers
    merged = model.merge_and_unload()
    print("[merge] Merged LoRA into base weights.")

    # Save final standalone model + tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    # Save small meta
    (out_dir / "sft_play_meta.json").write_text(json.dumps({
        "base_model": model_name,
        "dtype": args.dtype,
        "merged_from_adapters": str(adapters_dir),
    }, indent=2))

    print(f"[merge] Saved merged model -> {out_dir}")

if __name__ == "__main__":
    main()
