import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, List

import torch, yaml
from jinja2 import Template
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import PeftModel
from scripts.utils.model_store import prepare_local_model_dir
import os

# ---------- config ----------
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        run_cfg = yaml.safe_load(f)
    base_path = run_cfg.get("include")
    if base_path:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        return deep_merge(base_cfg, run_cfg)
    return run_cfg

# ---------- gen ----------
@torch.inference_mode()
def generate(model, tok, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float, model_type: str):
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tok.model_max_length)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=(temperature>0), temperature=temperature, top_p=top_p)
    if model_type != "seq2seq":
        gen_kwargs["pad_token_id"] = tok.pad_token_id
    out_ids = model.generate(**inputs, **gen_kwargs)
    return tok.batch_decode(out_ids, skip_special_tokens=True)

def load_model_and_tok(cfg, adapters_path: str = None):
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)

    model_name = local_model_dir   # from now on, load FROM DISK
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))
    model_type = cfg["model"]["type"]  # "causal" | "seq2seq"

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code)

    # attach adapters if path exists
    if adapters_path and Path(adapters_path).exists():
        try:
            model = PeftModel.from_pretrained(model, adapters_path)
            print(f"[infer] loaded adapters from {adapters_path}")
        except Exception as e:
            print(f"[infer] warning: failed to load adapters: {e}")

    model.eval()
    return model, tok

def render_prompt(tmpl: Template, system_txt: str, user_txt: str) -> str:
    return tmpl.render(system=(system_txt or "").strip(), user=user_txt.strip()).strip()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    ap.add_argument("--adapters", default="adapters/last", help="LoRA adapters dir (if used)")
    ap.add_argument("--mode", choices=["interactive","batch"], default="interactive")
    ap.add_argument("--input_file", help="Batch: file with one user prompt per line")
    ap.add_argument("--output_file", default="outputs/preds.txt", help="Batch: where to write generations")
    ap.add_argument("--system", default="You are a helpful domain assistant.", help="System prompt for inference")
    ap.add_argument("--max_new_tokens", type=int, help="Override config.gen.max_new_tokens")
    ap.add_argument("--temperature", type=float, help="Override config.gen.temperature")
    ap.add_argument("--top_p", type=float, help="Override config.gen.top_p")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tmpl = Template(Path(cfg["data"]["template_path"]).read_text())

    # gen params (allow CLI override)
    gcfg = cfg.get("gen", {})
    max_new_tokens = args.max_new_tokens or int(gcfg.get("max_new_tokens", 200))
    temperature    = args.temperature or float(gcfg.get("temperature", 0.2))
    top_p          = args.top_p or float(gcfg.get("top_p", 0.9))

    model, tok = load_model_and_tok(cfg, adapters_path=args.adapters)

    if args.mode == "interactive":
        print("=== SFT-Play :: Interactive ===")
        print("Type your question. Press Ctrl+C to exit.")
        while True:
            try:
                user = input("\nYou: ").strip()
                if not user:
                    continue
                prompt = render_prompt(tmpl, args.system, user)
                pred = generate(model, tok, [prompt], max_new_tokens, temperature, top_p, cfg["model"]["type"])[0]
                # For causal LMs, generation includes the prompt; try to strip it
                if cfg["model"]["type"] == "causal":
                    # naive split on last assistant tag if present
                    split_tag = "</|assistant|>"
                    start_tag = "<|assistant|>"
                    if start_tag in pred:
                        pred = pred.split(start_tag, 1)[-1]
                    if split_tag in pred:
                        pred = pred.split(split_tag, 1)[0]
                print(f"Model: {pred.strip()}")
            except KeyboardInterrupt:
                print("\nbye!")
                break
    else:
        if not args.input_file:
            print("[infer] batch mode requires --input_file", file=sys.stderr)
            sys.exit(1)
        lines = [ln.strip() for ln in Path(args.input_file).read_text().splitlines() if ln.strip()]
        prompts = [render_prompt(tmpl, args.system, u) for u in lines]
        preds = generate(model, tok, prompts, max_new_tokens, temperature, top_p, cfg["model"]["type"])
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            for p in preds:
                f.write(p.strip() + "\n")
        print(f"[infer] wrote {len(preds)} generations -> {args.output_file}")

if __name__ == "__main__":
    main()
