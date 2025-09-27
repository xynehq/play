#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HF Base → LoRA → Merge → Inference (single file)
- Optional Hugging Face login
- Load base model (bnb 4bit optional)
- Load LoRA from HF (repo + subfolder) OR local path
- Merge & optionally save merged model
- Run single prompt or interactive Q/A
"""
import os, argparse
from typing import Dict, Any, Optional
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF → LoRA → Merge → Inference")
    p.add_argument("--login", action="store_true", help="Login to Hugging Face")
    p.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token")

    p.add_argument("--model-id", type=str, default="google/gemma-3-27b-it")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--attn-impl", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])

    p.add_argument("--adapter-repo-id", type=str, default=None)
    p.add_argument("--adapter-subfolder", type=str, default=None)
    p.add_argument("--adapter-local-path", type=str, default=None)

    p.add_argument("--save-merged", type=str, default=None)

    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--do-sample", action="store_true")

    p.add_argument("--system-prompt", type=str, default="You are a helpful AI assistant.")
    p.add_argument("--user-prompt", type=str, default=None)
    p.add_argument("--interactive", action="store_true")

    return p.parse_args()

def maybe_login(login_flag: bool, token: Optional[str]):
    if not login_flag and not token:
        return
    try:
        from huggingface_hub import whoami
        whoami(token=token)
        print("Already authenticated (or token provided).")
    except Exception:
        from huggingface_hub import login
        if token:
            print("Logging in with provided token...")
            login(token=token, add_to_git_credential=True)
        else:
            print("Logging in to Hugging Face...")
            login()

def load_base_model(model_id, trust_remote_code, device_map, load_in_4bit, dtype_str, attn_impl):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    DTYPE = torch.float16 if dtype_str == "float16" else torch.bfloat16
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print(f"[Base] Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=(torch.bfloat16 if load_in_4bit else DTYPE),
        quantization_config=bnb_config,
        attn_implementation=attn_impl,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[Base] Loaded.")
    return model, tokenizer, DTYPE

def load_and_merge_lora(model, adapter_repo_id, adapter_subfolder, adapter_local_path, device_map, dtype):
    # If no adapter specified, return the base model as-is
    if not adapter_local_path and not adapter_repo_id:
        print("[LoRA] No adapter specified, using base model.")
        return model
        
    from peft import PeftModel
    kwargs = {}
    if adapter_local_path:
        src = adapter_local_path
        print(f"[LoRA] Loading adapters from local path: {src}")
    else:
        if not adapter_repo_id:
            raise ValueError("Provide --adapter-repo-id+--adapter-subfolder OR --adapter-local-path")
        src = adapter_repo_id
        if adapter_subfolder:
            kwargs["subfolder"] = adapter_subfolder
        print(f"[LoRA] Loading adapters from HF: {adapter_repo_id} (subfolder={adapter_subfolder})")
    model = PeftModel.from_pretrained(model, src, device_map=device_map, torch_dtype=dtype, **kwargs)
    print("[LoRA] Merging into base...")
    model = model.merge_and_unload()
    model.to(dtype=dtype)
    model.eval()   # add this
    print("[LoRA] Merge complete.")
    return model

def save_merged(model, tokenizer, out_dir):
    if not out_dir:
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[Save] Saving merged model to: {out.resolve()}")
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

def build_inputs(tokenizer, prompt: str, system_prompt: Optional[str] = None):
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    else:
        text = (system_prompt + "\n\n" if system_prompt else "") + prompt
        return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)

def generate_text(model, tokenizer, prompt: str, system_prompt: Optional[str], gen_kw: Dict[str, Any]) -> str:
    import torch
    from transformers import LogitsProcessorList
    try:
        from transformers.generation.logits_process import InfNanRemoveLogitsProcessor
        LOGITS_PROCESSORS = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
    except Exception:
        LOGITS_PROCESSORS = None
    
    with torch.inference_mode():
        # Build input text
        if hasattr(tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = (system_prompt + "\n\n" if system_prompt else "") + prompt
        
        # Tokenize with proper attention mask
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        # With device_map="auto", keep inputs on CPU; HF will scatter across shards
        inputs = {k: v for k, v in inputs.items()}
        
        # Filter valid generation parameters
        model_config = model.generation_config.to_dict() if hasattr(model, 'generation_config') else {}
        valid_gen_kw = {k: v for k, v in gen_kw.items() if k in model_config}
        
        # Add essential parameters that might be missing
        if 'eos_token_id' not in valid_gen_kw:
            valid_gen_kw['eos_token_id'] = tokenizer.eos_token_id
        if 'pad_token_id' not in valid_gen_kw:
            valid_gen_kw['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Generate with error handling
        try:
            outputs = model.generate(
                **inputs,
                **valid_gen_kw,
                logits_processor=LOGITS_PROCESSORS if LOGITS_PROCESSORS is not None else None,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            # Fallback to simple generation
            fallback_kwargs = {
                "max_new_tokens": min(100, gen_kw.get("max_new_tokens", 100)),
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "do_sample": False,
            }
            outputs = model.generate(
                **inputs,
                **fallback_kwargs,
                logits_processor=LOGITS_PROCESSORS if LOGITS_PROCESSORS is not None else None,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            return text.strip()

def build_generation_kwargs(args, tokenizer) -> Dict[str, Any]:
    """Build generation kwargs with validation"""
    gen_kw = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    
    # Only add sampling parameters if do_sample is True
    if args.do_sample:
        gen_kw.update({
            "temperature": max(0.1, args.temperature),  # Ensure temperature > 0
            "do_sample": True,
        })
        # Only add top_p and top_k if they're reasonable values
        if args.top_p > 0 and args.top_p <= 1.0:
            gen_kw["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kw["top_k"] = args.top_k
        if args.repetition_penalty > 0:
            gen_kw["repetition_penalty"] = args.repetition_penalty
    else:
        # Use greedy decoding
        gen_kw.update({
            "do_sample": False,
        })
    
    return gen_kw

def main():
    import torch
    # Enable TensorFloat32 for better performance
    torch.set_float32_matmul_precision('high')
    
    args = parse_args()
    maybe_login(args.login, args.hf_token)

    model, tokenizer, DTYPE = load_base_model(
        args.model_id, args.trust_remote_code, args.device_map, args.load_in_4bit, args.dtype, args.attn_impl
    )
    model = load_and_merge_lora(
        model, args.adapter_repo_id, args.adapter_subfolder, args.adapter_local_path, args.device_map, DTYPE
    )
    save_merged(model, tokenizer, args.save_merged)

    GEN_KW = build_generation_kwargs(args, tokenizer)

    if args.user_prompt:
        print("\n[Single] User:", args.user_prompt)
        print("\n[Answer]\n", generate_text(model, tokenizer, args.user_prompt, args.system_prompt, GEN_KW))
        return

    if args.interactive:
        print("\nInteractive Q/A. Type 'exit' to quit.")
        while True:
            try:
                q = input("\nAsk: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!"); break
            if q.lower() in {"exit", "quit"}: print("Bye!"); break
            if not q: continue
            try:
                print("\n", generate_text(model, tokenizer, q, args.system_prompt, GEN_KW))
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    main()
