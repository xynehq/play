#!/usr/bin/env python3
"""
Production-grade Distributed Training Script for SFT-Play
- Multi-GPU via accelerate launch + HF Trainer (DDP)
- QLoRA (bnb 4-bit) or LoRA/full finetuning
- Proper prompt masking for train & eval (causal LM)
- Generation-based eval (ROUGE-L) per-epoch or per-steps
"""

import os
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_ATTENTION", "1")
os.environ.setdefault("UNSLOTH_FORCE_SDPA", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF logs
# reasonable NCCL defaults (override from env/Makefile if needed)
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")

import sys, json, math, argparse, warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)

# Allow "scripts" imports when launched from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset
import yaml

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from datasets import interleave_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint

# repo-local helpers (keep your previous files)
from scripts.utils.model_store import prepare_local_model_dir
from scripts.datasets_cpt import load_cpt_dataset, load_chat_dataset_for_cpt
from scripts.collators_cpt import DataCollatorForCausalPairs  # used only for CPT/DAPT
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ----------------- Config helpers -----------------
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


# ----------------- Target module resolution -----------------
def resolve_target_modules(model_name: str, override):
    if override and override != "auto":
        return override
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "qwen", "phi", "gemma"]):
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    if "t5" in name or "flan" in name:
        return ["q","k","v","o","wi_0","wi_1","wo"]
    return ["q_proj","k_proj","v_proj","o_proj"]


# ----------------- Datasets -----------------
class ChatDataset(Dataset):
    """
    Accepts lines in any of these shapes (JSONL):
      { "input": "...", "target": "..." }
      { "system": "...", "user": "...", "assistant": "..." }
      { "messages": [{"role": "...", "content": "..."} ...] }
    """
    def __init__(self, path: str):
        self.rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def _extract_pair(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (system, user, assistant) strings from any supported row format.
    system may be "" if not provided.
    """
    if "input" in row and "target" in row:
        return "", row["input"], row["target"]
    if "user" in row and "assistant" in row:
        return row.get("system", "") or "", row["user"], row["assistant"]
    if "messages" in row:
        sys_msg, usr, asst = "", None, None
        for m in row["messages"]:
            r = m.get("role")
            if r == "system": sys_msg = m.get("content", "")
            elif r == "user": usr = m.get("content", "")
            elif r == "assistant": asst = m.get("content", "")
        if usr is None or asst is None:
            raise ValueError(f"messages row missing user/assistant: {row}")
        return sys_msg, usr, asst
    raise ValueError(f"Unsupported row keys: {list(row.keys())}")


class CausalCollator:
    """
    Robust, fast collator for causal LM:
      - builds prompt + target token ids
      - masks prompt tokens with -100 in labels
      - pads on the right
    """
    def __init__(self, tokenizer: AutoTokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len
        self.tok.padding_side = "right"

        # ensure eos/pad ids exist
        if self.tok.pad_token_id is None:
            if self.tok.eos_token_id is not None:
                self.tok.pad_token = self.tok.eos_token
            else:
                self.tok.add_special_tokens({"pad_token": "<|pad|>"})

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attn_mask = [], [], []

        for row in batch:
            sys_txt, user_txt, asst_txt = _extract_pair(row)
            if sys_txt:
                prompt = f"System: {sys_txt}\nUser: {user_txt}\nAssistant:"
            else:
                prompt = f"User: {user_txt}\nAssistant:"

            ids_prompt = self.tok(prompt, add_special_tokens=True).input_ids
            ids_target = self.tok(asst_txt, add_special_tokens=False).input_ids
            eos = [] if self.tok.eos_token_id is None else [self.tok.eos_token_id]

            ids = ids_prompt + ids_target + eos
            lbl = [-100] * len(ids_prompt) + ids_target + (eos if eos else [])

            # truncate (keep same length)
            ids = ids[:self.max_len]
            lbl = lbl[:self.max_len]

            input_ids.append(ids)
            labels.append(lbl)
            attn_mask.append([1] * len(ids))

        # pad batch
        maxL = max(len(x) for x in input_ids)
        pad_id = self.tok.pad_token_id
        for i in range(len(input_ids)):
            pad = maxL - len(input_ids[i])
            if pad > 0:
                input_ids[i] += [pad_id] * pad
                labels[i] += [-100] * pad
                attn_mask[i] += [0] * pad

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }


# ----------------- Metrics -----------------
def compute_metrics_builder(tokenizer):
    """
    Returns a function computing ROUGE-L on generated text.
    Works when predict_with_generate=True.
    """
    def _compute(eval_pred):
        try:
            from evaluate import load as load_metric
            rouge = load_metric("rouge")
        except Exception:
            # evaluate not installed; skip metrics
            return {}

        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        # Handle different prediction formats in newer Transformers versions
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Ensure preds is a numpy array or list of integers
        import numpy as np
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        
        # Handle the case where preds might be a list of lists or a 2D array
        if len(preds.shape) > 1:
            # If it's already 2D, use as is
            preds_list = preds.tolist()
        else:
            # If it's 1D, wrap in a list
            preds_list = [preds.tolist()] if isinstance(preds, np.ndarray) else [preds]
        
        # replace -100 in labels with pad so we can decode
        labels = [[(t if t != -100 else tokenizer.pad_token_id) for t in seq] for seq in labels]

        try:
            pred_text = tokenizer.batch_decode(preds_list, skip_special_tokens=True)
            label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            r = rouge.compute(predictions=pred_text, references=label_text, use_stemmer=True)
            return {"rougeL": r.get("rougeL", 0.0)}
        except Exception as e:
            print(f"[eval] Error in compute_metrics: {e}")
            print(f"[eval] preds type: {type(preds)}, shape: {getattr(preds, 'shape', 'no shape')}")
            print(f"[eval] preds sample: {preds_list[:1] if len(preds_list) > 0 else 'empty'}")
            return {"rougeL": 0.0}
    return _compute


# ----------------- CPT builder (unchanged) -----------------
def build_cpt_or_mixed(cfg, tokenizer):
    mode = cfg.get("task_mode", "sft")
    if mode == "cpt":
        cpt_path = cfg["datasets"][0]["path"]
        block_size = cfg.get("block_size", 2048)
        pack_factor = cfg.get("pack_factor", 4)
        cpt = load_cpt_dataset(cpt_path, tokenizer, block_size=block_size, pack_factor=pack_factor)
        return cpt, None, DataCollatorForCausalPairs(tokenizer)
    elif mode == "cpt_mixed":
        datasets_list, weights = [], []
        block_size = cfg.get("block_size", 2048)
        pack_factor = cfg.get("pack_factor", 4)
        for ds_cfg in cfg["datasets"]:
            if ds_cfg["type"] == "cpt":
                ds = load_cpt_dataset(ds_cfg["path"], tokenizer, block_size=block_size, pack_factor=pack_factor)
            elif ds_cfg["type"] == "chat":
                ds = load_chat_dataset_for_cpt(ds_cfg["path"], tokenizer, block_size=block_size)
            else:
                raise ValueError(f"Unknown dataset type: {ds_cfg['type']}")
            datasets_list.append(ds)
            weights.append(ds_cfg.get("weight", 1.0))
        s = sum(weights)
        weights = [w / s for w in weights]
        mixed = interleave_datasets(datasets_list, probabilities=weights, seed=cfg.get("seed", 42))
        return mixed, None, DataCollatorForCausalPairs(tokenizer)
    else:
        return None, None, None


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/run_distributed.yaml")
    ap.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed (pass config via YAML or --deepspeed_config)")
    ap.add_argument("--deepspeed_config", default=None, help="DeepSpeed JSON path (overrides YAML)")
    args = ap.parse_args()

    # Accelerator used for world/rank info & graceful shutdown. Trainer does DDP.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    # Model card → local dir (your helper can download/cache)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)
    model_name = local_model_dir
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    model_type = cfg["model"].get("type", "causal")
    max_len = int(cfg["model"].get("max_seq_len", 2048))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    tok.padding_side = "right"

    # Data
    ds_train, ds_val, data_collator = None, None, None
    if cfg.get("task_mode", "sft") in ("cpt", "cpt_mixed"):
        ds_train, ds_val, data_collator = build_cpt_or_mixed(cfg, tok)
    else:
        paths = cfg["data"]
        train_path = paths["train_path"]; val_path = paths["val_path"]
        # validate files exist
        if not os.path.exists(train_path): raise FileNotFoundError(f"train data not found: {train_path}")
        if not os.path.exists(val_path):   raise FileNotFoundError(f"val data not found: {val_path}")
        ds_train = ChatDataset(train_path)
        ds_val = ChatDataset(val_path)
        data_collator = CausalCollator(tok, max_len)

    # Backend and tuning mode
    backend = cfg["tuning"]["backend"]   # "bnb"|"unsloth"
    mode    = cfg["tuning"]["mode"]      # "qlora"|"lora"|"full"
    is_seq2seq = (model_type == "seq2seq")

    # Build model
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
            trust_remote_code=trust_remote_code
        )
    else:
        if mode == "qlora":
            # prefer BitsAndBytes for distributed stability
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # LoRA or Full finetune (no quantization)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

    # Apply LoRA where applicable
    if mode in ["qlora", "lora"]:
        lcfg = cfg["tuning"]["lora"]
        targets = resolve_target_modules(model_name, lcfg.get("target_modules"))
        lora_cfg = LoraConfig(
            r=int(lcfg.get("r", 32)),
            lora_alpha=int(lcfg.get("alpha", 64)),
            lora_dropout=float(lcfg.get("dropout", 0.1)),
            target_modules=targets,
            bias="none",
            task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    # Disable cache when using gradient checkpointing
    if bool(cfg["train"].get("gradient_checkpointing", True)) and hasattr(model, "config"):
        model.config.use_cache = False

    # Output dir & run name
    run_name = str(cfg["train"].get("run_name", "distributed-training"))
    base_outdir = str(cfg["train"].get("output_dir", "outputs/distributed-training"))
    if run_name == "distributed-training":
        from datetime import datetime
        run_name = f"distributed-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = f"outputs/{run_name}"

    # Backend stamp (nice for resuming sanity)
    if accelerator.is_main_process:
        os.makedirs(outdir, exist_ok=True)
        stamp = {"backend": backend, "dtype": "bf16" if cfg["train"].get("bf16", True) else "fp16",
                 "num_gpus": torch.cuda.device_count(), "distributed": True, "run_name": run_name}
        with open(os.path.join(outdir, "backend.json"), "w") as f:
            json.dump(stamp, f, indent=2)

    # Handle auto values for batch size
    batch_size = cfg["train"].get("batch_size", 1)
    if batch_size == "auto":
        per_device_train_batch_size = 1  # Default value for auto
    else:
        per_device_train_batch_size = int(batch_size)
    
    per_device_eval_batch_size = cfg["train"].get("per_device_eval_batch_size", "auto")
    if per_device_eval_batch_size == "auto":
        per_device_eval_batch_size = max(1, per_device_train_batch_size)
    else:
        per_device_eval_batch_size = int(per_device_eval_batch_size)

    # Handle auto values for gradient accumulation
    grad_accum = cfg["train"].get("grad_accum", 8)
    if grad_accum == "auto":
        gradient_accumulation_steps = 8  # Default value for auto
    else:
        gradient_accumulation_steps = int(grad_accum)

    # Build TrainingArguments with compatibility for different Transformers versions
    training_args_kwargs = {
        "output_dir": outdir,
        "run_name": f"sft-play-distributed-{torch.cuda.device_count()}gpu",
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": float(cfg["train"].get("epochs", 1)),
        "learning_rate": float(cfg["train"].get("learning_rate", cfg["train"].get("lr", 2e-4))),
        "warmup_ratio": float(cfg["train"].get("warmup_ratio", 0.06)),
        "weight_decay": float(cfg["train"].get("weight_decay", 0.01)),
        "bf16": bool(cfg["train"].get("bf16", True)),
        "fp16": bool(cfg["train"].get("fp16", False)),
        "logging_dir": "outputs/tb",
        "logging_steps": int(cfg["train"].get("logging_steps", 1)),
        "save_total_limit": int(cfg["train"].get("save_total_limit", 5)),
        "remove_unused_columns": False,

        # DDP-related stability & speed
        "dataloader_pin_memory": bool(cfg["train"].get("dataloader_pin_memory", False)),
        "dataloader_num_workers": int(cfg["train"].get("dataloader_num_workers", 4)),
        "gradient_checkpointing": bool(cfg["train"].get("gradient_checkpointing", True)),
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,

        # Eval/save strategy
        "eval_strategy": str(cfg["train"].get("evaluation_strategy", "no")),
        "eval_steps": int(cfg["train"].get("eval_steps", 200)),
        "save_strategy": str(cfg["train"].get("save_strategy", "epoch")),
        "save_steps": int(cfg["train"].get("save_steps", 200)),
        "load_best_model_at_end": bool(cfg["train"].get("load_best_model_at_end", False)),
        "metric_for_best_model": str(cfg["train"].get("metric_for_best_model", "eval_loss")) if cfg["train"].get("load_best_model_at_end", False) else None,
        "greater_is_better": bool(cfg["train"].get("greater_is_better", False)),

        # Generation-based eval (for ROUGE etc.) - add conditionally
        "eval_accumulation_steps": int(cfg["train"].get("eval_accumulation_steps", 8)),

        "report_to": ["tensorboard"],
    }

    # Add generation parameters only if they're supported in this Transformers version
    try:
        # Try to create a dummy TrainingArguments to see if generation_max_length is supported
        test_args = TrainingArguments(output_dir="test", generation_max_length=128)
        training_args_kwargs["generation_max_length"] = int(cfg["train"].get("generation_max_length", 128))
        training_args_kwargs["generation_num_beams"] = int(cfg["train"].get("generation_num_beams", 1))
    except TypeError:
        # If not supported, skip these parameters
        print("[train] Generation parameters not supported in this Transformers version, skipping...")

    tr_args = TrainingArguments(**training_args_kwargs)

    # Optional DeepSpeed
    ds_cfg = args.deepspeed_config or cfg["train"].get("deepspeed")
    if args.deepspeed or ds_cfg:
        tr_args.deepspeed = ds_cfg

    # Resume (if any)
    last_ckpt = get_last_checkpoint(outdir) if os.path.isdir(outdir) else None
    if accelerator.is_main_process and last_ckpt:
        print(f"[train] Resuming from checkpoint: {last_ckpt}")

    # Build Trainer (no accelerator.prepare; Trainer handles DDP under accelerate launch)
    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_val if tr_args.eval_strategy != "no" else None,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_builder(tok) if tr_args.eval_strategy != "no" else None,
    )

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Optional final evaluation even if strategy="no"
    if tr_args.eval_strategy == "no" and ds_val is not None and bool(cfg["train"].get("final_eval", False)):
        metrics = trainer.evaluate()
        if accelerator.is_main_process:
            print("[eval-final]", metrics)

    # Save adapters/checkpoints
    if mode in ["qlora", "lora"]:
        if accelerator.is_main_process:
            os.makedirs("adapters", exist_ok=True)
        trainer.model.save_pretrained("adapters/last")

    # Graceful shutdown to avoid NCCL warning
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

    if accelerator.is_main_process:
        print("[train] Distributed training completed.")

if __name__ == "__main__":
    main()
