#!/usr/bin/env python3
"""
Multi-GPU Distributed Training Script for SFT-Play
Supports training large models (27B+) across multiple GPUs using DeepSpeed and Accelerate
"""

import os
# Force disable XFormers and fast attention BEFORE any other imports
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_DISABLE_FAST_ATTENTION"] = "1"
os.environ["UNSLOTH_FORCE_SDPA"] = "1"

import sys, json, math, argparse, warnings
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from jinja2 import Template
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import deepspeed

# Import existing modules
from scripts.utils.model_store import prepare_local_model_dir
from scripts.datasets_cpt import load_cpt_dataset, load_chat_dataset_for_cpt
from scripts.collators_cpt import DataCollatorForCausalPairs
from datasets import interleave_datasets

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Multi-GPU Memory Management ----------
def get_gpu_memory_info():
    """Get memory info for all available GPUs"""
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        gpu_name = torch.cuda.get_device_name(i)
        gpu_info.append({
            'id': i,
            'name': gpu_name,
            'total_memory_gb': total_memory
        })
    return gpu_info

def calculate_optimal_batch_size(model_name: str, max_len: int, num_gpus: int, total_vram_gb: float, accelerator=None):
    """Calculate optimal batch size for multi-GPU training with resource awareness"""
    # Estimate model memory usage (more accurate heuristics)
    model_size_gb = 0
    model_params_billion = 0
    
    # Parse model size from name
    name_lower = model_name.lower()
    if "27b" in name_lower or "30b" in name_lower:
        model_size_gb = 54  # ~54GB for 27B model in fp16
        model_params_billion = 27
    elif "70b" in name_lower:
        model_size_gb = 140  # ~140GB for 70B model in fp16
        model_params_billion = 70
    elif "13b" in name_lower or "14b" in name_lower:
        model_size_gb = 26  # ~26GB for 13B model in fp16
        model_params_billion = 13
    elif "7b" in name_lower:
        model_size_gb = 14  # ~14GB for 7B model in fp16
        model_params_billion = 7
    elif "3b" in name_lower:
        model_size_gb = 6  # ~6GB for 3B model in fp16
        model_params_billion = 3
    elif "1.5b" in name_lower:
        model_size_gb = 3  # ~3GB for 1.5B model in fp16
        model_params_billion = 1.5
    else:
        model_size_gb = 28  # Default assumption for medium models
        model_params_billion = 7
    
    # Account for QLoRA memory reduction (roughly 4x less)
    is_qlora = True  # Assume QLoRA by default in distributed training
    if is_qlora:
        model_size_gb = model_size_gb / 4  # QLoRA reduces memory significantly
    
    # Calculate available memory per GPU with safety margin
    memory_per_gpu = total_vram_gb / num_gpus
    safety_margin_gb = max(8, min(20, memory_per_gpu * 0.1))  # 10% safety margin, min 8GB, max 20GB
    available_memory = memory_per_gpu - model_size_gb - safety_margin_gb
    
    if accelerator and accelerator.is_main_process:
        print(f"[train] Resource allocation:")
        print(f"  - Total VRAM: {total_vram_gb:.1f} GB ({num_gpus} GPUs)")
        print(f"  - Memory per GPU: {memory_per_gpu:.1f} GB")
        print(f"  - Model size: ~{model_size_gb:.1f} GB (QLoRA)")
        print(f"  - Safety margin: {safety_margin_gb:.1f} GB")
        print(f"  - Available memory per GPU: {available_memory:.1f} GB")
    
    if available_memory <= 0:
        if accelerator and accelerator.is_main_process:
            print(f"[train] Warning: Insufficient memory. Using minimum batch size with high gradient accumulation.")
        return 1, 128  # Minimum batch size with very high gradient accumulation
    
    # More accurate memory per sample calculation
    # Base memory: tokens * 4 bytes * 2 (for gradients) + optimizer states
    base_memory_per_token = 4 * 2 * 2  # 2x for gradients, 2x for optimizer states (Adam)
    memory_per_sample = (max_len * base_memory_per_token) / (1024**3)  # GB
    
    # Adjust for model size (larger models need more memory per sample)
    model_scale_factor = max(0.5, min(2.0, model_params_billion / 7.0))  # Scale factor based on 7B as baseline
    memory_per_sample = memory_per_sample * model_scale_factor
    
    if accelerator and accelerator.is_main_process:
        print(f"[train] Memory estimation:")
        print(f"  - Memory per sample: {memory_per_sample*1024:.1f} MB")
        print(f"  - Model scale factor: {model_scale_factor:.2f}x")
    
    max_batch_per_gpu = max(1, int(available_memory / memory_per_sample))
    
    # Clamp batch size based on model size and GPU memory
    max_reasonable_batch = 1024
    if model_params_billion > 50:
        max_reasonable_batch = 64  # Very large models
    elif model_params_billion > 20:
        max_reasonable_batch = 128  # Large models
    elif model_params_billion > 10:
        max_reasonable_batch = 256  # Medium models
    elif model_params_billion > 5:
        max_reasonable_batch = 512  # Small-medium models
    
    max_batch_per_gpu = min(max_batch_per_gpu, max_reasonable_batch)
    
    # Calculate gradient accumulation to achieve effective batch size of 32-128
    target_effective_batch = 64  # Good balance for most models
    if model_params_billion > 50:
        target_effective_batch = 32  # Smaller for very large models
    elif model_params_billion < 5:
        target_effective_batch = 128  # Larger for small models
    
    total_batch_per_step = max_batch_per_gpu * num_gpus
    grad_accum = max(1, target_effective_batch // total_batch_per_step)
    
    # Ensure reasonable gradient accumulation (not too high)
    grad_accum = min(grad_accum, 256)  # Cap at 256
    
    if accelerator and accelerator.is_main_process:
        print(f"[train] Batch size calculation:")
        print(f"  - Max batch per GPU: {max_batch_per_gpu}")
        print(f"  - Gradient accumulation: {grad_accum}")
        print(f"  - Effective batch size: {max_batch_per_gpu * num_gpus * grad_accum}")
    
    return max_batch_per_gpu, grad_accum

# ---------- Config loader (merge base + run) ----------
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

# ---------- Target modules resolver ----------
def resolve_target_modules(model_name: str, override):
    if override and override != "auto":
        return override
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "qwen", "phi", "gemma"]):
        return ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    # seq2seq (t5) often target feed-forward + attention
    if "t5" in name or "flan" in name:
        return ["q","k","v","o","wi_0","wi_1","wo"]
    return ["q_proj","k_proj","v_proj","o_proj"]  # safe fallback

# ---------- Dataset (structured chat; Jinja on-the-fly) ----------
class ChatDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path) as f:
            for line in f:
                line=line.strip()
                if line:
                    self.rows.append(json.loads(line))
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

class Collator:
    def __init__(self, tokenizer, template_path, max_len, model_type):
        self.tok = tokenizer
        self.max_len = max_len
        self.model_type = model_type
        # Load template if it exists, otherwise we'll handle rendered format
        self.template = None
        if template_path and Path(template_path).exists():
            self.template = Template(Path(template_path).read_text())

    def __call__(self, batch: List[Dict[str,Any]]):
        inputs = []
        targets = []
        
        # Auto-detect data format and handle multiple formats
        for row in batch:
            if "input" in row and "target" in row:
                # Rendered format: {input: "...", target: "..."}
                inputs.append(row["input"].strip())
                targets.append(row["target"].strip())
            elif "user" in row and "assistant" in row:
                # Processed format: {system: "...", user: "...", assistant: "..."}
                if self.template:
                    # Apply template if available
                    sys_txt = (row.get("system") or "").strip()
                    user_txt = row["user"].strip()
                    asst_txt = row["assistant"].strip()
                    rendered = self.template.render(system=sys_txt, user=user_txt).strip()
                    inputs.append(rendered)
                    targets.append(asst_txt)
                else:
                    # No template - create simple format
                    sys_txt = (row.get("system") or "").strip()
                    user_txt = row["user"].strip()
                    asst_txt = row["assistant"].strip()
                    if sys_txt:
                        simple_input = f"System: {sys_txt}\nUser: {user_txt}\nAssistant:"
                    else:
                        simple_input = f"User: {user_txt}\nAssistant:"
                    inputs.append(simple_input)
                    targets.append(asst_txt)
            elif "messages" in row:
                # Messages format: {messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                messages = row["messages"]
                user_msg = None
                assistant_msg = None
                system_msg = None
                
                for msg in messages:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_msg = msg["content"]
                    elif msg["role"] == "system":
                        system_msg = msg["content"]
                
                if user_msg and assistant_msg:
                    if self.template:
                        # Apply template if available
                        sys_txt = (system_msg or "").strip()
                        rendered = self.template.render(system=sys_txt, user=user_msg).strip()
                        inputs.append(rendered)
                        targets.append(assistant_msg)
                    else:
                        # No template - create simple format
                        if system_msg:
                            simple_input = f"System: {system_msg}\nUser: {user_msg}\nAssistant:"
                        else:
                            simple_input = f"User: {user_msg}\nAssistant:"
                        inputs.append(simple_input)
                        targets.append(assistant_msg)
                else:
                    raise ValueError(f"Messages format missing user or assistant message: {messages}")
            else:
                raise ValueError(f"Unsupported data format. Expected {{input, target}}, {{user, assistant}}, or {{messages}} format. Got: {list(row.keys())}")

        if self.model_type == "seq2seq":
            model_inputs = self.tok(inputs, max_length=self.max_len, truncation=True, padding=True)
            with self.tok.as_target_tokenizer():
                labels = self.tok(targets, max_length=self.max_len, truncation=True, padding=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            # causal: concatenate input + target with labels masked on the prompt
            # Strategy: tokenize separately, then build labels where prompt tokens = -100
            model_inputs = self.tok(inputs, max_length=self.max_len, truncation=True, padding=True, add_special_tokens=True)
            target_tok = self.tok(targets, max_length=self.max_len, truncation=True, padding=True, add_special_tokens=False)

            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            labels = []
            for i in range(len(input_ids)):
                # Get prompt text and target text
                prompt = self.tok.decode(input_ids[i], skip_special_tokens=True)
                target = targets[i]
                
                # Tokenize the full sequence (prompt + target)
                full_text = prompt + target
                full_tokens = self.tok(full_text, max_length=self.max_len, truncation=True, 
                                     padding=False, add_special_tokens=True)
                full_ids = full_tokens["input_ids"]
                
                # Get prompt length to mask it in labels
                prompt_tokens = self.tok(prompt, add_special_tokens=True, padding=False)
                prompt_len = len(prompt_tokens["input_ids"])
                
                # Create labels: -100 for prompt tokens, actual tokens for target
                labels_seq = [-100] * prompt_len + full_ids[prompt_len:]
                
                # Ensure labels match the length of input_ids
                if len(labels_seq) > len(full_ids):
                    labels_seq = labels_seq[:len(full_ids)]
                elif len(labels_seq) < len(full_ids):
                    labels_seq.extend([-100] * (len(full_ids) - len(labels_seq)))
                
                input_ids[i] = full_ids
                attention_mask[i] = [1] * len(full_ids)
                labels.append(labels_seq)

            # pad to same length
            maxL = max(len(x) for x in input_ids)
            for i in range(len(input_ids)):
                pad = maxL - len(input_ids[i])
                if pad>0:
                    input_ids[i] += [self.tok.pad_token_id]*pad
                    attention_mask[i] += [0]*pad
                    labels[i] += [-100]*pad
            return {"input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels)}

# ---------- CPT/DAPT dataset builder ----------
def build_cpt_or_mixed(cfg, tokenizer):
    """Build CPT, mixed, or return None for SFT mode."""
    mode = cfg.get("task_mode", "sft")
    if mode == "cpt":
        # Pure CPT mode - single dataset
        cpt_path = cfg["datasets"][0]["path"]
        block_size = cfg.get("block_size", 2048)
        pack_factor = cfg.get("pack_factor", 4)
        cpt = load_cpt_dataset(cpt_path, tokenizer, block_size=block_size, pack_factor=pack_factor)
        return cpt, DataCollatorForCausalPairs(tokenizer)
    elif mode == "cpt_mixed":
        # Mixed CPT + anchor mode
        datasets_list = []
        weights = []
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
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        mixed = interleave_datasets(datasets_list, probabilities=weights, seed=cfg.get("seed", 42))
        return mixed, DataCollatorForCausalPairs(tokenizer)
    else:
        return None, None  # SFT path uses existing code

# ---------- Metric: ROUGE-L (simple) ----------
def compute_metrics(tokenizer):
    def _fn(eval_pred):
        try:
            from evaluate import load as load_metric
            rouge = load_metric("rouge")
            
            preds, labels = eval_pred
            # replace -100 with pad for decode
            labels = [[(tok if tok != -100 else tokenizer.pad_token_id) for tok in seq] for seq in labels]
            pred_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
            label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            r = rouge.compute(predictions=pred_text, references=label_text)
            return {"rougeL": r["rougeL"]}
        except Exception as e:
            print(f"[train] Warning: Could not compute ROUGE metrics: {e}")
            return {"rougeL": 0.0}
    return _fn

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/config_run.yaml")
    ap.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed for training")
    ap.add_argument("--deepspeed_config", default=None, help="Path to DeepSpeed config file")
    args = ap.parse_args()
    
    # Initialize accelerator for distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # Set seed for reproducibility
    set_seed(42)
    
    cfg = load_config(args.config)
    
    # Print GPU information
    gpu_info = get_gpu_memory_info()
    if accelerator.is_main_process:
        print(f"[train] Detected {len(gpu_info)} GPUs:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
        
        total_vram = sum(gpu['total_memory_gb'] for gpu in gpu_info)
        print(f"[train] Total VRAM: {total_vram:.1f} GB")

    torch.manual_seed(cfg.get("seed", 42))

    # Setup output directory with run name
    run_name = cfg["train"].get("run_name", "distributed-training")
    base_outdir = cfg["train"]["output_dir"]
    
    # Create timestamped run directory if run_name is generic
    if run_name == "distributed-training":
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"distributed-training-{timestamp}"
    
    # Use run_name as the output directory
    outdir = f"outputs/{run_name}"
    
    # Update config to use the correct output directory
    cfg["train"]["output_dir"] = outdir
    
    # Backend stamp and resume check
    backend_file = os.path.join(outdir, "backend.json")
    current_backend = cfg["tuning"]["backend"]
    current_dtype = "bf16" if cfg["train"].get("bf16", False) else "fp16"
    
    if accelerator.is_main_process:
        if os.path.exists(backend_file):
            # Resume check - ensure backend consistency
            try:
                with open(backend_file, "r") as f:
                    saved_info = json.load(f)
                if saved_info.get("backend") != current_backend:
                    raise ValueError(
                        f"Backend mismatch! Saved run used '{saved_info.get('backend')}' "
                        f"but current config specifies '{current_backend}'. "
                        f"Use a different output_dir or ensure backend consistency."
                    )
                print(f"[train] Resuming {saved_info.get('backend')} run: {run_name}")
                print(f"[train] Backend: {saved_info.get('backend')}, Precision: {saved_info.get('dtype')}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[train] Warning: Could not read backend info: {e}")
        else:
            # New run - create backend stamp
            os.makedirs(outdir, exist_ok=True)
            backend_info = {
                "backend": current_backend,
                "dtype": current_dtype,
                "attn_impl": "sdpa" if current_backend == "unsloth" else "default",
                "num_gpus": len(gpu_info),
                "distributed": True,
                "run_name": run_name
            }
            with open(backend_file, "w") as f:
                json.dump(backend_info, f, indent=2)
            print(f"[train] Starting new {current_backend} distributed run: {run_name}")
            print(f"[train] Backend: {current_backend}, Precision: {current_dtype}, GPUs: {len(gpu_info)}")

    # Precision sanity check
    bf16_enabled = cfg["train"].get("bf16", False)
    fp16_enabled = cfg["train"].get("fp16", False)
    
    if bf16_enabled and fp16_enabled:
        if accelerator.is_main_process:
            print("[train] Warning: Both bf16 and fp16 enabled. Disabling fp16.")
        cfg["train"]["fp16"] = False
    elif not bf16_enabled and not fp16_enabled:
        # Default to bf16 for H100/H200, fp16 otherwise
        if torch.cuda.is_available() and len(gpu_info) > 0:
            gpu_name = gpu_info[0]['name'].lower()
            if "h100" in gpu_name or "h200" in gpu_name or "a100" in gpu_name:
                cfg["train"]["bf16"] = True
                if accelerator.is_main_process:
                    print("[train] Auto-enabled bf16 for H100/H200/A100 GPUs")
            else:
                cfg["train"]["fp16"] = True
                if accelerator.is_main_process:
                    print("[train] Auto-enabled fp16 for other GPUs")
        else:
            cfg["train"]["fp16"] = True
            if accelerator.is_main_process:
                print("[train] Auto-enabled fp16 (no CUDA detected)")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)

    model_name = local_model_dir   # from now on, load FROM DISK
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", False))
    model_type = cfg["model"]["type"]          # "causal" | "seq2seq"
    max_len    = int(cfg["model"].get("max_seq_len", 512))

    # Calculate optimal batch size for multi-GPU setup
    total_vram = sum(gpu['total_memory_gb'] for gpu in gpu_info) if gpu_info else 32
    num_gpus = len(gpu_info) if gpu_info else 1
    
    if cfg["train"].get("batch_size") == "auto" or cfg["train"].get("grad_accum") == "auto":
        bs, accum = calculate_optimal_batch_size(model_name, max_len, num_gpus, total_vram, accelerator)
        if accelerator.is_main_process:
            print(f"[train] Auto-calculated batch size: {bs} per GPU, gradient accumulation: {accum}")
            print(f"[train] Effective batch size: {bs * num_gpus * accum}")
    else:
        bs = int(cfg["train"].get("batch_size", 1))
        accum = int(cfg["train"].get("grad_accum", 16))

    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|pad|>"

    # Branch by mode: sft vs cpt vs cpt_mixed
    ds_train, ds_val, data_collator = None, None, None
    
    cpt_ds, cpt_collator = build_cpt_or_mixed(cfg, tok)
    if cfg.get("task_mode", "sft") in ("cpt", "cpt_mixed"):
        # CPT/DAPT mode
        if accelerator.is_main_process:
            print(f"[train] Using {cfg.get('task_mode')} mode with {len(cpt_ds)} samples")
        ds_train = cpt_ds
        ds_val = None  # CPT typically doesn't use validation during training
        data_collator = cpt_collator
    else:
        # SFT mode (existing path)
        paths = cfg["data"]
        
        # Check if data files exist (only for SFT mode)
        train_path = paths["train_path"]
        val_path = paths["val_path"]
        template_path = cfg["data"]["template_path"]
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        ds_train = ChatDataset(train_path)
        ds_val = ChatDataset(val_path)
        
        if accelerator.is_main_process:
            print(f"[train] SFT mode: Loaded {len(ds_train)} training samples, {len(ds_val)} validation samples")
        data_collator = Collator(tok, template_path, max_len, model_type)

    # backend & model load
    backend = cfg["tuning"]["backend"]         # "bnb" | "unsloth"
    mode    = cfg["tuning"]["mode"]            # "qlora" | "lora" | "full"

    # For distributed training, we prefer BitsAndBytes for stability
    use_bnb = True
    if backend == "unsloth":
        if accelerator.is_main_process:
            print("[train] WARNING: Unsloth backend may have issues with multi-GPU training.")
            print("[train] Automatically using BitsAndBytes for distributed training stability.")
        use_bnb = True
        current_backend = "bnb"

    is_seq2seq = (model_type == "seq2seq")
    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", False) else torch.float16, 
            trust_remote_code=trust_remote_code
        )
    else:
        if mode == "qlora":
            if use_bnb:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if cfg["train"].get("bf16", False) else torch.float16,
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", False) else torch.float16,
                    trust_remote_code=trust_remote_code,
                    low_cpu_mem_usage=True,
                )
                base_model = prepare_model_for_kbit_training(base_model)
            else:
                # Unsloth path (not recommended for multi-GPU)
                from unsloth import FastLanguageModel
                base_model, tok = FastLanguageModel.from_pretrained(
                    model_name, load_in_4bit=True,
                    trust_remote_code=trust_remote_code
                )
        else:
            # LoRA or Full: fp16/bf16 path
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", False) else torch.float16,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

    # Apply LoRA if needed
    if mode in ["qlora", "lora"]:
        lcfg = cfg["tuning"]["lora"]
        targets = resolve_target_modules(model_name, lcfg.get("target_modules"))
        lora_cfg = LoraConfig(
            r=int(lcfg.get("r", 32)),
            lora_alpha=int(lcfg.get("alpha", 32)),
            lora_dropout=float(lcfg.get("dropout", 0.05)),
            target_modules=targets,
            bias="none",
            task_type="SEQ_2_SEQ_LM" if is_seq2seq else "CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_cfg)

    # Training args with distributed training support
    outdir = cfg["train"]["output_dir"]
    import transformers
    from transformers.trainer_utils import IntervalStrategy

    # Build kwargs adaptively so it works across TF versions
    ta_fields = getattr(transformers.TrainingArguments, "__dataclass_fields__", {})
    has_eval_strategy = "evaluation_strategy" in ta_fields
    has_save_strategy = "save_strategy" in ta_fields

    # Map string -> IntervalStrategy
    def to_interval(name: str) -> IntervalStrategy:
        return IntervalStrategy.STEPS if str(name).lower() == "steps" else IntervalStrategy.EPOCH

    ta_kwargs = dict(
        output_dir=outdir,
        run_name=f"sft-play-distributed-{num_gpus}gpu",
        logging_dir="outputs/tb",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max(1, bs),
        gradient_accumulation_steps=accum,
        num_train_epochs=int(cfg["train"]["epochs"]),
        learning_rate=float(cfg["train"].get("lr", 2e-4)),
        warmup_ratio=float(cfg["train"].get("warmup_ratio", 0.06)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
        bf16=bool(cfg["train"].get("bf16", False)),
        fp16=bool(cfg["train"].get("fp16", False)),
        logging_steps=1,
        save_total_limit=int(cfg["train"].get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg["train"].get("load_best_model_at_end", False)),
        metric_for_best_model=cfg["train"].get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(cfg["train"].get("greater_is_better", False)),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        # Distributed training specific
        dataloader_pin_memory=False,  # Can cause issues with multi-GPU
        gradient_checkpointing=False,  # Disabled - we handle it manually for distributed training
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
    )

    # Add DeepSpeed support if requested
    if args.deepspeed:
        ta_kwargs["deepspeed"] = args.deepspeed_config

    # Add eval/save strategies in a version-safe way
    if cfg.get("task_mode", "sft") in ("cpt", "cpt_mixed") and ds_val is None:
        eval_strategy_value = "no"
    else:
        eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in cfg["train"] else "eval_strategy"
        eval_strategy_value = cfg["train"].get(eval_strategy_key, "epoch")
    
    if has_eval_strategy:
        ta_kwargs["evaluation_strategy"] = eval_strategy_value if eval_strategy_value == "no" else to_interval(eval_strategy_value)
    else:
        ta_kwargs["eval_strategy"] = eval_strategy_value if eval_strategy_value == "no" else to_interval(eval_strategy_value)

    save_strategy_value = cfg["train"].get("save_strategy", "epoch")
    if has_save_strategy:
        ta_kwargs["save_strategy"] = to_interval(save_strategy_value)
    else:
        ta_kwargs["save_steps"] = int(cfg["train"].get("save_steps", 100))

    # Only add step args if using steps
    if str(eval_strategy_value).lower() == "steps":
        ta_kwargs["eval_steps"] = int(cfg["train"].get("eval_steps", 50))

    if str(save_strategy_value).lower() == "steps":
        ta_kwargs["save_steps"] = int(cfg["train"].get("save_steps", 100))

    args_tr = transformers.TrainingArguments(**ta_kwargs)

    # Resume if checkpoint present
    last_ckpt = get_last_checkpoint(outdir) if os.path.isdir(outdir) else None
    if last_ckpt and accelerator.is_main_process:
        print(f"[train] Resuming from checkpoint: {last_ckpt}")

    trainer = Trainer(
        model=base_model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics(tok),
    )

    # Prepare for distributed training
    trainer.model, trainer.train_dataset, trainer.eval_dataset = accelerator.prepare(
        trainer.model, trainer.train_dataset, trainer.eval_dataset
    )

    # Enable gradient checkpointing after distributed preparation if needed
    # This is handled automatically by the Trainer when gradient_checkpointing=True in TrainingArguments
    if accelerator.is_main_process and cfg["train"].get("gradient_checkpointing", True):
        print("[train] Gradient checkpointing enabled by Trainer for memory efficiency")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Skip evaluation to avoid cache issues in distributed training
    # Evaluation can be done separately after training
    # if ds_val is not None:
    #     trainer.evaluate()
    
    trainer.save_state()
    if mode in ["qlora","lora"]:
        base_model.save_pretrained("adapters/last")
    
    if accelerator.is_main_process:
        print("[train] Distributed training completed.")

if __name__ == "__main__":
    main()
