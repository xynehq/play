import os
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_ATTENTION", "1")
os.environ.setdefault("UNSLOTH_FORCE_SDPA", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")

import sys, json, argparse, warnings, signal, math
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
      - builds prompt + target token ids (uses chat template if available)
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

        self.use_template = hasattr(self.tok, "apply_chat_template")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attn_mask = [], [], []
        eos_id = self.tok.eos_token_id

        for row in batch:
            sys_txt, user_txt, asst_txt = _extract_pair(row)
            if self.use_template:
                msgs = []
                if sys_txt:
                    msgs.append({"role": "system", "content": sys_txt})
                msgs.append({"role": "user", "content": user_txt})
                prompt = self.tok.apply_chat_template(
                    msgs,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                prompt = f"System: {sys_txt}\nUser: {user_txt}\nAssistant:" if sys_txt else f"User: {user_txt}\nAssistant:"

            # tokenize with truncation
            enc_prompt = self.tok(prompt, add_special_tokens=True, truncation=True, max_length=self.max_len)
            enc_target = self.tok(asst_txt, add_special_tokens=False, truncation=True, max_length=self.max_len)

            ids = enc_prompt.input_ids + enc_target.input_ids
            if eos_id is not None:
                ids = (ids + [eos_id])[:self.max_len]
            else:
                ids = ids[:self.max_len]

            # labels: mask the prompt, learn the answer (+ optional eos)
            lbl = [-100] * len(enc_prompt.input_ids) + enc_target.input_ids
            if len(lbl) < len(ids):  # account for eos or truncation boundary
                lbl += [ids[len(lbl)]]
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
# Load once to avoid repeated downloads
try:
    from evaluate import load as load_metric
    ROUGE_METRIC = load_metric("rouge")
except Exception:
    ROUGE_METRIC = None

def compute_metrics_builder(tokenizer):
    """
    Returns a function computing ROUGE-L on generated text.
    Requires predict_with_generate=True.
    """
    def _compute(eval_pred):
        if ROUGE_METRIC is None:
            return {}
        preds = eval_pred.predictions
        labels = eval_pred.label_ids

        # HF ensures token ids when predict_with_generate=True
        # Replace -100 with pad in labels so we can decode
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        
        # Handle numpy arrays safely
        import numpy as np
        
        # Convert to list of lists if it's a numpy array, then process each sequence
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        
        # Replace -100 with pad_id in each sequence individually
        import numpy as np
        processed_labels = []
        for seq in labels:
            # Convert sequence to numpy array if it's not already
            seq_array = np.array(seq)
            # Replace -100 with pad_id using numpy
            seq_array = np.where(seq_array == -100, pad_id, seq_array)
            # Convert back to list
            processed_seq = seq_array.tolist()
            processed_labels.append(processed_seq)
        labels = processed_labels

        # Some versions return numpy arrays; decode wants lists
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        # Ensure preds is a list of token sequences (not nested)
        # Handle various prediction formats from different Transformers versions
        if preds and isinstance(preds[0], list) and len(preds[0]) > 0 and isinstance(preds[0][0], (int, np.integer)):
            # preds is already in the right format: [[token1, token2, ...], ...]
            pass
        elif preds and isinstance(preds[0], list) and len(preds[0]) > 0 and isinstance(preds[0][0], list):
            # preds is nested: [[[token1, token2, ...]], ...] - flatten one level
            preds = [seq[0] if len(seq) > 0 else [] for seq in preds]
        elif preds and isinstance(preds[0], (int, np.integer)):
            # preds is flat: [token1, token2, ...] - wrap in list
            preds = [preds]
        elif preds and isinstance(preds[0], np.ndarray):
            # preds contains numpy arrays - convert to lists
            preds = [seq.tolist() for seq in preds]
        
        # Final safety check: ensure all elements are integers
        processed_preds = []
        for seq in preds:
            if isinstance(seq, list):
                processed_seq = []
                for token in seq:
                    if isinstance(token, (int, np.integer)):
                        processed_seq.append(int(token))
                    elif hasattr(token, 'item'):  # numpy scalar
                        processed_seq.append(int(token.item()))
                    else:
                        # Skip non-integer tokens or convert if possible
                        try:
                            processed_seq.append(int(token))
                        except (ValueError, TypeError):
                            continue
                processed_preds.append(processed_seq)
            else:
                # Skip non-list sequences
                continue
        preds = processed_preds

        # Only proceed if we have valid predictions
        if not preds:
            return {"rougeL": 0.0}

        # Filter out empty predictions
        valid_preds = []
        valid_labels = []
        for i, pred_seq in enumerate(preds):
            if pred_seq and len(pred_seq) > 0:  # Only non-empty sequences
                valid_preds.append(pred_seq)
                if i < len(labels):
                    valid_labels.append(labels[i])

        if not valid_preds:
            return {"rougeL": 0.0}

        try:
            pred_text = tokenizer.batch_decode(valid_preds, skip_special_tokens=True)
        except Exception as e:
            print(f"[eval] Error decoding predictions: {e}")
            print(f"[eval] Prediction structure: {type(valid_preds)} with {len(valid_preds)} sequences")
            if valid_preds:
                print(f"[eval] First sequence type: {type(valid_preds[0])}, length: {len(valid_preds[0])}")
                if valid_preds[0]:
                    print(f"[eval] First token type: {type(valid_preds[0][0])}, value: {valid_preds[0][0]}")
            return {"rougeL": 0.0}
        
        try:
            label_text = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
        except Exception as e:
            print(f"[eval] Error decoding labels: {e}")
            print(f"[eval] Label structure: {type(valid_labels)} with {len(valid_labels)} sequences")
            if valid_labels:
                print(f"[eval] First label sequence type: {type(valid_labels[0])}, length: {len(valid_labels[0])}")
                if valid_labels[0]:
                    print(f"[eval] First label token type: {type(valid_labels[0][0])}, value: {valid_labels[0][0]}")
            return {"rougeL": 0.0}
        
        # Filter out empty predictions/labels
        valid_pairs = []
        for p, l in zip(pred_text, label_text):
            if p.strip() and l.strip():  # Only include non-empty pairs
                valid_pairs.append((p, l))
        
        if not valid_pairs:
            return {"rougeL": 0.0}
        
        pred_text_clean, label_text_clean = zip(*valid_pairs)
        
        try:
            r = ROUGE_METRIC.compute(predictions=list(pred_text_clean), references=list(label_text_clean), use_stemmer=True)
            # ROUGE-L might be a score obj; normalize to float
            rl = r["rougeL"].mid.fmeasure if hasattr(r["rougeL"], "mid") else r.get("rougeL", 0.0)
            return {"rougeL": float(rl)}
        except Exception as e:
            print(f"[eval] ROUGE computation error: {e}")
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


# ----------------- Autoscaling -----------------
def autoscale(cfg, model, seq_len) -> Tuple[int, int]:
    """
    VRAM-aware derivation of per-device batch and grad_accum to meet a target tokens step.
    Very rough heuristic; tune autoscale_alpha and target tokens per device step.
    """
    world = max(torch.cuda.device_count(), 1)
    tgt_tokens = int(cfg["train"].get("target_tokens_per_device_step", 4096 * 4))  # e.g., 16K tokens/device/step
    # embed width (fallback 4096)
    hidden = getattr(getattr(model, "config", None), "hidden_size", 4096)
    # crude memory model coefficient for bf16/4bit compute
    alpha = float(cfg["train"].get("autoscale_alpha", 2e-6))

    # available memory per device
    max_mem = []
    for d in range(world):
        total, free = torch.cuda.mem_get_info(d)
        max_mem.append(int(free * 0.85))  # 85% headroom
    mem_cap = min(max_mem) if max_mem else 8 * 1024**3  # default 8GB if no CUDA

    per_device_tokens = max(int(seq_len), 1)
    init_bsz = max(1, tgt_tokens // per_device_tokens)
    bsz = init_bsz

    def mem_need(b):
        # memory ≈ alpha * batch * seq * hidden (bytes)
        return int(alpha * b * per_device_tokens * hidden)

    while bsz > 1 and mem_need(bsz) > mem_cap:
        bsz //= 2

    # derive grad_accum to hit target tokens
    total_tokens = bsz * per_device_tokens
    grad_accum = max(1, (tgt_tokens + total_tokens - 1) // total_tokens)

    return bsz, grad_accum


# ----------------- Eval speed helper -----------------
from transformers import TrainerCallback

class EvalSpeedCallback(TrainerCallback):
    """
    Toggle use_cache True during eval/predict to speed generation
    (we disable it during training for checkpointing).
    """
    def __init__(self, model):
        self.model = model
        self.prev = None

    def on_evaluate(self, args, state, control, **kwargs):
        self.prev = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = True

    def on_predict(self, args, state, control, **kwargs):
        self.prev = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = True

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.prev is not None:
            self.model.config.use_cache = self.prev

    def on_train_end(self, args, state, control, **kwargs):
        if self.prev is not None:
            self.model.config.use_cache = self.prev


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

    # SIGINT/SIGTERM → save state cleanly, then barrier
    interrupted = {"flag": False}
    def _graceful(sig, frame):
        interrupted["flag"] = True
        print(f"[signal] Caught {sig}. Will request Trainer to stop after current step.")
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Model card → local dir (your helper can download/cache)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
    local_model_dir = prepare_local_model_dir(cfg["model"], hf_token=hf_token)
    model_name = local_model_dir
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    model_type = cfg["model"].get("type", "causal")
    max_len = int(cfg["model"].get("max_seq_len", 2048))

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    added = False
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            added = True
    tok.padding_side = "right"

    # Data
    ds_train, ds_val, data_collator = None, None, None
    if cfg.get("task_mode", "sft") in ("cpt", "cpt_mixed"):
        ds_train, ds_val, data_collator = build_cpt_or_mixed(cfg, tok)
    else:
        paths = cfg["data"]
        train_path = paths["train_path"]; val_path = paths["val_path"]
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
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if cfg["train"].get("bf16", True) else torch.float16,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

    # Resize embeddings if we added a new pad token
    if added:
        model.resize_token_embeddings(len(tok))

    # Optional attention impl & TP tweaks
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    # Enable TF32 for speed (Ampere+)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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

    # Disable cache when using gradient checkpointing during training
    if bool(cfg["train"].get("gradient_checkpointing", True)) and hasattr(model, "config"):
        model.config.use_cache = False

    # Output dir & run name
    from datetime import datetime
    run_name = str(cfg["train"].get("run_name", f"distributed-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    outdir = f"outputs/{run_name}"

    # Backend stamp (nice for resuming sanity)
    if accelerator.is_main_process:
        os.makedirs(outdir, exist_ok=True)
        stamp = {
            "backend": backend,
            "dtype": "bf16" if cfg["train"].get("bf16", True) else "fp16",
            "num_gpus": torch.cuda.device_count(),
            "distributed": True,
            "run_name": run_name
        }
        with open(os.path.join(outdir, "backend.json"), "w") as f:
            json.dump(stamp, f, indent=2)
        # also persist the merged config
        with open(os.path.join(outdir, "run_config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)

    # ----------------- Autoscaling -----------------
    bs_cfg = cfg["train"].get("batch_size", 1)
    ga_cfg = cfg["train"].get("grad_accum", 8)
    if bs_cfg == "auto" or ga_cfg == "auto":
        per_device_train_batch_size, gradient_accumulation_steps = autoscale(cfg, model, max_len)
    else:
        per_device_train_batch_size = int(bs_cfg)
        gradient_accumulation_steps = int(ga_cfg)

    # Eval batch
    per_device_eval_batch_size = cfg["train"].get("per_device_eval_batch_size", "auto")
    if per_device_eval_batch_size == "auto":
        per_device_eval_batch_size = max(1, per_device_train_batch_size)
    else:
        per_device_eval_batch_size = int(per_device_eval_batch_size)

    # ----------------- TrainingArguments -----------------
    # Fast eval defaults
    gen_new_tokens = int(cfg["train"].get("generation_max_new_tokens", 64))
    num_beams = int(cfg["train"].get("generation_num_beams", 1))

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

        "logging_dir": f"outputs/tb/{run_name}",
        "logging_steps": int(cfg["train"].get("logging_steps", 1)),
        "save_total_limit": int(cfg["train"].get("save_total_limit", 5)),
        "remove_unused_columns": False,

        # DDP-related stability & speed
        "dataloader_pin_memory": True,
        "dataloader_num_workers": int(cfg["train"].get("dataloader_num_workers", 4)),
        "gradient_checkpointing": bool(cfg["train"].get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "ddp_bucket_cap_mb": int(cfg["train"].get("ddp_bucket_cap_mb", 200)),

        # Length bucketing to cut padding waste (disabled for custom datasets)
        "group_by_length": False,

        # Eval/save strategy - handle both old and new parameter names for compatibility
        "eval_strategy": str(cfg["train"].get("evaluation_strategy", cfg["train"].get("eval_strategy", "no"))),
        "eval_steps": int(cfg["train"].get("eval_steps", 200)),
        "save_strategy": str(cfg["train"].get("save_strategy", "epoch")),
        "save_steps": int(cfg["train"].get("save_steps", 200)),
        "load_best_model_at_end": bool(cfg["train"].get("load_best_model_at_end", False)),
        "metric_for_best_model": str(cfg["train"].get("metric_for_best_model", "eval_loss")) if cfg["train"].get("load_best_model_at_end", False) else None,
        "greater_is_better": bool(cfg["train"].get("greater_is_better", False)),

        "eval_accumulation_steps": int(cfg["train"].get("eval_accumulation_steps", 8)),
        "eval_do_concat_batches": False,  # lower peak RAM during eval

        # Generation-based eval - conditional for compatibility
        "predict_with_generate": bool(cfg["train"].get("predict_with_generate", True)),
        "generation_num_beams": num_beams,
        # prefer max_new_tokens to keep latency predictable (use smaller for step evals)
        "generation_max_new_tokens": gen_new_tokens,
        "eval_do_concat_batches": False,

        "report_to": ["tensorboard"],
        "save_safetensors": True,
    }

    # Build TrainingArguments with compatibility checks
    GEN_SUPPORTED = True
    try:
        tr_args = TrainingArguments(**training_args_kwargs)
    except TypeError as e:
        # Handle compatibility issues with older Transformers versions
        if "predict_with_generate" in str(e):
            # Remove generation-related parameters for older versions
            training_args_kwargs.pop("predict_with_generate", None)
            training_args_kwargs.pop("generation_num_beams", None)
            training_args_kwargs.pop("generation_max_new_tokens", None)
            GEN_SUPPORTED = False
            training_args_kwargs["prediction_loss_only"] = True
            tr_args = TrainingArguments(**training_args_kwargs)
            if accelerator.is_main_process:
                print("[train] Generation params not supported. Falling back to loss-only eval.")
        else:
            raise e

    # Optional DeepSpeed
    ds_cfg = args.deepspeed_config or cfg["train"].get("deepspeed")
    if args.deepspeed and not ds_cfg:
        raise ValueError("Use --deepspeed_config or set train.deepspeed in YAML.")
    if ds_cfg:
        tr_args.deepspeed = ds_cfg

    # Resume (if any)
    last_ckpt = get_last_checkpoint(outdir) if os.path.isdir(outdir) else None
    if accelerator.is_main_process and last_ckpt:
        print(f"[train] Resuming from checkpoint: {last_ckpt}")

    # Build Trainer
    callbacks = [EvalSpeedCallback(model)] if GEN_SUPPORTED else None
    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_val if tr_args.eval_strategy != "no" else None,
        data_collator=data_collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_builder(tok) if (tr_args.eval_strategy != "no" and GEN_SUPPORTED) else None,
        callbacks=callbacks,
    )

    # If we caught a signal, ask Trainer to stop cleanly after current step
    if interrupted["flag"]:
        trainer.control.should_training_stop = True

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Optional final evaluation even if strategy="no"
    if tr_args.eval_strategy == "no" and ds_val is not None and bool(cfg["train"].get("final_eval", False)):
        metrics = trainer.evaluate()
        # Post-eval sync to avoid long-tail stalls
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        if accelerator.is_main_process:
            print("[eval-final]", metrics)

    # Save adapters/checkpoints (rank-0 only)
    if mode in ["qlora", "lora"] and accelerator.is_main_process:
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
        # useful summary
        world = torch.cuda.device_count()
        global_bsz = per_device_train_batch_size * max(1, world) * gradient_accumulation_steps
        print(json.dumps({
            "status": "completed",
            "run_name": run_name,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "global_batch_size": global_bsz,
            "eval_gen_new_tokens": gen_new_tokens,
            "eval_num_beams": num_beams
        }, indent=2))
        print("[train] Distributed training completed.")

if __name__ == "__main__":
    main()
