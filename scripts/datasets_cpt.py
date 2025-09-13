# scripts/datasets_cpt.py
from typing import List, Tuple, Dict, Any
from datasets import load_dataset, Dataset

# ----------------------------
# helpers
# ----------------------------
def _windowize(ids: List[int], block_size: int, stride: int) -> List[List[int]]:
    """Slice a long token stream into overlapping windows."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    step = block_size if (stride is None or stride < 0 or stride >= block_size) else (block_size - stride)
    n = len(ids)
    if n < block_size:
        return []
    return [ids[i:i + block_size] for i in range(0, n - block_size + 1, step)]

def _windowize_with_offset(ids: List[int], block_size: int, stride: int, offset: int) -> List[List[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    step = block_size if (stride is None or stride < 0 or stride >= block_size) else (block_size - stride)
    n = len(ids)
    if n < block_size + offset:
        return []
    out = []
    for start in range(offset, n - block_size + 1, step):
        out.append(ids[start:start + block_size])
    return out

def _multi_offsets(ids: List[int], block_size: int, stride: int) -> List[List[int]]:
    """Three staggered passes: 0, 1/4, 1/2 of block_size."""
    o1 = 0
    o2 = max(1, block_size // 4)
    o3 = max(1, block_size // 2)
    return (
        _windowize_with_offset(ids, block_size, stride, o1)
        + _windowize_with_offset(ids, block_size, stride, o2)
        + _windowize_with_offset(ids, block_size, stride, o3)
    )

def _sep_ids(tokenizer) -> List[int]:
    """Separator tokens for global packing. Prefer EOS; else two newlines."""
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return [tokenizer.eos_token_id, tokenizer.eos_token_id]
    return tokenizer("\n\n", add_special_tokens=False)["input_ids"]

# ----------------------------
# CPT (unsupervised) dataset
# ----------------------------
def load_cpt_dataset(
    jsonl_path: str,
    tokenizer,
    block_size: int = 512,
    pack_factor: int = 4,   # kept for BC (unused with global packing)
    add_eos: bool = True,
    stride: int = 256,
) -> Dataset:
    """
    Load raw CPT JSONL and produce *globally packed* overlapping windows
    with staggered offsets. Accepts {"text": "..."} or {"instruction","input","output"}.
    """
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    def tok_batch(batch):
        if "text" in batch:
            texts = batch["text"]
        elif "instruction" in batch:
            texts = []
            N = len(batch["instruction"])
            for i in range(N):
                instr = batch["instruction"][i] or ""
                inpt  = batch.get("input",  [""] * N)[i] or ""
                out   = batch.get("output", [""] * N)[i] or ""
                text = f"{instr}\n\nInput: {inpt}\n\nResponse: {out}" if inpt.strip() else f"{instr}\n\nResponse: {out}"
                texts.append(text)
        else:
            raise ValueError(f"Unsupported fields: {list(batch.keys())}")

        if add_eos and tokenizer.eos_token:
            texts = [t + tokenizer.eos_token for t in texts]

        toks = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,              # keep full text, we pack later
            return_attention_mask=False,
        )
        return {"input_ids": toks["input_ids"]}

    tokenized = ds.map(tok_batch, batched=True, remove_columns=ds.column_names)

    # ---- GLOBAL PACK ACROSS ALL EXAMPLES ----
    sep = _sep_ids(tokenizer)
    all_ids: List[int] = []
    for ids in tokenized["input_ids"]:
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.extend(sep)  # boundary between docs

    # Multi-offset windowing (more samples from tiny corpora)
    x_windows = _multi_offsets(all_ids, block_size, stride)

    if not x_windows:
        # fallback: per-example windowing (extremely small corpora)
        for ids in tokenized["input_ids"]:
            x_windows.extend(_multi_offsets(ids, block_size, stride))

    y_windows = [w[:] for w in x_windows]  # LM target = input
    return Dataset.from_dict({"input_ids": x_windows, "labels": y_windows})

# ----------------------------
# Chat (anchors) formatted for CPT-mix
# ----------------------------
def load_chat_dataset_for_cpt(
    jsonl_path: str,
    tokenizer,
    block_size: int = 512,
    stride: int = 256,
) -> Dataset:
    """
    Loads {"instruction","input","output"} anchors.
    Formats with chat template, masks user tokens (-100), then *globally packs*
    with staggered offsets for BOTH inputs and labels.
    """
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    def format_one(instr: str, inpt: str, output: str) -> Tuple[List[int], List[int]]:
        instr  = (instr or "").strip()
        inpt   = (inpt  or "").strip()
        output = (output or "").strip()

        user = f"{instr}\n{inpt}".strip() if inpt else instr
        conv = [{"role": "user", "content": user},
                {"role": "assistant", "content": output}]

        # format with chat template when available
        if getattr(tokenizer, "chat_template", None):
            try:
                text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"<|user|>\n{user}<|end|>\n<|assistant|>\n{output}<|end|>\n"
        else:
            text = f"<|user|>\n{user}<|end|>\n<|assistant|>\n{output}<|end|>\n"

        user_part = text.split("<|assistant|>")[0] + "<|assistant|>\n"
        full_ids  = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        user_len  = len(tokenizer(user_part, add_special_tokens=True)["input_ids"])

        labels = [-100] * user_len + full_ids[user_len:]
        # pad/trim to equal length
        if len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]
        elif len(labels) < len(full_ids):
            labels += full_ids[len(labels):]
        return full_ids, labels

    # tokenize all, then global pack with masked separators
    sep = _sep_ids(tokenizer)
    all_x: List[int] = []
    all_y: List[int] = []

    for ex in ds:
        x, y = format_one(ex.get("instruction", ""), ex.get("input", ""), ex.get("output", ""))
        if not x:
            continue
        all_x.extend(x)
        all_y.extend(y)
        # add neutral separator (mask labels for separator)
        all_x.extend(sep)
        all_y.extend([-100] * len(sep))

    # Multi-offset windowing for BOTH x and y (keep alignment)
    x_windows = _multi_offsets(all_x, block_size, stride)
    y_windows = _multi_offsets(all_y, block_size, stride)

    n = min(len(x_windows), len(y_windows))
    if n == 0:
        # fallback: per-example windowing with offsets
        x_windows, y_windows = [], []
        for ex in ds:
            x, y = format_one(ex.get("instruction",""), ex.get("input",""), ex.get("output",""))
            xs = _multi_offsets(x, block_size, stride)
            ys = _multi_offsets(y, block_size, stride)
            for xi, yi in zip(xs, ys):
                x_windows.append(xi); y_windows.append(yi)
        n = len(x_windows)

    return Dataset.from_dict({"input_ids": x_windows[:n], "labels": y_windows[:n]})

# ----------------------------
# Convenience: read params from config dict
# ----------------------------
def load_cpt_dataset_from_cfg(cfg: Dict[str, Any], tokenizer) -> Dataset:
    bs = int(cfg.get("block_size", 512))
    st = int(cfg.get("stride", 256))
    path = cfg.get("path") or cfg.get("jsonl_path")
    if not path:
        raise ValueError("CPT cfg must include 'path' to a JSONL file.")
    return load_cpt_dataset(jsonl_path=path, tokenizer=tokenizer, block_size=bs, stride=st)

def load_chat_dataset_for_cpt_from_cfg(cfg: Dict[str, Any], tokenizer) -> Dataset:
    bs = int(cfg.get("block_size", 512))
    st = int(cfg.get("stride", 256))
    path = cfg.get("path") or cfg.get("jsonl_path")
    if not path:
        raise ValueError("Chat cfg must include 'path' to a JSONL file.")
    return load_chat_dataset_for_cpt(jsonl_path=path, tokenizer=tokenizer, block_size=bs, stride=st)
