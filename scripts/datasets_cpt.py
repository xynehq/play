# scripts/datasets_cpt.py
from datasets import load_dataset

def load_cpt_dataset(jsonl_path, tokenizer, block_size=2048, pack_factor=4, add_eos=True):
    """Load CPT dataset from JSONL, tokenize, and pack to fixed block_size."""
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    max_len = block_size * pack_factor

    def tok_fn(batch):
        texts = batch["text"]
        if add_eos and tokenizer.eos_token:
            texts = [t + tokenizer.eos_token for t in texts]
        out = tokenizer(texts, truncation=True, max_length=max_len, add_special_tokens=False)
        return out

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = tokenized.map(group_texts, batched=True)
    return tokenized

def load_chat_dataset_for_cpt(jsonl_path, tokenizer, block_size=2048):
    """Load chat dataset and prepare for CPT mixing (with proper label masking)."""
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    
    def build_anchor_prompt(rec):
        instr = rec.get("instruction", "").strip()
        inpt = rec.get("input", "").strip()
        if inpt:
            return f"[INST] {instr}\n{inpt} [/INST]\n"
        return f"[INST] {instr} [/INST]\n"

    def tok_map(batch):
        prompts = [build_anchor_prompt(r) for r in batch]
        outs = [r.get("output", "") for r in batch]
        
        prompt_tok = tokenizer(prompts, add_special_tokens=False)
        out_tok = tokenizer(outs, add_special_tokens=False)
        
        input_ids, labels = [], []
        for p_ids, o_ids in zip(prompt_tok["input_ids"], out_tok["input_ids"]):
            ids = p_ids + o_ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
            lab = [-100]*len(p_ids) + o_ids + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
            
            # Truncate if too long
            if len(ids) > block_size:
                ids = ids[:block_size]
                lab = lab[:block_size]
            
            input_ids.append(ids)
            labels.append(lab)
        
        return {"input_ids": input_ids, "labels": labels}

    tokenized = ds.map(tok_map, batched=True, remove_columns=ds.column_names)
    
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        return result

    tokenized = tokenized.map(group_texts, batched=True)
    return tokenized
