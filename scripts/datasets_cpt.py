# scripts/datasets_cpt.py
from datasets import load_dataset

def load_cpt_dataset(jsonl_path, tokenizer, block_size=2048, pack_factor=4, add_eos=True):
    """Load CPT dataset from JSONL, tokenize, and pack to fixed block_size."""
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    max_len = block_size * pack_factor

    def tok_fn(batch):
        # Handle different data formats
        if "text" in batch:
            # Raw text format
            texts = batch["text"]
        elif "instruction" in batch:
            # Instruction format - convert to text
            texts = []
            for i in range(len(batch["instruction"])):
                instruction = batch["instruction"][i]
                input_text = batch.get("input", [""] * len(batch["instruction"]))[i]
                output_text = batch.get("output", [""] * len(batch["instruction"]))[i]
                
                # Create a natural text format for CPT
                if input_text.strip():
                    full_text = f"{instruction}\n\nInput: {input_text}\n\nResponse: {output_text}"
                else:
                    full_text = f"{instruction}\n\nResponse: {output_text}"
                texts.append(full_text)
        else:
            raise ValueError(f"Unsupported data format. Expected 'text' or 'instruction' fields. Got: {list(batch.keys())}")
        
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
    
    def tok_map(batch):
        input_ids, labels = [], []
        
        for i in range(len(batch["instruction"])):
            # Build conversation in chat format
            instr = batch["instruction"][i].strip()
            inpt = batch.get("input", [""] * len(batch["instruction"]))[i].strip()
            output = batch.get("output", [""] * len(batch["instruction"]))[i].strip()
            
            # Create user message
            user_content = f"{instr}\n{inpt}".strip() if inpt else instr
            conversation = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            
            # Apply chat template if available
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                try:
                    formatted_text = tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                except Exception:
                    # Fallback to simple format if template fails
                    formatted_text = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{output}<|end|>\n"
            else:
                # Fallback format when no chat template
                formatted_text = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{output}<|end|>\n"
            
            # Tokenize the full conversation
            full_tokens = tokenizer(formatted_text, add_special_tokens=True, truncation=True, max_length=block_size)
            full_ids = full_tokens["input_ids"]
            
            # Find where assistant response starts for label masking
            # Tokenize just the user part to find the boundary
            user_part = formatted_text.split("<|assistant|>")[0] + "<|assistant|>\n"
            user_tokens = tokenizer(user_part, add_special_tokens=True)
            user_len = len(user_tokens["input_ids"])
            
            # Create labels: mask user part (-100), supervise assistant part
            lab = [-100] * user_len + full_ids[user_len:]
            
            # Ensure same length
            if len(lab) > len(full_ids):
                lab = lab[:len(full_ids)]
            elif len(lab) < len(full_ids):
                lab.extend(full_ids[len(lab):])
            
            input_ids.append(full_ids)
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
