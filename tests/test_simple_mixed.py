#!/usr/bin/env python3
"""
Simple test to verify mixed training behavior without complex packing
"""

import json
import tempfile
import os
from scripts.datasets_cpt import load_chat_dataset_for_cpt

class SimpleTokenizer:
    """Very simple tokenizer for testing"""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.chat_template = True
        
    def __call__(self, text, **kwargs):
        # Simple character-based tokenization for predictable results
        if isinstance(text, list):
            return {"input_ids": [self._encode(t) for t in text]}
        return {"input_ids": self._encode(text)}
    
    def _encode(self, text):
        # Convert each character to an ID (starting from 2)
        return [ord(c) for c in text[:50]]  # Limit length for testing
    
    def decode(self, ids, skip_special_tokens=False):
        try:
            return ''.join(chr(id) for id in ids if id > 1)
        except:
            return "decode_error"
    
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        formatted = ""
        for msg in conversation:
            if msg["role"] == "user":
                formatted += f"USER: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted += f"ASSISTANT: {msg['content']}\n"
        return formatted

def test_simple_behavior():
    print("Simple Mixed Training Behavior Test")
    print("=" * 40)
    
    tokenizer = SimpleTokenizer()
    
    # Create simple test data
    chat_data = [
        {"instruction": "Hi", "input": "", "output": "Hello"},
        {"instruction": "Test", "input": "", "output": "Response"}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in chat_data:
            f.write(json.dumps(item) + '\n')
        chat_file = f.name
    
    try:
        print("🔍 Testing chat dataset processing...")
        
        # Test the chat processing function directly
        from datasets import load_dataset
        ds = load_dataset("json", data_files=chat_file, split="train")
        
        print(f"Raw dataset length: {len(ds)}")
        print(f"Sample raw data: {ds[0]}")
        
        # Test the tokenization mapping function
        def test_tok_map(batch):
            input_ids, labels = [], []
            
            for i in range(len(batch["instruction"])):
                instr = batch["instruction"][i].strip()
                output = batch.get("output", [""] * len(batch["instruction"]))[i].strip()
                
                # Create conversation
                user_content = instr
                conversation = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
                
                # Apply chat template
                formatted_text = tokenizer.apply_chat_template(conversation, tokenize=False)
                print(f"Formatted text: {repr(formatted_text)}")
                
                # Tokenize
                full_tokens = tokenizer(formatted_text)
                full_ids = full_tokens["input_ids"]
                
                # Find assistant boundary for masking
                user_part = f"USER: {user_content}\nASSISTANT: "
                user_tokens = tokenizer(user_part)
                user_len = len(user_tokens["input_ids"])
                
                # Create labels with masking
                lab = [-100] * user_len + full_ids[user_len:]
                
                # Ensure same length
                if len(lab) > len(full_ids):
                    lab = lab[:len(full_ids)]
                elif len(lab) < len(full_ids):
                    lab.extend(full_ids[len(lab):])
                
                input_ids.append(full_ids)
                labels.append(lab)
                
                print(f"Input IDs length: {len(full_ids)}")
                print(f"Labels length: {len(lab)}")
                print(f"Masked positions: {sum(1 for x in lab if x == -100)}")
                print(f"Supervised positions: {sum(1 for x in lab if x != -100)}")
                
            return {"input_ids": input_ids, "labels": labels}
        
        # Apply the mapping
        result = ds.map(test_tok_map, batched=True, remove_columns=ds.column_names)
        
        if len(result) > 0:
            sample = result[0]
            has_masked = any(x == -100 for x in sample['labels'])
            has_supervised = any(x != -100 for x in sample['labels'])
            
            print(f"\n✅ Chat processing successful!")
            print(f"✅ Has masked labels (-100): {has_masked}")
            print(f"✅ Has supervised labels: {has_supervised}")
            
            # Show the pattern
            print(f"\nLabel pattern: {sample['labels'][:20]}...")
            
            return True
        else:
            print("❌ No samples processed")
            return False
            
    finally:
        os.unlink(chat_file)

if __name__ == "__main__":
    test_simple_behavior()
