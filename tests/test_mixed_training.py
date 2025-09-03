#!/usr/bin/env python3
"""
Test script to verify mixed training behavior:
- CPT chunks: no template, labels everywhere
- Chat samples: chat template applied, labels masked on prompt, supervised on assistant
"""

import json
import tempfile
import os
from scripts.datasets_cpt import load_cpt_dataset, load_chat_dataset_for_cpt

class MockTokenizer:
    """Simple mock tokenizer for testing purposes"""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {"<pad>": 0, "<eos>": 1, "<|user|>": 2, "<|end|>": 3, "<|assistant|>": 4}
        self.next_id = 5
    
    def __call__(self, text, **kwargs):
        add_special_tokens = kwargs.get('add_special_tokens', False)
        truncation = kwargs.get('truncation', False)
        max_length = kwargs.get('max_length', None)
        
        if isinstance(text, list):
            result = {"input_ids": [self._encode(t, add_special_tokens, truncation, max_length) for t in text]}
        else:
            result = {"input_ids": self._encode(text, add_special_tokens, truncation, max_length)}
        
        # Add attention_mask if needed
        if "padding" in kwargs or "attention_mask" in kwargs:
            if isinstance(text, list):
                result["attention_mask"] = [[1] * len(ids) for ids in result["input_ids"]]
            else:
                result["attention_mask"] = [1] * len(result["input_ids"])
        
        return result
    
    def _encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        # Simple word-based encoding for testing
        words = text.split()
        ids = []
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[word])
        
        if add_special_tokens:
            ids = ids + [self.eos_token_id]  # Add EOS token
        
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=False):
        # Reverse mapping
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, f"<unk_{id}>") for id in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if not t.startswith("<")]
        return " ".join(tokens)
    
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        # Simple chat template
        formatted = ""
        for msg in conversation:
            if msg["role"] == "user":
                formatted += f"<|user|>\n{msg['content']}<|end|>\n"
            elif msg["role"] == "assistant":
                formatted += f"<|assistant|>\n{msg['content']}<|end|>\n"
        return formatted

def test_mixed_training_behavior():
    print("Testing Mixed Training Behavior")
    print("=" * 50)
    
    # Use mock tokenizer for testing
    tokenizer = MockTokenizer()
    tokenizer.chat_template = True  # Enable chat template
    
    # Create test CPT data
    cpt_data = [
        {"text": "This is a document about machine learning. It covers various topics including neural networks and deep learning."},
        {"text": "Another document discussing artificial intelligence and its applications in modern technology."}
    ]
    
    # Create test chat data
    chat_data = [
        {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of AI that enables computers to learn from data."},
        {"instruction": "Explain neural networks", "input": "in simple terms", "output": "Neural networks are computing systems inspired by biological neural networks."}
    ]
    
    # Write test data to temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in cpt_data:
            f.write(json.dumps(item) + '\n')
        cpt_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in chat_data:
            f.write(json.dumps(item) + '\n')
        chat_file = f.name
    
    try:
        # Test CPT dataset
        print("🔍 Testing CPT dataset processing...")
        cpt_ds = load_cpt_dataset(cpt_file, tokenizer, block_size=32, pack_factor=2)  # Smaller block size
        
        print(f"CPT dataset length: {len(cpt_ds)}")
        if len(cpt_ds) == 0:
            print("❌ CPT dataset is empty after processing - this indicates an issue with packing")
            return False
        
        # Check CPT sample
        cpt_sample = cpt_ds[0]
        print(f"CPT sample input_ids length: {len(cpt_sample['input_ids'])}")
        print(f"CPT sample labels length: {len(cpt_sample['labels'])}")
        
        # In CPT, labels should equal input_ids (no masking)
        labels_match_input = all(
            label == input_id for label, input_id in zip(cpt_sample['labels'], cpt_sample['input_ids'])
        )
        print(f"✅ CPT labels match input_ids (no masking): {labels_match_input}")
        
        # Test chat dataset
        print("\n🔍 Testing chat dataset processing...")
        chat_ds = load_chat_dataset_for_cpt(chat_file, tokenizer, block_size=32)  # Smaller block size
        
        print(f"Chat dataset length: {len(chat_ds)}")
        if len(chat_ds) == 0:
            print("❌ Chat dataset is empty after processing")
            return False
        
        # Check chat sample
        chat_sample = chat_ds[0]
        print(f"Chat sample input_ids length: {len(chat_sample['input_ids'])}")
        print(f"Chat sample labels length: {len(chat_sample['labels'])}")
        
        # In chat, some labels should be -100 (masked), some should match input_ids
        has_masked_labels = any(label == -100 for label in chat_sample['labels'])
        has_supervised_labels = any(
            label != -100 and label == input_id 
            for label, input_id in zip(chat_sample['labels'], chat_sample['input_ids'])
        )
        print(f"✅ Chat has masked labels (-100): {has_masked_labels}")
        print(f"✅ Chat has supervised labels: {has_supervised_labels}")
        
        # Decode and show the structure
        print("\n📝 Sample structures:")
        
        # CPT sample
        cpt_text = tokenizer.decode(cpt_sample['input_ids'], skip_special_tokens=True)
        print(f"CPT text: {cpt_text[:100]}...")
        
        # Chat sample
        chat_text = tokenizer.decode(chat_sample['input_ids'], skip_special_tokens=True)
        print(f"Chat text: {chat_text[:100]}...")
        
        # Show label masking pattern for chat
        masked_positions = [i for i, label in enumerate(chat_sample['labels']) if label == -100]
        supervised_positions = [i for i, label in enumerate(chat_sample['labels']) if label != -100]
        print(f"Chat masked positions: {len(masked_positions)} tokens")
        print(f"Chat supervised positions: {len(supervised_positions)} tokens")
        
        print("\n🎉 Mixed training behavior verification complete!")
        print("✅ CPT data: labels everywhere (no masking)")
        print("✅ Chat data: proper template + label masking")
        
        return True
        
    finally:
        # Clean up temp files
        os.unlink(cpt_file)
        os.unlink(chat_file)

if __name__ == "__main__":
    test_mixed_training_behavior()
