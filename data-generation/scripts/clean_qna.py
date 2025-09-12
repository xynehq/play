#!/usr/bin/env python3
"""
Script to clean qna_raw.jsonl file by:
1. Removing <think> </think> content from answer_raw
2. Deleting entire rows if only opening <think> tag is present
3. Making existing entries into perfect JSON form
"""

import json
import re
import argparse
from pathlib import Path

def clean_think_tags(text):
    """Remove <think> </think> blocks from text"""
    if not text:
        return text
    
    # Remove complete <think> </think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Clean up any remaining whitespace
    cleaned = cleaned.strip()
    
    return cleaned

def has_only_opening_think_tag(text):
    """Check if text has only opening <think> tag without closing tag"""
    if not text:
        return False
    
    # Check if there's an opening <think> tag
    has_opening = '<think>' in text
    # Check if there's a closing </think> tag
    has_closing = '</think>' in text
    
    # Return True if there's opening but no closing tag
    return has_opening and not has_closing

def clean_json_entry(entry):
    """Clean a single JSON entry"""
    if 'answer_raw' not in entry:
        return entry
    
    answer_raw = entry['answer_raw']
    
    # Check if we should delete this entry (only opening think tag)
    if has_only_opening_think_tag(answer_raw):
        return None
    
    # Clean the answer_raw field
    cleaned_answer = clean_think_tags(answer_raw)
    entry['answer_raw'] = cleaned_answer
    
    return entry

def process_jsonl_file(input_path, output_path):
    """Process the JSONL file and clean it"""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file {input_path} does not exist")
        return
    
    processed_count = 0
    deleted_count = 0
    cleaned_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON
                entry = json.loads(line)
                processed_count += 1
                
                # Clean the entry
                cleaned_entry = clean_json_entry(entry)
                
                if cleaned_entry is None:
                    # Entry was deleted due to incomplete think tag
                    deleted_count += 1
                    print(f"Deleted entry at line {line_num}: incomplete <think> tag")
                else:
                    # Check if we actually cleaned something
                    if 'answer_raw' in entry and '<think>' in entry['answer_raw']:
                        cleaned_count += 1
                        print(f"Cleaned <think> tags from entry at line {line_num}")
                    
                    # Write cleaned entry
                    json.dump(cleaned_entry, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"\nProcessing complete:")
    print(f"  Total entries processed: {processed_count}")
    print(f"  Entries with <think> tags cleaned: {cleaned_count}")
    print(f"  Entries deleted (incomplete <think>): {deleted_count}")
    print(f"  Output written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean QnA JSONL file by removing <think> tags')
    parser.add_argument('--input', '-i', default='data/generated/qna_raw.jsonl',
                        help='Input JSONL file path (default: data/generated/qna_raw.jsonl)')
    parser.add_argument('--output', '-o', default='data/generated/qna_cleaned.jsonl',
                        help='Output JSONL file path (default: data/generated/qna_cleaned.jsonl)')
    
    args = parser.parse_args()
    
    print(f"Cleaning QnA file: {args.input} -> {args.output}")
    process_jsonl_file(args.input, args.output)

if __name__ == "__main__":
    main()
