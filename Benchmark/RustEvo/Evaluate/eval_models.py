from unit import call_LLM, run_rust_code,get_test_program_prompt,get_code_solution_prompt,extract_rust_code
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import re
MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "gpt-4o",
    "gpt-4o-mini"
]

API_KEY = os.getenv('API_KEY','')
BASE_URL=os.getenv('BASE_URL','')

def needs_testing(entry: Dict[str, Any], model: str) -> bool:
    """Check if processing is required for the given model."""
    code_solution_key = f"{model}_code_solution"
    test_program_key = f"{model}_test_program"
    result_key = f"{model}_result"
    # TODO return not entry.get(code_solution_key) or not entry.get(test_program_key) or not entry.get(result_key)
    return 1

def initialize_model_fields(entry: Dict[str, Any]) -> None:
    """Ensure each model's fields are initialized explicitly."""
    for model in MODELS:
        for suffix in ['code_solution', 'test_program', 'result']:
            entry.setdefault(f"{model}_{suffix}", '')

def process_model(entry: Dict[str, Any], model: str, api_key: str, base_url: str) -> None:
    """Process a single model (baseline only) sequentially."""
    query = entry.get('query', '')
    signature = entry.get('signature', '')
    api_info = entry.get('api_info', '')
    api = entry.get('api', '')

    # Generate code solution
    raw_code_solution = call_LLM(
        get_code_solution_prompt(query, signature, api_info),
        model, api_key, base_url
    )

    code_solution = extract_rust_code(raw_code_solution)
    entry[f"{model}_code_solution"] = code_solution

    # Generate test program
    raw_test_program = call_LLM(
        get_test_program_prompt(query, code_solution, api),
        model, api_key, base_url
    )

    test_program = extract_rust_code(raw_test_program)
    entry[f"{model}_test_program"] = test_program

    # Execute test program
    try:
        entry[f"{model}_result"] = run_rust_code(test_program)
    except Exception as e:
        entry[f"{model}_result"] = f"Execution error: {str(e)}"

def process_entry(entry: Dict[str, Any], api_key: str, base_url: str) -> Dict[str, Any]:
    """Process all models sequentially within each entry."""
    initialize_model_fields(entry)

    for model in MODELS:
        if needs_testing(entry, model):
            process_model(entry, model, api_key, base_url)

    return entry


def main():
    with open("example.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)

    # Adjust batch size as appropriate for your hardware/API limits
    batch_size = 5

    with tqdm(total=total_entries, desc="Overall Progress") as overall_pbar:
        for batch_start in range(0, total_entries, batch_size):
            batch_end = min(batch_start + batch_size, total_entries)
            current_batch = data[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(process_entry, entry, API_KEY, BASE_URL): idx
                    for idx, entry in enumerate(current_batch, start=batch_start)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        data[idx] = future.result()
                    except Exception as e:
                        tqdm.write(f"Error processing entry {idx}: {str(e)}")
                    finally:
                        overall_pbar.update(1)

            # Explicitly save after each batch is processed
            with open("example.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()