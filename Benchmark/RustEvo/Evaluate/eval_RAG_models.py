import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, List
from unit import call_LLM, run_rust_code, get_test_program_prompt, get_code_solution_prompt
from RAG_unit import get_RAG_document
from dotenv import load_dotenv

load_dotenv()

# API_KEY = os.environ["AGICTO_API_KEY"]
# BASE_URL = os.environ["AGICTO_BASE_URL"]

API_KEY = os.environ["XIAOAI_API_KEY"]
BASE_URL = os.environ["XIAOAI_BASE_URL"]

MAX_WORKERS=5

MODELS = [
    "gpt-4o-mini"
]

def needs_testing(entry: Dict[str, Any], model: str, suffix: str) -> bool:
    """Check if the model with given suffix (baseline/rag) needs testing."""
    code_solution_key = f"{model}_{suffix}_code_solution"
    test_program_key = f"{model}_{suffix}_test_program"
    result_key = f"{model}_{suffix}_result"
    return not entry.get(code_solution_key) or not entry.get(test_program_key) or not entry.get(result_key)

def initialize_model_fields(entry: Dict[str, Any]) -> None:
    for model in MODELS:
        for suffix in ['baseline', 'rag']:
            for field in ['code_solution', 'test_program', 'result']:
                key = f"{model}_{suffix}_{field}"
                entry.setdefault(key, '')

def process_single_model(entry: Dict[str, Any], model: str, api_key: str, base_url: str, use_rag: bool) -> None:
    suffix = "rag" if use_rag else "baseline"
    query = entry.get('query', '')
    signature = entry.get('signature', '')
    api_info = entry.get('api_info', '')
    api = entry.get('api', '')

    if use_rag := (suffix == 'rag'):
        retrieved_docs = get_RAG_document(query)
        api_info += f"\n\nRetrieved Documentation:\n{retrieved_docs}"

    # Generate code solution
    code_solution = call_LLM(
        get_code_solution_prompt(query, signature, api_info),
        model, api_key=API_KEY, BASE_URL=base_url
    )
    entry[f"{model}_{suffix}_code_solution"] = code_solution

    # Generate test program
    test_program = call_LLM(
        get_test_program_prompt(query, code_solution, api),
        model, API_KEY, BASE_URL
    )
    entry[f"{model}_{suffix}_test_program"] = test_program

    # Execute test program
    try:
        entry[f"{model}_{suffix}_result"] = run_rust_code(test_program)
    except Exception as e:
        entry[f"{model}_{suffix}_result"] = f"Execution error: {str(e)}"

def process_entry(entry: Dict[str, Any], api_key: str, base_url: str) -> Dict[str, Any]:
    initialize_model_fields(entry)

    for model in MODELS:
        # First run baseline
        if needs_testing(entry, model, 'baseline'):
            process_model(entry, model, api_key=api_key, base_url=base_url, suffix='baseline')

        # Then run RAG-enhanced
        if needs_testing(entry, model, 'rag'):
            process_model(entry, model, api_key=api_key, base_url=base_url, suffix='rag')

    return entry

def process_model(entry, model, api_key, base_url, suffix):
    query = entry.get('query', '')
    signature = entry.get('signature', '')
    api_info = entry.get('api_info', '')
    api = entry.get('api', '')

    if suffix == 'rag':
        retrieved_docs = get_RAG_document(query)
        prompt_api_info = f"{api_info}\n\nRetrieved Documentation:\n{retrieved_docs}"
    else:
        prompt_api_info = api_info

    code_solution = call_LLM(
        get_code_solution_prompt(query, signature, prompt_api_info),
        model, api_key, base_url
    )
    entry[f"{model}_{suffix}_code_solution"] = code_solution = code_solution

    test_program = call_LLM(
        get_test_program_prompt(query, code_solution, api),
        model, api_key, base_url
    )
    entry[f"{model}_{suffix}_test_program"] = test_program

    try:
        entry[f"{model}_{suffix}_result"] = run_rust_code(test_program)
    except Exception as e:
        entry[f"{model}_{suffix}_result"] = f"Failed to run: {str(e)}"

def main():

    with open("example.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    with tqdm(total=total_entries, desc="Overall Progress") as overall_pbar:
        for round_start in range(0, total_entries, MAX_WORKERS):  # process 5 entries concurrently
            batch_entries = data[round_start: round_start+MAX_WORKERS]

            # Concurrently handle multiple entries
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(process_entry, entry, API_KEY, BASE_URL): entry for entry in data[round_start:round_start+5]}

                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                    finally:
                        overall_pbar.update(1)

            # Save data explicitly after each concurrent round
            with open("example.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()