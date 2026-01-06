from __future__ import annotations
# Run the following in terminal first
# export LITE_LLM_URL="https://grid.ai.juspay.net"
# export LITE_LLM_API_KEY="sk-..."

# Excecute with this command
"""
python3 script.py \

"""

#!/usr/bin/env python3
"""
Evaluate filename prediction using OpenAI-compatible API (LiteLLM, vLLM API, etc.)
Generates n predictions per example and calculates pass@n metrics.
"""


import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from datasets import Dataset, load_dataset, load_from_disk
import yaml
# ---------- Constants ----------
SYSTEM_PROMPT = (
    "You are an expert at analyzing code changes and predicting which files will be modified "
    "in a Rust codebase. Think step by step about which files are likely to be affected "
    "based on the problem description and hints."
)


# ---------- Data classes ----------
@dataclass
class ExampleResult:
    index: int
    instance_id: str
    all_predictions: List[List[str]]  # n predictions
    all_reasoning: List[str] = None
    actual_filenames: List[str] = None
    success: bool = True
    pass_at_n: Dict[int, bool] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.pass_at_n is None:
            self.pass_at_n = {}
        if self.all_reasoning is None:
            self.all_reasoning = []
        if self.actual_filenames is None:
            self.actual_filenames = []


@dataclass
class EvalSummary:
    total_examples: int
    successful_predictions: int
    error_rate: float
    pass_at_n: Dict[int, float] = None

    def __post_init__(self):
        if self.pass_at_n is None:
            self.pass_at_n = {}


# ---------- Utils ----------
def create_prompt(problem_statement: str, hints_text: Optional[str], max_chars: Optional[int] = None) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n"
    prompt += "You are analyzing a code change request for the Hyperswitch payment processing system (a Rust project).\n\n"
    prompt += f"**Problem Statement:**\n{problem_statement.strip()}\n\n"
    if hints_text and hints_text.strip():
        prompt += f"**Hints:**\n{hints_text.strip()}\n\n"
    prompt += (
        "Based on this information, predict which files in the codebase will need to be modified.\n"
        "Respond with a JSON object containing:\n"
        '{"filenames": ["file1.rs", "file2.rs", ...], "reasoning": "explanation"}\n\n'
        "JSON:"
    )

    if max_chars and len(prompt) > max_chars:
        header = f"{SYSTEM_PROMPT}\n\nYou are analyzing a code change request for the Hyperswitch payment processing system (a Rust project).\n\n**Problem Statement:**\n"
        footer = (
            "\n\nBased on this information, predict which files in the codebase will need to be modified.\n"
            "Respond with a JSON object containing:\n"
            '{"filenames": ["file1.rs", "file2.rs", ...], "reasoning": "explanation"}\n\n'
            "JSON:"
        )
        available_chars = max_chars - len(header) - len(footer) - 50
        content = problem_statement.strip()
        if hints_text and hints_text.strip():
            content += f"\n\n**Hints:**\n{hints_text.strip()}"
        if len(content) > available_chars:
            content = content[:available_chars] + "...[truncated]"
        prompt = header + content + footer
    return prompt


def parse_json_response(text: str) -> tuple[List[str], str]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return [], ""
        json_str = text[start:end]
        data = json.loads(json_str)
        return data.get("filenames", []), data.get("reasoning", "")
    except Exception as e:
        logging.warning(f"Failed to parse JSON: {e}\nRaw text: {text[:500]}")
        return [], ""


def calculate_pass_at_k(predictions: List[List[str]], actual: List[str], k: int) -> bool:
    if not actual:
        return True
    actual_set = set(actual)
    for pred_list in predictions[:k]:
        pred_set = set(pred_list)
        if pred_set & actual_set:
            return True
    return False


# ---------- API Client ----------
def generate_openai_api(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    n: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 2048,
) -> List[str]:
    """Call OpenAI-compatible API (LiteLLM, vLLM, etc.)"""
    base = api_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    url = f"{base}/v1/chat/completions"
    # url = f"{api_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
    }

    logging.debug(f"Sending request to {url} with model={model}, n={n}")
    start_time = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=300)
    duration = time.time() - start_time
    logging.debug(f"API call completed in {duration:.2f}s")

    if resp.status_code != 200:
        logging.error(f"API Error {resp.status_code}: {resp.text[:500]}")
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    completions = []
    for i, choice in enumerate(data.get("choices", [])):
        content = choice["message"]["content"]
        logging.debug(f"Response {i+1}/{n}:\n{content[:500]}\n{'-'*40}")
        completions.append(content)
    return completions


# ---------- Evaluation ----------
def evaluate_with_api(
    dataset: Dataset,
    api_url: str,
    api_key: str,
    model: str,
    n: int = 10,
    max_examples: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 2048,
) -> List[ExampleResult]:
    limit = min(len(dataset), max_examples or 100)
    results: List[ExampleResult] = []
    logging.info(f"Generating {n} predictions per example for {limit} examples via API")

    for idx in range(limit):
        example = dataset[idx]
        problem = example["problem_statement"]
        hints = example.get("hints_text", "")
        prompt = create_prompt(problem, hints)
        actual = example.get("filenames", []) or []

        logging.info(f"\n--- Processing Example {idx} ({example.get('instance_id', 'N/A')}) ---")
        logging.debug(f"Prompt:\n{prompt[:1000]}\n{'='*60}")

        all_predictions, all_reasoning = [], []

        try:
            completions = generate_openai_api(
                api_url=api_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            for text in completions:
                filenames, reasoning = parse_json_response(text)
                all_predictions.append(filenames)
                all_reasoning.append(reasoning)
                logging.debug(f"Parsed filenames: {filenames}\nReasoning: {reasoning[:300]}")

            result = ExampleResult(
                index=idx,
                instance_id=str(example.get("instance_id", "")),
                all_predictions=all_predictions,
                all_reasoning=all_reasoning,
                actual_filenames=actual,
                success=True,
            )
            logging.info(f"✅ Example {idx} completed successfully")

        except Exception as exc:
            logging.error(f"❌ Error processing example {idx}: {exc}")
            result = ExampleResult(
                index=idx,
                instance_id=str(example.get("instance_id", "")),
                all_predictions=[],
                actual_filenames=actual,
                success=False,
                error=str(exc),
            )

        results.append(result)
        time.sleep(0.3)  # avoid API rate limits

    return results


# ---------- Aggregation ----------
def aggregate_metrics(results: List[ExampleResult], k_values: List[int]) -> EvalSummary:
    total = len(results)
    successes = [r for r in results if r.success]
    successful_count = len(successes)
    error_rate = 1.0 - (successful_count / total) if total > 0 else 1.0

    if successful_count == 0:
        return EvalSummary(total_examples=total, successful_predictions=0, error_rate=error_rate, pass_at_n={})

    pass_at_k = {}
    for k in k_values:
        passed = sum(1 for r in successes if calculate_pass_at_k(r.all_predictions, r.actual_filenames, k))
        pass_at_k[k] = passed / successful_count

    return EvalSummary(
        total_examples=total,
        successful_predictions=successful_count,
        error_rate=error_rate,
        pass_at_n=pass_at_k,
    )


# ---------- I/O ----------
def save_results(results: List[ExampleResult], summary: EvalSummary, output_path: Path) -> None:
    payload = {"metrics": asdict(summary), "results": [asdict(r) for r in results]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logging.info(f"Results written to {output_path}")


def print_summary(summary: EvalSummary) -> None:
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print(f"Total Examples:           {summary.total_examples}")
    print(f"Successful Predictions:   {summary.successful_predictions}")
    print(f"Error Rate:               {summary.error_rate * 100:.2f}%")
    if summary.successful_predictions > 0:
        print(f"\n--- Pass@K Metrics ---")
        for k in sorted(summary.pass_at_n.keys()):
            print(f"Pass@{k:2d}:                 {summary.pass_at_n[k] * 100:.2f}%")
    else:
        print("\nNo successful predictions")
    print("=" * 80)


def load_dataset_safely(dataset_ref: str, split: str) -> Dataset:
    try:
        ds = load_dataset(dataset_ref, split=split)
        logging.info(f"Loaded dataset from HuggingFace: {dataset_ref} (len={len(ds)})")
        return ds
    except Exception:
        logging.info(f"Could not load from HuggingFace, trying disk path: {dataset_ref}")
        ds = load_from_disk(dataset_ref)
        logging.info(f"Loaded dataset from disk: {dataset_ref} (len={len(ds)})")
        return ds


# ---------- CLI & main ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate filename prediction with OpenAI-compatible API pass@n")
    p.add_argument("--dataset", type=str, default="archit11/evals")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--api_url", type=str, default=os.getenv("LITE_LLM_URL", "http://localhost:8005"))
    p.add_argument("--api_key", type=str, default=os.getenv("LITE_LLM_API_KEY", "dummy-key"))
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--output", type=Path, default=Path("eval_passn_results.json"))
    p.add_argument("--pass_k", type=str, default="1,3,8")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config_path = Path(Path(__file__).parent.parent / "model_config.yaml")
    config_file = Path(config_path)
    with open(config_file, 'r') as f:
        yaml_data = yaml.safe_load(f)
        archit_config = yaml_data.get('archit_config', {})
        k_values = archit_config.get('k_values')
        k_values = [k for k in k_values if k <= archit_config.get('archit_n')]
        logging.info(f"Computing pass@k for k={k_values}")

        dataset = load_dataset_safely(archit_config.get('archit_dataset'), archit_config.get('archit_split'))

        results = evaluate_with_api(
            dataset=dataset,
            api_url=yaml_data.get('api_base'),
            api_key=yaml_data.get('api_key'),
            model=archit_config.get('model_name'),
            n=archit_config.get('archit_n',8),
            max_examples=archit_config.get('max_examples', 300),
            temperature=archit_config.get('temperature', 0.0),
            top_p=archit_config.get('top_p', 0.95),
            max_tokens=archit_config.get('max_tokens', 2048),
        )

        summary = aggregate_metrics(results, k_values)
        print_summary(summary)
        save_results(results, summary, archit_config.get('archit_output'))

        logging.info("\nSample predictions (first 3 examples):")
        for r in results[:3]:
            logging.info(f"\nExample {r.index} ({r.instance_id}):")
            logging.info(f"  Actual: {r.actual_filenames[:5]}")
            logging.info(f"  Pred 1: {r.all_predictions[0][:5] if r.all_predictions else 'N/A'}")
            logging.info(f"  Pred 2: {r.all_predictions[1][:5] if len(r.all_predictions) > 1 else 'N/A'}")


if __name__ == "__main__":
    main()