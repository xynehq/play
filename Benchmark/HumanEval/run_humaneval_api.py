#!/usr/bin/env python3
"""
HumanEval Benchmark Script - API Version
Evaluates language models via API on the HumanEval coding benchmark.
Supports both Python and Rust (MultiPL-E) benchmarking.
"""

import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from human_eval.data import read_problems, write_jsonl, stream_jsonl
from human_eval.evaluation import evaluate_functional_correctness, evaluate_rust_correctness, check_rust_correctness, clean_rust_completion
import requests
from tqdm import tqdm

# API Configuration
API_KEY = ''
BASE_URL = 'https://grid.ai.juspay.net'

# Model Configuration  
BASE_MODEL = "kat-dev-base-72b"
FINE_TUNED_MODEL = "kat-dev-hs-72b"


def setup_logging(model_name, output_suffix):
    """
    Setup logging for real-time task monitoring.
    
    Args:
        model_name: Model name being evaluated
        output_suffix: Output suffix (rust/python)
        
    Returns:
        Tuple of (text_log_path, json_log_path)
    """
    # Create logs directory
    os.makedirs("result/logs", exist_ok=True)
    
    # Setup text logging
    log_filename = f"result/logs/{model_name}_{output_suffix}_task_log.txt"
    json_log_filename = f"result/logs/{model_name}_{output_suffix}_realtime_log.jsonl"
    
    # Clear existing logs
    open(log_filename, 'w').close()
    open(json_log_filename, 'w').close()
    
    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return log_filename, json_log_filename


def log_task_result(task_id, status, duration, error_msg=None, json_log_file=None):
    """
    Log task result immediately to both text log and JSON log.
    
    Args:
        task_id: Task identifier
        status: PASS, FAIL, or ERROR
        duration: Task duration in seconds
        error_msg: Error message if any
        json_log_file: Path to JSON log file
    """
    # Log to text file
    if error_msg:
        logging.error(f"{task_id} | {status} | Duration: {duration:.2f}s | Error: {error_msg}")
    else:
        logging.info(f"{task_id} | {status} | Duration: {duration:.2f}s")
    
    # Log to console with color
    status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    print(f"{status_symbol} [{task_id}] {status} ({duration:.2f}s)")
    
    # Log to JSON file for structured analysis
    if json_log_file:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": status,
            "duration_seconds": round(duration, 2)
        }
        if error_msg:
            log_entry["error"] = error_msg
        
        with open(json_log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')


def call_api(prompt, model, max_tokens=512, temperature=0.2, n=1, timeout=60):
    """
    Call the OpenAI-compatible API to generate code completions.
    
    Args:
        prompt: The code prompt
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n: Number of completions to generate
        timeout: Request timeout in seconds
        
    Returns:
        List of generated completions
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "stop": None
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        completions = [choice["text"] for choice in result["choices"]]
        return completions
        
    except requests.exceptions.Timeout:
        print(f"⚠️  API request timed out after {timeout}s")
        return ["" for _ in range(n)]  # Return empty completions
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return ["" for _ in range(n)]
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return ["" for _ in range(n)]


def run_eval_api(model_name, output_file="samples.jsonl", k=[1, 10], num_samples=10, 
                 max_new_tokens=512, temperature=0.2, language="python", problem_file=None):
    """
    Run evaluation on a model via API with real-time logging.
    
    Args:
        model_name: Model name for API
        output_file: Output file for generated samples
        k: List of k values for pass@k metrics
        num_samples: Number of samples to generate per problem
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        language: Programming language for evaluation ("python" or "rust")
        problem_file: Optional path to problem file (for MultiPL-E datasets)
    """
    # Setup logging
    output_suffix = "rust" if language.lower() == "rust" else "python"
    text_log, json_log = setup_logging(model_name, output_suffix)
    
    print(f"🚀 Using API model: {model_name}")
    print(f"   API Endpoint: {BASE_URL}")
    print(f"   Task Log: {text_log}")
    print(f"   JSON Log: {json_log}")
    
    # Read HumanEval problems
    if problem_file:
        problems = read_problems(problem_file)
        print(f"📚 Loaded {len(problems)} {language.upper()} problems from {problem_file}")
    else:
        problems = read_problems()
        print(f"📚 Loaded {len(problems)} problems from HumanEval")
    
    samples = []
    total_problems = len(problems)
    
    print(f"\n🔄 Generating {num_samples} samples per problem via API...")
    print(f"   This will make {total_problems} API calls with {num_samples} completions each")
    print(f"   Real-time results will be logged as tasks complete\n")
    
    # Use tqdm for progress bar
    for task_id, task in tqdm(problems.items(), desc="Generating samples", unit="problem"):
        prompt = task["prompt"]
        task_start_time = time.time()
        
        try:
            # Generate completions via API
            completions = call_api(
                prompt=prompt,
                model=model_name,
                max_tokens=max_new_tokens,
                temperature=temperature,
                n=num_samples
            )
            
            # Save samples
            for completion in completions:
                samples.append({"task_id": task_id, "completion": completion})
            
            # Log generation success
            gen_duration = time.time() - task_start_time
            log_task_result(
                task_id=f"{task_id}_GENERATION",
                status="PASS",
                duration=gen_duration,
                json_log_file=json_log
            )
            
        except Exception as e:
            gen_duration = time.time() - task_start_time
            error_msg = str(e)[:200]
            log_task_result(
                task_id=f"{task_id}_GENERATION",
                status="ERROR",
                duration=gen_duration,
                error_msg=error_msg,
                json_log_file=json_log
            )
            # Add empty completions on error
            for _ in range(num_samples):
                samples.append({"task_id": task_id, "completion": ""})
        
        # Rate limiting - small delay between requests
        time.sleep(0.5)
    
    # Save samples
    write_jsonl(output_file, samples)
    print(f"\n✅ Samples saved to {output_file}")
    print(f"   Total samples generated: {len(samples)}")
    
    # Evaluate
    print(f"\n🧪 Evaluating functional correctness...")
    try:
        if language.lower() == "rust":
            results = evaluate_rust_correctness(
                sample_file=output_file,
                problem_file=problem_file if problem_file else problems,
                k=k,
                n_workers=2  # Use 2 workers to avoid Cargo lock issues
            )
        else:
            results = evaluate_functional_correctness(
                sample_file=output_file,
                k=k,
                n_workers=4,
                problem_file=problem_file if problem_file else None
            )
        
        print(f"\n📊 Results for {model_name}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f} ({value*100:.2f}%)")
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run HumanEval benchmark via API."""
    
    # Benchmark configuration
    LANGUAGE = "rust"  # Set to "rust" for Rust benchmarking, "python" for Python
    RUST_PROBLEMS_FILE = "data/humaneval-rust.jsonl.gz"
    COMPARE_MODELS = True  # Set to False to only evaluate fine-tuned model
    
    print("="*70)
    print("HumanEval Benchmark Evaluation - API Version")
    print("="*70)
    print(f"API Endpoint: {BASE_URL}")
    print(f"Fine-tuned Model: {FINE_TUNED_MODEL}")
    if COMPARE_MODELS:
        print(f"Base Model: {BASE_MODEL}")
    print(f"Language: {LANGUAGE.upper()}")
    print(f"Compare Models: {COMPARE_MODELS}")
    print("="*70)
    
    # Create result directory
    os.makedirs("result", exist_ok=True)
    
    # Determine problem file and output suffix
    if LANGUAGE.lower() == "rust":
        if not os.path.exists(RUST_PROBLEMS_FILE):
            print(f"\n⚠️  WARNING: Rust problems file not found: {RUST_PROBLEMS_FILE}")
            print("\nTo use Rust benchmarking, you need the MultiPL-E Rust dataset.")
            print("The file should be at: data/humaneval-rust.jsonl.gz")
            return 1
        problem_file = RUST_PROBLEMS_FILE
        output_suffix = "rust"
    else:
        problem_file = None
        output_suffix = "python"
    
    # Evaluate fine-tuned model
    print(f"\n{'='*70}")
    print(f"🚀 Evaluating Fine-tuned Model: {FINE_TUNED_MODEL}")
    print(f"{'='*70}")
    
    fine_tuned_results = run_eval_api(
        model_name=FINE_TUNED_MODEL,
        output_file=f"result/{FINE_TUNED_MODEL}_{output_suffix}.jsonl",
        num_samples=10,
        max_new_tokens=512,
        temperature=0.2,
        k=[1, 10],
        language=LANGUAGE,
        problem_file=problem_file
    )
    
    # Evaluate base model if comparison enabled
    base_results = None
    if COMPARE_MODELS:
        print(f"\n{'='*70}")
        print(f"🚀 Evaluating Base Model: {BASE_MODEL}")
        print(f"{'='*70}")
        
        base_results = run_eval_api(
            model_name=BASE_MODEL,
            output_file=f"result/{BASE_MODEL}_{output_suffix}.jsonl",
            num_samples=10,
            max_new_tokens=512,
            temperature=0.2,
            k=[1, 10],
            language=LANGUAGE,
            problem_file=problem_file
        )
    
    # Print final results
    if fine_tuned_results:
        print("\n" + "="*70)
        print(f"📊 FINAL RESULTS - Fine-tuned Model")
        print("="*70)
        for metric, value in fine_tuned_results.items():
            print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        
        if base_results:
            print("\n" + "="*70)
            print(f"📊 BASE MODEL RESULTS")
            print("="*70)
            for metric, value in base_results.items():
                print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
            
            # Print comparison
            print("\n" + "="*70)
            print(f"📈 IMPROVEMENT ANALYSIS")
            print("="*70)
            for metric in fine_tuned_results.keys():
                if metric in base_results:
                    improvement = (fine_tuned_results[metric] - base_results[metric]) * 100
                    improvement_pct = ((fine_tuned_results[metric] / base_results[metric]) - 1) * 100 if base_results[metric] > 0 else 0
                    print(f"{metric}:")
                    print(f"  Fine-tuned: {fine_tuned_results[metric]:.4f} ({fine_tuned_results[metric]*100:.2f}%)")
                    print(f"  Base:       {base_results[metric]:.4f} ({base_results[metric]*100:.2f}%)")
                    print(f"  Δ Absolute: {improvement:+.2f} percentage points")
                    print(f"  Δ Relative: {improvement_pct:+.2f}% improvement")
        
        # Save results to file
        results_file = f"result/{FINE_TUNED_MODEL}_vs_{BASE_MODEL}_{output_suffix}_results.json" if base_results else f"result/{FINE_TUNED_MODEL}_{output_suffix}_results.json"
        results_data = {
            "api_endpoint": BASE_URL,
            "language": LANGUAGE,
            "fine_tuned_model": FINE_TUNED_MODEL,
            "fine_tuned_metrics": fine_tuned_results
        }
        if base_results:
            results_data["base_model"] = BASE_MODEL
            results_data["base_metrics"] = base_results
            results_data["improvements"] = {
                metric: {
                    "absolute": (fine_tuned_results[metric] - base_results[metric]) * 100,
                    "relative": ((fine_tuned_results[metric] / base_results[metric]) - 1) * 100 if base_results[metric] > 0 else 0
                }
                for metric in fine_tuned_results.keys() if metric in base_results
            }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✅ Results saved to {results_file}")
        
        # Print location of detailed results
        print(f"\n📋 Detailed results:")
        print(f"  Fine-tuned: result/{FINE_TUNED_MODEL}_{output_suffix}.jsonl_results.jsonl")
        print(f"  Progress:   result/{FINE_TUNED_MODEL}_{output_suffix}.jsonl_progress.log")
        if base_results:
            print(f"  Base:       result/{BASE_MODEL}_{output_suffix}.jsonl_results.jsonl")
            print(f"  Progress:   result/{BASE_MODEL}_{output_suffix}.jsonl_progress.log")
        
        if LANGUAGE.lower() == "rust":
            print("\n" + "="*70)
            print("ℹ️  Note: Rust evaluation requires:")
            print("  • Rust toolchain (cargo) installed")
            print("  • MultiPL-E Rust dataset at data/humaneval-rust.jsonl.gz")
            print("="*70)
        
        return 0
    else:
        print("\n❌ Benchmark failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
