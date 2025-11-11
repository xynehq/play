from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import re
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import tqdm

from human_eval.data import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k


def clean_rust_completion(completion: str) -> str:
    """
    Clean the model-generated Rust completion to remove problematic patterns.
    
    Models sometimes generate extra code like main() functions or test code
    that conflicts with the test harness. This function removes such patterns.
    
    Args:
        completion: Raw completion from the model
        
    Returns:
        Cleaned completion ready to be combined with prompt and tests
    """
    # Remove any fn main() functions and their contents
    completion = re.sub(r'fn\s+main\s*\([^)]*\)\s*\{[^}]*\}', '', completion, flags=re.DOTALL)
    
    # Remove standalone main() calls
    completion = re.sub(r'\n\s*main\s*\([^)]*\)\s*;?\s*$', '', completion)
    
    # Remove any trailing test modules that the model might have added
    completion = re.sub(r'#\[cfg\(test\)\][\s\S]*$', '', completion)
    
    # Remove extra closing braces at the end that don't match any opening
    lines = completion.split('\n')
    cleaned_lines = []
    brace_count = 0
    
    for line in lines:
        # Count braces in this line
        open_braces = line.count('{')
        close_braces = line.count('}')
        brace_count += open_braces - close_braces
        
        # Only add lines that aren't just a standalone closing brace when we're already balanced
        stripped = line.strip()
        if stripped == '}' and brace_count < 0:
            brace_count += 1  # Don't add this line, but adjust counter
            continue
        
        cleaned_lines.append(line)
    
    completion = '\n'.join(cleaned_lines)
    
    # Remove any println! or assert! statements that are outside functions
    lines = completion.split('\n')
    cleaned_lines = []
    in_function = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track if we're inside a function
        if 'fn ' in line and '{' in line:
            in_function = True
        
        # Skip standalone assertions/prints outside functions
        if not in_function and (stripped.startswith('assert') or stripped.startswith('println!')):
            continue
        
        cleaned_lines.append(line)
        
        # Check if function ends
        if in_function and '}' in line:
            if stripped == '}':
                in_function = False
    
    return '\n'.join(cleaned_lines)


def check_rust_correctness(problem: Dict, completion: str, timeout: float, completion_id: int = None) -> Dict:
    """
    Evaluate Rust code by compiling and running tests.
    
    Args:
        problem: Problem dict with 'prompt', 'tests', 'task_id'
        completion: Generated Rust code
        timeout: Timeout in seconds
        completion_id: Optional completion ID
        
    Returns:
        Dict with task_id, passed status, result message, and completion_id
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Cargo project
        project_name = "test_project"
        project_dir = Path(tmpdir) / project_name
        
        # Set isolated Cargo home to avoid file lock conflicts
        cargo_home = Path(tmpdir) / ".cargo"
        cargo_home.mkdir(exist_ok=True)
        
        # Create environment with isolated CARGO_HOME
        env = os.environ.copy()
        env["CARGO_HOME"] = str(cargo_home)
        env["CARGO_TARGET_DIR"] = str(project_dir / "target")
        env["CARGO_NET_OFFLINE"] = "true"
        env["CARGO_HTTP_MULTIPLEXING"] = "false"
        env["CARGO_REGISTRIES_CRATES_IO_PROTOCOL"] = "sparse"
        
        # Initialize Cargo project
        try:
            subprocess.run(
                ["cargo", "new", "--lib", project_name],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                timeout=timeout,
                env=env
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            return {
                "task_id": problem.get("task_id", "unknown"),
                "passed": False,
                "result": f"failed: cargo init error - {str(e)}",
                "completion_id": completion_id,
            }
        
        # Write the code to lib.rs
        lib_file = project_dir / "src" / "lib.rs"
        
        # Clean the completion
        cleaned_completion = clean_rust_completion(completion)
        
        # Construct the full Rust program
        rust_code = problem.get("prompt", "") + "\n" + cleaned_completion + "\n\n" + problem.get("tests", "")
        
        try:
            lib_file.write_text(rust_code)
        except Exception as e:
            return {
                "task_id": problem.get("task_id", "unknown"),
                "passed": False,
                "result": f"failed: write error - {str(e)}",
                "completion_id": completion_id,
            }
        
        # Run cargo test
        try:
            result = subprocess.run(
                ["cargo", "test", "--", "--nocapture"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            if result.returncode == 0:
                return {
                    "task_id": problem.get("task_id", "unknown"),
                    "passed": True,
                    "result": "passed",
                    "completion_id": completion_id,
                }
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                return {
                    "task_id": problem.get("task_id", "unknown"),
                    "passed": False,
                    "result": f"failed: {error_msg[:200]}",
                    "completion_id": completion_id,
                }
        except subprocess.TimeoutExpired:
            return {
                "task_id": problem.get("task_id", "unknown"),
                "passed": False,
                "result": "timed out",
                "completion_id": completion_id,
            }
        except Exception as e:
            return {
                "task_id": problem.get("task_id", "unknown"),
                "passed": False,
                "result": f"failed: {str(e)}",
                "completion_id": completion_id,
            }


def evaluate_rust_correctness(
    sample_file: str,
    problem_file: str,
    k: List[int] = [1, 10],
    n_workers: int = 2,
    timeout: float = 10.0
) -> Dict:
    """
    Evaluate Rust code completions by compiling and running tests.
    Results are written incrementally as each test completes.
    
    Args:
        sample_file: JSONL file with generated Rust code samples
        problem_file: Path to Rust problems file or dict of problems
        k: List of k values for pass@k
        n_workers: Number of parallel workers (reduced to avoid lock contention)
        timeout: Timeout for each test execution
        
    Returns:
        Dict with pass@k metrics
    """
    # Load problems if file path provided
    if isinstance(problem_file, str):
        problems = read_problems(problem_file)
    else:
        problems = problem_file
    
    # Prepare incremental results file
    out_file = sample_file + "_results.jsonl"
    progress_file = sample_file + "_progress.log"
    
    # Create/clear the results file
    with open(out_file, 'w') as f:
        pass
    
    # Read all samples into memory with their original order
    all_samples = list(stream_jsonl(sample_file))
    sample_dict = {(s["task_id"], i): s for i, s in enumerate(all_samples)}
    
    # Check generated samples against test suites
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)
        
        print(f"\n{'='*70}")
        print(f"📋 Starting Rust Evaluation")
        print(f"{'='*70}")
        print(f"Total samples: {len(all_samples)}")
        print(f"Parallel workers: {n_workers}")
        print(f"Results file: {out_file}")
        print(f"Progress log: {progress_file}")
        print(f"{'='*70}\n")
        
        # Submit all tasks
        for sample in all_samples:
            task_id = sample["task_id"]
            completion = sample["completion"]
            comp_id = completion_id[task_id]
            args = (problems[task_id], completion, timeout, comp_id)
            future = executor.submit(check_rust_correctness, *args)
            futures[future] = (task_id, comp_id, n_samples)
            completion_id[task_id] += 1
            n_samples += 1
        
        assert len(completion_id) == len(problems), "Some problems are not attempted."
        
        # Process results as they complete
        passed_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        with open(progress_file, 'w') as log_f:
            log_f.write(f"Benchmarking started at {start_time}\n")
            log_f.write(f"Total samples: {n_samples}\n")
            log_f.write(f"Workers: {n_workers}\n\n")
            
            for i, future in enumerate(as_completed(futures), 1):
                task_id, comp_id, sample_idx = futures[future]
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))
                
                # Track statistics
                if result["passed"]:
                    passed_count += 1
                    status_emoji = "✅"
                else:
                    failed_count += 1
                    status_emoji = "❌"
                
                # Calculate progress
                progress_pct = (i / n_samples) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / i if i > 0 else 0
                est_remaining = avg_time * (n_samples - i)
                
                # Prepare result entry
                original_sample = sample_dict[(task_id, sample_idx)]
                result_entry = original_sample.copy()
                result_entry["result"] = result["result"]
                result_entry["passed"] = result["passed"]
                
                # Write result immediately to file
                with open(out_file, 'a') as f:
                    import json
                    f.write(json.dumps(result_entry) + '\n')
                
                # Display progress
                status_line = (
                    f"{status_emoji} [{i}/{n_samples}] {progress_pct:5.1f}% | "
                    f"{task_id} (#{comp_id}) | "
                    f"✓ {passed_count} ✗ {failed_count} | "
                    f"⏱️  {elapsed:.0f}s / ~{est_remaining:.0f}s left"
                )
                print(status_line)
                
                # Log to file
                log_f.write(f"{datetime.now()} | {status_line}\n")
                if not result["passed"] and result["result"] != "timed out":
                    log_f.write(f"  Error: {result['result'][:150]}\n")
                log_f.flush()
    
    # Calculate pass@k
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    
    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    
    return pass_at_k
