import json
import os
import sys
import time
import re
import datetime
import traceback
import requests
import numpy as np
import logging
import yaml
import subprocess
import tempfile
from scipy import stats
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from math import exp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ==============================
# LOGGING CONFIGURATION
# ==============================
def setup_logging(log_file: Optional[str] = None):
    """Configure logging with file and console handlers."""
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silence noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# ==============================
# CONFIG LOADING
# ==============================
def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"⚠️ Config not found at {config_path}. Using internal defaults.")
        return {}

_CONFIG = load_config()

# ==============================
# 🔧 GLOBAL CONFIGURATION
# ==============================
API_KEY = _CONFIG.get('api', {}).get('api_key')
BASE_URL = _CONFIG.get('api', {}).get('base_url')
MODEL_NAME = _CONFIG.get('models', {}).get('evaluation_model')

TEMPERATURE = _CONFIG.get('evaluation', {}).get('temperature', 0.0)
MAX_TOKENS = _CONFIG.get('evaluation', {}).get('max_tokens', 1024)

# Constants
CHECKPOINT_FILE = "evaluation_checkpoint.json"
DEFAULT_PARALLEL_WORKERS = 5  # Increased default for speed

# ==============================
# UTILITY FUNCTIONS
# ==============================

def query_model_with_reasoning(prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Call LLM for evaluation with JSON response.
    Handles retries, JSON parsing, and cleans invisible characters.
    """
    if not API_KEY or not BASE_URL:
        return {}

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        # Add JSON enforcement instruction if retrying
        json_instruction = ""
        if attempt > 0:
            json_instruction = "\n\n🚨 CRITICAL: Your previous response was invalid. You MUST respond with ONLY a valid JSON object. NO code snippets, NO explanations. Start response with '{'."
        
        # Use system message for better instruction following
        messages = [
            {
                "role": "system", 
                "content": "You are a precise evaluator. You MUST respond with ONLY valid JSON. Never include code snippets, explanations, or any text outside the JSON object."
            },
            {
                "role": "user", 
                "content": prompt + json_instruction
            }
        ]
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": TEMPERATURE,
            # INCREASED TIMEOUT: Reasoning models can take longer
            "max_tokens": MAX_TOKENS,
            "stream": False
        }

        try:
            # Increased timeout to 90s for reasoning models
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            result = response.json()
            
            # Handle Anthropic API response format
            if "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                content = result["content"][0].get("text", "").strip()
            # Try OpenAI format as fallback
            elif "choices" in result:
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                logging.error(f"Unexpected API response format: {result}")
                continue
            
            if not content:
                logging.error(f"Empty content from API response: {result}")
                continue
            
            # Remove thinking/reasoning tags if present (Claude/DeepSeek format)
            if "<think>" in content:
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[-1].strip()
            
            # Extract JSON from Markdown block if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content and "{" in content:
                # More aggressive extraction for any code block with JSON
                parts = content.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        content = part.strip()
                        break
            
            # If starts with non-JSON characters, find the JSON object using bracket counting
            if not content.startswith("{"):
                json_start = content.find("{")
                if json_start >= 0:
                    brace_count = 0
                    json_end = -1
                    for i in range(json_start, len(content)):
                        if content[i] == "{":
                            brace_count += 1
                        elif content[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    if json_end > 0:
                        content = content[json_start:json_end]
            
            # --- IMPROVEMENT: SANITIZE INVISIBLE CHARACTERS ---
            # Removes null bytes and control characters that break json.loads
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            
            # Parse JSON
            parsed = json.loads(content)
            
            # Validate that it's a dictionary with expected structure
            if isinstance(parsed, dict) and len(parsed) > 0:
                return parsed
            else:
                logging.warning(f"Parsed JSON but got unexpected structure: {parsed}")
                if attempt < max_retries - 1:
                    continue
                    
        except json.JSONDecodeError as e:
            logging.error(f"JSON Parse Error (attempt {attempt+1}/{max_retries}): {e}. Content: {content[:200] if 'content' in locals() else 'N/A'}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            logging.error(f"LLM Query Failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    # All retries failed
    return {}
def calculate_bleu_score(reference: str, candidate: str, max_n: int = 4) -> Dict[str, Any]:
    """Calculate BLEU score for code comparison."""
    def tokenize(code):
        return re.findall(r'\w+|[^\w\s]', code or "")

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if not ref_tokens or not cand_tokens:
        return {"bleu_score": 0.0}
    
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1))
        cand_ngrams = Counter(tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1))
        
        matches = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)
    
    # Geometric mean
    if all(p > 0 for p in precisions):
        geo_mean = exp(sum(np.log(p) for p in precisions) / max_n)
    else:
        geo_mean = 0.0
        
    # Brevity penalty
    ratio = len(cand_tokens) / len(ref_tokens) if len(ref_tokens) > 0 else 0
    bp = 1.0 if ratio > 1.0 else exp(1 - 1/ratio) if ratio > 0 else 0.0
    
    return {"bleu_score": round(bp * geo_mean, 4)}

def check_syntax_validity(code: str) -> bool:
    """Check Rust syntax validity using rustfmt."""
    if not code or len(code.strip()) < 10: return False
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
            f.write(code)
            fname = f.name
            
        result = subprocess.run(
            ['rustfmt', '--check', fname], 
            capture_output=True, timeout=5
        )
        os.unlink(fname)
        # return code 0 (valid) or 1 (formatting needed but valid)
        return result.returncode <= 1
    except:
        return False

# ==============================
# UNIFIED EVALUATION FUNCTIONS
# ==============================

def evaluate_debugging_combined(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates 'Bug Fixed' and 'Root Cause Identification' in a single API call.
    """
    buggy_code = item.get('buggy_code', '')
    context_code = item.get('context_code', '')
    prompt = item.get('prompt', '')
    model_output = item.get('model_output', '')
    expected_output = item.get('expected_output', '')
    
    if not model_output:
        return {"bug_fixed": 0.0, "root_cause_identified": 0.0, "bug_fixed_reasoning": "No Output"}

    system_prompt = f"""Evaluate this Rust debugging solution and respond with ONLY a JSON object. Strictly follow the JSON format.

[BUGGY CODE]
{buggy_code[:1000]}

[TASK]
{prompt[:500]}

[REFERENCE FIX]
{expected_output[:1000]}

[MODEL SOLUTION]
{model_output[:1000]}

Rate on scale 1-5:

1. bug_fixed: Does the solution correctly fix the bug? Consider logic and approach, ignore minor syntax.

2. root_cause_identified: Does it explain WHY the bug occurs with technical accuracy?

Respond ONLY with this JSON (no other text):
{{
  "bug_fixed": <number 1-5>,
  "bug_fixed_reasoning": "<brief explanation>",
  "root_cause_identified": <number 1-5>,
  "root_cause_reasoning": "<brief explanation>"
}}"""
    result = query_model_with_reasoning(system_prompt)
    
    # Fallback keys if LLM varies response slightly
    return {
        "bug_fixed": float(result.get("bug_fixed", 0.0)),
        "bug_fixed_reasoning": result.get("bug_fixed_reasoning", ""),
        "root_cause_identified": float(result.get("root_cause_identified", 0.0)),
        "root_cause_reasoning": result.get("root_cause_reasoning", "")
    }

def evaluate_generation_combined(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Functional Correctness, Task Completion, and Syntax in a single API call.
    """
    prompt = item.get('prompt', '')
    constraints = item.get('constraints', '')
    context = item.get('context_code', '')
    model_output = item.get('model_output', '')
    expected = item.get('expected_output', '')

    if not model_output:
        return {"functional_correctness": 0.0, "task_completion": 0.0, "syntax_validity": 0.0}

    system_prompt = f"""Evaluate the generated Rust code on three metrics. and Strictly follow the JSON format.

[PROMPT]
{prompt}

[CONSTRAINTS]
{constraints}

[CONTEXT]
{context}

[REFERENCE]
{expected}

[GENERATED CODE]
{model_output}

METRICS:
1. FUNCTIONAL_CORRECTNESS (1-5): Does the code perform the logic required?

Focus: Conceptual & Logical Soundness
Evaluate whether the code's underlying logic and algorithmic approach are correct, assuming proper implementation.
What to Evaluate:

Is the algorithmic approach valid for the problem?
Does the control flow make logical sense?
Are the core operations and transformations conceptually correct?
Does the solution demonstrate proper understanding of the problem domain?

What to Ignore:

Missing import statements
Minor syntax typos or formatting issues
Code style preferences

Scoring Guidance:
Assess the conceptual correctness of the solution's logic. A sound solution uses valid algorithms and demonstrates clear understanding of problem-solving approach. Partially correct solutions may have the right general idea but contain logical gaps or minor bugs. Fundamentally flawed solutions show misunderstanding of the problem's logical requirements.

2. TASK_COMPLETION (1-5): Did it follow all constraints and complete the task?

Focus: Requirement Satisfaction & Completeness
Evaluate whether the solution addresses all specified requirements, constraints, and scope elements from the prompt.
What to Evaluate:

Are all explicit requirements addressed?
Are specified constraints respected (e.g., input formats, boundaries, dependencies)?
Is the solution complete, or just a partial stub?
Does it align with the provided context and use case?
Are edge cases mentioned in the prompt handled?

What to Ignore:

Optimization or elegance (unless explicitly requested)
Implementation efficiency
Code readability or documentation quality

Scoring Guidance:
Evaluate comprehensiveness against the stated requirements. Complete solutions address every requirement and respect all constraints. Partial solutions may miss secondary requirements or handle only the primary use case. Incomplete solutions ignore major requirements or deliver something tangential to the request.

3. SYNTAX_VALIDITY (0 or 1): 1=Valid Rust, 0=Invalid/Hallucinated.

CRITICAL: Return ONLY the JSON object below. Do not include any explanations, code examples, or text outside the JSON.

OUTPUT JSON ONLY:
{{
  "functional_correctness": <float>,
  "functional_reasoning": "<string>",
  "task_completion": <float>,
  "task_completion_reasoning": "<string>",
  "syntax_validity": <int>,
  "syntax_reasoning": "<string>"
}}
"""
    result = query_model_with_reasoning(system_prompt)
    
    return {
        "functional_correctness": float(result.get("functional_correctness", 0.0)),
        "functional_reasoning": result.get("functional_reasoning", ""),
        "task_completion": float(result.get("task_completion", 0.0)),
        "task_completion_reasoning": result.get("task_completion_reasoning", ""),
        "syntax_validity": float(result.get("syntax_validity", 0.0)),
        "syntax_valid": float(result.get("syntax_validity", 0)) > 0.5
    }

def evaluate_understanding_combined(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates Comprehension Accuracy and Context Awareness in a single API call.
    """
    code = item.get('context_code', '')
    prompt = item.get('prompt', '')
    file_path = item.get('file_path', '')
    model_output = item.get('model_output', '')
    expected = item.get('expected_output', '')

    if not model_output:
        return {"comprehension_accuracy": 0.0, "repo_context_awareness": 0.0}

    system_prompt = f"""Evaluate the code explanation considering as developer's explanation of code. Strictly follow the JSON format.

[FILE] {file_path}

[CODE]
{code}

[QUESTION]
{prompt}

[REFERENCE ANSWER]
{expected}

[MODEL ANSWER]
{model_output}

METRICS:
1. COMPREHENSION_ACCURACY (1-5): Is the technical explanation correct?
Focus: Technical Correctness & Code Fidelity
Evaluate whether the explanation is technically accurate and faithfully represents the actual code provided.
What to Evaluate:

Is the control flow explained correctly?
Are type systems and data structures described accurately?
Are language-specific concepts (e.g., Option, Result, Traits, ownership) used correctly?
Does the explanation reflect what the code actually does?
Are claims about functions, variables, and behaviors verifiable in the provided code?

Critical Check - Hallucination Detection:
Strictly penalize any claims about:

Functions or methods that don't exist in the code
Variables or fields not present
Behaviors or logic not supported by the actual implementation
Features or capabilities the code doesn't have

Scoring Guidance:
Accurate explanations demonstrate precise understanding of the code's mechanics and stay grounded in what's actually present. Partially accurate explanations may have the general idea right but use imprecise terminology or miss important details. Inaccurate explanations contain technical errors, misrepresent the code's behavior, or make unsupported claims about functionality.

2. REPO_CONTEXT_AWARENESS (1-5): Does it show knowledge of Hyperswitch/Rust domain?
Focus: Domain Knowledge & System-Specific Understanding
Evaluate whether the explanation demonstrates appropriate awareness of the code's context within the Hyperswitch payment system.
What to Evaluate:

Does the explanation recognize relevant payment domain concepts (e.g., connectors, routing, payment flows, validation)?
Are Hyperswitch-specific architectural patterns or components identified when present?
Is domain terminology used appropriately where applicable?
Does the explanation connect the code to its role in the broader system when relevant?

Context Sensitivity:

Domain-relevant code: Should reference payment concepts, system architecture, or business logic context
Generic utility code: May appropriately receive generic explanations (e.g., string parsing, data structures)
Boundary cases: Code that interfaces between generic and domain-specific layers should acknowledge both aspects

Scoring Guidance:
Strong context awareness integrates relevant domain knowledge and recognizes how the code fits within Hyperswitch's architecture. Moderate awareness provides technically correct explanations but treats domain-specific code as generic, missing opportunities to clarify business purpose. Weak awareness misidentifies the system context or applies incorrect domain assumptions.

CRITICAL: Return ONLY the JSON object below. Do not include any explanations, code examples, or text outside the JSON.



OUTPUT JSON ONLY:
{{
  "comprehension_accuracy": <float>,
  "comprehension_reasoning": "<string>",
  "repo_context_awareness": <float>,
  "context_reasoning": "<string>"
}}
"""
    result = query_model_with_reasoning(system_prompt)
    
    return {
        "comprehension_accuracy": float(result.get("comprehension_accuracy", 0.0)),
        "comprehension_reasoning": result.get("comprehension_reasoning", ""),
        "repo_context_awareness": float(result.get("repo_context_awareness", 0.0)),
        "context_reasoning": result.get("context_reasoning", "")
    }

# ==============================
# DISPATCHER FUNCTIONS
# ==============================

def evaluate_debugging_task(item: Dict[str, Any]) -> Dict[str, Any]:
    # 1. LLM Evaluation (Unified)
    eval_result = evaluate_debugging_combined(item)
    
    # 2. BLEU Score (Deterministic)
    bleu = calculate_bleu_score(item.get('expected_output', ''), item.get('model_output', ''))
    
    # 3. Merge
    return {**item, **eval_result, **bleu}

def evaluate_generation_task(item: Dict[str, Any]) -> Dict[str, Any]:
    # 1. LLM Evaluation (Unified)
    eval_result = evaluate_generation_combined(item)
    
    # 2. BLEU Score
    bleu = calculate_bleu_score(item.get('expected_output', ''), item.get('model_output', ''))
    
    return {**item, **eval_result, **bleu}

def evaluate_understanding_task(item: Dict[str, Any]) -> Dict[str, Any]:
    # 1. LLM Evaluation (Unified)
    eval_result = evaluate_understanding_combined(item)
    
    # 2. BLEU Score
    bleu = calculate_bleu_score(item.get('expected_output', ''), item.get('model_output', ''))
    
    return {**item, **eval_result, **bleu}

# ==============================
# PROCESSING LOGIC (Pass@K & Parallel)
# ==============================

def calculate_pass_at_k(results_list: List[Dict[str, Any]], k_values: List[int], task_type: str) -> Dict[str, Any]:
    """Calculate Pass@K for metrics available in the task type."""
    if not results_list: return {}
    
    metrics_map = {
        'debug': ['bug_fixed', 'root_cause_identified', 'bleu_score'],
        'generation': ['functional_correctness', 'task_completion', 'syntax_validity', 'bleu_score'],
        'understanding': ['comprehension_accuracy', 'repo_context_awareness', 'bleu_score']
    }
    
    # Identify relevant metrics based on task string
    relevant_metrics = []
    for key, metrics in metrics_map.items():
        if key in task_type.lower():
            relevant_metrics = metrics
            break
            
    pass_at_k = {}
    for k in k_values:
        if k > len(results_list): continue
        pass_at_k[f"pass@{k}"] = {}
        
        for metric in relevant_metrics:
            # Sort scores descending
            scores = sorted([r.get(metric, 0) for r in results_list], reverse=True)
            # Average the top K
            pass_at_k[f"pass@{k}"][metric] = round(sum(scores[:k]) / k, 3)
            
    return pass_at_k

def evaluate_item(item: Dict[str, Any], pass_k_config: List[int]) -> Optional[Dict[str, Any]]:
    """Determine how to evaluate a single item (Single vs Multi-output)."""
    task_type = item.get('task_type', '').lower()
    
    # Detect outputs (output1, output2...)
    output_keys = sorted([k for k in item.keys() if k.startswith('output') and k[6:].isdigit()], key=lambda x: int(x[6:]))
    
    # If no numbered outputs, check for 'model_output'
    if not output_keys:
        if 'model_output' in item:
            output_keys = ['model_output']
        else:
            return None

    # Evaluate all outputs
    output_results = []
    for key in output_keys:
        temp_item = item.copy()
        temp_item['model_output'] = item[key]
        
        if 'debug' in task_type:
            res = evaluate_debugging_task(temp_item)
        elif 'gen' in task_type:
            res = evaluate_generation_task(temp_item)
        elif 'understanding' in task_type:
            res = evaluate_understanding_task(temp_item)
        else:
            return None
        
        output_results.append(res)

    # Consolidate Results - Store each output's scores separately
    combined = {
        "id": item.get('id'),
        "task_type": task_type,
        "file_path": item.get("file_path", ""),
        "prompt": item.get("prompt", ""),
        "expected_output": item.get("expected_output", ""),
    }
    
    # Extract common metadata from first result (same for all outputs)
    if output_results:
        first_result = output_results[0]
        # Add error_handling_pattern if it exists
        if 'error_handling_pattern' in first_result:
            combined["error_handling_pattern"] = first_result['error_handling_pattern']
        # Add metadata if it exists
        if 'metadata' in first_result:
            combined["metadata"] = first_result['metadata']
    
    # Store individual output evaluations
    combined["outputs"] = {}
    
    # Define keys to exclude (these are either in combined already or shouldn't be in metrics)
    exclude_keys = {
        'id', 'task_type', 'file_path', 'prompt', 'expected_output', 
        'model_output', 'context_code', 'buggy_code', 'constraints',
        'error_handling_pattern', 'metadata',
        # Also exclude output1, output2, output3 text that gets stored in results
        'output1', 'output2', 'output3', 'output4', 'output5'
    }
    
    # Store each output's scores individually
    for i, res in enumerate(output_results):
        output_key = output_keys[i]
        # Extract only the evaluation metrics (scores and reasoning)
        eval_metrics = {k: v for k, v in res.items() if k not in exclude_keys}
        
        combined["outputs"][output_key] = {
            "output_text": item[output_key],
            "metrics": eval_metrics
        }
    
    # Add Pass@K if multiple outputs
    if len(output_results) > 1:
        combined["pass_at_k"] = calculate_pass_at_k(output_results, pass_k_config, task_type)
        
    return combined

# ==============================
# CHECKPOINT & RESUME LOGIC
# ==============================

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint data if it exists."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_checkpoint(checkpoint_path: str, processed_ids: set, results: List[Dict], run_id: str, model_name: Optional[str] = None):
    """Save checkpoint data."""
    checkpoint_data = {
        "run_id": run_id,
        "model_name": model_name,
        "processed_ids": list(processed_ids),
        "total_processed": len(results),
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def save_result_immediately(result: Dict[str, Any], output_file: str, lock: threading.Lock, model_name: Optional[str] = None):
    """Save a single result immediately to the output file."""
    with lock:
        try:
            # Read existing results
            existing_data: Dict[str, Any] = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        existing_data = json.load(f)
                        # Handle legacy format (just a list)
                        if isinstance(existing_data, list):
                            existing_data = {"results": existing_data}
                except:
                    existing_data = {}
            
            # Ensure structure
            if "model_name" not in existing_data and model_name:
                existing_data["model_name"] = model_name
            if "results" not in existing_data:
                existing_data["results"] = []
            
            # Append new result
            existing_data["results"].append(result)
            
            # Write back
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save result immediately: {e}")

# ==============================
# SUMMARY DISPLAY
# ==============================

def display_final_summary(result_file: str):
    """Display task-type-grouped average scores."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            results = data.get('results', [])
        
        if not results:
            print("\n⚠️  No results to summarize")
            return
        
        # Group by task type
        by_task = defaultdict(list)
        for r in results:
            task_type = r.get('task_type', 'unknown')
            by_task[task_type].append(r)
        
        print(f"\n{'='*70}")
        print("📊 EVALUATION SUMMARY BY TASK TYPE")
        print(f"{'='*70}")
        
        for task_type, items in sorted(by_task.items()):
            print(f"\n{'─'*70}")
            print(f"📋 {task_type.upper().replace('_', ' ')} ({len(items)} items)")
            print(f"{'─'*70}")
            
            # Collect all metrics from all outputs
            if 'debug' in task_type.lower():
                bug_scores = []
                root_scores = []
                bleu_scores = []
                
                for item in items:
                    for output_key, output_data in item.get('outputs', {}).items():
                        metrics = output_data.get('metrics', {})
                        if 'bug_fixed' in metrics:
                            bug_scores.append(metrics['bug_fixed'])
                        if 'root_cause_identified' in metrics:
                            root_scores.append(metrics['root_cause_identified'])
                        if 'bleu_score' in metrics:
                            bleu_scores.append(metrics['bleu_score'])
                
                if bug_scores:
                    print(f"  🐛 Bug Fixed:                 {sum(bug_scores)/len(bug_scores):.3f} (avg of {len(bug_scores)} outputs)")
                if root_scores:
                    print(f"  🔍 Root Cause Identified:     {sum(root_scores)/len(root_scores):.3f} (avg of {len(root_scores)} outputs)")
                if bleu_scores:
                    print(f"  📝 BLEU Score:                {sum(bleu_scores)/len(bleu_scores):.4f} (avg of {len(bleu_scores)} outputs)")
                    
            elif 'gen' in task_type.lower():
                func_scores = []
                task_scores = []
                syntax_scores = []
                bleu_scores = []
                
                for item in items:
                    for output_key, output_data in item.get('outputs', {}).items():
                        metrics = output_data.get('metrics', {})
                        if 'functional_correctness' in metrics:
                            func_scores.append(metrics['functional_correctness'])
                        if 'task_completion' in metrics:
                            task_scores.append(metrics['task_completion'])
                        if 'syntax_validity' in metrics:
                            syntax_scores.append(metrics['syntax_validity'])
                        if 'bleu_score' in metrics:
                            bleu_scores.append(metrics['bleu_score'])
                
                if func_scores:
                    print(f"  ⚙️  Functional Correctness:    {sum(func_scores)/len(func_scores):.3f} (avg of {len(func_scores)} outputs)")
                if task_scores:
                    print(f"  ✅ Task Completion:           {sum(task_scores)/len(task_scores):.3f} (avg of {len(task_scores)} outputs)")
                if syntax_scores:
                    print(f"  📐 Syntax Validity:           {sum(syntax_scores)/len(syntax_scores):.3f} (avg of {len(syntax_scores)} outputs)")
                if bleu_scores:
                    print(f"  📝 BLEU Score:                {sum(bleu_scores)/len(bleu_scores):.4f} (avg of {len(bleu_scores)} outputs)")
                    
            elif 'understanding' in task_type.lower() or 'exp' in task_type.lower():
                comp_scores = []
                context_scores = []
                bleu_scores = []
                
                for item in items:
                    for output_key, output_data in item.get('outputs', {}).items():
                        metrics = output_data.get('metrics', {})
                        if 'comprehension_accuracy' in metrics:
                            comp_scores.append(metrics['comprehension_accuracy'])
                        if 'repo_context_awareness' in metrics:
                            context_scores.append(metrics['repo_context_awareness'])
                        if 'bleu_score' in metrics:
                            bleu_scores.append(metrics['bleu_score'])
                
                if comp_scores:
                    print(f"  🎯 Comprehension Accuracy:    {sum(comp_scores)/len(comp_scores):.3f} (avg of {len(comp_scores)} outputs)")
                if context_scores:
                    print(f"  🏗️  Repo Context Awareness:    {sum(context_scores)/len(context_scores):.3f} (avg of {len(context_scores)} outputs)")
                if bleu_scores:
                    print(f"  📝 BLEU Score:                {sum(bleu_scores)/len(bleu_scores):.4f} (avg of {len(bleu_scores)} outputs)")
            
            # Display Pass@K if available
            if items and 'pass_at_k' in items[0]:
                print(f"\n  Pass@K Metrics:")
                pass_k_data = items[0].get('pass_at_k', {})
                for k_name, k_scores in sorted(pass_k_data.items()):
                    print(f"    {k_name}:")
                    for metric, score in k_scores.items():
                        print(f"      {metric}: {score:.3f}")
        
        print(f"\n{'='*70}")
        print(f"✅ Total items evaluated: {len(results)}")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n⚠️  Could not display summary: {e}")

# ==============================
# MAIN EXECUTION
# ==============================

def get_user_input():
    print("=" * 60)
    print("🚀 UNIFIED CODE EVALUATOR (Optimized)")
    print("=" * 60)
    
    input_file = input("Input JSON file path: ").strip()
    if not os.path.exists(input_file):
        print("❌ File not found.")
        return None

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'results' in data: data = data['results']
            if not isinstance(data, list): data = [data]
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    # Ask for model name
    model_name = input("Model name being evaluated (e.g., claude-3.5-sonnet, gpt-4, deepseek-coder): ").strip()
    if not model_name:
        print("❌ Model name is required.")
        return None
    
    # Create model_results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "model_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate run ID
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check for multi-output
    is_multi = any('output2' in x for x in data[:1])
    pass_k = []
    if is_multi:
        k_in = input("Enter Pass@K values (e.g., 1,3,5): ").strip()
        pass_k = [int(x) for x in k_in.split(',') if x.isdigit()] or [1]
    else:
        pass_k = [1]

    # Concurrency
    workers_in = input(f"Parallel Workers [Default {DEFAULT_PARALLEL_WORKERS}]: ").strip()
    workers = int(workers_in) if workers_in.isdigit() else DEFAULT_PARALLEL_WORKERS
    
    # Check for checkpoint
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    checkpoint = load_checkpoint(checkpoint_path)
    
    resume = False
    if checkpoint and checkpoint.get('processed_ids'):
        resume_input = input(f"\n📋 Found checkpoint with {len(checkpoint['processed_ids'])} processed items. Resume? (y/n): ").strip().lower()
        resume = resume_input == 'y'
        if resume:
            run_id = checkpoint.get('run_id', run_id)
            # Try to get model name from checkpoint
            if not model_name and checkpoint.get('model_name'):
                model_name = checkpoint.get('model_name')

    return {
        "data": data,
        "output_dir": output_dir,
        "pass_k": pass_k,
        "workers": workers,
        "run_id": run_id,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "resume": resume,
        "processed_ids": set(checkpoint.get('processed_ids', [])) if resume else set()
    }

def main():
    logger = setup_logging()
    config = get_user_input()
    if not config: return

    # Output file with run ID
    result_file = os.path.join(config['output_dir'], f"eval_{config['run_id']}.json")
    
    # Filter items to process
    items_to_process = [
        item for item in config['data'] 
        if item.get('id') not in config['processed_ids']
    ]
    
    if config['resume']:
        print(f"📂 Resuming run: {config['run_id']}")
        print(f"🤖 Model: {config['model_name']}")
        print(f"⏭️  Skipping {len(config['processed_ids'])} already processed items")
        print(f"🔄 Processing {len(items_to_process)} remaining items")
    else:
        print(f"🆕 Starting new run: {config['run_id']}")
        print(f"🤖 Model being evaluated: {config['model_name']}")
        print(f"📊 Total items to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("✅ All items already processed!")
        return
    
    print(f"📁 Output file: {result_file}")
    print(f"⚡ Using {config['workers']} parallel workers\n")
    
    start_time = time.time()
    
    # Thread-safe operations
    save_lock = threading.Lock()
    processed_ids = config['processed_ids'].copy()
    completed_count = 0

    def process_and_save(item):
        nonlocal completed_count
        try:
            result = evaluate_item(item, config['pass_k'])
            if result:
                # Save immediately with model name
                save_result_immediately(result, result_file, save_lock, config['model_name'])
                
                # Update checkpoint
                item_id = item.get('id')
                if item_id:
                    processed_ids.add(item_id)
                
                completed_count += 1
                
                # Save checkpoint every 5 items
                if completed_count % 5 == 0:
                    save_checkpoint(
                        config['checkpoint_path'],
                        processed_ids,
                        [],
                        config['run_id'],
                        config['model_name']
                    )
                
                return result
        except Exception as e:
            logger.error(f"Failed to process item {item.get('id', 'unknown')}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=config['workers']) as executor:
        futures = {executor.submit(process_and_save, item): item for item in items_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", unit="item"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Task execution failed: {e}")

    # Final checkpoint save
    save_checkpoint(
        config['checkpoint_path'],
        processed_ids,
        [],
        config['run_id'],
        config['model_name']
    )
    
    # Clean up checkpoint if all done
    if len(processed_ids) >= len(config['data']):
        try:
            os.remove(config['checkpoint_path'])
            print("\n🗑️  Checkpoint file removed (all items processed)")
        except:
            pass
    
    duration = round(time.time() - start_time, 2)
    print(f"\n✅ Completed {completed_count} items in {duration}s")
    print(f"📄 Results saved to: {result_file}")
    print(f"📊 Total processed in this run: {len(processed_ids)}/{len(config['data'])}")
    print(f"🤖 Model evaluated: {config['model_name']}")
    
    # Display task-type-grouped summary
    display_final_summary(result_file)

if __name__ == "__main__":
    main()