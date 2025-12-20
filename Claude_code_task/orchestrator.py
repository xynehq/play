from pathlib import Path
import sys
import shutil
import subprocess
import json
import re
import os

# Add current directory to path to import existing modules
AUTOMATION_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(AUTOMATION_DIR))

from prompt_builder import load_templates, fill_placeholders
from llm_client import LLMClient
from pr_setup import setup_pr, PRSetupError


def run_pr(pr_number: str, models: list = None, skip_setup: bool = False):
    """
    Main orchestrator function for processing a PR.
    
    Args:
        pr_number: The PR number to process
        models: List of model names to run (e.g., ["glm-full", "minimax-m2"]). 
                If None, defaults to ["minimaxai/minimax-m2"]
        skip_setup: If True, skip Stage 0 (PR setup). Use this if PR setup was already done manually.
    """
    if models is None:
        models = ["minimaxai/minimax-m2"]
    
    print(f"[PR {pr_number}] Starting pipeline")
    print(f"[PR {pr_number}] Models to run: {models}")
    
    # Stage 0: Automated PR Setup
    if not skip_setup:
        try:
            repo_dir = AUTOMATION_DIR.parent  # claude-work2 directory
            pr_dir = AUTOMATION_DIR / "pr"
            setup_metadata = setup_pr(pr_number, repo_dir, pr_dir)
            print(f"[PR {pr_number}] Stage 0 completed successfully")
        except PRSetupError as e:
            print(f"[PR {pr_number}] CRITICAL ERROR in Stage 0 (PR Setup):")
            print(f"{str(e)}")
            sys.exit(1)
    else:
        print(f"[PR {pr_number}] Skipping Stage 0 (PR setup) - using existing artifacts")
    
    # Load PR inputs (relative to automation_2 directory)
    pr_desc_file = AUTOMATION_DIR / "pr" / f"task_pr_{pr_number}.md"
    human_diff_file = AUTOMATION_DIR / "pr" / "original_changes.diff"
    
    if not pr_desc_file.exists():
        print(f"[PR {pr_number}] Error: PR description not found at {pr_desc_file}")
        sys.exit(1)
    
    if not human_diff_file.exists():
        print(f"[PR {pr_number}] Error: Human diff not found at {human_diff_file}")
        sys.exit(1)
    
    # Read PR description and human diff
    pr_text = pr_desc_file.read_text(encoding="utf-8")
    human_patch = human_diff_file.read_text(encoding="utf-8")
    
    # Extract PR metadata
    pr_metadata = _extract_pr_metadata(pr_text)
    
    # Load templates (relative to automation_2 directory)
    templates = load_templates(AUTOMATION_DIR / "Prompt.md")
    
    # Generate human approach summary once (used for all models)
    # Prompt generation ALWAYS uses minimaxai/minimax-m2 (strict requirement)
    print(f"[PR {pr_number}] Generating human approach summary...")
    human_approach_client = LLMClient(
        model="minimaxai/minimax-m2",  # Always use minimaxai/minimax-m2 for prompt generation
        api_key=" ",
        base_url="https://grid.ai.juspay.net"
    )
    human_approach_summary = _generate_human_approach(templates, pr_metadata, human_patch, human_approach_client)
    
    # Create PR-level runs directory for shared artifacts
    pr_run_dir = AUTOMATION_DIR / "runs" / f"PR-{pr_number}"
    pr_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save human approach summary to PR-level directory (shared across all models)
    human_approach_file = pr_run_dir / "human_approach.txt"
    human_approach_file.write_text(human_approach_summary, encoding="utf-8")
    print(f"[PR {pr_number}] Human approach summary saved to {human_approach_file}")
    
    # Run for each model
    all_results = {}
    for model in models:
        print(f"\n{'='*80}")
        print(f"[PR {pr_number}] Running with model: {model}")
        print(f"{'='*80}\n")
        
        # Create model-specific run directory (relative to automation_2 directory)
        model_run_dir = AUTOMATION_DIR / "runs" / f"PR-{pr_number}" / model.replace("/", "-")
        model_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create input directory and copy inputs
        input_dir = model_run_dir / "input"
        input_dir.mkdir(exist_ok=True)
        shutil.copy2(pr_desc_file, input_dir / "pr.md")
        shutil.copy2(human_diff_file, input_dir / "human.diff")
        # Save human approach summary to model-specific input directory
        (input_dir / "human_approach.txt").write_text(human_approach_summary, encoding="utf-8")
        
        # Create LLM client for prompt generation (ALWAYS uses minimaxai/minimax-m2)
        # Prompt generation model ≠ execution model (strict requirement)
        prompt_gen_client = LLMClient(
            model="minimaxai/minimax-m2",  # Always use minimaxai/minimax-m2 for prompt generation
            api_key=" ",
            base_url="https://grid.ai.juspay.net"
        )
        
        # Create LLM client for evaluation (uses execution model)
        eval_client = LLMClient(
            model=model,
            api_key=" ",
            base_url="https://grid.ai.juspay.net"
        )
        
        # Run attempts loop
        final_result = _run_attempts_loop(
            pr_number, model_run_dir, templates, pr_text, human_patch, 
            prompt_gen_client, eval_client, pr_metadata, human_approach_summary, model
        )
        
        # Save model-specific final.json
        final_file = model_run_dir / "final.json"
        final_file.write_text(json.dumps(final_result, indent=2), encoding="utf-8")
        
        all_results[model] = final_result
        
        print(f"[PR {pr_number}] Model {model} - Final verdict: {final_result['final_verdict']}")
        print(f"[PR {pr_number}] Model {model} - Attempts: {final_result['attempts']}")
    
    # Generate comparison report if multiple models
    if len(models) > 1:
        comparison_file = AUTOMATION_DIR / "runs" / f"PR-{pr_number}" / "model_comparison.json"
        comparison_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n[PR {pr_number}] Model comparison saved to {comparison_file}")
    
    # Overall verdict (PASS if any model passed)
    overall_verdict = "PASS" if any(r['final_verdict'] == 'PASS' for r in all_results.values()) else "FAIL"
    print(f"\n[PR {pr_number}] Overall verdict: {overall_verdict}")
    print(f"[PR {pr_number}] Pipeline complete")
    
    # Exit with appropriate code
    exit_code = 0 if overall_verdict == 'PASS' else 1
    sys.exit(exit_code)


def _extract_pr_metadata(pr_text: str) -> dict:
    """Extract PR title, description, and issue from task file."""
    lines = pr_text.splitlines()
    title = ""
    description_lines = []
    issue_description = ""
    in_issue_section = False
    in_description_section = False
    
    for line in lines:
        if not title and line.startswith("# Task:"):
            title = line.lstrip("# Task:").strip()
            continue
        if line.startswith("## Issue #") or "Issue #" in line:
            in_issue_section = True
            in_description_section = False
            continue
        if line.startswith("## Description") or line.startswith("## PR Information"):
            in_description_section = True
            in_issue_section = False
            continue
        if line.startswith("## Requirements"):
            break
        if in_issue_section and line.strip():
            issue_description += line + "\n"
        elif not in_issue_section and (in_description_section or not line.startswith("##")):
            description_lines.append(line)
    
    return {
        "PR_TITLE": title or "Unknown title",
        "PR_DESCRIPTION": "\n".join(description_lines).strip() or "No description provided.",
        "PR_ISSUE_DESCRIPTION": issue_description.strip() or "No issue description provided.",
    }


def _generate_human_approach(templates, pr_metadata: dict, human_patch: str, client: LLMClient) -> str:
    """Generate human approach summary using the human_approach template."""
    replacements = {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_DESCRIPTION": pr_metadata["PR_DESCRIPTION"],
        "GROUND_TRUTH_DIFF_TEXT": human_patch,
    }
    meta_prompt = fill_placeholders(templates.human_approach, replacements)
    return client.run(meta_prompt).strip()


def _build_replacements(pr_text: str, human_patch: str, pr_metadata: dict = None) -> dict:
    """Build replacements for template filling."""
    if pr_metadata is None:
        pr_metadata = _extract_pr_metadata(pr_text)
    
    return {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_DESCRIPTION": pr_metadata["PR_DESCRIPTION"],
        "GROUND_TRUTH_DIFF_TEXT": human_patch,
    }


def _validate_git_state():
    """Validate git repository state before Claude execution.
    
    Pre-execution git sanity checks:
    - Clean working tree
    - Correct base commit (merge^1)
    
    Raises:
        RuntimeError: If repository state is unsafe
    """
    claude_work_dir = AUTOMATION_DIR.parent
    
    # Check for clean working tree
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    if status_result.stdout.strip():
        raise RuntimeError(
            f"Git working tree is not clean. Repository state is unsafe.\n"
            f"Working tree status:\n{status_result.stdout}"
        )
    
    # Verify we're on a valid commit (not detached HEAD or similar issues)
    rev_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    if rev_result.returncode != 0:
        raise RuntimeError(
            f"Failed to verify git HEAD. Repository state is unsafe.\n"
            f"Error: {rev_result.stderr}"
        )
    
    print("  [Stage 3] Git state validated: clean working tree")


def _reset_repository():
    """Reset repository to clean state before Claude execution."""
    print("  [Stage 3] Resetting repository")
    
    # Get the claude-work directory (parent of automation_2)
    claude_work_dir = AUTOMATION_DIR.parent
    
    # Reset to HEAD (run from claude-work directory)
    reset_result = subprocess.run(
        ["git", "reset", "--hard"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    if reset_result.returncode != 0:
        raise RuntimeError(
            f"Git reset failed. Repository state is unsafe.\n"
            f"Error: {reset_result.stderr}"
        )
    
    # Clean untracked files
    clean_result = subprocess.run(
        ["git", "clean", "-fd"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    if clean_result.returncode != 0:
        raise RuntimeError(
            f"Git clean failed. Repository state is unsafe.\n"
            f"Error: {clean_result.stderr}"
        )
    
    # Re-validate that we have a clean state after reset
    _validate_git_state()


def _map_model_name(model: str) -> str:
    """Map CLI model name to Claude Code execution model name.
    
    This is the explicit, validated execution-model mapping.
    Rules:
    - run_pr --models glm-full → execution model = "glm-full"
    - run_pr --models minimax-m2 → execution model = "minimaxai/minimax-m2"
    
    Raises:
        ValueError: If model name is unknown or unmapped
    """
    # Explicit model mapping - DO NOT guess or default
    MODEL_MAPPING = {
        "glm-full": "glm-full",
        "minimax-m2": "minimaxai/minimax-m2",
        "minimaxai/minimax-m2": "minimaxai/minimax-m2",  # Allow already-mapped names
        # Add more models here as needed
    }
    
    if model not in MODEL_MAPPING:
        raise ValueError(
            f"Unknown model name: '{model}'. "
            f"Valid models: {list(MODEL_MAPPING.keys())}"
        )
    
    return MODEL_MAPPING[model]


def _validate_environment_variables():
    """Validate required environment variables for Claude Code execution.
    
    Raises:
        RuntimeError: If required environment variables are missing
    """
    required_vars = {
        "ANTHROPIC_BASE_URL": "https://grid.ai.juspay.net",
        "ANTHROPIC_AUTH_TOKEN": " "
    }
    
    # We don't check os.environ here because we inject them ourselves
    # This function exists to document the contract and can be extended if needed
    for var_name, var_value in required_vars.items():
        if not var_value:
            raise RuntimeError(
                f"Required environment variable '{var_name}' is not configured. "
                f"Claude Code execution cannot proceed."
            )


def _execute_claude_code(prompt_file: Path, stdout_file: Path, stderr_file: Path, execution_model: str):
    """Execute Claude Code non-interactively and capture output.
    
    Uses the required execution contract:
    ANTHROPIC_BASE_URL="https://grid.ai.juspay.net" \
    ANTHROPIC_AUTH_TOKEN="sk-uJfk3pIE2KcP9DoGx4UeHA" \
    claude --model "<MODEL_NAME>" --prompt-file <PROMPT_PATH>
    
    Raises:
        RuntimeError: If Claude Code execution fails
    """
    print("  [Stage 3] Running Claude Code")
    
    # Validate environment variables
    _validate_environment_variables()
    
    # Get the claude-work directory (parent of automation_2)
    claude_work_dir = AUTOMATION_DIR.parent
    
    # Map model name (minimax-m2 -> minimaxai/minimax-m2, others stay as-is)
    # This will raise ValueError if model is unknown
    mapped_model = _map_model_name(execution_model)
    print(f"  [Stage 3] Using execution model: {mapped_model}")
    
    # Set environment variables for Claude Code (explicit injection)
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "https://grid.ai.juspay.net"
    env["ANTHROPIC_AUTH_TOKEN"] = " "
    
    # Read prompt content and clean it (remove any reasoning/metadata tags)
    prompt_content = prompt_file.read_text(encoding="utf-8")
    # Remove any reasoning tags that might be in the prompt
    # Remove <think> or <think> tags and their content
    prompt_content = re.sub(r'<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>', '', prompt_content, flags=re.DOTALL | re.IGNORECASE)
    # Clean up extra whitespace
    prompt_content = re.sub(r'\n{3,}', '\n\n', prompt_content)
    prompt_content = prompt_content.strip()
    
    # If prompt is empty after cleaning, use the original (fallback)
    if not prompt_content:
        prompt_content = prompt_file.read_text(encoding="utf-8").strip()
    
    # Execute Claude Code with --model, --print, and --permission-mode to allow edits
    # Use stdin to pass prompt (more reliable for long prompts)
    # --print enables non-interactive mode, --permission-mode=acceptEdits allows file edits
    try:
        result = subprocess.run(
            ["claude", "--model", mapped_model, "--print", "--permission-mode", "acceptEdits"],
            input=prompt_content,
            cwd=claude_work_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Claude Code execution timed out after 600 seconds. "
            f"This is a CRITICAL execution failure."
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Claude Code CLI not found. Cannot execute. "
            f"Error: {str(e)}"
        )
    
    # Save stdout and stderr
    stdout_file.write_text(result.stdout, encoding="utf-8")
    stderr_file.write_text(result.stderr, encoding="utf-8")
    
    # CRITICAL: Check subprocess return code (Problem 1️⃣)
    if result.returncode != 0:
        error_msg = (
            f"Claude Code execution FAILED with return code {result.returncode}.\n"
            f"stderr: {result.stderr[:500]}\n"
            f"stdout: {result.stdout[:500]}"
        )
        print(f"  [Stage 3] CRITICAL ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    print(f"  [Stage 3] Claude Code executed successfully (return code 0)")
    print(f"  [Stage 3] Output saved to {stdout_file} and {stderr_file}")


def _capture_git_diff(diff_file: Path):
    """Capture git diff after Claude execution."""
    print("  [Stage 3] Capturing git diff")
    
    # Get the claude-work directory (parent of automation_2)
    claude_work_dir = AUTOMATION_DIR.parent
    
    # Get git diff (run from claude-work directory)
    result = subprocess.run(
        ["git", "diff"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    # Save diff
    diff_content = result.stdout
    diff_file.write_text(diff_content, encoding="utf-8")
    
    # Log explicitly if diff is empty
    if not diff_content.strip():
        print(f"  [Stage 3] Warning: Claude Code made no changes (empty diff)")
    else:
        diff_lines = len(diff_content.splitlines())
        print(f"  [Stage 3] Diff captured: {diff_lines} lines")
    
    print(f"  [Stage 3] Diff saved to {diff_file}")


def _evaluate_solution(templates, pr_metadata: dict, human_patch: str, human_approach_summary: str, 
                      claude_diff_file: Path, eval_file: Path, client: LLMClient):
    """Evaluate Claude's solution using the evaluation template."""
    print("  [Stage 4] Loading Claude's diff")
    
    # Read Claude's diff
    claude_patch = claude_diff_file.read_text(encoding="utf-8")
    
    # Build evaluation prompt with all required placeholders
    eval_replacements = {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_ISSUE_DESCRIPTION": pr_metadata["PR_ISSUE_DESCRIPTION"],
        "HUMAN_APPROACH_SUMMARY": human_approach_summary,
        "GROUND_TRUTH_DIFF_TEXT": human_patch,
        "MODEL_DIFF_TEXT": claude_patch,
    }
    
    eval_prompt = fill_placeholders(templates.evaluation, eval_replacements)
    
    print("  [Stage 4] Calling evaluation LLM")
    
    # Call LLM for evaluation
    eval_response = client.run(eval_prompt)
    
    # Parse verdict from response
    verdict_data = _parse_verdict(eval_response)
    
    # Save evaluation result
    eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
    
    print(f"  [Stage 4] Verdict: {verdict_data['verdict']}")
    print(f"  [Stage 4] Evaluation saved to {eval_file}")


def _parse_verdict(eval_response: str) -> dict:
    """Parse verdict from LLM evaluation response."""
    # Look for PASS or FAIL in the response
    verdict = "UNKNOWN"
    
    # Check for explicit verdict markers
    if re.search(r'\bPASS\b', eval_response, re.IGNORECASE):
        verdict = "PASS"
    elif re.search(r'\bFAIL\b', eval_response, re.IGNORECASE):
        verdict = "FAIL"
    
    # Extract reason (use the full response as reason)
    reason = eval_response.strip()
    
    return {
        "verdict": verdict,
        "reason": reason
    }


def _run_attempts_loop(pr_number: str, run_dir: Path, templates, pr_text: str, human_patch: str, 
                      prompt_gen_client: LLMClient, eval_client: LLMClient, pr_metadata: dict, 
                      human_approach_summary: str, execution_model: str) -> dict:
    """Run up to 3 attempts with controlled retries."""
    attempts = 0
    final_verdict = "FAIL"
    passed_on = None
    
    for attempt_num in range(1, 4):  # 1, 2, 3
        attempts += 1
        print(f"[PR {pr_number}] Attempt {attempt_num}")
        
        # Run single attempt
        verdict = _run_single_attempt(
            pr_number, run_dir, templates, pr_text, human_patch, prompt_gen_client, eval_client, 
            attempt_num, pr_metadata, human_approach_summary, execution_model
        )
        
        if verdict == "PASS":
            final_verdict = "PASS"
            passed_on = attempt_num
            print(f"[PR {pr_number}] PASS achieved on attempt {attempt_num}")
            break
        else:
            print(f"[PR {pr_number}] Attempt {attempt_num} failed")
    
    return {
        "pr": pr_number,
        "final_verdict": final_verdict,
        "attempts": attempts,
        "passed_on": passed_on
    }


def _run_single_attempt(pr_number: str, run_dir: Path, templates, pr_text: str, human_patch: str, 
                      prompt_gen_client: LLMClient, eval_client: LLMClient, attempt_num: int, 
                      pr_metadata: dict, human_approach_summary: str, execution_model: str) -> str:
    """Run a single attempt: generate prompt, run Claude, evaluate.
    
    Returns:
        Verdict string: "PASS", "FAIL", or "EXECUTION_FAILED"
    """
    attempt_dir = run_dir / f"p{attempt_num}"
    attempt_dir.mkdir(exist_ok=True)
    
    # Generate prompt for this attempt (ALWAYS uses prompt_gen_client = minimaxai/minimax-m2)
    if attempt_num == 1:
        # First attempt uses original prompt generation
        replacements = _build_replacements(pr_text, human_patch, pr_metadata)
        meta_prompt = fill_placeholders(templates.prompt1, replacements)
    else:
        # Subsequent attempts use retry templates with previous attempt data
        previous_attempt_dir = run_dir / f"p{attempt_num - 1}"
        previous_eval_file = previous_attempt_dir / "eval.json"
        previous_diff_file = previous_attempt_dir / "claude.diff"
        previous_prompt_file = previous_attempt_dir / f"p{attempt_num - 1}_prompt.txt"
        
        if not previous_eval_file.exists() or not previous_diff_file.exists():
            print(f"[PR {pr_number}] Warning: Previous attempt data not found")
            return "FAIL"
        
        # Load previous attempt data
        previous_eval = json.loads(previous_eval_file.read_text(encoding="utf-8"))
        previous_diff = previous_diff_file.read_text(encoding="utf-8")
        previous_prompt = previous_prompt_file.read_text(encoding="utf-8") if previous_prompt_file.exists() else ""
        
        # Build retry replacements
        replacements = _build_retry_replacements(
            pr_text, human_patch, previous_diff, previous_eval['reason'], 
            pr_metadata, human_approach_summary, previous_prompt
        )
        
        # Use appropriate retry template
        if attempt_num == 2:
            meta_prompt = fill_placeholders(templates.prompt2, replacements)
        else:  # attempt_num == 3
            meta_prompt = fill_placeholders(templates.prompt3, replacements)
    
    # Generate the actual prompt (using prompt_gen_client = minimaxai/minimax-m2)
    generated_prompt = prompt_gen_client.run(meta_prompt)
    
    # Save the generated prompt
    prompt_file = attempt_dir / f"p{attempt_num}_prompt.txt"
    prompt_file.write_text(generated_prompt, encoding="utf-8")
    
    print(f"[PR {pr_number}] Prompt {attempt_num} saved to {prompt_file}")
    
    # Reset repository before execution
    _reset_repository()
    
    # Execute Claude Code
    stdout_file = attempt_dir / "stdout.txt"
    stderr_file = attempt_dir / "stderr.txt"
    diff_file = attempt_dir / "claude.diff"
    
    # Execution with error handling
    try:
        _execute_claude_code(prompt_file, stdout_file, stderr_file, execution_model)
        _capture_git_diff(diff_file)
    except (RuntimeError, ValueError) as e:
        # Hard failure on execution errors (Problem 1️⃣)
        error_data = {
            "verdict": "EXECUTION_FAILED",
            "reason": f"Claude Code execution failed: {str(e)}",
            "error_type": type(e).__name__
        }
        error_file = attempt_dir / "eval.json"
        error_file.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] EXECUTION FAILED: {str(e)}")
        return "EXECUTION_FAILED"
    
    # CRITICAL: Check if claude.diff is empty or missing (Problem 3️⃣)
    if not diff_file.exists():
        failure_data = {
            "verdict": "FAIL",
            "reason": "Claude Code execution completed but no diff file was generated. This is a FAILED attempt.",
            "failure_type": "MISSING_DIFF"
        }
        failure_file = attempt_dir / "eval.json"
        failure_file.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] FAILED: claude.diff is missing")
        return "FAIL"
    
    diff_content = diff_file.read_text(encoding="utf-8")
    if not diff_content.strip():
        failure_data = {
            "verdict": "FAIL",
            "reason": "Claude Code made no changes (empty diff). This is a FAILED attempt. Silent success is NOT allowed.",
            "failure_type": "EMPTY_DIFF"
        }
        failure_file = attempt_dir / "eval.json"
        failure_file.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] FAILED: claude.diff is empty (no changes made)")
        return "FAIL"
    
    # Evaluate the solution (using eval_client = execution model)
    eval_file = attempt_dir / "eval.json"
    _evaluate_solution(templates, pr_metadata, human_patch, human_approach_summary, diff_file, eval_file, eval_client)
    
    # Return verdict
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
    return eval_data['verdict']


def _build_retry_replacements(pr_text: str, human_patch: str, previous_diff: str, previous_reason: str, 
                             pr_metadata: dict, human_approach_summary: str, previous_prompt: str = "") -> dict:
    """Build replacements for retry prompts (P2, P3) with previous attempt data."""
    return {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_DESCRIPTION": pr_metadata["PR_DESCRIPTION"],
        "GROUND_TRUTH_DIFF_TEXT": human_patch,
        "PREVIOUS_PROMPT_TEXT": previous_prompt,
        "DIFF of Previous ATTEMPT": previous_diff,
        "EVALUATION_SUMMARY_FOR_PREVIOUS_ATTEMPT": previous_reason,
        "HUMAN_APPROACH_SUMMARY": human_approach_summary,
    }
