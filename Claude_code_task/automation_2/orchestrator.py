from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional

if TYPE_CHECKING:
    from scripts.llm_client import LLMClient

# Add current directory to path to import existing modules
AUTOMATION_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(AUTOMATION_DIR))


RAW_DIFF_CHAR_LIMIT = 35_000      # Conservative for most frontier models
CONTEXT_DIFF_CHAR_LIMIT = 20_000  # Leaves room for system prompt + evaluation

# Tier multipliers for scoring (applied to base correctness score)
TIER_MULTIPLIERS = {
    1: 1.00,   # Tier-1 (Raw diff) - full precision
    2: 0.95,   # Tier-2 (Context diff) - function-level
    3: 0.85    # Tier-3 (Summary) - compressed
}

# Dual-Layer Timeout System for Evaluation Reliability
EVAL_REQUEST_TIMEOUT_SECONDS = 300   # 5 minute timeout per eval request


MAX_ATTEMPTS = 3


from scripts.tier3_diff import (
    # Re-export for backward compatibility with tests
    MAX_DIFF_CHARS,  # Maximum diff size for safe analysis (50KB)
    _normalize_code_line,
    _compare_raw_diffs_strict,
    _split_diff_into_chunks,
    _validate_tier3_summary_completeness,
    compare_chunked_with_claude,
)

# Import evaluation pipeline
from scripts.eval_pipeline import (
    _select_diff_tier,
    _generate_tier3_summary,
    _count_changed_files,
    build_tier1_eval_prompt,
    build_tier2_eval_prompt,
    build_tier3_eval_prompt,
    _evaluate_solution,
    _parse_verdict,
)

from scripts.prompt_builder import load_templates, fill_placeholders
from scripts.llm_client import LLMClient


from scripts.api_config import (
    create_client, 
    get_prompt_generation_model, 
    validate_model,
    get_anthropic_auth_key,
    get_anthropic_base_url
)
from scripts.pr_setup import setup_pr, PRSetupError
from scripts.checkpoint_manager import (
    get_resume_info,
    mark_attempt_in_progress,
    mark_attempt_completed,
    mark_attempt_failed,
    validate_attempt_artifacts
)

def run_pr(pr_number: str, models: list = None, skip_setup: bool = False):
    """
    Main orchestrator function for processing a PR.
    
    Args:
        pr_number: The PR number to process
        models: List of model names to run (CLI-driven execution models).
                This parameter is REQUIRED - execution model is NOT hardcoded.
        skip_setup: If True, skip Stage 0 (PR setup). Use this if PR setup was already done manually.
    """
    # CRITICAL: Validate models parameter (no hardcoded defaults)
    if not models:
        raise ValueError(
            "Models parameter is REQUIRED. Use --models to specify execution models. "
            "Prompt generation will ALWAYS use minimaxai/minimax-m2, "
            "execution will use the models specified via CLI."
        )
    
    # Validate each model
    for model in models:
        validate_model(model)
    
    print(f"[PR {pr_number}] Starting pipeline")
    print(f"[PR {pr_number}] CLI-driven execution models: {models}")
    print(f"[PR {pr_number}] Prompt generation model: {get_prompt_generation_model()}")
    
    # Log the model separation clearly
    print(f"[PR {pr_number}] Model separation:")
    print(f"  📝 PROMPT_GENERATION = {get_prompt_generation_model()} (hardcoded)")
    print(f"  ⚡ EXECUTION = {', '.join(models)} (CLI-driven)")
    
    # Stage 0: Automated PR Setup
    if not skip_setup:
        try:
            repo_dir = AUTOMATION_DIR.parent  # claude-work2 directory
            pr_dir = AUTOMATION_DIR / "Data" / "pr_data"
            setup_metadata = setup_pr(pr_number, repo_dir, pr_dir)
            print(f"[PR {pr_number}] Stage 0 completed successfully")
        except PRSetupError as e:
            print(f"[PR {pr_number}] CRITICAL ERROR in Stage 0 (PR Setup):")
            print(f"{str(e)}")
            sys.exit(1)
    else:
        print(f"[PR {pr_number}] Skipping Stage 0 (PR setup) - using existing artifacts")
    
    # Load PR inputs (relative to automation_2 directory)
    pr_desc_file = AUTOMATION_DIR / "Data" / "pr_data" / f"task_pr_{pr_number}.md"
    human_diff_file = AUTOMATION_DIR / "Data" / "pr_data" / "original_changes.diff"
    
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
    templates = load_templates(AUTOMATION_DIR / "templates" / "Prompt.md")
    
    # Generate human approach summary once (used for all models)
    # Prompt generation ALWAYS uses minimaxai/minimax-m2 (strict requirement)
    print(f"[PR {pr_number}] Generating human approach summary...")
    human_approach_client = create_client(get_prompt_generation_model())
    human_approach_summary = _generate_human_approach(templates, pr_metadata, human_patch, human_approach_client)
    
    # Create PR-level runs directory for shared artifacts
    pr_run_dir = AUTOMATION_DIR / "Data" / "runs" / f"PR-{pr_number}"
    pr_run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shared input directory for all models (single source of truth)
    shared_input_dir = pr_run_dir / "shared_input"
    shared_input_dir.mkdir(exist_ok=True)
    
    # Save human approach summary to shared input (ONCE for all models)
    human_approach_file = shared_input_dir / "human_approach.txt"
    human_approach_file.write_text(human_approach_summary, encoding="utf-8")
    print(f"[PR {pr_number}] Human approach summary saved to {human_approach_file}")
    
    # Copy PR files to shared input (ONCE for all models)
    shutil.copy2(pr_desc_file, shared_input_dir / "pr.md")
    shutil.copy2(human_diff_file, shared_input_dir / "human.diff")
    print(f"[PR {pr_number}] PR files copied to shared input")
    
    # Generate human context diff (ground truth function-level diff) - ONCE for all models
    print(f"[PR {pr_number}] Generating human context diff...")
    human_context_diff_file = shared_input_dir / "human_context.diff"
    try:
        # Get merge commit info from setup
        repo_dir = AUTOMATION_DIR.parent
        pr_dir = AUTOMATION_DIR / "Data" / "pr_data"
        merge_commit_file = pr_dir / f"merge_commit_{pr_number}.txt"
        
        if merge_commit_file.exists():
            merge_commit = merge_commit_file.read_text(encoding="utf-8").strip()
            # Generate context diff for merge^1..merge (the PR changes)
            base_ref = f"{merge_commit}^1"
            target_ref = merge_commit
            _generate_context_diff(human_context_diff_file, base_ref=base_ref, target_ref=target_ref)
            print(f"[PR {pr_number}] Human context diff saved to {human_context_diff_file}")
        else:
            print(f"[PR {pr_number}] Warning: merge_commit file not found, cannot generate human context diff")
            human_context_diff_file = None
    except RuntimeError as e:
        print(f"[PR {pr_number}] Warning: Human context diff generation failed: {e}")
        human_context_diff_file = None
    
    # Generate first prompt ONCE for all models (deterministic across models)
    print(f"[PR {pr_number}] Generating shared first prompt...")
    first_prompt_file = shared_input_dir / "first_prompt.txt"
    if not first_prompt_file.exists():
        # Generate first prompt using minimax-m2
        reasoning_client = create_client(get_prompt_generation_model())
        replacements = _build_replacements(pr_text, human_patch, pr_metadata)
        meta_prompt = fill_placeholders(templates.prompt1, replacements)
        first_prompt = reasoning_client.run(meta_prompt)
        first_prompt_file.write_text(first_prompt, encoding="utf-8")
        print(f"[PR {pr_number}] First prompt generated and saved to {first_prompt_file}")
    else:
        print(f"[PR {pr_number}] First prompt already exists, reusing from {first_prompt_file}")
    
    # Run for each model
    all_results = {}
    for model in models:
        print(f"\n{'='*80}")
        print(f"[PR {pr_number}] Running with model: {model}")
        print(f"{'='*80}\n")
        
        # Reset repository before starting each new model
        # This ensures each model starts with a clean slate
        print(f"[PR {pr_number}] Resetting repository before model {model}")
        _reset_repository()
        
        # Create model-specific run directory (relative to automation_2 directory)
        model_run_dir = AUTOMATION_DIR / "Data" / "runs" / f"PR-{pr_number}" / model.replace("/", "-")
        model_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lightweight model-specific input reference (NOT a copy - just a symlink/reference)
        # Model consumes shared PR context, no duplication
        input_ref_file = model_run_dir / "input_ref.json"
        input_ref = {
            "shared_input_dir": str(shared_input_dir.relative_to(AUTOMATION_DIR / "Data" / "runs")),
            "pr_md": "shared_input/pr.md",
            "human_diff": "shared_input/human.diff",
            "human_approach": "shared_input/human_approach.txt",
            "human_context_diff": "shared_input/human_context.diff" if human_context_diff_file else None,
            "first_prompt": "shared_input/first_prompt.txt"
        }
        input_ref_file.write_text(json.dumps(input_ref, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] Model input reference created (points to shared_input)")
        
        # Create LLM clients using unified API configuration
        # STRICT SEPARATION: ALL reasoning uses minimaxai/minimax-m2
        # - Prompt generation uses minimax-m2
        # - Evaluation uses minimax-m2
        # - Human approach summary uses minimax-m2
        # The execution model is ONLY used for Claude Code execution
        reasoning_client = create_client(get_prompt_generation_model())
        
        print(f"[PR {pr_number}] Client setup:")
        print(f"  🧠 BRAIN (Reasoning): {get_prompt_generation_model()}")
        print(f"  🤖 HANDS (Execution): {model}")
        print(f"  📝 Prompt Generation: minimax-m2")
        print(f"  ⚖️  Evaluation: minimax-m2")
        print(f"  ⚡ Code Execution: {model}")
        
        # Run attempts loop - pass shared_input_dir for first prompt reuse
        # The execution model is only used for Claude Code execution (not LLM reasoning)
        final_result = _run_attempts_loop(
            pr_number, model_run_dir, templates, pr_text, human_patch, 
            reasoning_client, reasoning_client, pr_metadata, human_approach_summary, model,
            human_context_diff_file, shared_input_dir
        )
        
        # Save model-specific final.json
        final_file = model_run_dir / "final.json"
        final_file.write_text(json.dumps(final_result, indent=2), encoding="utf-8")
        
        all_results[model] = final_result
        
        print(f"[PR {pr_number}] Model {model} - Final verdict: {final_result['final_verdict']}")
        print(f"[PR {pr_number}] Model {model} - Attempts: {final_result['attempts']}")
    
    # Generate comparison report if multiple models
    if len(models) > 1:
        comparison_file = AUTOMATION_DIR / "Data" / "runs" / f"PR-{pr_number}" / "model_comparison.json"
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
    - Clean working tree (ignoring automation_2/ which is gitignored)
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
    
    # Filter out automation_2/ directory (it's in .gitignore)
    status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
    relevant_changes = [line for line in status_lines if line and not line.endswith('automation_2/')]
    
    if relevant_changes:
        raise RuntimeError(
            f"Git working tree is not clean. Repository state is unsafe.\n"
            f"Working tree status:\n" + '\n'.join(relevant_changes)
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
    
    # Clean untracked files (SAFE: preserve automation_2 directory and all its subdirectories)
    clean_result = subprocess.run(
        ["git", "clean", "-fd", "-e", "automation_2", "-e", "automation_2/**"],
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
    - run_pr --models glm-45-air-curriculum-learning → execution model = "glm-45-air-curriculum-learning"
    - run_pr --models minimax-m2 → execution model = "minimaxai/minimax-m2"
    
    Raises:
        ValueError: If model name is unknown or unmapped
    """
    # Flexible model mapping that accepts any model name
    # Special handling for known minimax models to use full namespace
    if model == "minimax-m2":
        return "minimaxai/minimax-m2"
    
    # For all other models, use the name as-is (including minimaxai/minimax-m2)
    return model


def _validate_environment_variables():
    """Validate required environment variables for Claude Code execution.
    
    This function validates that all required environment variables are available
    from the centralized api_config module.
    
    Raises:
        SystemExit: If required environment variables are missing (from api_config)
    """
    # Validate by attempting to retrieve values from centralized config
    # These calls will fail fast with clear error messages if env vars are not set
    try:
        _ = get_anthropic_auth_key()
        _ = get_anthropic_base_url()
    except SystemExit:
        # Re-raise the SystemExit from api_config (already has clear error message)
        raise


def _execute_claude_code(prompt_file: Path, stdout_file: Path, stderr_file: Path, execution_model: str):
    """Execute Claude Code non-interactively and capture output.
    
    Uses the required execution contract:
    ANTHROPIC_BASE_URL="https://grid.ai.juspay.net" \
    ANTHROPIC_AUTH_TOKEN="sk.." \
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
    
    # Set environment variables for Claude Code (explicit injection from centralized config)
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = get_anthropic_base_url()
    env["ANTHROPIC_AUTH_TOKEN"] = get_anthropic_auth_key()
    
    # Read prompt content and clean it (remove any reasoning/metadata tags)
    prompt_content = prompt_file.read_text(encoding="utf-8")
    
    # Remove any reasoning tags that might be in the LLM response
    # Match both <think>...</think> and <redacted_reasoning>...</redacted_reasoning>
    cleaned_content = re.sub(r'<(?:think|redacted_reasoning)>.*?</(?:think|redacted_reasoning)>', '', prompt_content, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content).strip()
    
    # If prompt is empty after cleaning, fail explicitly (don't fallback)
    if not cleaned_content:
        raise RuntimeError(
            f"Prompt is empty after cleaning <think> tags. "
            f"The LLM may have only generated reasoning without an actual prompt. "
            f"Original prompt file: {prompt_file}"
        )
    
    prompt_content = cleaned_content
    
    print(f"  [Stage 3] Cleaned prompt: {len(prompt_content)} chars")
    
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
            timeout=2400  # ~16.7 minute timeout
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Claude Code execution timed out after 1000 seconds. "
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


def _generate_context_diff(output_file: Path, base_ref: str = None, target_ref: str = None, staged: bool = False):
    """Generate context-aware diff using automation_2/context_diff.py
    
    Args:
        output_file: Path to save the context diff
        base_ref: Base git ref (for human diff)
        target_ref: Target git ref (for human diff)
        staged: If True, generate diff from staged changes (for model diff)
    
    Raises:
        RuntimeError: If context diff generation fails
    """
    claude_work_dir = AUTOMATION_DIR.parent
    context_diff_script = AUTOMATION_DIR / "scripts" / "context_diff.py"
    
    if not context_diff_script.exists():
        raise RuntimeError(
            f"Context diff script not found at {context_diff_script}. "
            f"Cannot generate function-level context diffs."
        )
    
    # Build command
    cmd = ["python3", str(context_diff_script), "-o", str(output_file)]
    
    if staged:
        cmd.append("--staged")
    elif base_ref and target_ref:
        cmd.extend([base_ref, target_ref])
    elif base_ref:
        cmd.append(base_ref)
    
    # Run context_diff
    result = subprocess.run(
        cmd,
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Context diff generation failed (return code {result.returncode}).\n"
            f"Error: {result.stderr}"
        )
    
    # Verify output file was created and is non-empty
    if not output_file.exists():
        raise RuntimeError(
            f"Context diff output file not created: {output_file}. "
            f"This is a CRITICAL failure."
        )
    
    content = output_file.read_text(encoding="utf-8")
    if not content.strip():
        raise RuntimeError(
            f"Context diff is empty. This is a FAILED attempt. "
            f"Git may have no changes to process."
        )
    
    return content


def _capture_git_diff(diff_file: Path, context_diff_file: Path):
    """Capture git diff and context diff after Claude execution.
    
    Args:
        diff_file: Path to save raw git diff
        context_diff_file: Path to save context-aware diff
    """
    print("  [Stage 3] Capturing git diff")
    
    # Get the claude-work directory (parent of automation_2)
    claude_work_dir = AUTOMATION_DIR.parent
    
    # Get raw git diff (run from claude-work directory)
    result = subprocess.run(
        ["git", "diff"],
        cwd=claude_work_dir,
        capture_output=True,
        text=True
    )
    
    # Save raw diff
    diff_content = result.stdout
    diff_file.write_text(diff_content, encoding="utf-8")
    
    # Log explicitly if diff is empty
    if not diff_content.strip():
        print(f"  [Stage 3] Warning: Claude Code made no changes (empty diff)")
        print(f"  [Stage 3] Diff saved to {diff_file}")
        return
    else:
        diff_lines = len(diff_content.splitlines())
        print(f"  [Stage 3] Raw diff captured: {diff_lines} lines")
        print(f"  [Stage 3] Raw diff saved to {diff_file}")
    
    # Generate context diff from staged changes
    print("  [Stage 3] Generating model context diff (function-level)")
    try:
        # Stage all changes first
        subprocess.run(["git", "add", "-A"], cwd=claude_work_dir, check=True)
        
        # Generate context diff from staged changes
        _generate_context_diff(context_diff_file, staged=True)
        
        # Unstage changes
        subprocess.run(["git", "reset"], cwd=claude_work_dir, check=True)
        
        context_lines = len(context_diff_file.read_text(encoding="utf-8").splitlines())
        print(f"  [Stage 3] Context diff captured: {context_lines} lines")
        print(f"  [Stage 3] Context diff saved to {context_diff_file}")
    except RuntimeError as e:
        print(f"  [Stage 3] Warning: Context diff generation failed: {e}")
        print(f"  [Stage 3] Continuing with raw diff only")


def _run_attempts_loop(pr_number: str, run_dir: Path, templates, pr_text: str, human_patch: str,
                      prompt_gen_client: LLMClient, eval_client: LLMClient, pr_metadata: dict,
                      human_approach_summary: str, execution_model: str, human_context_diff_file: Path = None,
                      shared_input_dir: Path = None) -> dict:
    """Run up to 3 attempts with controlled retries and checkpoint/resume support."""

    # SAFETY FIX: Initialize variables BEFORE any conditional logic (prevents NameError)
    start_attempt = 1
    actually_resumed = False

    # Checkpoint detection: Check for previously completed attempts
    resume_info = get_resume_info(run_dir)
    last_completed = resume_info.get("last_completed_attempt", 0)  # Default to 0 if not present

    if resume_info["should_resume"]:
        print(f"[PR {pr_number}] ✓ Checkpoint detected: Found {last_completed} completed attempt(s)")
        print(f"[PR {pr_number}] ✓ Last verdict: {resume_info['last_verdict']}")

        # SAFETY FIX: Resume validation - verify prompt hash, tier, diff hash
        # This prevents resume with stale/mismatched artifacts
        # Use explicit validation_passed flag instead of mutating resume_info
        validation_passed = True

        # Validate prompt hash (if first prompt exists)
        if shared_input_dir:
            shared_first_prompt = shared_input_dir / "first_prompt.txt"
            if shared_first_prompt.exists():
                current_prompt = shared_first_prompt.read_text(encoding="utf-8")
                current_prompt_hash = hashlib.sha256(current_prompt.encode()).hexdigest()[:16]

                # Check if checkpointed prompt matches
                last_attempt_dir = run_dir / f"p{last_completed}"
                last_prompt_file = last_attempt_dir / f"p{last_completed}_prompt.txt"
                if last_prompt_file.exists():
                    last_prompt = last_prompt_file.read_text(encoding="utf-8")
                    last_prompt_hash = hashlib.sha256(last_prompt.encode()).hexdigest()[:16]

                    if current_prompt_hash != last_prompt_hash and last_completed == 1:
                        print(f"[PR {pr_number}] ⚠ RESUME ABORTED: Prompt hash mismatch")
                        print(f"[PR {pr_number}] ⚠ Expected: {last_prompt_hash}, Got: {current_prompt_hash}")
                        print(f"[PR {pr_number}] ⚠ Restarting from attempt 1 for safety")
                        validation_passed = False

        # Validate diff hash (ensure ground truth hasn't changed)
        if validation_passed:
            current_diff_hash = hashlib.sha256(human_patch.encode()).hexdigest()[:16]
            last_attempt_dir = run_dir / f"p{last_completed}"
            last_eval_file = last_attempt_dir / "eval.json"
            if last_eval_file.exists():
                last_eval = json.loads(last_eval_file.read_text(encoding="utf-8"))
                if "human_diff_hash" in last_eval:
                    if last_eval["human_diff_hash"] != current_diff_hash:
                        print(f"[PR {pr_number}] ⚠ RESUME ABORTED: Ground truth diff hash mismatch")
                        print(f"[PR {pr_number}] ⚠ Ground truth may have changed - unsafe to resume")
                        print(f"[PR {pr_number}] ⚠ Restarting from attempt 1 for safety")
                        validation_passed = False

        # Validate tier consistency (ensure tier selection is deterministic)
        if validation_passed:
            human_context = ""
            if human_context_diff_file and human_context_diff_file.exists():
                human_context = human_context_diff_file.read_text(encoding="utf-8")
            current_tier, _ = _select_diff_tier(human_patch, human_context)
            last_attempt_dir = run_dir / f"p{last_completed}"
            last_eval_file = last_attempt_dir / "eval.json"
            if last_eval_file.exists():
                last_eval = json.loads(last_eval_file.read_text(encoding="utf-8"))
                if "tier_used" in last_eval:
                    if last_eval["tier_used"] != current_tier:
                        print(f"[PR {pr_number}] ⚠ RESUME ABORTED: Tier mismatch")
                        print(f"[PR {pr_number}] ⚠ Expected tier {last_eval['tier_used']}, got tier {current_tier}")
                        print(f"[PR {pr_number}] ⚠ Restarting from attempt 1 for safety")
                        validation_passed = False

        # Resume validation result - proceed based on validation outcome
        if not validation_passed:
            # SAFETY FIX: Reset last_completed to 0 when restarting (prevents incorrect total_attempts)
            print(f"[PR {pr_number}] ✓ Starting fresh from attempt 1 due to validation failure")
            start_attempt = 1
            last_completed = 0
            actually_resumed = False
        elif resume_info["last_verdict"] == "PASS":
            # Last attempt passed, no need to continue
            print(f"[PR {pr_number}] ✓ Last attempt PASSED - resuming with success (no re-execution)")
            return {
                "pr": pr_number,
                "final_verdict": "PASS",
                "attempts": last_completed,
                "passed_on": last_completed,
                "resumed_from_checkpoint": True
            }
        else:
            # Validate last attempt has proper artifacts
            last_attempt_dir = run_dir / f"p{last_completed}"
            if not validate_attempt_artifacts(last_attempt_dir, last_completed):
                print(f"[PR {pr_number}] ⚠ Warning: Last attempt artifacts incomplete, will re-run from attempt 1")
                start_attempt = 1
                last_completed = 0
                actually_resumed = False
            else:
                start_attempt = resume_info["resume_from_attempt"]
                actually_resumed = True
                print(f"[PR {pr_number}] ✓ Resuming from attempt {start_attempt}")
    else:
        print(f"[PR {pr_number}] No checkpoint found - starting from attempt 1")
        start_attempt = 1
        last_completed = 0
        actually_resumed = False

    # Run attempts from start_attempt to MAX_ATTEMPTS
    max_attempts = MAX_ATTEMPTS  # SAFETY: Use module constant (defined at top)

    # SAFETY FIX: Initialize per-loop variables (prevent state leakage)
    # These variables are reset here to ensure no data from previous runs
    verdicts = []           # Collect all verdicts for deterministic aggregation
    passed_on = None        # Track which attempt number achieved PASS (if any)
    last_attempt = start_attempt  # Track the last attempt number executed

    for attempt_num in range(start_attempt, max_attempts + 1):
        print(f"[PR {pr_number}] Attempt {attempt_num}")

        # Run single attempt - pass shared_input_dir for first prompt reuse
        verdict = _run_single_attempt(
            pr_number, run_dir, templates, pr_text, human_patch, prompt_gen_client, eval_client,
            attempt_num, pr_metadata, human_approach_summary, execution_model, human_context_diff_file,
            shared_input_dir
        )

        last_attempt = attempt_num  # Update last attempted number
        verdicts.append(verdict)

        # SAFETY FIX: PASS stops retries immediately - success achieved
        if verdict == "PASS":
            passed_on = attempt_num
            print(f"[PR {pr_number}] PASS achieved on attempt {attempt_num}")
            break

        # SKIPPED_CONTEXT_OVERFLOW stops retries immediately
        # Context overflow is an infrastructure limitation that retrying won't fix
        if verdict == "SKIPPED_CONTEXT_OVERFLOW":
            print(f"[PR {pr_number}] SKIPPED_CONTEXT_OVERFLOW - stopping retries (infrastructure limitation)")
            break

        # For all other verdicts (FAIL, FAIL_INFRA, EXECUTION_FAILED, ERROR):
        # Continue to next attempt up to MAX_ATTEMPTS to collect complete data
        if verdict in ["FAIL", "FAIL_INFRA", "EXECUTION_FAILED", "ERROR"]:
            print(f"[PR {pr_number}] Attempt {attempt_num} verdict: {verdict} - continuing to next attempt if available")
        else:
            # Unknown verdict - stop retries for safety (should never happen)
            print(f"[PR {pr_number}] WARNING: Unknown verdict '{verdict}' - stopping retries for safety")
            break

    
    if actually_resumed:
        total_attempts = max(last_completed, last_attempt)
    else:
        total_attempts = last_attempt

    # ============================================================================
    # VERDICT AGGREGATION WITH DETERMINISTIC PRIORITY
    # ============================================================================
    #
    # VERDICT_PRIORITY (highest to lowest):
    # 1. PASS    - Any successful attempt = overall success
    # 2. FAIL    - Logical failure (model got it wrong)
    # 3. SKIPPED - Infrastructure limitation (context overflow)
    # 4. ERROR   - Generic infrastructure failure
    # 5. FAIL_INFRA - Infrastructure/evaluation failure (lowest priority)
    # ============================================================================

    # Define verdict priority (first match wins)
    VERDICT_PRIORITY = ["PASS", "FAIL", "SKIPPED_CONTEXT_OVERFLOW", "ERROR", "FAIL_INFRA", "EXECUTION_FAILED"]

    # Find highest priority verdict present in results
    final_verdict = None
    for priority_verdict in VERDICT_PRIORITY:
        if priority_verdict in verdicts:
            final_verdict = priority_verdict
            break

    if final_verdict is None:
        if verdicts:
            final_verdict = verdicts[0]
        else:
            print(f"[PR {pr_number}] CRITICAL: No verdicts recorded - defaulting to ERROR")
            final_verdict = "ERROR"

    # SAFETY: Ensure final_verdict is one of the expected values (final validation)
    valid_verdicts = {"PASS", "FAIL", "FAIL_INFRA", "SKIPPED_CONTEXT_OVERFLOW", "EXECUTION_FAILED", "ERROR"}
    if final_verdict not in valid_verdicts:
        print(f"[PR {pr_number}] WARNING: Invalid final verdict '{final_verdict}' - defaulting to ERROR")
        final_verdict = "ERROR"

    return {
        "pr": pr_number,
        "final_verdict": final_verdict,
        "attempts": total_attempts,
        "passed_on": passed_on,
        "resumed_from_checkpoint": actually_resumed,  # SAFETY FIX: Use actually_resumed flag
        "all_verdicts": verdicts  # SAFETY: Include all verdicts for traceability
    }


def _run_single_attempt(pr_number: str, run_dir: Path, templates, pr_text: str, human_patch: str,
                      prompt_gen_client: LLMClient, eval_client: LLMClient, attempt_num: int,
                      pr_metadata: dict, human_approach_summary: str, execution_model: str,
                      human_context_diff_file: Path = None, shared_input_dir: Path = None) -> str:
    """Run a single attempt: generate prompt, run Claude, evaluate.

    EMPTY_DIFF Retry Logic:
    - Each attempt (p1, p2, p3) must produce a non-empty diff
    - If empty diff detected: retry ONCE with forced-change instruction
    - Retry state tracked in-memory (not via marker files)
    - Retry prompt saved for traceability but never reused

    SAFETY GUARANTEES:
    - No exception path returns PASS (only EXECUTION_FAILED, ERROR, or FAIL)
    - Per-attempt variables are function-scoped (no state leakage between attempts)
    - Retry can only happen ONCE per attempt (enforced by retry_attempted flag)

    Returns:
        Verdict string: "PASS", "FAIL", "EXECUTION_FAILED", or "ERROR"
    """
    retry_attempted = False  # Retry state for empty diff handling (resets per attempt)
    attempt_dir = run_dir / f"p{attempt_num}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Mark attempt as in-progress
    mark_attempt_in_progress(attempt_dir)
    print(f"[PR {pr_number}] ✓ Attempt {attempt_num} marked as in-progress")

    # Generate/load prompt for this attempt
    if attempt_num == 1:
        # First attempt: REUSE shared first prompt (deterministic across all models)
        if shared_input_dir:
            shared_first_prompt = shared_input_dir / "first_prompt.txt"
            if shared_first_prompt.exists():
                generated_prompt = shared_first_prompt.read_text(encoding="utf-8")
                print(f"[PR {pr_number}] ✓ Reusing shared first prompt from {shared_first_prompt}")
            else:
                # Fallback: Generate if shared prompt doesn't exist (shouldn't happen)
                print(f"[PR {pr_number}] Warning: Shared first prompt not found, generating new one")
                replacements = _build_replacements(pr_text, human_patch, pr_metadata)
                meta_prompt = fill_placeholders(templates.prompt1, replacements)
                generated_prompt = prompt_gen_client.run(meta_prompt)
        else:
            # Fallback for backwards compatibility
            replacements = _build_replacements(pr_text, human_patch, pr_metadata)
            meta_prompt = fill_placeholders(templates.prompt1, replacements)
            generated_prompt = prompt_gen_client.run(meta_prompt)
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
    
        # Generate the actual prompt for retry attempts
        generated_prompt = prompt_gen_client.run(meta_prompt)
    
    # Save the prompt to attempt directory
    prompt_file = attempt_dir / f"p{attempt_num}_prompt.txt"
    prompt_file.write_text(generated_prompt, encoding="utf-8")
    
    print(f"[PR {pr_number}] Prompt {attempt_num} saved to {prompt_file}")
    
    # Reset repository ONLY before first attempt
    # Subsequent attempts (p2, p3) must build upon changes from previous attempts
    if attempt_num == 1:
        _reset_repository()
    else:
        print(f"[PR {pr_number}] ✓ Skipping git reset - continuing from previous attempt's changes")
    
    # Execute Claude Code
    stdout_file = attempt_dir / "stdout.txt"
    stderr_file = attempt_dir / "stderr.txt"
    diff_file = attempt_dir / "claude.diff"
    context_diff_file = attempt_dir / "claude_context.diff"
    
    try:
        _execute_claude_code(prompt_file, stdout_file, stderr_file, execution_model)
        _capture_git_diff(diff_file, context_diff_file)
    except (RuntimeError, ValueError) as e:
        error_data = {
            "verdict": "EXECUTION_FAILED",
            "reason": f"Claude Code execution failed: {str(e)}",
            "error_type": type(e).__name__
        }
        error_file = attempt_dir / "eval.json"
        error_file.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] EXECUTION FAILED: {str(e)}")
        mark_attempt_failed(attempt_dir)
        return "EXECUTION_FAILED"
    except Exception as e:
        # SAFETY: Catch-all for unexpected errors - log and return ERROR (never PASS)
        print(f"[PR {pr_number}] UNEXPECTED ERROR during execution: {type(e).__name__}: {str(e)}")
        error_data = {
            "verdict": "ERROR",
            "reason": f"Unexpected error during execution: {type(e).__name__}: {str(e)}",
            "error_type": type(e).__name__
        }
        error_file = attempt_dir / "eval.json"
        error_file.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
        mark_attempt_failed(attempt_dir)
        return "ERROR"
    
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
        # EMPTY_DIFF detected - check if we can retry (using in-memory state, not marker files)
        if retry_attempted:
            # Already retried once for this attempt - hard FAIL (GUARD: prevent retry-on-retry)
            failure_data = {
                "verdict": "FAIL",
                "reason": "Claude Code made no changes (empty diff) even after forced retry. This is a FAILED attempt. Silent success is NOT allowed.",
                "failure_type": "EMPTY_DIFF",
                "empty_diff_retry_attempted": True  # Canonical metadata key
            }
            failure_file = attempt_dir / "eval.json"
            failure_file.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")
            print(f"[PR {pr_number}] FAILED: claude.diff is empty after retry (no changes made)")
            mark_attempt_failed(attempt_dir)
            return "FAIL"
        else:
            # First time seeing empty diff - retry once with forced-change instruction
            print(f"[PR {pr_number}] WARNING: Empty diff detected - retrying with forced-change instruction")
            
            # Mark that we're retrying (in-memory, not filesystem)
            retry_attempted = True
            
            # Append improved forced-change instruction to the prompt
            original_prompt = prompt_file.read_text(encoding="utf-8")
            forced_change_instruction = """

IMPORTANT:
Your previous run produced no file changes.
You MUST modify at least one file.
If no functional change is required, make a minimal but meaningful change such as:
- adding a clarifying comment
- adding a TODO
- adding a validation/assertion
- a small refactor that improves clarity
If you believe the code is already correct, add a comment explaining why no further changes are required.
Do not respond without modifying a file.
"""
            retry_prompt = original_prompt + forced_change_instruction
            
            # Save retry prompt (for traceability only - NEVER reused for p2/p3)
            retry_prompt_file = attempt_dir / f"p{attempt_num}_prompt_retry.txt"
            retry_prompt_file.write_text(retry_prompt, encoding="utf-8")
            print(f"[PR {pr_number}] Retry prompt saved to {retry_prompt_file} (ephemeral, not reused)")
            
            # Execute Claude Code again with retry prompt (NO git reset between runs)
            # Reuse the same stdout/stderr/diff files (will overwrite)
            # SAFETY: Same exception handling as main execution
            try:
                _execute_claude_code(retry_prompt_file, stdout_file, stderr_file, execution_model)
                _capture_git_diff(diff_file, context_diff_file)
            except (RuntimeError, ValueError) as e:
                # SAFETY: Retry execution failure - return EXECUTION_FAILED (never PASS)
                error_data = {
                    "verdict": "EXECUTION_FAILED",
                    "reason": f"Claude Code retry execution failed: {str(e)}",
                    "error_type": type(e).__name__,
                    "empty_diff_retry_attempted": True  # Canonical metadata key
                }
                error_file = attempt_dir / "eval.json"
                error_file.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
                print(f"[PR {pr_number}] RETRY EXECUTION FAILED: {str(e)}")
                mark_attempt_failed(attempt_dir)
                return "EXECUTION_FAILED"
            except Exception as e:
                # SAFETY: Unexpected error in retry - return ERROR (never PASS)
                print(f"[PR {pr_number}] UNEXPECTED ERROR during retry execution: {type(e).__name__}: {str(e)}")
                error_data = {
                    "verdict": "ERROR",
                    "reason": f"Unexpected error during retry execution: {type(e).__name__}: {str(e)}",
                    "error_type": type(e).__name__,
                    "empty_diff_retry_attempted": True
                }
                error_file = attempt_dir / "eval.json"
                error_file.write_text(json.dumps(error_data, indent=2), encoding="utf-8")
                mark_attempt_failed(attempt_dir)
                return "ERROR"
            
            # Check retry result (GUARD: retry can only happen once per attempt)
            retry_diff_content = diff_file.read_text(encoding="utf-8")
            if not retry_diff_content.strip():
                # Retry also produced empty diff - hard FAIL (GUARD ensures this is terminal)
                failure_data = {
                    "verdict": "FAIL",
                    "reason": "Claude Code made no changes (empty diff) even after forced retry. This is a FAILED attempt. Silent success is NOT allowed.",
                    "failure_type": "EMPTY_DIFF",
                    "empty_diff_retry_attempted": True  # Canonical metadata key
                }
                failure_file = attempt_dir / "eval.json"
                failure_file.write_text(json.dumps(failure_data, indent=2), encoding="utf-8")
                print(f"[PR {pr_number}] FAILED: claude.diff is still empty after retry")
                mark_attempt_failed(attempt_dir)
                return "FAIL"
            else:
                # Retry succeeded - proceed with evaluation
                print(f"[PR {pr_number}] ✓ EMPTY_DIFF recovered via forced retry ({len(retry_diff_content)} chars)")
                # Continue to evaluation below (retry_attempted flag will be saved in metadata)
    
    claude_patch = diff_file.read_text(encoding="utf-8")
    human_context = ""
    model_context = ""
    
    if human_context_diff_file and human_context_diff_file.exists():
        human_context = human_context_diff_file.read_text(encoding="utf-8")
    if context_diff_file and context_diff_file.exists():
        model_context = context_diff_file.read_text(encoding="utf-8")
    
    # Check if Tier-3 will be needed
    tier, tier_reason = _select_diff_tier(human_patch, human_context)
    
    if tier == 3:
        # Pre-generate and cache Tier-3 summary
        print(f"  [Stage 3.5] Pre-generating Tier-3 summary (performance optimization)")
        tier3_summary_file = attempt_dir / "tier3_summary.json"
        
        tier3_summary = _generate_tier3_summary(
            human_patch, claude_patch, human_context, model_context, eval_client
        )
        
        tier3_summary_file.write_text(json.dumps(tier3_summary, indent=2), encoding="utf-8")
        print(f"  [Stage 3.5] Tier-3 summary cached to {tier3_summary_file}")
    
    # SAFETY FIX: Compute diff hash for resume validation
    human_diff_hash = hashlib.sha256(human_patch.encode()).hexdigest()[:16]

    # Evaluate the solution with dynamic ground-truth selection
    eval_file = attempt_dir / "eval.json"
    _evaluate_solution(
        templates, pr_metadata, human_patch, human_approach_summary,
        diff_file, eval_file, eval_client,
        human_context_diff_file, context_diff_file,
        attempt_dir  # Pass attempt_dir for ground-truth metadata persistence
    )

    # SAFETY FIX: Add diff hash to eval.json for resume validation
    if eval_file.exists():
        eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
        eval_data["human_diff_hash"] = human_diff_hash
        eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    
    # Read verdict and add retry metadata if applicable
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
    verdict = eval_data['verdict']
    
    # If retry was attempted and succeeded, add metadata to eval.json
    if retry_attempted:
        eval_data["empty_diff_retry_attempted"] = True
        eval_data["empty_diff_recovery_status"] = "recovered" if verdict != "FAIL" else "failed"
        eval_file.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
        print(f"[PR {pr_number}] ✓ Added empty_diff retry metadata to eval.json")
    
    # Mark attempt status based on verdict
    if verdict == "PASS":
        mark_attempt_completed(attempt_dir)
        print(f"[PR {pr_number}] ✓ Attempt {attempt_num} marked as completed (PASS)")
    else:
        mark_attempt_failed(attempt_dir)
        print(f"[PR {pr_number}] ✓ Attempt {attempt_num} marked as failed ({verdict})")
    
    return verdict


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
