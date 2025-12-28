"""Evaluation Pipeline: Tier Selection and Routing.

This module performs strict, deterministic evaluation with zero tolerance for ambiguity.

Core responsibilities:
- Tier selection based on diff sizes (Tier-1: raw, Tier-2: context, Tier-3: behavioral)
- Tier-specific prompt building with strict isolation
- Evaluation orchestration with hard context guards
- Verdict parsing with exact regex matching

Safety guarantees:
- Context overflow checked BEFORE model invocation
- Malformed verdicts default to FAIL/ERROR (never PASS)
- All code paths are deterministic and reproducible
- Tier-3 delegates to strict textual comparison engine
"""

import re
import json
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Dict

if TYPE_CHECKING:
    from scripts.llm_client import LLMClient

# ============================================================================
# TIER SELECTION CONSTANTS
# ============================================================================

# Tier selection thresholds (production-grade, model-agnostic)
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

# Maximum prompt size (safety guard)
MAX_PROMPT_CHARS = 120_000


# ============================================================================
# TIER SELECTION
# ============================================================================

def _count_changed_files(diff_content: str) -> int:
    """Count number of unique files changed in a diff.

    Args:
        diff_content: Raw git diff content

    Returns:
        Number of unique files changed
    """
    changed_files = set()
    for line in diff_content.splitlines():
        if line.startswith('diff --git'):
            # Extract filename from "diff --git a/path/file.py b/path/file.py"
            parts = line.split()
            if len(parts) >= 3:
                # Use the 'a/' version (before changes)
                file_path = parts[2].lstrip('a/')
                changed_files.add(file_path)
    return len(changed_files)


def _select_diff_tier(raw_diff: str, context_diff: str) -> tuple:
    """Select optimal diff tier based on size (Tier-3 system).

    Applies size-based selection rules:
    - Tier-1: raw_diff <= 35,000 chars
    - Tier-2: raw_diff > 35,000 AND context_diff <= 20,000 chars
    - Tier-3: both exceed limits

    Args:
        raw_diff: Raw git diff content
        context_diff: Context diff content (may be empty)

    Returns:
        tuple: (tier_number, tier_reason)
    """
    raw_chars = len(raw_diff)
    context_chars = len(context_diff) if context_diff else 0

    # Tier-1: Raw diff fits within limit
    if raw_chars <= RAW_DIFF_CHAR_LIMIT:
        return (1, f"Raw diff within limit ({raw_chars} <= {RAW_DIFF_CHAR_LIMIT})")

    # Tier-2: Context diff fits (when raw exceeds)
    if context_chars > 0 and context_chars <= CONTEXT_DIFF_CHAR_LIMIT:
        return (2, f"Context diff within limit ({context_chars} <= {CONTEXT_DIFF_CHAR_LIMIT}, raw={raw_chars})")

    # Tier-3: Both exceed limits
    return (3, f"Both diffs exceed limits (raw={raw_chars}, context={context_chars})")


# ============================================================================
# TIER-SPECIFIC EVALUATION PROMPT BUILDERS
# ============================================================================

def build_tier1_eval_prompt(templates, pr_metadata: dict, human_approach_summary: str,
                            human_patch: str, claude_patch: str) -> str:
    """Build Tier-1 evaluation prompt (raw diffs only)."""
    from scripts.prompt_builder import fill_placeholders

    tier_declaration = "EVALUATION TIER: Tier-1 (Raw Diff)\n\n"
    replacements = {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_ISSUE_DESCRIPTION": pr_metadata["PR_ISSUE_DESCRIPTION"],
        "HUMAN_APPROACH_SUMMARY": human_approach_summary,
        "DIFF_SUMMARY": "",
        "HUMAN_CONTEXT_DIFF": "",
        "MODEL_CONTEXT_DIFF": "",
        "GROUND_TRUTH_DIFF_TEXT": human_patch,
        "MODEL_DIFF_TEXT": claude_patch,
    }
    eval_prompt = tier_declaration + fill_placeholders(templates.evaluation, replacements)
    return eval_prompt


def build_tier2_eval_prompt(templates, pr_metadata: dict, human_approach_summary: str,
                            human_context_diff: str, model_context_diff: str) -> str:
    """Build Tier-2 evaluation prompt (context diffs only)."""
    from scripts.prompt_builder import fill_placeholders

    tier_declaration = "EVALUATION TIER: Tier-2 (Context Diff)\n\n"
    replacements = {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_ISSUE_DESCRIPTION": pr_metadata["PR_ISSUE_DESCRIPTION"],
        "HUMAN_APPROACH_SUMMARY": human_approach_summary,
        "DIFF_SUMMARY": "",
        "HUMAN_CONTEXT_DIFF": human_context_diff,
        "MODEL_CONTEXT_DIFF": model_context_diff,
        "GROUND_TRUTH_DIFF_TEXT": "",
        "MODEL_DIFF_TEXT": "",
    }
    eval_prompt = tier_declaration + fill_placeholders(templates.evaluation, replacements)
    return eval_prompt


def build_tier3_eval_prompt(templates, pr_metadata: dict, human_approach_summary: str,
                            diff_summary: str) -> str:
    """Build Tier-3 evaluation prompt (structured summary only)."""
    from scripts.prompt_builder import fill_placeholders

    tier_declaration = "EVALUATION TIER: Tier-3 (Structured Summary)\n\n"
    replacements = {
        "PR_TITLE": pr_metadata["PR_TITLE"],
        "PR_ISSUE_DESCRIPTION": pr_metadata["PR_ISSUE_DESCRIPTION"],
        "HUMAN_APPROACH_SUMMARY": human_approach_summary,
        "DIFF_SUMMARY": diff_summary,
        "HUMAN_CONTEXT_DIFF": "",
        "MODEL_CONTEXT_DIFF": "",
        "GROUND_TRUTH_DIFF_TEXT": "",
        "MODEL_DIFF_TEXT": "",
    }
    eval_prompt = tier_declaration + fill_placeholders(templates.evaluation, replacements)
    return eval_prompt


# ============================================================================
# VERDICT PARSING
# ============================================================================

def _parse_verdict(eval_response: str, is_tier3: bool = False) -> dict:
    r"""Parse verdict from LLM evaluation response with strict enforcement.

    SAFETY FIX: Enforces EXACTLY ONE machine-readable verdict block.

    Extraction rules:
    - ALL TIERS: MUST have strict regex ^FINAL_VERDICT:\s*(PASS|FAIL|FAIL_INFRA|SKIPPED_CONTEXT_OVERFLOW)
    - EXACTLY ONE match required (zero or multiple matches → ERROR)
    - Tier-3: Missing/malformed/multiple verdicts → FAIL_INFRA
    - Tier-1/2: Missing/malformed/multiple verdicts → FAIL
    - No heuristic or substring-based detection allowed

    Returns:
        dict with "verdict" and "reason" keys
    """
    reason = eval_response.strip()

    # Detect repetitive/looping responses (LLM stuck in a loop)
    if len(reason) > 50000:
        lines = reason.split('\n')
        if len(lines) > 1000:
            print("  [Stage 4] Warning: Detected repetitive/looping evaluation response")
            return {
                "verdict": "FAIL_INFRA" if is_tier3 else "FAIL",
                "reason": "Evaluation LLM generated repetitive/looping response. Deterministic verdict extraction required."
            }

    # SAFETY FIX: Strict regex extraction with EXACTLY ONE match enforcement
    # Accept both FINAL_VERDICT and VERDICT for backward compatibility
    verdict_matches = re.findall(
        r'^(?:FINAL_VERDICT|VERDICT):\s*(PASS|FAIL|FAIL_INFRA|SKIPPED_CONTEXT_OVERFLOW)\s*$',
        eval_response,
        re.MULTILINE | re.IGNORECASE
    )

    # SAFETY FIX: Require EXACTLY ONE verdict line
    if len(verdict_matches) == 0:
        # No verdict found - default to FAIL (FAIL_INFRA for Tier-3)
        default_verdict = "FAIL_INFRA" if is_tier3 else "FAIL"
        print(f"  [Stage 4] ERROR: No machine-readable verdict found, defaulting to {default_verdict}")
        return {
            "verdict": default_verdict,
            "reason": f"Evaluation response missing required 'FINAL_VERDICT: PASS|FAIL|FAIL_INFRA|SKIPPED_CONTEXT_OVERFLOW' line. Original response length: {len(reason)} chars. Defaulting to {default_verdict} for safety."
        }
    elif len(verdict_matches) > 1:
        # Multiple verdicts found - ambiguous, default to ERROR
        default_verdict = "FAIL_INFRA" if is_tier3 else "FAIL"
        print(f"  [Stage 4] ERROR: Found {len(verdict_matches)} verdict lines (expected exactly 1), defaulting to {default_verdict}")
        print(f"  [Stage 4] ERROR: Verdicts found: {verdict_matches}")
        return {
            "verdict": default_verdict,
            "reason": f"Evaluation response contains {len(verdict_matches)} FINAL_VERDICT lines (expected exactly 1). Verdicts: {verdict_matches}. This is ambiguous. Defaulting to {default_verdict} for safety."
        }

    # EXACTLY ONE verdict found - safe to proceed
    verdict = verdict_matches[0].upper()

    # Extract reason (everything after VERDICT line, or full response if structured)
    reason_match = re.search(r'^REASON:\s*(.*)$', eval_response, re.MULTILINE | re.DOTALL)
    extracted_reason = reason_match.group(1).strip() if reason_match else reason

    return {
        "verdict": verdict,
        "reason": extracted_reason
    }


# ============================================================================
# LEGACY TIER-3 SUMMARY GENERATION (for backward compatibility)
# ============================================================================

def _generate_tier3_summary(human_diff: str, model_diff: str,
                           human_context_diff: str, model_context_diff: str,
                           client: "LLMClient") -> dict:
    """Generate Tier-3 structured diff summary when both diffs exceed limits.

    FAIL-INFRA GUARANTEE: Tier-3 summary generation can never produce a partial
    summary that risks false PASS. Any failure mode returns FAIL_INFRA marker.

    Risk-aware defaults:
    - files_changed > 1 without full diff analysis → FAIL_INFRA
    - JSON parse failure → FAIL_INFRA (no partial summaries)
    - LLM call failure → FAIL_INFRA

    Args:
        human_diff: Human raw diff (ground truth)
        model_diff: Model raw diff (attempt)
        human_context_diff: Human context diff
        model_context_diff: Model context diff
        client: LLM client for generating summary

    Returns:
        dict: Structured diff summary or FAIL_INFRA marker
    """
    # Count files changed (used for risk-aware default)
    files_changed = _count_changed_files(human_diff)

    # RISK-AWARE DEFAULT: If multiple files changed and we're in Tier-3,
    # we cannot reliably evaluate without full diff access
    if files_changed > 1:
        print(f"  [Tier-3] RISK: {files_changed} files changed, full diff unavailable")
        print(f"  [Tier-3] FAIL_INFRA: Cannot safely evaluate multi-file changes without full diff")
        return {
            "tier": 3,
            "summary_type": "FAIL_INFRA",
            "failure_reason": "RISK_AWARE_REJECTION",
            "details": {
                "files_changed": files_changed,
                "raw_diff_chars": len(human_diff),
                "context_diff_chars": len(human_context_diff) if human_context_diff else 0,
                "reason": "Full diff analysis unavailable for multi-file changes. Tier-3 evaluation cannot safely determine correctness. Large diffs must bias toward rejection."
            },
            "behavioral_changes": [],
            "non_changes": [],
            "known_omissions": ["Full diff analysis unavailable"],
            "_tier3_fail_infra": True
        }

    # Use context diffs if available (more compact than raw diffs)
    human_input = human_context_diff if human_context_diff else human_diff
    model_input = model_context_diff if model_context_diff else model_diff

    summary_prompt = f"""You are generating a STRUCTURED DIFF SUMMARY for evaluation.

Your job is to COMPRESS changes, not judge them.

Rules:
- Output VALID JSON only
- No opinions, no reasoning, no code blocks, no speculation
- List: files changed, symbols/functions changed, type of change
- Observable behavioral changes
- Explicit non-changes
- Known omissions due to compression
- CRITICAL: Detect propagation patterns (if change affects multiple files/functions)

HUMAN DIFF (Ground Truth - COMPLETE):
{human_input}

MODEL DIFF (Attempt - COMPLETE):
{model_input}

Output structured summary as JSON matching this schema:
{{
  "tier": 3,
  "summary_type": "STRUCTURED_DIFF_SUMMARY",
  "compression_reason": "Both raw and context diffs exceeded limits",
  "stats": {{
    "raw_diff_chars": {len(human_diff)},
    "context_diff_chars": {len(human_context_diff)},
    "files_changed": {files_changed},
    "functions_changed": <count>
  }},
  "files": [
    {{
      "path": "...",
      "change_type": "added|modified|deleted",
      "symbols": [
        {{
          "name": "...",
          "kind": "function|struct|enum|file",
          "change": "logic_update|refactor|validation|comment",
          "description": "1-2 factual sentences"
        }}
      ]
    }}
  ],
  "behavioral_changes": ["explicit observable behavior only"],
  "non_changes": ["explicitly state what did NOT change"],
  "known_omissions": ["what details were dropped due to compression"]
}}"""

    # Call LLM with deterministic settings
    print(f"  [Tier-3] Generating summary via LLM...")
    response = client.run(summary_prompt)

    # Parse JSON response
    try:
        summary = json.loads(response)
        print(f"  [Tier-3] ✓ Summary generated: {len(json.dumps(summary))} chars")
        return summary
    except json.JSONDecodeError as e:
        # FAIL_INFRA: JSON parse failure cannot produce partial summary
        print(f"  [Tier-3] FAIL_INFRA: LLM response invalid JSON: {e}")
        return {
            "tier": 3,
            "summary_type": "FAIL_INFRA",
            "failure_reason": "JSON_PARSE_FAILURE",
            "details": {
                "raw_diff_chars": len(human_diff),
                "context_diff_chars": len(human_context_diff) if human_context_diff else 0,
                "files_changed": files_changed,
                "parse_error": str(e),
                "response_preview": response[:500] if response else "Empty response"
            },
            "behavioral_changes": [],
            "non_changes": [],
            "known_omissions": ["Complete diff analysis due to JSON parse failure"],
            "_tier3_fail_infra": True
        }


# ============================================================================
# MAIN EVALUATION ORCHESTRATION
# ============================================================================

def _evaluate_solution(eval_templates, pr_metadata: dict, human_patch: str, human_approach_summary: str,
                      claude_diff_file: Path, eval_file: Path, client: "LLMClient",
                      human_context_diff_file: Path = None, model_context_diff_file: Path = None,
                      attempt_dir: Path = None):
    """Evaluate using 3-tier diff selection system with tier-specific prompt builders.

    This function implements size-based tier selection:
    - Tier-1: raw_diff <= 35,000 chars → Full precision evaluation using RAW diffs
    - Tier-2: raw_diff > 35,000 AND context_diff <= 20,000 → Function-level evaluation using CONTEXT diffs
    - Tier-3: both exceed limits → Compressed behavioral summary (LLM-generated) using STRUCTURED SUMMARY

    Key Features:
    - Tier-specific prompt builders ensure STRICT ISOLATION
    - Only active tier's rules and diffs are visible to evaluation LLM
    - Prompt size safety guard (120k chars)
    - Tier metadata saved to eval.json for traceability
    - Multipliers applied to scoring (1.00, 0.95, 0.85)
    """
    # Import Tier-3 engine
    from scripts.tier3_diff import compare_chunked_with_claude

    print("  [Stage 4] Starting tier-based evaluation with isolated prompts")

    # Read raw diffs (always needed)
    human_patch_content = human_patch
    claude_patch = claude_diff_file.read_text(encoding="utf-8")

    # Check if claude.diff is empty
    if not claude_patch.strip():
        verdict_data = {
            "verdict": "FAIL",
            "reason": "Model made no changes (empty diff). Cannot evaluate.",
            "diff_tier_used": 0
        }
        eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
        print(f"  [Stage 4] Verdict: FAIL (empty model diff)")
        return

    # Load context diffs if available
    human_context = ""
    model_context = ""

    if human_context_diff_file and human_context_diff_file.exists():
        human_context = human_context_diff_file.read_text(encoding="utf-8")
    if model_context_diff_file and model_context_diff_file.exists():
        model_context = model_context_diff_file.read_text(encoding="utf-8")

    # Select tier based on size using the new system
    tier, tier_reason = _select_diff_tier(human_patch_content, human_context)

    # SAFETY FIX: Validate tier before routing (must be 1, 2, or 3)
    if tier not in [1, 2, 3]:
        print(f"  [Stage 4] ERROR: Invalid tier {tier} - expected 1, 2, or 3")
        verdict_data = {
            "verdict": "ERROR",
            "reason": f"Invalid tier selection: {tier}. Expected 1, 2, or 3. Tier reason: {tier_reason}",
            "failure_type": "INVALID_TIER",
            "diff_tier_used": tier,
            "tier_reason": tier_reason
        }
        eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
        print(f"  [Stage 4] Verdict: ERROR (invalid tier)")
        return

    print(f"  [Stage 4] Selected Tier-{tier}: {tier_reason}")
    print(f"  [Stage 4] Raw diff: {len(human_patch_content)} chars, Context diff: {len(human_context)} chars")

    # Initialize variables for evaluation
    eval_prompt = ""
    summary_text = ""

    # Build tier-specific evaluation prompt using tier-specific builders
    # Each builder includes ONLY the relevant data for that tier
    if tier == 1:
        # Tier-1: Use raw diffs only
        eval_prompt = build_tier1_eval_prompt(
            eval_templates, pr_metadata, human_approach_summary,
            human_patch_content, claude_patch
        )

    elif tier == 2:
        # Tier-2: Use context diffs only
        eval_prompt = build_tier2_eval_prompt(
            eval_templates, pr_metadata, human_approach_summary,
            human_context, model_context
        )

    else:  # tier == 3
        # Tier-3: BEHAVIORAL COMPARISON (with chunking for oversized diffs)
        print(f"  [Stage 4] Running Tier-3 behavioral comparison (chunked)")

        # Use chunked comparison which handles both small and large diffs
        verdict, reason, confidence = compare_chunked_with_claude(
            human_patch_content, claude_patch, client,
            timeout=EVAL_REQUEST_TIMEOUT_SECONDS
        )

        # Generate final output (strict plain text)
        final_output = f"""=== TIER 3 EVALUATION ===
VERDICT: {verdict}

REASON:
{reason}

CONFIDENCE: {confidence}
"""

        # Build verdict data - include all fields for schema consistency with Tier-1/2
        verdict_data = {
            "verdict": verdict,
            "reason": reason,
            "confidence": confidence,
            "diff_tier_used": tier,
            "tier_used": tier,
            "tier_reason": tier_reason,
            "tier3_triggered": True,
            "tier3_reason": tier_reason,
            "raw_diff_chars": len(human_patch_content),
            "context_diff_chars": len(human_context),
            "human_diff_chars": len(human_patch_content),
            "claude_diff_chars": len(claude_patch),
            "tier_multiplier": TIER_MULTIPLIERS[tier],
            "tier3_output": final_output.strip(),
            "eval_prompt_chars": 0,  # Tier-3 uses chunked engine, no eval prompt
            "tier3_summary_chars": len(final_output.strip()) if final_output else 0
        }
        eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
        print(f"  [Stage 4] Verdict: {verdict} ({reason})")
        print(f"  [Stage 4] Evaluation saved to {eval_file}")
        return

    # SAFETY FIX: Context overflow hard guard - check BEFORE model invocation
    # If prompt exceeds limit, return SKIPPED immediately (do NOT call model)
    if len(eval_prompt) > MAX_PROMPT_CHARS:
        print(f"  [Stage 4] CONTEXT OVERFLOW: Prompt too large ({len(eval_prompt)} > {MAX_PROMPT_CHARS})")
        print(f"  [Stage 4] SKIPPED_CONTEXT_OVERFLOW - cannot evaluate safely")

        # SAFETY: Return SKIPPED_CONTEXT_OVERFLOW without calling model
        verdict_data = {
            "verdict": "SKIPPED_CONTEXT_OVERFLOW",
            "reason": f"Evaluation prompt exceeds context limit ({len(eval_prompt)} chars > {MAX_PROMPT_CHARS} chars). Cannot evaluate safely. This is a context overflow, not a model failure.",
            "failure_type": "CONTEXT_OVERFLOW",
            "diff_tier_used": tier,
            "tier_used": tier,
            "tier_reason": tier_reason,
            "tier3_triggered": False,
            "raw_diff_chars": len(human_patch_content),
            "context_diff_chars": len(human_context),
            "eval_prompt_chars": len(eval_prompt),
            "tier_multiplier": TIER_MULTIPLIERS[tier]
        }
        eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
        print(f"  [Stage 4] Verdict: SKIPPED_CONTEXT_OVERFLOW (prompt too large)")
        print(f"  [Stage 4] Evaluation saved to {eval_file}")
        return

    # Call evaluator with timeout and error handling (Issues #1 and #3)
    print(f"  [Stage 4] Calling evaluation LLM (Tier-{tier}) with {EVAL_REQUEST_TIMEOUT_SECONDS}s timeout")
    try:
        eval_response = client.run(eval_prompt, timeout=EVAL_REQUEST_TIMEOUT_SECONDS)
    except Exception as e:
        # Handle timeout or other LLM errors gracefully
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            print(f"  [Stage 4] ERROR: Evaluation timed out")
            # Tier-3 must return FAIL_INFRA for any failure
            verdict = "FAIL_INFRA" if tier == 3 else "FAIL"
            verdict_data = {
                "verdict": verdict,
                "reason": f"Evaluation timed out after {EVAL_REQUEST_TIMEOUT_SECONDS}s. LLM did not respond in time. Tier-3 infrastructure failure.",
                "failure_type": "EVAL_TIMEOUT",
                "diff_tier_used": tier,
                "tier_used": tier,
                "tier_reason": tier_reason,
                "tier3_triggered": (tier == 3),
                "tier3_reason": tier_reason if tier == 3 else None,
                "raw_diff_chars": len(human_patch_content),
                "context_diff_chars": len(human_context),
                "tier_multiplier": TIER_MULTIPLIERS[tier]
            }
            eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")
            print(f"  [Stage 4] Verdict: {verdict} (timeout)")
            return
        else:
            # Other LLM errors - re-raise for debugging
            print(f"  [Stage 4] ERROR: Evaluation LLM failed: {e}")
            raise

    # Parse verdict - pass is_tier3 flag for strict extraction
    verdict_data = _parse_verdict(eval_response, is_tier3=(tier == 3))

    # Add tier metadata to verdict data
    verdict_data["diff_tier_used"] = tier
    verdict_data["tier_used"] = tier
    verdict_data["tier_reason"] = tier_reason
    verdict_data["eval_prompt_chars"] = len(eval_prompt)
    verdict_data["tier3_triggered"] = (tier == 3)
    verdict_data["tier3_reason"] = tier_reason if tier == 3 else None
    verdict_data["raw_diff_chars"] = len(human_patch_content)
    verdict_data["context_diff_chars"] = len(human_context)
    verdict_data["tier3_summary_chars"] = len(summary_text) if tier == 3 else None
    verdict_data["tier_multiplier"] = TIER_MULTIPLIERS[tier]

    # Save evaluation result
    eval_file.write_text(json.dumps(verdict_data, indent=2), encoding="utf-8")

    print(f"  [Stage 4] Verdict: {verdict_data['verdict']} (Tier-{tier}, multiplier={TIER_MULTIPLIERS[tier]})")
    print(f"  [Stage 4] Evaluation saved to {eval_file}")
