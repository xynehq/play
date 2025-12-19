# End-to-End Automation Flow

## Overview
This document describes the complete automation flow for PR benchmarking using Claude Code.

## Architecture Components

```
┌─────────────────┐
│   run_pr CLI    │  Entry point: `python3 run_pr <PR_NUMBER> --models <MODEL>`
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  orchestrator   │  Main control flow
└────────┬────────┘
         │
         ├──► Prompt Generation (LLM)
         ├──► Claude Code Execution
         └──► Evaluation (LLM)
```

## Stage-by-Stage Flow

### Stage 0: PR Setup (pr_analyzer.sh)
**Location**: `pr_analyzer.sh` (run separately before automation)

**Actions**:
1. Fetch PR metadata via `gh pr view`
2. Identify merge commit
3. Checkout base commit (`merge_commit^1`)
4. Create isolated branch
5. Generate `pr/task_pr_<PR>.md`
6. Generate `pr/original_changes.diff`

**Outputs**:
- `pr/task_pr_<PR>.md` - Natural language task description
- `pr/original_changes.diff` - Ground truth diff

---

### Stage 1: Initialization
**Function**: `run_pr()` in `orchestrator.py`

**Actions**:
1. Load PR description from `pr/task_pr_<PR>.md`
2. Load human diff from `pr/original_changes.diff`
3. Extract PR metadata (title, description, issue)
4. Load prompt templates from `Prompt.md`
5. Generate human approach summary (LLM call using `minimaxai/minimax-m2`)

**Outputs**:
- `runs/PR-<PR>/<model>/input/pr.md` - Copied PR description
- `runs/PR-<PR>/<model>/input/human.diff` - Copied human diff
- Human approach summary (in memory, used for evaluation)

---

### Stage 2: Attempt Loop (Max 3 Attempts)
**Function**: `_run_attempts_loop()`

**Flow**:
```
Attempt 1 → Generate P1 → Execute → Evaluate → PASS? → Stop
                ↓ FAIL
Attempt 2 → Generate P2 → Execute → Evaluate → PASS? → Stop
                ↓ FAIL
Attempt 3 → Generate P3 → Execute → Evaluate → PASS/FAIL → Stop
```

---

### Stage 3: Prompt Generation (Per Attempt)
**Function**: `_run_single_attempt()` → Prompt generation section

**For Attempt 1 (P1)**:
- **Inputs**: PR description + Human diff only
- **Template**: `templates.prompt1`
- **LLM**: `minimaxai/minimax-m2` (ALWAYS, regardless of execution model)
- **Output**: `p1_prompt.txt`

**For Attempt 2 (P2)**:
- **Inputs**: PR description + Human diff + Claude P1 diff + P1 evaluation feedback
- **Template**: `templates.prompt2`
- **LLM**: `minimaxai/minimax-m2` (ALWAYS)
- **Output**: `p2_prompt.txt`

**For Attempt 3 (P3)**:
- **Inputs**: PR description + Human diff + Claude P2 diff + P2 evaluation feedback
- **Template**: `templates.prompt3`
- **LLM**: `minimaxai/minimax-m2` (ALWAYS)
- **Output**: `p3_prompt.txt`

**Critical Rule**: Prompt generation ALWAYS uses `minimaxai/minimax-m2`, never the execution model.

---

### Stage 4: Repository Reset
**Function**: `_reset_repository()`

**Actions**:
1. `git reset --hard` (reset to HEAD)
2. `git clean -fd` (remove untracked files)

**Purpose**: Ensure clean state before each Claude Code execution.

---

### Stage 5: Claude Code Execution
**Function**: `_execute_claude_code()`

**Execution Contract**:
```bash
ANTHROPIC_BASE_URL="https://grid.ai.juspay.net" \
ANTHROPIC_AUTH_TOKEN="sk-uJfk3pIE2KcP9DoGx4UeHA" \
claude --model "<MAPPED_MODEL>" --print --permission-mode acceptEdits
```

**Model Mapping**:
- `glm-full` → `glm-full` (as-is)
- `minimax-m2` → `minimaxai/minimax-m2`

**Actions**:
1. Read prompt from `p<N>_prompt.txt`
2. Clean prompt (remove `<think>` tags)
3. Set environment variables
4. Execute Claude Code with prompt via stdin
5. Capture stdout → `stdout.txt`
6. Capture stderr → `stderr.txt`

**Outputs**:
- `p<N>/stdout.txt` - Claude Code output
- `p<N>/stderr.txt` - Error output (if any)

---

### Stage 6: Diff Capture
**Function**: `_capture_git_diff()`

**Actions**:
1. Run `git diff` in claude-work directory
2. Save to `p<N>/claude.diff`
3. Log if diff is empty (warning)

**Outputs**:
- `p<N>/claude.diff` - Git diff of changes made by Claude Code

**Critical**: If diff is empty, Claude Code made no changes.

---

### Stage 7: Evaluation
**Function**: `_evaluate_solution()`

**Inputs**:
- PR metadata (title, description, issue)
- Human approach summary
- Ground truth diff (`original_changes.diff`)
- Model diff (`claude.diff`)

**LLM Call**:
- **Model**: Execution model (same as Claude Code execution)
- **Template**: `templates.evaluation`
- **Purpose**: Compare model diff vs ground truth diff

**Evaluation Rules** (from Prompt.md):
- **PASS**: Changes are logically equivalent, semantically aligned
- **FAIL**: Changes don't solve core problem, different strategy, missing core logic

**Outputs**:
- `p<N>/eval.json` - Contains `verdict` (PASS/FAIL) and `reason`

**Critical Bug**: Currently, empty diffs can still be evaluated. This should be fixed.

---

### Stage 8: Retry Decision
**Function**: `_run_attempts_loop()`

**Logic**:
- If verdict == "PASS" → Stop immediately, return PASS
- If verdict == "FAIL" and attempts < 3 → Continue to next attempt
- If verdict == "FAIL" and attempts == 3 → Stop, return FAIL

---

### Stage 9: Final Results
**Function**: `run_pr()`

**Outputs**:
- `runs/PR-<PR>/<model>/final.json` - Final verdict and attempt count
- `runs/PR-<PR>/model_comparison.json` - Comparison if multiple models

**Final JSON Structure**:
```json
{
  "pr": "8916",
  "final_verdict": "PASS" | "FAIL",
  "attempts": 1-3,
  "passed_on": 1-3 | null
}
```

---

## Critical Invariants (Must Never Break)

1. ✅ **Prompt generation ALWAYS uses `minimaxai/minimax-m2`**
2. ✅ **Claude Code NEVER sees human diff** (only sees generated prompts)
3. ✅ **Git reset before every attempt**
4. ✅ **Base commit is `merge_commit^1`**
5. ✅ **Evaluation is diff-based and model-agnostic**
6. ✅ **Stop immediately on PASS**
7. ✅ **Stop after 3 attempts regardless**
8. ❌ **BUG: Empty diffs should auto-FAIL** (currently not enforced)

---

## Artifact Structure

```
runs/PR-8916/glm-full/
├── input/
│   ├── pr.md              # PR description
│   └── human.diff         # Ground truth
├── p1/
│   ├── p1_prompt.txt      # Generated prompt (LLM output)
│   ├── claude.diff        # Claude Code changes
│   ├── eval.json          # Evaluation result
│   ├── stdout.txt         # Claude Code stdout
│   └── stderr.txt         # Claude Code stderr
├── p2/ (same structure)
├── p3/ (same structure)
└── final.json             # Final verdict
```

---

## Data Flow Diagram

```
PR Description + Human Diff
        │
        ▼
[Prompt Generation LLM] (minimaxai/minimax-m2)
        │
        ▼
Generated Prompt (p1_prompt.txt)
        │
        ▼
[Claude Code Execution] (execution model, e.g., glm-full)
        │
        ▼
Code Changes (git diff)
        │
        ▼
[Evaluation LLM] (execution model)
        │
        ▼
PASS/FAIL Verdict
```

---

## Why P3 Got PASS with Empty Diff

**Root Cause**: The evaluation LLM received an empty `MODEL_DIFF_TEXT` but still evaluated it and passed. This is a bug because:

1. Empty diff = Claude Code made no changes
2. No changes = Cannot solve the problem
3. Should auto-FAIL, not evaluate

**Fix Needed**: Add validation to auto-FAIL if `claude.diff` is empty before calling evaluation LLM.

