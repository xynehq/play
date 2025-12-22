# PR Benchmarking Automation System

## Overview

This automation system benchmarks how well different AI models can implement real PR fixes when given only a natural-language task description. It uses a strict separation between reasoning (LLM), execution (Claude Code), and evaluation (LLM) to provide reproducible, fair, and auditable testing of coding AI models.

### Key Principle

**Claude Code NEVER thinks. Claude Code ONLY executes.**

- **LLM #1** → Generates execution prompt
- **Claude Code** → Applies code changes  
- **LLM #2** → Evaluates output

---

## Table of Contents

1. [Architecture & Flow](#architecture--flow)
2. [Key Features](#key-features)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Stage-by-Stage Flow](#stage-by-stage-flow)
6. [Command Reference](#command-reference)
7. [Artifact Structure](#artifact-structure)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Architecture & Flow

### System Architecture

```
┌─────────────────┐
│   run_pr CLI    │  Entry point: `./run_pr <PR_NUMBER> --models <MODEL>`
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  orchestrator   │  Main control flow
└────────┬────────┘
         │
         ├──► Prompt Generation (LLM: minimaxai/minimax-m2)
         ├──► Claude Code Execution (Model: specified via --models)
         └──► Evaluation (LLM: execution model)
```

### Process Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Stage 0     │    │ Stage 1-3   │    │ Stage 4-8   │
│ PR Setup    │───►│ Model Loop  │───►│ Evaluation  │
│ (Automated) │    │ (3 attempts)│    │ & Results   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Key Features

### 🎯 **Multi-Model Benchmarking**
- Test the same PR with multiple AI models
- Compare performance across models
- Single command for multiple models: `--models glm-full,minimax-m2`

### 🔄 **Iterative Retry Logic**
- Up to 3 attempts per model
- Each attempt learns from previous failures
- Immediate stop on PASS

### 🧠 **Smart Evaluation**
- Natural language-based evaluation
- Function-level context diffs for precise comparison
- Eliminates noise from formatting differences

### 🚀 **Automated Stage 0**
- Zero manual setup required
- Automatic PR metadata fetching
- Ground truth diff generation

### 📊 **Complete Artifact Tracking**
- Full audit trail of every step
- Generated prompts, diffs, evaluations
- Human approach summaries

### 🔒 **Strict Separation of Concerns**
- LLM reasoning never mixed with execution
- Claude Code never sees human solutions
- Reproducible and fair testing

---


### Environment Setup

The automation uses these API endpoints:
- **Base URL**: `https://grid.ai.juspay.net`
- **API Keys**: Configured in `scripts/api_config.py`

---

## Quick Start

### Automated Workflow (Recommended)

**Single command - everything runs automatically:**

```bash
cd /path/to/hyperswitch
./run_pr <PR_NUMBER> --models <MODEL_NAME>
```

**Examples:**

```bash
# Single model
./run_pr 8916 --models glm-full

# Multiple models (benchmarking)
./run_pr 8916 --models glm-full,minimax-m2

# Skip setup if already done
./run_pr 8916 --models glm-full --skip-setup
```

### What Happens Automatically

1. **Fetches PR metadata** from GitHub
2. **Creates isolated branch** for testing
3. **Generates task description** from PR
4. **Creates ground truth diff** from human fix
5. **Runs up to 3 attempts** per model
6. **Evaluates each attempt** with detailed feedback
7. **Provides final verdict** and comparison

---

## Stage-by-Stage Flow

### Stage 0: PR Setup (Automated)

**Module**: `scripts/pr_setup.py`

**Status**: ✅ **Fully Automated** - runs automatically

**Actions**:
- Fetch PR metadata from GitHub API
- Verify PR is merged
- Create isolated test branch
- Generate task description (`Data/pr_data/task_pr_<PR>.md`)
- Generate ground truth diff (`Data/pr_data/original_changes.diff`)
- Create function-level context diff

**Outputs**:
- `Data/pr_data/task_pr_<PR>.md` - Natural language task
- `Data/pr_data/original_changes.diff` - Human fix
- `Data/runs/PR-<PR>/human_context.diff` - Function-level ground truth

---

### Stage 1: Initialization

**Purpose**: Load inputs and generate shared artifacts

**Actions**:
1. Load PR description and metadata
2. Load human diff for reference
3. Load prompt templates
4. Generate human approach summary (using `minimaxai/minimax-m2`)

**Output**: `Data/runs/PR-<PR>/human_approach.txt`

---

### Stage 2: Model Loop

**Purpose**: Run automation for each specified model

**Flow**:
```
For each model in --models:
    ├── Create Data/runs/PR-<PR>/<model>/
    ├── Copy inputs to model directory
    └── Run attempt loop (max 3 attempts)
```

---

### Stage 3: Attempt Loop (Max 3 Attempts)

**Purpose**: Retry until PASS or exhaust attempts

**Flow**:
```
Attempt 1: Generate P1 prompt → Execute → Evaluate
    ↓ (if FAIL)
Attempt 2: Generate P2 prompt (with P1 feedback) → Execute → Evaluate  
    ↓ (if FAIL)
Attempt 3: Generate P3 prompt (with P2 feedback) → Execute → Evaluate
    ↓ (stop regardless)
```

**Stopping Conditions**:
- ✅ **PASS** → Stop immediately
- ✅ **3 attempts** → Stop regardless

---

### Stage 4: Prompt Generation

**Per Attempt**:
- **P1**: Uses PR description + human diff
- **P2**: Adds P1 diff + P1 evaluation feedback
- **P3**: Adds P2 diff + P2 evaluation feedback

**LLM Used**: `minimaxai/minimax-m2` (always, regardless of execution model)

**Output**: `p<N>/p<N>_prompt.txt`

---

### Stage 5: Claude Code Execution

**Function**: `_execute_claude_code()`

**Process**:
1. Reset repository to clean state
2. Execute Claude Code with generated prompt
3. Capture stdout and stderr
4. Generate diffs (raw + context-aware)

**Model Mapping**:
- `glm-full` → `glm-full` (as-is)
- `minimax-m2` → `minimaxai/minimax-m2`

**Outputs**:
- `p<N>/claude.diff` - Raw git diff
- `p<N>/claude_context.diff` - Function-level context diff
- `p<N>/stdout.txt` - Execution output
- `p<N>/stderr.txt` - Error output

---

### Stage 6: Evaluation

**Function**: `_evaluate_solution()`

**Process**:
1. Compare model changes with human fix
2. Use function-level context diffs as primary signal
3. Generate natural language evaluation
4. Decide PASS/FAIL with detailed reasoning

**LLM Used**: Execution model (same as Claude Code)

**Output**: `p<N>/eval.json` with verdict and reason

---

## Command Reference

### Main Commands

**Run automation**:
```bash
# Single model
./run_pr 8916 --models glm-full

# Multiple models (benchmarking)
./run_pr 8916 --models glm-full,minimax-m2

# Skip Stage 0 setup
./run_pr 8916 --models glm-full --skip-setup

# View help
./run_pr --help
```

**Context diff operations**:
```bash
# Generate context diff for current changes
python3 scripts/context_diff.py --staged

# Generate context diff between commits
python3 scripts/context_diff.py HEAD~1 HEAD

# Save context diff to file
python3 scripts/context_diff.py --staged -o context.diff
```

---

## Check Results

### Final Results

**View final verdict**:
```bash
cat Data/runs/PR-8916/glm-full/final.json
```

**Multi-model comparison**:
```bash
cat Data/runs/PR-8916/model_comparison.json
```

### Per-Attempt Details

**View generated prompt**:
```bash
cat Data/runs/PR-8916/glm-full/p1/p1_prompt.txt
```

**View Claude Code changes**:
```bash
cat Data/runs/PR-8916/glm-full/p1/claude.diff
```

**View function-level diff**:
```bash
cat Data/runs/PR-8916/glm-full/p1/claude_context.diff
```

**View evaluation result**:
```bash
cat Data/runs/PR-8916/glm-full/p1/eval.json
```

**View human approach**:
```bash
cat Data/runs/PR-8916/human_approach.txt
```

---

## Artifact Structure

### Complete Directory Tree

```
automation_2/
├── Data/
│   └── pr_data/                      # Stage 0 outputs
│       ├── task_pr_8916.md          # PR task description
│       └── original_changes.diff     # Ground truth diff
│
├── Data/runs/                        # All run outputs
│   └── PR-8916/
│       ├── human_approach.txt        # Human approach summary
│       ├── human_context.diff        # Function-level ground truth
│       ├── model_comparison.json     # Multi-model results
│       │
│       └── glm-full/                 # Model-specific run
│           ├── input/                # Input files for this model
│           ├── p1/                   # Attempt 1
│           ├── p2/                   # Attempt 2 (if needed)
│           ├── p3/                   # Attempt 3 (if needed)
│           └── final.json            # Final verdict
│
├── scripts/                          # Core modules
│   ├── api_config.py                 # API configuration
│   ├── llm_client.py                 # LLM client wrapper
│   ├── prompt_builder.py             # Template processing
│   ├── pr_setup.py                   # PR setup automation
│   └── context_diff.py               # Context diff generation
│
├── templates/
│   └── Prompt.md                     # LLM prompt templates
│
└── orchestrator.py                   # Main automation logic
```

---

## Examples

### Example 1: Single Model Test

```bash
# Test PR 8916 with glm-full model
./run_pr 8916 --models glm-full

# Expected output structure:
Data/runs/PR-8916/glm-full/
├── final.json          # {"pr": "8916", "final_verdict": "PASS", "attempts": 2, "passed_on": 2}
├── p1/p1_prompt.txt    # Generated execution prompt
├── p1/claude.diff      # Claude Code changes
├── p1/eval.json        # {"verdict": "FAIL", "reason": "..."}
├── p2/p2_prompt.txt    # Improved prompt with feedback
├── p2/claude.diff      # Updated changes
├── p2/eval.json        # {"verdict": "PASS", "reason": "..."}
└── human_approach.txt  # Human solution summary
```

### Example 2: Multi-Model Benchmark

```bash
# Compare multiple models on same PR
./run_pr 8916 --models glm-full,minimax-m2

# Results in model_comparison.json:
{
  "glm-full": {
    "pr": "8916",
    "final_verdict": "PASS",
    "attempts": 2,
    "passed_on": 2
  },
  "minimax-m2": {
    "pr": "8916", 
    "final_verdict": "FAIL",
    "attempts": 3,
    "passed_on": null
  }
}
```

### Example 3: Custom Setup

```bash
# If you need custom issue context, use manual setup first
./pr_analyzer.sh 8916

# Then run automation (skips Stage 0)
./run_pr 8916 --models glm-full --skip-setup
```

---

## Core Invariants

### These Rules Must Never Be Broken

1. ✅ **Prompts are ALWAYS generated by LLM, never handwritten**
2. ✅ **Claude Code NEVER sees human diff** (only generated prompts)
3. ✅ **Git state is reset before every attempt**
4. ✅ **Base commit is `merge_commit^1`** (one commit before PR merge)
5. ✅ **Prompt generation ALWAYS uses `minimaxai/minimax-m2`** (regardless of execution model)
6. ✅ **Stop immediately on PASS**
7. ✅ **Stop after 3 attempts regardless of outcome**

---

## Troubleshooting

### Common Issues

#### Branch Already Exists
```bash
# Error: Branch 'test-claude-pr-XXXX' already exists
git branch -D test-claude-pr-8916
./run_pr 8916 --models glm-full
```

#### Empty Claude Diff
```bash
# Check if Claude Code executed
cat Data/runs/PR-8916/glm-full/p1/stderr.txt
cat Data/runs/PR-8916/glm-full/p1/stdout.txt

# Test Claude Code manually
ANTHROPIC_BASE_URL="https://grid.ai.juspay.net" \
ANTHROPIC_AUTH_TOKEN="<token>" \
echo "Add a comment" | claude --model glm-full --print --permission-mode acceptEdits
```

#### Missing Context Diff
```bash
# Ensure changes are staged
git status --porcelain
python3 scripts/context_diff.py --staged -o test.diff
```

#### Evaluation Issues
```bash
# Check evaluation prompt structure
grep -A 10 "PRIMARY SIGNAL" templates/Prompt.md

# Verify context diffs are being passed
grep -A 5 "HUMAN_CONTEXT_DIFF" templates/Prompt.md
```

---

## Best Practices

1. **Always run Stage 0 first** (automatic or manual setup)
2. **Check base commit** - Ensure you're on `merge_commit^1`
3. **Verify inputs exist** - Check generated files are non-empty
4. **Monitor execution** - Watch stdout/stderr for errors
5. **Review diffs** - Check both raw and context diffs
6. **Multi-model testing** - Use comma-separated models for benchmarking

---

## Summary

This automation system provides a comprehensive, reproducible benchmark for testing AI coding models on real-world bug fixes.

**Key Benefits**:
- 🎯 **Fair Comparison** - Same PR, same evaluation criteria
- 🔄 **Retry Logic** - Models get multiple chances to succeed
- 📊 **Detailed Metrics** - Complete audit trail of attempts
- 🧠 **Natural Evaluation** - Human-like assessment of solutions
- 🚀 **Zero Setup** - Automated from start to finish

**Perfect For**:
- Comparing AI coding models
- Benchmarking model improvements
- Testing new models on historical bugs
- Academic research on AI coding

For questions or issues, refer to the troubleshooting section or check the code comments in `orchestrator.py`.
