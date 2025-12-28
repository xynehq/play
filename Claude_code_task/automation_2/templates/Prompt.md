———————Prompt for getting Human Approach from real diff:———————————
## [HUMAN_APPROACH]
You are a senior backend engineer reviewing a pull request that fixed a real bug.

I will give you:
- The PR title and high-level description.
- The final git diff that was merged to fix the issue.

Your task is to reconstruct the HUMAN APPROACH: explain in a single 4–5 line paragraph, in natural language, how a human developer understood the issue and approached fixing it, focusing on intent, reasoning, and conceptual flow.
Do not include any code, file paths, function names, or low-level details. Emphasize how the developer ensured the fix was complete, checked propagation, considered symmetry, and handled parallel or secondary flows.
Do NOT generate <think> tags, reasoning, or step-by-step thought processes; output ONLY the requested 4–5 line text or JSON format.
===== PR METADATA =====
## [pr]
Title:
<PR_TITLE>

Description (from PR):
<PR_DESCRIPTION>

===== FINAL MERGED DIFF (GROUND TRUTH) =====
<GROUND_TRUTH_DIFF_TEXT>

===== INSTRUCTIONS =====
Write a single 4-5 line paragraph that explains, at a conceptual level, what the human developer understood about the issue and how they approached fixing it. Avoid all low-level or code-specific details.

——————————————Prompt to get First Prompt:————————————
## [PROMPT_1]
You are helping me design a high-quality coding instruction for Claude Code. Claude Code is an agentic coding tool running in the terminal that deeply understands the repository and its execution flow.

Goal: Given only the buggy codebase (one commit before the real fix), we want Claude Code to implement a correction/solution that mirrors the real human solution. The final fix does not need to match line-for-line, but it must follow the same logic, produce the same behavior, and avoid introducing any new or unintended side effects.
I will give you:
- PR title
- PR description
if the description is missing or insufficient, you must infer the developer’s intent using very minimal, high-level signals from the human context diff, only enough for a competent engineer to recognize what class of bug exists and what behavior must change, without revealing structure, functions, or implementation details.

Do NOT generate any internal reasoning, <think> tags, explanations, or step-by-step thought processes; output ONLY the requested text in the specified format.
You must produce a single prompt that I can copy-paste into Claude Code so that it can arrive at a solution .
===== PR METADATA =====
Title:
<PR_TITLE>
Original PR description:
<PR_DESCRIPTION>

===== INSTRUCTIONS FOR YOU =====
1. Write the prompt as if speaking directly to Claude Code.
2. Include:
 - A restatement of the core problem in your own words.
 - A clear description of the intended behavior and constraints.
 - Hints about which parts of the system Claude Code should focus on.
3. Keep the final prompt between 4–5 lines.
Output ONLY the prompt text, with no commentary and no markdown.

———————————Prompt to get Second and Third Prompt—————————
## [PROMPT_2and3]
You are helping me generate a corrective follow-up prompt for Claude Code after a prior attempt failed due to incomplete, shallow, or narrowly scoped logic. Claude Code must discard its earlier assumptions and re-reason from first principles, as a human engineer would after a detailed review.
Using the human approach summary, evaluation feedback, and minimal hints inferred from the human context diff, rewrite the prompt as a single 4–5 line paragraph that steers Claude Code toward the full intended behavior without revealing the complete human approach. 
Apply **hint strength based on attempt number**: for the second attempt, give light guidance—just subtle cues about propagation, symmetry, or secondary flows; for the third attempt, provide near-moderate guidance—stronger directional nudges about missing paths, supporting logic, and behavioral completeness, still without revealing implementation details.
Direct Claude Code to validate end-to-end behavior across all equivalent or parallel flows, assume any missing or asymmetric handling is a bug, and approach the fix as a human would when preparing a change confident enough to merge without follow-up patches.
Just give hints inferred from the context diff; do not reveal the complete human approach.

Explicitly ensure that every function, structure, and data path involved is updated, covering all edge cases, parallel flows, and secondary propagation; assume any missing handling is a bug.
Output ONLY the new 4–5 line prompt text, with no commentary, markdown, file paths, symbols, code, or references to diffs or prior attempts.
Do NOT generate any internal reasoning, <think> tags, explanations, or step-by-step thought processes; output ONLY the requested text in the specified format.

===== PREVIOUS PROMPT =====
<PREVIOUS_PROMPT_TEXT>
===== DIFF OF PREVIOUS ATTEMPT =====
<DIFF of Previous ATTEMPT>
===== EVALUATION OF MODEL ATTEMPT =====
<EVALUATION_SUMMARY_FOR_PREVIOUS_ATTEMPT>
=====Human Approach (Summary)=======
<HUMAN_APPROACH_SUMMARY>

===== INSTRUCTIONS FOR YOU =====
	1	Start from the structure and intent of the previous prompt, but rewrite it into a 4–5 line paragraph using human approach summary.
	2	Tell Claude Code not to repeat the types of mistakes it made previously, again in broad terms.
	3	Emphasize that it should focus more precisely on the intended behavior and correct overall logic, while staying within the relevant area of the codebase.
	4	Avoid explicit file paths, symbols, code, patches, or human approach details.
	5	Output ONLY the new 4–5 line paragraph prompt, with no commentary or markdown.

———————————————Summarize Claude Code's approach, compare with real changes, and decide Pass/Fail———————————
# [EVALUATION] — Developer-Focused, Strict, Unbiased Version

You are a senior Hyperswitch engineer reviewing a model-generated fix exactly as you would during a real PR review.

Your task is to judge **correctness, completeness, and behavioral accuracy** of the model’s fix.
Do NOT judge intent, explanation quality, or reasoning style.

Claude Code is an execution tool. Evaluate **ONLY observable behavior**.

You are given:
- PR title and issue description
- Human-written approach summary (context only)
- Ground-truth diff representation (tier-dependent)
- Model-generated diff (tier-dependent)

──────────────── DIFF REPRESENTATION TIERS ────────────────

Exactly **ONE tier** is used for this evaluation and is declared at the start.

**Tier-1 (Raw Diff)**
- Full diff, full precision
- Used when: raw_diff ≤ 35k chars
- Multiplier: 1.00

**Tier-2 (Context Diff)**
- Function / logical-flow level diff
- Used when: raw_diff > 35k AND context_diff ≤ 20k
- Multiplier: 0.95

**Tier-3 (Structured Summary)**
- Behavioral summary only
- Used when both diffs exceed limits
- Multiplier: 0.85

The declared tier determines the **ONLY PRIMARY SIGNAL** for evaluation.

──────────────── PRIMARY SIGNAL RULE ────────────────

You MUST base your verdict **only** on the PRIMARY SIGNAL for the declared tier.
Reference material (if provided) is for sanity checking only and MUST NOT override conclusions from the primary signal.

──────────────── TIER-SPECIFIC RULES ────────────────

### Tier-1 (Raw Diff)
Primary Signal:
- `GROUND_TRUTH_DIFF_TEXT` vs `MODEL_DIFF_TEXT`

Rules:
1. Compare **line-by-line and semantic behavior**.
2. All required behavior, edge cases, and **propagation across all affected files, functions, and modules** must be present.
3. Refactors allowed only if behavior is identical.
4. Ignore harmless formatting or whitespace differences.
5. **PASS:** The fix matches the ground truth exactly in behavior, logic, and propagation; no missing or unsafe updates.
6. **FAIL:** Any missing logic, broken propagation, unsafe/incorrect changes, or partial/asymmetric fix.

### Tier-2 (Context Diff)
Primary Signal:
- `HUMAN_CONTEXT_DIFF` vs `MODEL_CONTEXT_DIFF`

Rules:
1. Compare behavior at function / logical-flow level.
2. Ignore formatting, ordering, naming, or whitespace.
3. Core bug must propagate through **all relevant code paths and modules**.
4. **PASS:** All required behavior implemented, propagation correct across all functions/modules, no unsafe or missing updates.
5. **FAIL:** Missing or incomplete behavior, broken propagation, unsafe/incorrect logic, or partial/inconsistent fix.

### Tier-3 (Structured Summary)
Primary Signal:
- `DIFF_SUMMARY.behavioral_changes`

Rules:
1. Only listed behaviors in summary are considered. Missing implementation details are not automatic FAIL unless they affect critical behavior.
2. Evaluate behavior correctness, safety, and alignment with PR goal.
3. **PASS:** All listed behaviors implemented correctly and fully; no unsafe deviations; propagation across critical paths verified.
4. **FAIL:** Any listed behavior missing, incorrect, unsafe, or critical call sites affected by core logic are missing.

──────────────── DEVELOPER INTUITION GUIDE ────────────────

When judging the fix, think like a senior engineer:

1. Does the model fix **fully address the core bug** described in the PR?
2. Are all **required behaviors propagated correctly** across all affected code paths, call sites, and modules?
3. Has the model introduced any **unsafe, incorrect, or unexpected behavior**?
4. Are differences limited to harmless **refactors, formatting, or ordering**?
5. If any critical logic is missing or behavior is uncertain, mark **FAIL**.
6. **Bias-free, all-or-nothing evaluation:**
   - Only FAIL if the fix is truly incomplete, unsafe, or misses required behavior.
   - "Harmless differences" refers ONLY to: formatting, variable naming, or equivalent refactors that preserve behavior.
   - Missing logic, incomplete propagation, or broken behavior is NEVER a "harmless difference."
   - The fix must be fully complete and correct to PASS - partial or incomplete fixes are FAIL.

──────────────── ABSOLUTE STRICTNESS RULES ────────────────

1. ALL affected files, functions, modules, and call sites in the ground truth (Tier-1) or context diff (Tier-2) **must be updated**. Missing any → FAIL.
2. Partial fixes or skipped modules are **not acceptable**.
3. Any incorrect, unsafe, or redundant logic deviating from the ground truth → FAIL.
4. Tier-3 summaries: if any critical behavior from the summary or inferred from context is missing → FAIL.
5. Evaluation is **all-or-nothing**: the fix must be fully complete and correct to PASS.
6. Harmless refactors, formatting, or variable renames are **not penalized**.

──────────────── CONTEXT INPUTS ────────────────

PR Title  
<PR_TITLE>

Issue  
<PR_ISSUE_DESCRIPTION>

Human Approach Summary (context only)  
<HUMAN_APPROACH_SUMMARY>

Tier-3 Summary (if applicable)  
<DIFF_SUMMARY>

Tier-2 Context Diff (if applicable)  
<HUMAN_CONTEXT_DIFF>  
<MODEL_CONTEXT_DIFF>

Tier-1 Raw Diff (if applicable)  
<GROUND_TRUTH_DIFF_TEXT>  
<MODEL_DIFF_TEXT>

──────────────── VERDICT RULES ────────────────

Return exactly ONE verdict.

**PASS:**
- Fix fully replicates required behavior per tier primary signal.
- Core bug fully resolved.
- Behavior propagated correctly across **all affected files, functions, and modules**.
- No unsafe, incorrect, or invalid changes.
- Differences limited to harmless refactors, formatting, or ordering.

**FAIL:**
- Core bug not fixed or partially fixed.
- Missing behavior, incomplete propagation, or skipped call sites/files.
- Unsafe, incorrect, or redundant logic introduced.
- Any uncertainty about correctness → FAIL.

──────────────── OUTPUT FORMAT (STRICT) ────────────────

For Tier-3 evaluation, you MUST output PLAIN TEXT with a VERDICT line. NO JSON, NO YAML, NO MARKDOWN TABLES.

=== TIER 3 EVALUATION ===
VERDICT: <PASS | FAIL | FAIL_INFRA>

REASON:
[Explain the behavioral comparison result]

CONFIDENCE: <0.0 to 1.0>

──────────────── BEHAVIORAL COMPARISON RULES ────────────────

Tier-3 evaluation compares BEHAVIORAL SUMMARIES of human vs Claude diff:

- If Claude summary misses any PRIMARY behavior from Human → FAIL
- If Claude summary introduces NEW behavior not in Human → FAIL
- If behavior is partially matched → FAIL
- If behavior matches exactly in substance → PASS
- If comparison cannot be done safely → FAIL_INFRA

──────────────── FAIL_INFRA RULES ────────────────

Tier-3 MUST return FAIL_INFRA when:
- Behavioral summary cannot be generated (output is empty, unparsable, or "BEHAVIOR: UNKNOWN")
- Comparison is unsafe or indeterminate
- Any input is missing or malformed

Do NOT hedge or qualify. If you cannot determine PASS or FAIL, the verdict MUST be FAIL_INFRA.


============================== TIER 3 BEHAVIORAL SUMMARY PROMPT ==============================

## [BEHAVIORAL_SUMMARY]

You are generating a BEHAVIORAL SUMMARY from a code diff.

THIS SUMMARY IS USED FOR STRICT IMPLEMENTATION COMPARISON.
Behaviorally equivalent but structurally different changes MUST be treated as DIFFERENT.

--------------------------------
STRICT RULES (NON-NEGOTIABLE)
--------------------------------

Your summary MUST match the diff EXACTLY.

INCLUDE (everything that changes in the diff):
- All files modified (exact paths)
- All functions, methods, or blocks changed
- Exact logic changes introduced or removed
- Control flow changes (branches, loops, early returns)
- State, lifecycle, or variable initialization changes
- Error handling changes
- Edge cases explicitly handled

DO NOT INCLUDE:
- Line counts or formatting
- Refactors without behavioral impact
- Speculation or inferred intent
- Equivalent behavior implemented differently
- Any change not explicitly visible in the diff

--------------------------------
CRITICAL CONSTRAINTS
--------------------------------

- If the diff changes implementation structure, your summary MUST reflect that structure.
- If two diffs achieve the same outcome via different logic, they are NOT equivalent.
- Behavioral similarity does NOT imply correctness.
- Missing a change = FAILURE.
- Adding a change not in the diff = FAILURE.
- Do not infer correctness or mark changes as equivalent.

--------------------------------
VALIDATION GUARANTEES
--------------------------------

Your summary will be validated against the raw diff.

FAIL CONDITIONS:
- Any diff change missing from the summary
- Any summary claim not supported by the diff
- Any disagreement between summary and diff
- Any attempt to generalize, abstract, or infer logic

--------------------------------
OUTPUT FORMAT (STRICT — NO DEVIATION)
--------------------------------

BEHAVIORAL SUMMARY:
- Files changed:
- Primary behavior change:
- Secondary effects (ONLY direct impact caused by diff changes, do NOT infer external effects):
- Error handling changes:
- Edge cases affected:

--------------------------------
UNKNOWN SAFETY
--------------------------------

If the behavior or logic cannot be determined safely and completely from the diff, output EXACTLY:

BEHAVIOR: UNKNOWN

⚠️ When this output occurs:
- The orchestrator/eval pipeline must treat this attempt as SKIPPED_CONTEXT_OVERFLOW or ERROR.
- Do NOT invent changes or infer behavior.

--------------------------------
LARGE DIFF / CHUNK HANDLING
--------------------------------

- If the diff is too large to fit in model context:
    - Split the diff into chunks.
    - Summarize **each chunk independently** using this same strict format.
    - Do not omit any changes; do not invent behavior.
    - Combine all chunk summaries at the end in the same strict output format.

--------------------------------
DIFF INPUT
--------------------------------

<DIFF>  

Notes:
- <DIFF> is the raw git unified diff including file paths, line numbers, and context lines.
- The model must rely only on the explicit content of this diff.
