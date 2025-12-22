———————Prompt for getting Human Approach from real diff:———————————
## [HUMAN_APPROACH]
You are a senior backend engineer reviewing a pull request that fixed a real bug.

I will give you:
- The PR title and high-level description.
- The final git diff that was merged to fix the issue.

Your job is to reconstruct the HUMAN APPROACH: a explanation in simple terms written as a natural paragraph.
Do not include any code, file paths, or specific variable or function names.
Focus on intent, reasoning, and conceptual flow of the solution.

Your output must be one paragraph of 4-5 lines, nothing else.
No lists, no headings, no summaries.

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
You are helping me design a high-quality coding instruction for Claude Code. Claude Code is an agentic coding tool in the terminal that understands the codebase.
Goal: Given only the buggy codebase (one commit before the real fix), we want Claude Code to implement a correction/solution that mirrors the real human solution. The final fix does not need to match line-for-line, but it must follow the same logic, produce the same behavior, and avoid introducing any new or unintended side effects.
I will give you:
- PR title
- PR description
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
You are helping me iteratively improve a prompt for Claude Code. Claude Code is an agentic coding tool in the terminal that understands the codebase. It will be the next consecutive prompt after the previous prompt was given, where the model's earlier attempt was incomplete or incorrect. I will give you the previous prompt, Claude Code's diff, and a natural-language evaluation of what went wrong and the real human approach. Your task is to produce a NEW prompt for the next attempt that provides guidance using the human approach summary to arrive at the same solution as the human and fits into a single 4–5 line paragraph.
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

## [EVALUATION]

You are a senior Hyperswitch engineer evaluating a model-generated fix against a known correct human fix.

You are given:
- PR title and issue description
- A human-written approach summary (supporting context only)
- Function-level context diffs for BOTH the human fix and the model fix
- Raw git diffs for reference only

Claude Code is an agentic coding tool that attempted to fix the issue.

===== EVALUATION PRINCIPLES =====

CRITICAL RULE:
The HUMAN_CONTEXT_DIFF is the absolute ground truth.
It defines the required behavior, propagation, and structure.
The MODEL_CONTEXT_DIFF must be evaluated strictly relative to it.

Human approach summaries and PR descriptions provide background context only.
They must NOT override or excuse missing behavior in the human context diff.

Context diffs operate at the function / logical-flow level.
Ignore formatting, ordering, refactors, naming, and whitespace.
Do NOT reason line-by-line.

Raw diffs are reference-only.
They must NEVER override conclusions drawn from context diffs.

===== CONTEXT INPUTS =====

PR Title
<PR_TITLE>

Issue
<PR_ISSUE_DESCRIPTION>

Human Approach (Summary)
<HUMAN_APPROACH_SUMMARY>

PRIMARY SIGNAL – Function-Level Context Diffs

Human Context Diff (Ground Truth)
<HUMAN_CONTEXT_DIFF>

Model Context Diff (Claude Code Attempt)
<MODEL_CONTEXT_DIFF>

REFERENCE ONLY – Raw Diffs

Real Merged Diff (Ground Truth – Raw)
<GROUND_TRUTH_DIFF_TEXT>

Model Diff (Claude Code Attempt – Raw)
<MODEL_DIFF_TEXT>

===== HOW TO EVALUATE =====

1. Compare HUMAN_CONTEXT_DIFF vs MODEL_CONTEXT_DIFF
   - Identify which functions were changed
   - Verify that every required functional change present in the human diff
     is present or behaviorally equivalent in the model diff

2. Focus on:
   - End-to-end propagation across layers
   - Enum exhaustiveness and symmetry
   - Control flow, guards, and error handling
   - Supporting logic required for correctness

3. Ignore:
   - Drive-by cleanups in the human diff
   - Unrelated refactors
   - Harmless structural differences

===== VERDICT RULES =====

Return exactly ONE verdict:

PASS:
- All behaviorally required changes in the human context diff are present
- Model implementation is logically and semantically equivalent
- No required propagation or supporting logic is missing

FAIL:
- Any required behavior or propagation from the human context diff is missing or incorrect
- Core logic deviates from the human fix
- Any function-level change required by the human diff is absent

If there is any uncertainty about completeness, choose FAIL.

===== OUTPUT FORMAT (STRICT) =====

Return exactly this JSON and nothing else:

{
  "verdict": "PASS | FAIL",
  "confidence": 0.0,
  "summary": "4–5 line paragraph summarizing the model’s functional approach and how it compares to the human fix based strictly on context diff analysis"
}

