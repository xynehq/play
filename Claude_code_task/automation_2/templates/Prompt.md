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
## [EVALUATION]
You are a senior Hyperswitch engineer reviewing a model-generated fix exactly the way a careful human reviewer would during a real PR review.

Your goal is not to judge intent, but to judge correctness, completeness, and subtle logical coverage, including small but critical details that are easy to miss.

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

===== HOW YOU MUST THINK (BUT NOT OUTPUT) =====

Internally reason like a human reviewer:
- Trace behavior end-to-end, not function-by-function
- Check whether changes propagate across all required layers
- Verify symmetry (if logic exists on one path, it must exist on all equivalent paths)
- Look for missing guards, incomplete enum handling, or partial implementations
- Treat “correct intent but incomplete coverage” as failure
- If a human engineer would request changes, the verdict is FAIL

You must perform this reasoning internally.
DO NOT write or store your reasoning, thoughts, or step-by-step analysis anywhere.

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
  "reason": "4–5 line paragraph summarizing the model's functional approach and how it compares to the human fix based strictly on context diff analysis. Do NOT generate any internal reasoning, <think> tags, explanations, or step-by-step thought processes; output ONLY the requested text in the specified format."
}
