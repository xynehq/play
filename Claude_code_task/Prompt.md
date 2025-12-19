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
You are an expert code reviewer benchmarked against a known "ground truth" fix.
I will give you:
The PR title.
The related Issue description.
A high-level human approach summary derived from the real merged PR.
The real ground-truth git diff.
The git diff produced by Claude Code for the same task. Claude Code is an agentic coding tool in the terminal that understands the codebase.
Your job is to:
Produce a single 4–5 line Summary paragraph describing what the model tried to do and how it compares to the real fix.
Decide PASS or FAIL according to the benchmark rules below.

===== BENCHMARK RULES =====
PASS when:
The model's changes are logically equivalent, semantically aligned, or follow the same intent as the human fix based on the PR title, issue, and human approach summary.
The model is clearly on the right track, even if some names, struct shapes, or small details differ.
Any differences are minor refactors, stylistic differences, or harmless omissions.
The model does not introduce unrelated or harmful behavior.
If the human diff includes changes that are:
renames,
cleanup,
refactors,
small behavior tweaks unrelated to the PR title/issue/human-approach summary,
then do NOT mark those as missing—these are irrelevant for evaluation.
FAIL when:
The model's changes do not solve the core problem described by the PR title/issue/human-approach summary.
The model uses a different strategy that does not actually address the bug.
The model introduces incorrect, dangerous, or invalid behavior.
The model misses the core fix logic required by the PR title/issue.
If unsure, lean toward FAIL, but do NOT penalize harmless or unrelated differences.
Important evaluation clarification
Only compare overlapping, relevant portions of the diff.
If the human diff includes changes not implied by the PR title/issue/human approach (drive-by cleanups, renames, unrelated tweaks),
ignore them—they should NOT affect the verdict.
Evaluate only the core change described by the PR title, issue, and human approach summary.

===== CONTEXT INPUTS =====
PR Title
<PR_TITLE>
Issue
<PR_ISSUE_DESCRIPTION>
Human Approach (Summary)
<HUMAN_APPROACH_SUMMARY>
Real Merged Diff (Ground Truth)
<GROUND_TRUTH_DIFF_TEXT>
Model Diff (Claude Code Attempt)
<MODEL_DIFF_TEXT>

===== INSTRUCTIONS FOR YOUR OUTPUT =====
Produce exactly two sections:
1. Summary (4–5 line paragraph)
Write a concise 4–5 line paragraph summarizing Claude's approach and how the model's work aligns with or diverges from the real human fix, focusing only on changes relevant to the PR title, issue, and human approach summary. Mention whether the important logic is present, whether the model's implementation follows the same intent, and whether any meaningful parts are missing—while ignoring unrelated human diff changes.
2. Verdict
A single line:
Verdict: PASS
or
Verdict: FAIL
Then 1–3 short sentences explaining why, based solely on the benchmark rules and relevance to the PR title + issue (not unrelated differences).