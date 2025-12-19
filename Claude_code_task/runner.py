from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from llm_client import LLMClient
from prompt_builder import Templates, fill_placeholders, load_templates


def parse_pr_description(desc_text: str) -> Dict[str, str]:
    """
    Extract title and description from pr_description.md.
    Takes the first markdown heading as title; the rest is description.
    """
    lines = desc_text.splitlines()
    title = ""
    description_lines = []
    for line in lines:
        if not title and line.startswith("#"):
            title = line.lstrip("# ").strip()
            continue
        description_lines.append(line)
    return {
        "PR_TITLE": title or "Unknown title",
        "PR_DESCRIPTION": "\n".join(description_lines).strip() or "No description provided.",
    }


@dataclass
class StageResult:
    prompt_text: str
    model_patch: str


def _save(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _stage(
    stage_prefix: str,
    template_text: str,
    replacements: Dict[str, str],
    model_patch: str,
    run_dir: Path,
    client: LLMClient,
) -> StageResult:
    print(f"\n{'='*80}")
    print(f"STAGE {stage_prefix.upper()}")
    print(f"{'='*80}")
    
    print(f"\n[1/2] Building meta-prompt from template...")
    meta_prompt = fill_placeholders(template_text, replacements)
    _save(run_dir / f"{stage_prefix}_input_prompt.txt", meta_prompt)
    print(f"✓ Meta-prompt saved to {run_dir / f'{stage_prefix}_input_prompt.txt'}")
    print(f"\n--- PROMPT PREVIEW (first 500 chars) ---")
    print(meta_prompt[:500] + "..." if len(meta_prompt) > 500 else meta_prompt)
    print(f"--- END PREVIEW ---\n")

    print(f"[2/2] Calling LLM to generate prompt text...")
    prompt_text = client.run(meta_prompt)
    _save(run_dir / f"{stage_prefix}_prompt.txt", prompt_text)
    print(f"✓ Generated prompt saved to {run_dir / f'{stage_prefix}_prompt.txt'}")

    print(f"[INFO] Persisting provided model diff...")
    _save(run_dir / f"{stage_prefix}_model_diff.patch", model_patch)
    print(f"✓ Model patch saved to {run_dir / f'{stage_prefix}_model_diff.patch'}")
    if model_patch:
        print(f"   Patch length: {len(model_patch)} chars")
        print(f"   Patch preview (first 300 chars):")
        print(f"   {model_patch[:300]}...")
    else:
        print(f"   ⚠ WARNING: Empty model diff provided.")

    return StageResult(
        prompt_text=prompt_text,
        model_patch=model_patch,
    )


def _build_replacements(
    pr_text: str,
    human_patch: str,
    model_patch: str = "",
    previous_prompt: str = "",
) -> Dict[str, str]:
    meta = parse_pr_description(pr_text)
    return {
        "TASK_DETAILS": pr_text,
        "HUMAN_DIFF": human_patch,
        "MODEL_DIFF": model_patch,
        "PREVIOUS_PROMPT_TEXT": previous_prompt,
        "EVALUATION_SUMMARY_FOR_PREVIOUS_ATTEMPT": "",
        "PR_TITLE": meta["PR_TITLE"],
        "PR_DESCRIPTION": meta["PR_DESCRIPTION"],
        "HUMAN_APPROACH_SUMMARY": "",
    }


def run_pipeline(
    pr_id: str,
    pr_description_file: Path,
    human_diff_file: Path,
    model_diff_stage2_file: Optional[Path],
    model_diff_stage3_file: Optional[Path],
    template_file: Path,
    runs_dir: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
) -> Dict:
    print(f"\n{'#'*80}")
    print(f"# MULTI-PROMPT LLM AUTOMATION PIPELINE")
    print(f"# PR ID: {pr_id}")
    print(f"{'#'*80}\n")
    
    print(f"[INIT] Setting up pipeline...")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / f"PR-{pr_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Run directory: {run_dir}")

    print(f"[INIT] Loading inputs...")
    pr_text = pr_description_file.read_text(encoding="utf-8")
    human_patch = human_diff_file.read_text(encoding="utf-8")
    model_patch_stage2 = ""
    if model_diff_stage2_file is not None:
        model_patch_stage2 = model_diff_stage2_file.read_text(encoding="utf-8")
    model_patch_stage3 = ""
    if model_diff_stage3_file is not None:
        model_patch_stage3 = model_diff_stage3_file.read_text(encoding="utf-8")
    print(f"✓ PR description: {pr_description_file} ({len(pr_text)} chars)")
    print(f"✓ Human diff: {human_diff_file} ({len(human_patch)} chars)")
    if model_diff_stage2_file is not None:
        print(f"✓ Model diff (stage 2): {model_diff_stage2_file} ({len(model_patch_stage2)} chars)")
    else:
        print(f"ℹ No model diff provided; Stage 1 only until supplied.")
    if model_diff_stage3_file is not None:
        print(f"✓ Model diff (stage 3): {model_diff_stage3_file} ({len(model_patch_stage3)} chars)")
    else:
        print(f"ℹ No model diff provided; Stage 1 only.")

    # Persist inputs for this run (skip if already in run_dir).
    if pr_description_file.resolve() != (run_dir / pr_description_file.name).resolve():
        shutil.copyfile(pr_description_file, run_dir / pr_description_file.name)
    if human_diff_file.resolve() != (run_dir / human_diff_file.name).resolve():
        shutil.copyfile(human_diff_file, run_dir / human_diff_file.name)
    if model_diff_stage2_file is not None:
        if model_diff_stage2_file.resolve() != (run_dir / f"model_diff_stage2.diff").resolve():
            shutil.copyfile(model_diff_stage2_file, run_dir / f"model_diff_stage2.diff")
    if model_diff_stage3_file is not None:
        if model_diff_stage3_file.resolve() != (run_dir / f"model_diff_stage3.diff").resolve():
            shutil.copyfile(model_diff_stage3_file, run_dir / f"model_diff_stage3.diff")
    print(f"✓ Inputs copied to run directory")

    print(f"[INIT] Loading templates...")
    templates: Templates = load_templates(template_file)
    print(f"✓ Templates loaded from {template_file}")

    client = LLMClient(model=model, api_key=api_key, base_url=base_url)
    
    # Stage 1
    print(f"\n>>> Starting Stage 1: Prompt #1 <<<")
    p1_repl = _build_replacements(pr_text, human_patch)
    p1 = _stage("p1", templates.prompt1, p1_repl, "", run_dir, client)

    if model_diff_stage2_file is None:
        print(f"\nℹ Model diff for Stage 2 not provided; stopping after Prompt-1.")
        print(f"Provide model diff from the model’s attempt using Prompt-1 and rerun with --model-diff2 to generate Prompt-2.")
        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETE - PROMPT 1 GENERATED")
        print(f"{'#'*80}\n")
        print("=== PROMPT-1 ===")
        print(p1.prompt_text.strip())
        return {
            "pr_id": pr_id,
            "p1_prompt_path": str(run_dir / "p1_prompt.txt"),
            "run_dir": str(run_dir),
            "next_step": "Rerun with --model-diff2 to produce Prompt-2 (and optionally Prompt-3)",
        }

    # Stage 2
    print(f"\n>>> Starting Stage 2: Prompt #2 <<<")
    p2_repl = _build_replacements(
        pr_text=pr_text,
        human_patch=human_patch,
        model_patch=model_patch_stage2,
        previous_prompt=p1.prompt_text,
    )
    p2 = _stage("p2", templates.prompt2, p2_repl, model_patch_stage2, run_dir, client)

    if model_diff_stage3_file is None:
        print(f"\nℹ Model diff for Stage 3 not provided; stopping after Prompt-2.")
        print(f"Provide model diff from the model’s attempt using Prompt-2 and rerun with --model-diff3 to generate Prompt-3.")
        print(f"\n{'#'*80}")
        print(f"# PIPELINE COMPLETE - PROMPTS 1-2 GENERATED")
        print(f"{'#'*80}\n")
        print("=== PROMPT-1 ===")
        print(p1.prompt_text.strip())
        print("\n=== PROMPT-2 ===")
        print(p2.prompt_text.strip())
        return {
            "pr_id": pr_id,
            "p1_prompt_path": str(run_dir / "p1_prompt.txt"),
            "p2_prompt_path": str(run_dir / "p2_prompt.txt"),
            "run_dir": str(run_dir),
            "next_step": "Rerun with --model-diff3 to produce Prompt-3",
        }

    # Stage 3
    print(f"\n>>> Starting Stage 3: Prompt #3 <<<")
    p3_repl = _build_replacements(
        pr_text=pr_text,
        human_patch=human_patch,
        model_patch=model_patch_stage3,
        previous_prompt=p2.prompt_text,
    )
    p3 = _stage("p3", templates.prompt3, p3_repl, model_patch_stage3, run_dir, client)

    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE - PROMPTS GENERATED")
    print(f"{'#'*80}\n")
    print("=== PROMPT-1 ===")
    print(p1.prompt_text.strip())
    print("\n=== PROMPT-2 ===")
    print(p2.prompt_text.strip())
    print("\n=== PROMPT-3 ===")
    print(p3.prompt_text.strip())

    return {
        "pr_id": pr_id,
        "p1_prompt_path": str(run_dir / "p1_prompt.txt"),
        "p2_prompt_path": str(run_dir / "p2_prompt.txt"),
        "p3_prompt_path": str(run_dir / "p3_prompt.txt"),
        "run_dir": str(run_dir),
    }
