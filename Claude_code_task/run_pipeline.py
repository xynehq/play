from __future__ import annotations

import argparse
from pathlib import Path

from runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-generation pipeline using LLM.")
    parser.add_argument("--pr-id", required=True, help="PR identifier, e.g., 1234")
    parser.add_argument("--pr-description", required=True, type=Path, help="Path to pr_description.md")
    parser.add_argument("--human-diff", required=True, type=Path, help="Path to human_diff.diff")
    parser.add_argument(
        "--model-diff",
        required=False,
        type=Path,
        help="Alias for --model-diff2 (model attempt after Prompt-1). Optional; if omitted, only Prompt-1 is generated.",
    )
    parser.add_argument(
        "--model-diff2",
        required=False,
        type=Path,
        help="Model diff after Prompt-1 (for Prompt-2 generation).",
    )
    parser.add_argument(
        "--model-diff3",
        required=False,
        type=Path,
        help="Model diff after Prompt-2 (for Prompt-3 generation).",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        default=Path("Prompt.md"),
        help="Path to prompt_template.md containing all templates",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Base directory to store pipeline artifacts",
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="API key for the LLM (falls back to env API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        required=False,
        help="Base URL for the LLM (falls back to env BASE_URL)",
    )
    parser.add_argument(
        "--model",
        required=False,
        default="minimaxai/minimax-m2",
        help="Model name to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_diff2 = args.model_diff2 or args.model_diff
    model_diff3 = args.model_diff3
    run_pipeline(
        pr_id=args.pr_id,
        pr_description_file=args.pr_description,
        human_diff_file=args.human_diff,
        model_diff_stage2_file=model_diff2,
        model_diff_stage3_file=model_diff3,
        template_file=args.templates,
        runs_dir=args.runs_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )


if __name__ == "__main__":
    main()

