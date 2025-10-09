"""
Evaluation module - wrapper around main repo's evaluation script.
For now, this provides the entry point. Full implementation will be ported later.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main evaluation entry point.

    For full implementation, this will call the actual evaluation logic.
    Currently serves as a placeholder/wrapper.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained models with SFT-Play"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration YAML file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for evaluation results",
    )

    args = parser.parse_args(argv)

    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        return 1

    print(f"SFT-Play Evaluation")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print()
    print("NOTE: Full evaluation implementation will be ported in next phase.")
    print("For now, please use the main repo's evaluation scripts directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
