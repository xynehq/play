"""
Inference module - wrapper around main repo's inference script.
For now, this provides the entry point. Full implementation will be ported later.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main inference entry point.

    For full implementation, this will call the actual inference logic.
    Currently serves as a placeholder/wrapper.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained models"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for inference (interactive mode if not provided)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args(argv)

    # Validate model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        return 1

    print(f"SFT-Play Inference")
    print(f"Model: {args.model}")
    print(f"Max Length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    print()

    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print()
    else:
        print("Interactive mode")
        print()

    print("NOTE: Full inference implementation will be ported in next phase.")
    print("For now, please use the main repo's inference scripts directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
