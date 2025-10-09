"""
Training module - wrapper around main repo's training script.
For now, this provides the entry point. Full implementation will be ported later.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main training entry point.

    For full implementation, this will call the actual training logic.
    Currently serves as a placeholder/wrapper.
    """
    parser = argparse.ArgumentParser(
        description="Train LLM models with SFT-Play"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )

    args = parser.parse_args(argv)

    # Validate config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    print(f"SFT-Play Training")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Resume: {args.resume}")
    print()
    print("NOTE: Full training implementation will be ported in next phase.")
    print("For now, please use the main repo's training scripts directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
