"""
Main CLI entry point for sft-play.
Provides unified interface to all training, evaluation, and inference commands.
"""

import argparse
import sys
from typing import List, Optional

from sft_play.version import __version__


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sft-play",
        description="Universal LLM Fine-Tuning CLI - Scale from RTX 4060 to H200 clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sft-play train --config configs/run_bnb.yaml
  sft-play eval --model ./outputs/checkpoints/final
  sft-play infer --model ./outputs/checkpoints/final

For more information, visit: https://github.com/xynehq/sft-play
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"sft-play {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="Available commands",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a model",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration YAML file",
    )

    # Infer command
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run interactive inference",
    )
    infer_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    infer_parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for batch inference",
    )

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process raw data for training",
    )
    process_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data directory or file",
    )
    process_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Import and delegate to specific command handlers
    if args.command == "train":
        from sft_play.train import main as train_main
        return train_main([f"--config={args.config}"])
    elif args.command == "eval":
        from sft_play.evaluate import main as eval_main
        eval_args = [f"--model={args.model}"]
        if args.config:
            eval_args.append(f"--config={args.config}")
        return eval_main(eval_args)
    elif args.command == "infer":
        from sft_play.infer import main as infer_main
        infer_args = [f"--model={args.model}"]
        if args.prompt:
            infer_args.append(f"--prompt={args.prompt}")
        return infer_main(infer_args)
    elif args.command == "process":
        from sft_play.process import main as process_main
        return process_main([f"--input={args.input}", f"--output={args.output}"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
