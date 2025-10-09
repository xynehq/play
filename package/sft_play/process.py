"""
Data processing module - wrapper around main repo's processing script.
For now, this provides the entry point. Full implementation will be ported later.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main data processing entry point.

    For full implementation, this will call the actual processing logic.
    Currently serves as a placeholder/wrapper.
    """
    parser = argparse.ArgumentParser(
        description="Process raw data for training"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data directory or file",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "parquet", "arrow"],
        default="jsonl",
        help="Output format",
    )

    args = parser.parse_args(argv)

    # Validate input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {args.input}", file=sys.stderr)
        return 1

    print(f"SFT-Play Data Processing")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print()
    print("NOTE: Full processing implementation will be ported in next phase.")
    print("For now, please use the main repo's processing scripts directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
