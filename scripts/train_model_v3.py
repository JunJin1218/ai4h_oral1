"""
Train assessor model_v3 from a streamlined annotation file (e.g. test-data.xlsx or data_reformatted.xlsx).

Uses lookalike_80_20 split and --balance-weights. By default trains on
data/Annotation/test-data.xlsx. To train on the consolidated lookalike_annotation data, use
data/Annotation/data_reformatted.xlsx. Saves to assessor/model_v3.

Usage (from project root ai4h_oral1):
  uv run python scripts/train_model_v3.py
  uv run python scripts/train_model_v3.py --annotation-file data/Annotation/test-data.xlsx
  uv run python scripts/train_model_v3.py --annotation-file data/Annotation/data_reformatted.xlsx
  uv run python scripts/train_model_v3.py --epochs 25
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train model_v3 on test-data.xlsx (streamlined format, 80/20 split, balance-weights)"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="data/Annotation/test-data.xlsx",
        help="Annotation file for training (default: data/Annotation/test-data.xlsx)",
    )
    parser.add_argument("--out-dir", type=str, default="assessor/model_v3", help="Output model directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    train_py = root / "train.py"
    if not train_py.exists():
        print(f"train.py not found at {train_py}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(train_py),
        "--split", "lookalike_80_20",
        "--balance-weights",
        "--annotation-file", args.annotation_file,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--out-dir", args.out_dir,
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
