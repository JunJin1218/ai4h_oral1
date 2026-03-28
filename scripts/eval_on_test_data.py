"""
Evaluate an assessor model on a test dataset (CSV or Excel).

Loads (query, candidate, label) pairs from the test file, resolves to embeddings,
runs the model, and prints accuracy, precision, recall, F1, and confusion matrix.

Test data format:
  - CSV: query_image,candidate_image,label (0 or 1). Header optional but names as above.
  - Excel: same structure as lookalike annotation (col 0 = query, col 2 = label, col 3+ = candidates),
    or a simple 3-column sheet: query, candidate, label.

Usage (from project root ai4h_oral1):
  uv run python scripts/eval_on_test_data.py --model-dir assessor/model_v3 --test-data data/test-data.csv
  uv run python scripts/eval_on_test_data.py --model-dir assessor/model_v3 --test-data data/Annotation/test-data.xlsx
  uv run python scripts/eval_on_test_data.py --test-data data/test-data
  (if --test-data is a folder, looks for test-data.csv or test-data.xlsx inside it)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Project root so "assessor" and "train" resolve when run as scripts/eval_on_test_data.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import torch

from assessor.model import load_assessor
from assessor.test_lookalike import find_embedding_path, load_annotations

from train import _resolve_pairs, _label_counts, evaluate


def load_test_csv(path: Path) -> list[tuple[str, str, int]]:
    """Load (query, candidate, label) from CSV. Header: query_image, candidate_image, label."""
    raw: list[tuple[str, str, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            q = (row.get("query_image") or row.get("query") or "").strip()
            c = (row.get("candidate_image") or row.get("candidate") or "").strip()
            try:
                label = int(row.get("label", row.get("human_label", "")))
            except (ValueError, TypeError):
                continue
            if label not in (0, 1):
                continue
            if not q or not c:
                continue
            raw.append((q, c, label))
    return raw


def load_test_excel(path: Path) -> list[tuple[str, str, int]]:
    """Load from Excel: try lookalike annotation format first, else simple 3-column (query, candidate, label)."""
    raw = load_annotations(path)
    if raw:
        return raw
    # Simple 3-column: col0=query, col1=candidate, col2=label
    df = pd.read_excel(path, header=0)
    if len(df.columns) < 3:
        return []
    raw = []
    for _, row in df.iterrows():
        q = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        c = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        try:
            label = int(float(row.iloc[2])) if pd.notna(row.iloc[2]) else None
        except (ValueError, TypeError):
            continue
        if label not in (0, 1) or not q or not c:
            continue
        raw.append((q, c, label))
    return raw


def load_test_data(path: Path) -> list[tuple[str, str, int]]:
    """Load test pairs from CSV or Excel."""
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        return load_test_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return load_test_excel(path)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on test-data (CSV or Excel)")
    parser.add_argument("--model-dir", type=str, default="assessor/model_v3", help="Model directory")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test-data",
        help="Path to test CSV/Excel, or folder containing test-data.csv / test-data.xlsx",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data dir for resolving embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_path = Path(args.test_data)
    if test_path.is_dir():
        for name in ("test-data.csv", "test-data.xlsx", "test_data.csv", "test_data.xlsx"):
            p = test_path / name
            if p.exists():
                test_path = p
                break
        else:
            print(f"No test-data.csv or test-data.xlsx found in {test_path}")
            return

    raw = load_test_data(test_path)
    if not raw:
        print(f"No pairs loaded from {test_path}. Check format (query_image, candidate_image, label).")
        return

    print(f"Loaded {len(raw)} test pairs from {test_path}")
    pairs = _resolve_pairs(raw, data_dir=data_dir)
    if not pairs:
        print("No pairs resolved to embeddings. Check that query/candidate names match .pt files under data/.")
        return

    counts = _label_counts(pairs)
    print(f"Resolved: {len(pairs)} pairs, label_counts={dict(counts)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_assessor(device=device, in_dir=args.model_dir)
    metrics = evaluate(
        model,
        pairs,
        cfg=cfg,
        device=device,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    print(f"Model: {args.model_dir}")
    print(f"  loss      = {metrics['loss']:.4f}")
    print(f"  accuracy  = {metrics['accuracy']:.4f}")
    print(f"  precision = {metrics['precision']:.4f}")
    print(f"  recall    = {metrics['recall']:.4f}")
    print(f"  f1        = {metrics['f1']:.4f}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]] = {metrics['cm'].tolist()}")


if __name__ == "__main__":
    main()
