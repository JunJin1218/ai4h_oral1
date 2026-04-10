"""
Compare two assessor models on the same test set.

Loads model A and model B from two directories, evaluates both on the same
holdout pairs (e.g. lookalike_80_20 test or by_query val), and prints
metrics side by side so you can see if the new model is better.

Usage (from project root ai4h_oral1):
  uv run python scripts/compare_models.py --model-a assessor/model --model-b assessor/model_new
  uv run python scripts/compare_models.py --model-b assessor/model_new --split by_query
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root so "assessor" and "train" resolve when run as scripts/compare_models.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch

from assessor.model import load_assessor

# Import train's data loading and evaluation (same test set logic)
from train import (
    evaluate,
    _resolve_pairs,
    collect_lookalike_only_raw,
    collect_all_raw_pairs,
    split_by_query_disjoint,
    group_pairs_by_query,
    _label_counts,
)


def get_test_pairs(data_dir: Path, split: str, seed: int, train_ratio: float):
    """Return resolved test (or val) pairs for the given split."""
    if split == "lookalike_80_20":
        raw_all = collect_lookalike_only_raw(data_dir)
        if not raw_all:
            return []
        train_raw, test_raw = split_by_query_disjoint(
            raw_all, train_ratio=train_ratio, val_ratio=1.0 - train_ratio, seed=seed
        )
        return _resolve_pairs(test_raw, data_dir=data_dir)
    elif split == "by_query":
        raw_all = collect_all_raw_pairs(data_dir)
        if not raw_all:
            return []
        train_raw, val_raw = split_by_query_disjoint(
            raw_all, train_ratio=train_ratio, val_ratio=1.0 - train_ratio, seed=seed
        )
        return _resolve_pairs(val_raw, data_dir=data_dir)
    else:
        raise ValueError(f"Unknown split: {split}. Use lookalike_80_20 or by_query.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two assessor models on the same test set"
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="assessor/model",
        help="Directory of existing (baseline) model",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="assessor/model_new",
        help="Directory of new model to compare",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory for annotations and embeddings",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="lookalike_80_20",
        choices=["lookalike_80_20", "by_query"],
        help="Split used to build the test set (must match how models were trained for fair comparison)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio used in split (default 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for eval")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: {args.split} (test set)")
    print()

    test_pairs = get_test_pairs(data_dir, args.split, args.seed, args.train_ratio)
    if not test_pairs:
        print("No test pairs resolved. Check data/Annotation and embedding paths.")
        return
    counts = _label_counts(test_pairs)
    print(f"Test set: {len(test_pairs)} pairs, label_counts={dict(counts)}\n")

    results = {}
    for label, model_dir in [("Model A (existing)", args.model_a), ("Model B (new)", args.model_b)]:
        path = Path(model_dir)
        if not (path / "assessor.pt").exists() or not (path / "config.json").exists():
            print(f"{label}: NOT FOUND at {path} (missing assessor.pt or config.json)")
            results[label] = None
            continue
        model, cfg = load_assessor(device=device, in_dir=path)
        metrics = evaluate(
            model,
            test_pairs,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        results[label] = (model_dir, metrics)
        print(f"{label} ({model_dir}):")
        print(f"  loss={metrics['loss']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}")
        print(f"  confusion_matrix [[TN,FP],[FN,TP]] = {metrics['cm'].tolist()}")
        print()

    if results.get("Model A (existing)") and results.get("Model B (new)"):
        _, a = results["Model A (existing)"]
        _, b = results["Model B (new)"]
        print("--- Side-by-side ---")
        print(f"                  {'Model A (existing)':>20}   {'Model B (new)':>20}")
        print(f"  loss             {a['loss']:>20.4f}   {b['loss']:>20.4f}")
        print(f"  accuracy         {a['accuracy']:>20.4f}   {b['accuracy']:>20.4f}")
        print(f"  precision        {a['precision']:>20.4f}   {b['precision']:>20.4f}")
        print(f"  recall           {a['recall']:>20.4f}   {b['recall']:>20.4f}")
        print(f"  f1               {a['f1']:>20.4f}   {b['f1']:>20.4f}")
        better = "Model B" if b["f1"] > a["f1"] else "Model A"
        print(f"\n  Higher F1 on this test set: {better}")


if __name__ == "__main__":
    main()
