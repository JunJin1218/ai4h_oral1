"""
Consolidate lookalike_annotation.xlsx into test-data style format (sheet-per-query).

Reads data/Annotation/lookalike_annotation.xlsx (or lookalike annotation.xlsx),
converts to the streamlined format: one sheet per query drug, with columns
"Lookalike" and "Non lookalike" and candidate names in rows.

Output: data/Annotation/data_reformatted.xlsx

Usage (from project root ai4h_oral1):
  uv run python scripts/reformat_lookalike_to_testdata.py
  uv run python scripts/reformat_lookalike_to_testdata.py --annotation-dir data/Annotation --out data/Annotation/data_reformatted.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# Project root so assessor resolves when run as scripts/reformat_...
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from assessor.test_lookalike import load_annotations


def sanitize_sheet_name(name: str, max_len: int = 31) -> str:
    """Excel sheet names: max 31 chars, no \\ / ? * [ ]."""
    s = re.sub(r'[\\/?*\[\]]', '_', str(name).strip())
    if len(s) > max_len:
        s = s[:max_len]
    return s or "Sheet"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reformat lookalike_annotation into test-data style (sheet-per-query)"
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="data/Annotation",
        help="Directory containing lookalike_annotation.xlsx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/Annotation/data_reformatted.xlsx",
        help="Output Excel path",
    )
    args = parser.parse_args()

    ann_dir = Path(args.annotation_dir)
    out_path = Path(args.out)

    # Find source file
    source_path = None
    for name in ("lookalike_annotation.xlsx", "lookalike annotation.xlsx"):
        p = ann_dir / name
        if p.exists():
            source_path = p
            break
    if not source_path:
        print(f"No lookalike_annotation.xlsx or lookalike annotation.xlsx in {ann_dir}")
        sys.exit(1)

    print(f"Reading {source_path} ...")
    raw = load_annotations(source_path)
    if not raw:
        print("No pairs loaded. Check file format (col0=query, col2=label, col3+=candidates).")
        sys.exit(1)

    # Group by query, then by label (1 = lookalike, 0 = non lookalike)
    by_query: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for query, candidate, label in raw:
        by_query[query].append((candidate, label))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for query in sorted(by_query.keys()):
            pairs = by_query[query]
            lookalikes = [c for c, lbl in pairs if lbl == 1]
            non_lookalikes = [c for c, lbl in pairs if lbl == 0]

            # Build a rectangular table: row0 = headers, row1+ = candidates
            # Column A = Lookalike, Column B = Non lookalike (pad with empty strings)
            n_rows = max(len(lookalikes), len(non_lookalikes), 1)
            col_lookalike = lookalikes + [""] * (n_rows - len(lookalikes))
            col_non = non_lookalikes + [""] * (n_rows - len(non_lookalikes))

            df = pd.DataFrame({
                "Lookalike": col_lookalike,
                "Non lookalike": col_non,
            })
            sheet_name = sanitize_sheet_name(query)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

    print(f"Wrote {len(by_query)} sheets to {out_path}")
    print(f"  Total pairs: {len(raw)} (lookalike: {sum(1 for _, _, l in raw if l == 1)}, non lookalike: {sum(1 for _, _, l in raw if l == 0)})")


if __name__ == "__main__":
    main()
