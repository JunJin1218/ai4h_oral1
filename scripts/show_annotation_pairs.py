"""
Load the streamlined annotation file and print pairs in a table.
Confirms at least one pair for a given query (e.g. 1 Acarbose).

Usage (from project root ai4h_oral1):
  uv run python scripts/show_annotation_pairs.py
  uv run python scripts/show_annotation_pairs.py --annotation-file data/Annotation/test-data.xlsx
  uv run python scripts/show_annotation_pairs.py --match "1 Acarbose" --max-rows 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root so "assessor" resolves when run as scripts/show_annotation_pairs.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from assessor.test_lookalike import load_annotations_streamlined


def main() -> None:
    parser = argparse.ArgumentParser(description="Show annotation pairs in a table")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="data/Annotation/test-data.xlsx",
        help="Path to annotation Excel",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="1 Acarbose",
        help="Show pairs whose query contains this string (default: 1 Acarbose)",
    )
    parser.add_argument("--max-rows", type=int, default=25, help="Max rows to print (default 25)")
    args = parser.parse_args()

    path = Path(args.annotation_file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    pairs = load_annotations_streamlined(path)
    if not pairs:
        print(f"No pairs loaded from {path}")
        return

    print(f"Loaded {len(pairs)} pairs from {path.name}\n")

    # Filter to match query
    match_lower = args.match.lower()
    matched = [(q, c, lbl) for q, c, lbl in pairs if match_lower in q.lower()]
    if not matched:
        print(f"No pairs found for query matching {args.match!r}.")
        print("First 5 query names in file:")
        seen = set()
        for q, _, _ in pairs:
            if q not in seen:
                seen.add(q)
                print(f"  - {q}")
            if len(seen) >= 5:
                break
        return

    print(f"Pairs where query contains {args.match!r}: {len(matched)} found.\n")
    display = matched[: args.max_rows]

    # Table: query | candidate | label
    col_q = max(6, min(50, max(len(str(p[0])) for p in display)))
    col_c = max(10, min(60, max(len(str(p[1])) for p in display)))
    fmt = f"  {{:<{col_q}}}  {{:<{col_c}}}  {{}}"
    print(fmt.format("query", "candidate", "label"))
    print("  " + "-" * (col_q + col_c + 6))
    for q, c, lbl in display:
        q_short = (q[: col_q - 3] + "...") if len(q) > col_q else q
        c_short = (c[: col_c - 3] + "...") if len(c) > col_c else c
        print(fmt.format(q_short, c_short, lbl))
    if len(matched) > args.max_rows:
        print(f"  ... and {len(matched) - args.max_rows} more")
    print(f"\nConfirmed: at least 1 pair for query matching {args.match!r}.")


if __name__ == "__main__":
    main()
