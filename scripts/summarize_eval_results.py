from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any


ASSESSOR_CSV = Path("assessor/evaluate_results.csv")
TEACHER_CSV = Path("assessor/teacher_eval/evaluate_teacher_llm_results.csv")
OUT_DIR = Path("assessor/summary")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def assessor_display_name(row: dict[str, str]) -> str:
    model = row["model"]
    threshold = to_float(row, "threshold")
    return f"{model} @ {threshold:.1f}"


def shorten_label(label: str, max_len: int = 38) -> str:
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def top_rows(rows: list[dict[str, str]], metric: str, n: int) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: to_float(row, metric), reverse=True)[:n]


def md_table(rows: list[dict[str, str]], *, include_threshold: bool = True) -> str:
    if not rows:
        return "_No rows available._"
    headers = ["Rank", "Model"]
    if include_threshold:
        headers.append("Threshold")
    headers.extend(["Accuracy", "Precision", "Recall", "F1", "Pairs"])
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for idx, row in enumerate(rows, start=1):
        cells = [str(idx), row["model"]]
        if include_threshold:
            cells.append(f"{to_float(row, 'threshold'):.1f}")
        cells.extend(
            [
                format_metric(to_float(row, "accuracy")),
                format_metric(to_float(row, "precision")),
                format_metric(to_float(row, "recall")),
                format_metric(to_float(row, "f1")),
                str(to_int(row, "n_eval_pairs")),
            ]
        )
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def teacher_md_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No teacher rows available._"
    headers = ["Model", "Top-K", "Similarity", "Accuracy", "Precision", "Recall", "F1", "Pairs"]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append(
            "| " + " | ".join(
                [
                    row["model"],
                    row["top_k"],
                    row["similarity_type"],
                    format_metric(to_float(row, "accuracy")),
                    format_metric(to_float(row, "precision")),
                    format_metric(to_float(row, "recall")),
                    format_metric(to_float(row, "f1")),
                    str(to_int(row, "n_eval_pairs")),
                ]
            ) + " |"
        )
    return "\n".join(out)


def svg_text(x: float, y: float, text: str, *, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-weight="{weight}" fill="#1f2937">{html.escape(text)}</text>'
    )


def build_top_f1_svg(rows: list[dict[str, str]], out_path: Path) -> None:
    top = top_rows(rows, "f1", 10)
    width = 1100
    height = 70 + len(top) * 52
    left = 360
    right = 80
    bar_w = width - left - right
    max_f1 = max((to_float(row, "f1") for row in top), default=1.0)
    max_f1 = max(max_f1, 0.05)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        svg_text(24, 32, "Top Assessor Rows by F1", size=24, weight="bold"),
        svg_text(24, 52, "Bars show F1. Blue marker shows recall. Orange marker shows precision.", size=12),
    ]
    for tick in range(6):
        x = left + bar_w * tick / 5
        value = max_f1 * tick / 5
        svg.append(f'<line x1="{x:.1f}" y1="70" x2="{x:.1f}" y2="{height - 24:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(svg_text(x, height - 8, f"{value:.2f}", size=11, anchor="middle"))

    for idx, row in enumerate(top):
        y = 92 + idx * 52
        label = shorten_label(assessor_display_name(row), 46)
        f1 = to_float(row, "f1")
        recall = to_float(row, "recall")
        precision = to_float(row, "precision")
        bar_len = bar_w * f1 / max_f1
        recall_x = left + bar_w * recall / max_f1
        precision_x = left + bar_w * precision / max_f1
        svg.append(svg_text(left - 12, y + 14, label, size=12, anchor="end"))
        svg.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{bar_len:.1f}" height="22" rx="4" fill="#4f46e5"/>')
        svg.append(f'<circle cx="{recall_x:.1f}" cy="{y + 11:.1f}" r="5" fill="#0ea5e9"/>')
        svg.append(f'<circle cx="{precision_x:.1f}" cy="{y + 11:.1f}" r="5" fill="#f97316"/>')
        svg.append(svg_text(left + bar_len + 8, y + 15, f"F1 {f1:.3f}", size=11))
    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")


def build_pr_scatter_svg(rows: list[dict[str, str]], out_path: Path) -> None:
    width = 900
    height = 700
    left = 90
    right = 30
    top = 60
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_map(value: float) -> float:
        return left + plot_w * value

    def y_map(value: float) -> float:
        return top + plot_h * (1 - value)

    ranked_labels = {assessor_display_name(row) for row in top_rows(rows, "f1", 8)}
    ranked_labels.update(assessor_display_name(row) for row in top_rows(rows, "recall", 5))

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        svg_text(24, 32, "Assessor Precision vs Recall", size=24, weight="bold"),
        svg_text(24, 52, "Dot size tracks F1. Labels shown for top F1 and top recall rows.", size=12),
    ]
    for tick in range(6):
        v = tick / 5
        x = x_map(v)
        y = y_map(v)
        svg.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + plot_h:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_w:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(svg_text(x, height - 32, f"{v:.1f}", size=11, anchor="middle"))
        svg.append(svg_text(58, y + 4, f"{v:.1f}", size=11, anchor="middle"))
    svg.append(f'<line x1="{left:.1f}" y1="{top + plot_h:.1f}" x2="{left + plot_w:.1f}" y2="{top + plot_h:.1f}" stroke="#374151" stroke-width="1.5"/>')
    svg.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_h:.1f}" stroke="#374151" stroke-width="1.5"/>')
    svg.append(svg_text(left + plot_w / 2, height - 10, "Precision", size=13, anchor="middle", weight="bold"))
    svg.append(svg_text(24, top + plot_h / 2, "Recall", size=13, weight="bold"))

    for row in rows:
        label = assessor_display_name(row)
        precision = to_float(row, "precision")
        recall = to_float(row, "recall")
        f1 = to_float(row, "f1")
        radius = 4 + f1 * 70
        x = x_map(precision)
        y = y_map(recall)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#4f46e5" fill-opacity="0.35" stroke="#4338ca" stroke-width="1"/>')
        if label in ranked_labels:
            svg.append(svg_text(x + 8, y - 6, shorten_label(label, 32), size=11))
    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")


def build_teacher_compare_svg(assessor_rows: list[dict[str, str]], teacher_rows: list[dict[str, str]], out_path: Path) -> None:
    best_f1 = top_rows(assessor_rows, "f1", 1)[0] if assessor_rows else None
    best_recall = top_rows(assessor_rows, "recall", 1)[0] if assessor_rows else None
    teacher = teacher_rows[0] if teacher_rows else None
    series: list[tuple[str, dict[str, str], str]] = []
    if best_f1 is not None:
        series.append(("Best assessor by F1", best_f1, "#4f46e5"))
    if best_recall is not None:
        series.append(("Best assessor by Recall", best_recall, "#0ea5e9"))
    if teacher is not None:
        series.append(("Teacher LLM", teacher, "#f97316"))

    metrics = ["accuracy", "precision", "recall", "f1"]
    width = 980
    height = 420
    left = 120
    top = 70
    group_gap = 180
    bar_gap = 28
    max_bar_h = 220

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        svg_text(24, 32, "Teacher vs Assessor Snapshot", size=24, weight="bold"),
        svg_text(24, 52, "Teacher row uses a different evaluation set (top-50 + GT extras), so read as context, not a direct apples-to-apples ranking.", size=12),
    ]
    for tick in range(6):
        value = tick / 5
        y = top + max_bar_h * (1 - value)
        svg.append(f'<line x1="{left - 20:.1f}" y1="{y:.1f}" x2="{width - 24:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(svg_text(72, y + 4, f"{value:.1f}", size=11, anchor="middle"))

    for metric_idx, metric in enumerate(metrics):
        group_x = left + metric_idx * group_gap
        svg.append(svg_text(group_x + 40, top + max_bar_h + 28, metric.upper(), size=13, anchor="middle", weight="bold"))
        for series_idx, (label, row, color) in enumerate(series):
            value = to_float(row, metric)
            x = group_x + series_idx * bar_gap
            bar_h = max_bar_h * value
            y = top + max_bar_h - bar_h
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="20" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
            svg.append(svg_text(x + 10, y - 6, f"{value:.3f}", size=10, anchor="middle"))

    legend_y = height - 28
    for idx, (label, _, color) in enumerate(series):
        x = 36 + idx * 250
        svg.append(f'<rect x="{x:.1f}" y="{legend_y - 12:.1f}" width="14" height="14" fill="{color}" rx="3"/>')
        svg.append(svg_text(x + 22, legend_y, label, size=12))
    svg.append('</svg>')
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize assessor and teacher evaluation CSVs")
    parser.add_argument("--assessor-csv", type=str, default=str(ASSESSOR_CSV))
    parser.add_argument("--teacher-csv", type=str, default=str(TEACHER_CSV))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    assessor_rows = read_csv_rows(Path(args.assessor_csv))
    teacher_rows = read_csv_rows(Path(args.teacher_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not assessor_rows:
        raise RuntimeError(f"No assessor rows found in {args.assessor_csv}")

    top_f1 = top_rows(assessor_rows, "f1", args.top_n)
    top_recall = top_rows(assessor_rows, "recall", args.top_n)
    top_precision = top_rows(assessor_rows, "precision", args.top_n)
    top_accuracy = top_rows(assessor_rows, "accuracy", args.top_n)

    top_f1_svg = out_dir / "assessor_top_f1.svg"
    pr_scatter_svg = out_dir / "assessor_precision_recall_scatter.svg"
    teacher_compare_svg = out_dir / "teacher_vs_best_assessor.svg"
    build_top_f1_svg(assessor_rows, top_f1_svg)
    build_pr_scatter_svg(assessor_rows, pr_scatter_svg)
    build_teacher_compare_svg(assessor_rows, teacher_rows, teacher_compare_svg)

    summary_md = out_dir / "evaluation_summary.md"
    lines = [
        "# Evaluation Summary",
        "",
        f"- Assessor rows: {len(assessor_rows)}",
        f"- Teacher rows: {len(teacher_rows)}",
        "",
        "## Notes",
        "",
        "- Assessor CSV rows come from `assessor/evaluate_results.csv`.",
        "- Teacher CSV rows come from `assessor/teacher_eval/evaluate_teacher_llm_results.csv`.",
        "- Teacher evaluation currently uses `top-50 + GT extra positives`, so it is not directly apples-to-apples with the current assessor CSV.",
        "",
        "## Top Assessor by F1",
        "",
        md_table(top_f1),
        "",
        "## Top Assessor by Recall",
        "",
        md_table(top_recall),
        "",
        "## Top Assessor by Precision",
        "",
        md_table(top_precision),
        "",
        "## Top Assessor by Accuracy",
        "",
        md_table(top_accuracy),
        "",
        "## Teacher Snapshot",
        "",
        teacher_md_table(teacher_rows),
        "",
        "## Visuals",
        "",
        f"![Top assessor by F1](./{top_f1_svg.name})",
        "",
        f"![Assessor precision recall scatter](./{pr_scatter_svg.name})",
        "",
        f"![Teacher vs best assessor](./{teacher_compare_svg.name})",
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[summarize_eval_results] wrote {summary_md}")
    print(f"[summarize_eval_results] wrote {top_f1_svg}")
    print(f"[summarize_eval_results] wrote {pr_scatter_svg}")
    print(f"[summarize_eval_results] wrote {teacher_compare_svg}")


if __name__ == "__main__":
    main()
