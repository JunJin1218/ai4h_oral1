from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from assessor.model import load_assessor
from assessor.test_lookalike import find_embedding_path
from image_embedding.vit import get_image_embedding, load_vit_model
from utils import search_similar_with_metadata


DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
ASSESSOR_ROOT = Path("assessor")
TEST_QUERY_DIR = Path("test_query_img")
GT_CSV_PATH = TEST_QUERY_DIR / "gt_table_csv.csv"
DEFAULT_TOP_KS = (50, 100, 200)
DEFAULT_SIMILARITY_TYPES = ("l2", "cosine", "ip")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class QueryGroundTruth:
    query_id: str
    query_image_path: Path
    positive_candidate_names: set[str]


@dataclass(frozen=True)
class CorpusCandidate:
    vector_id: int
    file_name: str
    vector: np.ndarray


def extract_leading_query_id(path: Path) -> str | None:
    match = re.match(r"^(\d+)\b", path.stem)
    return None if match is None else match.group(1)


def canonical_candidate_name(raw_name: str, data_dir: Path) -> str | None:
    resolved = find_embedding_path(raw_name, data_dir)
    if resolved is not None:
        return resolved.name
    return _canonical_candidate_name_from_db(raw_name, DB_PATH)


def _stem_base(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


@lru_cache(maxsize=1)
def _load_db_name_indexes(db_path_str: str) -> tuple[set[str], set[str], dict[str, str]]:
    db_path = Path(db_path_str)
    if not db_path.exists():
        return set(), set(), {}

    exact_stems: set[str] = set()
    base_stems: set[str] = set()
    normalized_to_file_name: dict[str, str] = {}

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db")
        for (file_name,) in cur.fetchall():
            if not isinstance(file_name, str):
                continue
            stem = Path(file_name).stem
            stem_lower = stem.lower()
            base_lower = _stem_base(stem).lower()
            exact_stems.add(stem_lower)
            base_stems.add(base_lower)
            normalized_to_file_name.setdefault(_normalize_name(stem_lower), file_name)
            normalized_to_file_name.setdefault(_normalize_name(base_lower), file_name)

    return exact_stems, base_stems, normalized_to_file_name


def _canonical_candidate_name_from_db(raw_name: str, db_path: Path) -> str | None:
    exact_stems, base_stems, normalized_to_file_name = _load_db_name_indexes(str(db_path))
    if not exact_stems and not base_stems and not normalized_to_file_name:
        return None

    query_stem = Path(raw_name).stem
    query_stem_lower = query_stem.lower()
    query_base_lower = _stem_base(query_stem).lower()

    if query_stem_lower in exact_stems or query_stem_lower in base_stems:
        # Keep original style from DB, including .pt extension if present.
        return normalized_to_file_name.get(_normalize_name(query_stem_lower))
    if query_base_lower in exact_stems or query_base_lower in base_stems:
        return normalized_to_file_name.get(_normalize_name(query_base_lower))

    return normalized_to_file_name.get(_normalize_name(query_base_lower))


def load_ground_truth(csv_path: Path, query_dir: Path, data_dir: Path) -> list[QueryGroundTruth]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    query_image_map: dict[str, Path] = {}
    for path in sorted(query_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        query_id = extract_leading_query_id(path)
        if query_id is not None:
            query_image_map[query_id] = path

    positives_by_query: dict[str, set[str]] = {}
    unresolved_candidates = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row.get("query_id", "")).strip()
            candidate_raw = str(row.get("candidate_file_name", "")).strip()
            if not query_id or not candidate_raw:
                continue
            candidate_name = canonical_candidate_name(candidate_raw, data_dir)
            if candidate_name is None:
                unresolved_candidates += 1
                continue
            positives_by_query.setdefault(query_id, set()).add(candidate_name)

    out: list[QueryGroundTruth] = []
    missing_query_images: list[str] = []
    for query_id, positive_names in sorted(positives_by_query.items(), key=lambda x: int(x[0])):
        query_image_path = query_image_map.get(query_id)
        if query_image_path is None:
            missing_query_images.append(query_id)
            continue
        out.append(
            QueryGroundTruth(
                query_id=query_id,
                query_image_path=query_image_path,
                positive_candidate_names=positive_names,
            )
        )

    print(
        f"[evaluate] ground_truth_queries={len(out)} "
        f"unresolved_candidates={unresolved_candidates} "
        f"missing_query_images={len(missing_query_images)}"
    )
    if missing_query_images:
        preview = ", ".join(missing_query_images[:10])
        print(f"[evaluate] missing query image ids: {preview}")
    return out


def discover_model_checkpoints(assessor_root: Path) -> list[tuple[str, Path, Path]]:
    checkpoints: list[tuple[str, Path, Path]] = []
    for config_path in sorted(assessor_root.rglob("config.json")):
        model_dir = config_path.parent
        rel_dir = model_dir.relative_to(assessor_root)
        for ckpt_path in sorted(model_dir.glob("*.pt")):
            label = f"{rel_dir}/{ckpt_path.name}" if str(rel_dir) != "." else ckpt_path.name
            checkpoints.append((label, model_dir, ckpt_path))
    return checkpoints


def load_full_corpus_candidates(index_path: Path, db_path: Path) -> list[CorpusCandidate]:
    index = faiss.read_index(str(index_path))
    if hasattr(index, "id_map"):
        ids = faiss.vector_to_array(index.id_map).astype(np.int64, copy=False)
    else:
        ids = np.arange(int(index.ntotal), dtype=np.int64)

    file_name_map: dict[int, str] = {}
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT vector_id, file_name FROM image_db")
        for vector_id, file_name in cur.fetchall():
            file_name_map[int(vector_id)] = str(file_name)

    out: list[CorpusCandidate] = []
    for vector_id in ids:
        file_name = file_name_map.get(int(vector_id))
        if file_name is None:
            continue
        out.append(
            CorpusCandidate(
                vector_id=int(vector_id),
                file_name=file_name,
                vector=index.reconstruct(int(vector_id)).astype(np.float32, copy=False),
            )
        )
    print(f"[evaluate] full_corpus_candidates={len(out)}")
    return out


def evaluate_checkpoint(
    *,
    model_dir: Path,
    checkpoint_path: Path,
    ground_truth: list[QueryGroundTruth],
    top_k: int,
    similarity_type: str,
    score_all_candidates: bool,
    full_corpus_candidates: list[CorpusCandidate] | None,
    threshold: float,
    processor,
    vit_model,
    vit_device,
) -> dict[str, object]:
    model, cfg = load_assessor(
        device=vit_device,
        in_dir=model_dir,
        filename=checkpoint_path.name,
    )
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    n_retrieved = 0
    n_missing_gt = 0
    missing_gt_details: list[dict[str, object]] = []

    for item in ground_truth:
        query_vector_t = get_image_embedding(
            vit_model,
            processor,
            item.query_image_path,
            device=vit_device,
            pooling="cls",
        )
        query_vector = query_vector_t.detach().cpu().numpy().astype("float32", copy=False)

        if score_all_candidates:
            if full_corpus_candidates is None:
                raise RuntimeError("full_corpus_candidates is required when --score-all-candidates is enabled")
            n_retrieved += len(full_corpus_candidates)
            q_batch = query_vector_t.unsqueeze(0).repeat(len(full_corpus_candidates), 1).to(vit_device)
            c_batch = torch.from_numpy(
                np.stack([cand.vector for cand in full_corpus_candidates]).astype(np.float32, copy=False)
            ).to(vit_device)
            with torch.no_grad():
                scores = model(q_batch, c_batch).detach().cpu().numpy().tolist()
            for cand, score in zip(full_corpus_candidates, scores):
                pred_label = 1 if float(score) >= threshold else 0
                true_label = 1 if cand.file_name in item.positive_candidate_names else 0
                y_true.append(true_label)
                y_pred.append(pred_label)
        else:
            candidates = search_similar_with_metadata(
                query_vector=query_vector,
                similarity_type=similarity_type,
                top_n=top_k,
                index_path=INDEX_PATH,
                db_path=DB_PATH,
                exclude_vector_id=None,
            )
            n_retrieved += len(candidates)

            seen_positive_names: set[str] = set()
            for cand in candidates:
                if cand.file_name is None:
                    continue
                q_t = query_vector_t.unsqueeze(0).to(vit_device)
                c_t = torch.from_numpy(cand.vector).float().unsqueeze(0).to(vit_device)
                with torch.no_grad():
                    score = float(model(q_t, c_t).item())
                pred_label = 1 if score >= threshold else 0
                true_label = 1 if cand.file_name in item.positive_candidate_names else 0
                if true_label == 1:
                    seen_positive_names.add(cand.file_name)
                y_true.append(true_label)
                y_pred.append(pred_label)

            missing_positives = item.positive_candidate_names - seen_positive_names
            n_missing_gt += len(missing_positives)
            if missing_positives:
                missing_gt_details.append(
                    {
                        "query_id": item.query_id,
                        "query_image_path": str(item.query_image_path),
                        "missing_positive_count": len(missing_positives),
                        "missing_positive_names": sorted(missing_positives),
                    }
                )
            for _ in missing_positives:
                y_true.append(1)
                y_pred.append(0)

    if not y_true:
        raise RuntimeError("No evaluation pairs were produced.")

    return {
        "n_queries": len(ground_truth),
        "n_eval_pairs": len(y_true),
        "n_retrieved_pairs": n_retrieved,
        "n_missing_gt": n_missing_gt,
        "n_queries_with_missing_gt": len(missing_gt_details),
        "missing_gt_details": missing_gt_details,
        "similarity_type": similarity_type,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_results_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No rows to print.")
        return

    headers = [
        "model",
        "similarity_type",
        "top_k",
        "threshold",
        "queries",
        "eval_pairs",
        "missing_gt",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    print()
    print(" | ".join(headers))
    print("-|-".join("-" * len(h) for h in headers))
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["model"]),
                    str(row["similarity_type"]),
                    str(row["top_k"]),
                    f"{float(row['threshold']):.2f}",
                    str(row["n_queries"]),
                    str(row["n_eval_pairs"]),
                    str(row["n_missing_gt"]),
                    f"{float(row['accuracy']):.4f}",
                    f"{float(row['precision']):.4f}",
                    f"{float(row['recall']):.4f}",
                    f"{float(row['f1']):.4f}",
                ]
            )
        )


def save_results_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint_path",
        "similarity_type",
        "top_k",
        "threshold",
        "n_queries",
        "n_eval_pairs",
        "n_retrieved_pairs",
        "n_missing_gt",
        "n_queries_with_missing_gt",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def save_missing_gt_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint_path",
        "similarity_type",
        "top_k",
        "query_id",
        "query_image_path",
        "missing_positive_count",
        "missing_positive_names",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for missing in row.get("missing_gt_details", []):
                writer.writerow(
                    {
                        "model": row["model"],
                        "checkpoint_path": row["checkpoint_path"],
                        "similarity_type": row["similarity_type"],
                        "top_k": row["top_k"],
                        "threshold": row["threshold"],
                        "query_id": missing["query_id"],
                        "query_image_path": missing["query_image_path"],
                        "missing_positive_count": missing["missing_positive_count"],
                        "missing_positive_names": " | ".join(missing["missing_positive_names"]),
                    }
                )


def print_missing_gt_summary(rows: list[dict[str, object]], *, preview_per_row: int = 3) -> None:
    print("\nMissing GT outside top-k:")
    any_missing = False
    for row in rows:
        details = row.get("missing_gt_details", [])
        if not details:
            continue
        any_missing = True
        print(
                f"- {row['model']} top_k={row['top_k']}: "
            f"similarity={row['similarity_type']} "
            f"threshold={float(row['threshold']):.2f} "
            f"queries_with_missing_gt={row['n_queries_with_missing_gt']} "
            f"missing_gt={row['n_missing_gt']}"
        )
        for item in details[:preview_per_row]:
            preview = ", ".join(item["missing_positive_names"][:3])
            print(
                f"  query_id={item['query_id']} "
                f"missing={item['missing_positive_count']} "
                f"sample={preview}"
            )
    if not any_missing:
        print("- none")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all assessor checkpoints on test_query_img ground truth")
    parser.add_argument("--gt-csv", type=str, default=str(GT_CSV_PATH))
    parser.add_argument("--query-dir", type=str, default=str(TEST_QUERY_DIR))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--assessor-root", type=str, default=str(ASSESSOR_ROOT))
    parser.add_argument("--out-csv", type=str, default="assessor/evaluate_results.csv")
    parser.add_argument("--missing-gt-csv", type=str, default="assessor/evaluate_missing_gt.csv")
    parser.add_argument("--top-ks", type=str, default="200")
    parser.add_argument(
        "--score-all-candidates",
        action="store_true",
        help="Skip top-k retrieval and score the model against the full image_db candidate corpus for each query.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated decision thresholds to evaluate.",
    )
    args = parser.parse_args()

    top_ks = tuple(int(x.strip()) for x in args.top_ks.split(",") if x.strip())
    if not top_ks:
        raise ValueError("--top-ks must contain at least one integer")
    thresholds = tuple(float(x.strip()) for x in args.thresholds.split(",") if x.strip())
    if not thresholds:
        raise ValueError("--thresholds must contain at least one float")

    gt_csv = Path(args.gt_csv)
    query_dir = Path(args.query_dir)
    data_dir = Path(args.data_dir)
    assessor_root = Path(args.assessor_root)
    out_csv = Path(args.out_csv)
    missing_gt_csv = Path(args.missing_gt_csv)

    ground_truth = load_ground_truth(gt_csv, query_dir, data_dir)
    if not ground_truth:
        raise RuntimeError("No usable ground-truth queries found.")

    checkpoints = discover_model_checkpoints(assessor_root)
    if not checkpoints:
        raise RuntimeError(f"No model checkpoints found under {assessor_root}")

    print(f"[evaluate] discovered_checkpoints={len(checkpoints)}")
    processor, vit_model, vit_device = load_vit_model()
    full_corpus_candidates = load_full_corpus_candidates(INDEX_PATH, DB_PATH) if args.score_all_candidates else None

    rows: list[dict[str, object]] = []
    for model_label, model_dir, checkpoint_path in checkpoints:
        print(f"\n[evaluate] model={model_label}")
        similarity_types = ("all_candidates",) if args.score_all_candidates else DEFAULT_SIMILARITY_TYPES
        eval_top_ks = ((len(full_corpus_candidates),) if full_corpus_candidates is not None else top_ks)
        for similarity_type in similarity_types:
            for top_k in eval_top_ks:
                for threshold in thresholds:
                    metrics = evaluate_checkpoint(
                        model_dir=model_dir,
                        checkpoint_path=checkpoint_path,
                        ground_truth=ground_truth,
                        top_k=int(top_k),
                        similarity_type=similarity_type,
                        score_all_candidates=args.score_all_candidates,
                        full_corpus_candidates=full_corpus_candidates,
                        threshold=threshold,
                        processor=processor,
                        vit_model=vit_model,
                        vit_device=vit_device,
                    )
                    row = {
                        "model": model_label,
                        "checkpoint_path": str(checkpoint_path),
                        "top_k": top_k,
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"  similarity={similarity_type} top_k={top_k} threshold={threshold:.2f} "
                        f"acc={float(metrics['accuracy']):.4f} "
                        f"precision={float(metrics['precision']):.4f} "
                        f"recall={float(metrics['recall']):.4f} "
                        f"f1={float(metrics['f1']):.4f} "
                        f"missing_gt={metrics['n_missing_gt']} "
                        f"queries_with_missing_gt={metrics['n_queries_with_missing_gt']}"
                    )

    save_results_csv(rows, out_csv)
    save_missing_gt_csv(rows, missing_gt_csv)
    print_results_table(rows)
    print_missing_gt_summary(rows)
    print(f"\nSaved CSV: {out_csv}")
    print(f"Saved missing-GT CSV: {missing_gt_csv}")


if __name__ == "__main__":
    main()
