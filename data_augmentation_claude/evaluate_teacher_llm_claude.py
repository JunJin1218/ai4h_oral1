"""
Claude version of evaluate_teacher_llm.py - Teacher LLM evaluation using Claude Messages API.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from assessor.evaluate import IMAGE_EXTS, QueryGroundTruth, load_ground_truth
from data_augmentation.batch_generator import (
    build_image_index,
    load_few_shots,
    load_prompt,
    load_schema_format,
    resolve_image_path,
)
from data_augmentation_claude.claude_helpers import (
    build_few_shot_content_claude,
    build_vision_content,
    create_claude_batch_request,
    extract_claude_response,
)
from image_embedding.vit import get_image_embedding, load_vit_model
from utils import get_vector_id_by_file_name, search_similar_with_metadata


DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
TEST_QUERY_DIR = Path("test_query_img")
GT_CSV_PATH = TEST_QUERY_DIR / "gt_table_csv.csv"
DEFAULT_TOP_K = 50
DEFAULT_SIMILARITY_TYPE = "l2"
BASE_DIR = Path("assessor/teacher_eval_claude")
INPUT_DIR = BASE_DIR / "batches"
MANIFEST_DIR = BASE_DIR / "manifests"
RESULT_DIR = BASE_DIR / "results"
ACTIVE_LOG_PATH = BASE_DIR / "active_batches.jsonl"
DONE_LOG_PATH = BASE_DIR / "done_batches.jsonl"
REPORT_CSV_PATH = BASE_DIR / "evaluate_teacher_llm_results_claude.csv"
MISSING_CSV_PATH = BASE_DIR / "evaluate_teacher_llm_missing_claude.csv"
ACTIVE_STATUSES = {
    "in_progress",
    "processing",
}
TERMINAL_STATUSES = {
    "ended",
    "failed",
    "expired",
    "cancelled",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


@dataclass(frozen=True)
class CandidateSpec:
    candidate_file_name: str
    true_label: int
    source: str
    retrieval_rank: int | None
    status: str
    note: str | None
    vector_id: int | None
    candidate_image_path: str | None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def query_image_name_for_prompt(path: Path) -> str:
    return path.name


def build_query_image_map(query_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(query_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = path.stem.split(maxsplit=1)
        if not parts:
            continue
        query_id = parts[0]
        if query_id.isdigit():
            out[query_id] = path
    return out


def make_run_name() -> str:
    return datetime.now(timezone.utc).strftime("teacher_eval_claude_%Y%m%dT%H%M%SZ")


def build_candidate_specs(
    *,
    item: QueryGroundTruth,
    query_vector: Any,
    image_index: dict[str, Path],
    top_k: int,
    similarity_type: str,
) -> list[CandidateSpec]:
    candidates = search_similar_with_metadata(
        query_vector=query_vector,
        similarity_type=similarity_type,
        top_n=top_k,
        index_path=INDEX_PATH,
        db_path=DB_PATH,
        exclude_vector_id=None,
    )

    candidate_specs: list[CandidateSpec] = []
    seen_names: set[str] = set()
    for rank, cand in enumerate(candidates, start=1):
        if cand.file_name is None or cand.file_name in seen_names:
            continue
        seen_names.add(cand.file_name)
        candidate_path = resolve_image_path(cand.file_name, image_index)
        status = "submitted" if candidate_path is not None else "missing_candidate_image"
        note = None if candidate_path is not None else "candidate image path not found in image roots"
        candidate_specs.append(
            CandidateSpec(
                candidate_file_name=cand.file_name,
                true_label=1 if cand.file_name in item.positive_candidate_names else 0,
                source="retrieved",
                retrieval_rank=rank,
                status=status,
                note=note,
                vector_id=int(cand.vector_id),
                candidate_image_path=None if candidate_path is None else str(candidate_path),
            )
        )

    for positive_name in sorted(item.positive_candidate_names):
        if positive_name in seen_names:
            continue
        seen_names.add(positive_name)
        try:
            vector_id = get_vector_id_by_file_name(positive_name, db_path=DB_PATH)
        except Exception as exc:
            candidate_specs.append(
                CandidateSpec(
                    candidate_file_name=positive_name,
                    true_label=1,
                    source="gt_extra",
                    retrieval_rank=None,
                    status="missing_vector_id",
                    note=str(exc),
                    vector_id=None,
                    candidate_image_path=None,
                )
            )
            continue

        candidate_path = resolve_image_path(positive_name, image_index)
        status = "submitted" if candidate_path is not None else "missing_candidate_image"
        note = None if candidate_path is not None else "gt extra candidate image path not found in image roots"
        candidate_specs.append(
            CandidateSpec(
                candidate_file_name=positive_name,
                true_label=1,
                source="gt_extra",
                retrieval_rank=None,
                status=status,
                note=note,
                vector_id=int(vector_id),
                candidate_image_path=None if candidate_path is None else str(candidate_path),
            )
        )

    return candidate_specs


def build_request_content(
    *,
    prompt: str,
    few_shot_content: list[dict[str, Any]],
    query_image_name: str,
    query_file_id: str,
    candidate_file_name: str,
    candidate_file_id: str,
) -> list[dict[str, Any]]:
    return [
        {"type": "input_text", "text": prompt},
        *few_shot_content,
        {"type": "input_text", "text": f"Query file name: {query_image_name}"},
        {"type": "input_image", "file_id": query_file_id},
        {"type": "input_text", "text": f"Candidate file name: {candidate_file_name}"},
        {"type": "input_image", "file_id": candidate_file_id},
    ]


def submit(args: argparse.Namespace) -> None:
    started_at = time.time()
    load_dotenv()
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    if not args.dry_run:
        from anthropic import Anthropic

    run_name = args.run_name or make_run_name()
    jsonl_path = INPUT_DIR / f"{run_name}.jsonl"
    manifest_path = MANIFEST_DIR / f"{run_name}.jsonl"
    ensure_parent(jsonl_path)
    ensure_parent(manifest_path)
    _log(f"[teacher_eval_claude.submit] run_name={run_name} model={args.model}")

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    few_shot_content: list[dict[str, Any]] = []
    if args.no_few_shots:
        _log("[teacher_eval_claude.submit] skipping few-shots (--no-few-shots)")
    else:
        try:
            few_shot_content = build_few_shot_content_claude(few_shots, build_image_index())
        except RuntimeError as exc:
            if args.skip_missing_few_shots:
                _log(
                    "[teacher_eval_claude.submit] warning: failed to resolve few-shots; "
                    f"continuing without few-shots. error={exc}"
                )
                few_shot_content = []
            else:
                raise
    image_index = build_image_index()
    ground_truth = load_ground_truth(Path(args.gt_csv), Path(args.query_dir), Path(args.data_dir))
    query_image_map = build_query_image_map(Path(args.query_dir))
    _log(
        f"[teacher_eval_claude.submit] loaded_ground_truth={len(ground_truth)} "
        f"query_images={len(query_image_map)}"
    )

    processor, vit_model, vit_device = load_vit_model()
    client = None if args.dry_run else Anthropic()

    manifest_records: list[dict[str, Any]] = []
    batch_requests: list[dict[str, Any]] = []
    written_requests = 0

    total_queries = len(ground_truth)
    for query_idx, item in enumerate(ground_truth, start=1):
        elapsed_s = time.time() - started_at
        _log(
            f"[teacher_eval_claude.submit] query_progress={query_idx}/{total_queries} "
            f"query_id={item.query_id} elapsed_s={elapsed_s:.1f}"
        )
        query_image_path = query_image_map.get(item.query_id, item.query_image_path)
        query_vector_t = get_image_embedding(
            vit_model,
            processor,
            query_image_path,
            device=vit_device,
            pooling="cls",
        )
        query_vector = query_vector_t.detach().cpu().numpy().astype("float32", copy=False)
        candidate_specs = build_candidate_specs(
            item=item,
            query_vector=query_vector,
            image_index=image_index,
            top_k=args.top_k,
            similarity_type=args.similarity_type,
        )

        for spec in candidate_specs:
            custom_id = (
                f"{run_name}__query_{item.query_id}__vec_{spec.vector_id}"
                if spec.status == "submitted" and spec.vector_id is not None
                else None
            )
            manifest_records.append(
                {
                    "run_name": run_name,
                    "query_id": item.query_id,
                    "query_image_path": str(query_image_path),
                    "query_image_name": query_image_name_for_prompt(query_image_path),
                    "candidate_file_name": spec.candidate_file_name,
                    "candidate_image_path": spec.candidate_image_path,
                    "true_label": spec.true_label,
                    "source": spec.source,
                    "retrieval_rank": spec.retrieval_rank,
                    "status": spec.status,
                    "note": spec.note,
                    "vector_id": spec.vector_id,
                    "custom_id": custom_id,
                }
            )

            if custom_id is None or args.dry_run:
                continue

            candidate_path = Path(spec.candidate_image_path)
            batch_request = create_claude_batch_request(
                custom_id=custom_id,
                prompt=prompt,
                query_image_path=query_image_path,
                candidate_image_path=candidate_path,
                query_file_name=query_image_name_for_prompt(query_image_path),
                candidate_file_name=spec.candidate_file_name,
                few_shot_content=few_shot_content,
                model=args.model
            )
            batch_requests.append(batch_request)
            written_requests += 1

    # Save batch requests to JSONL file
    with jsonl_path.open("w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=True) + "\n")

    write_jsonl(manifest_path, manifest_records)

    submitted_count = sum(1 for row in manifest_records if row["status"] == "submitted")
    gt_extra_count = sum(1 for row in manifest_records if row["source"] == "gt_extra")
    unresolved_positive_count = sum(
        1 for row in manifest_records if row["true_label"] == 1 and row["status"] != "submitted"
    )
    _log(
        f"[teacher_eval_claude.submit] run_name={run_name} queries={len(ground_truth)} "
        f"manifest_rows={len(manifest_records)} submitted_pairs={submitted_count} "
        f"written_requests={written_requests} gt_extra_rows={gt_extra_count} "
        f"unresolved_positive_rows={unresolved_positive_count}"
    )
    _log(f"[teacher_eval_claude.submit] jsonl_path={jsonl_path}")
    _log(f"[teacher_eval_claude.submit] manifest_path={manifest_path}")

    if args.dry_run:
        _log("[teacher_eval_claude.submit] dry-run enabled; batch upload skipped")
        return

    if written_requests == 0:
        _log("[teacher_eval_claude.submit] no new requests were written; skipping batch creation")
        return

    assert client is not None
    batch = client.messages.batch.create(requests=batch_requests)

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "batch_id": batch.id,
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "query_count": len(ground_truth),
        "request_count": written_requests,
        "model": args.model,
        "top_k": args.top_k,
        "similarity_type": args.similarity_type,
        "gt_csv": str(args.gt_csv),
        "query_dir": str(args.query_dir),
    }
    append_jsonl(ACTIVE_LOG_PATH, [record])
    total_elapsed_s = time.time() - started_at
    _log(f"[teacher_eval_claude.submit] batch_id={batch.id}")
    _log(f"[teacher_eval_claude.submit] active_log={ACTIVE_LOG_PATH}")
    _log(f"[teacher_eval_claude.submit] completed elapsed_s={total_elapsed_s:.1f}")


def poll(args: argparse.Namespace) -> None:
    poll_started_at = time.time()
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")

    from anthropic import Anthropic

    client = Anthropic()
    active_records = read_jsonl(ACTIVE_LOG_PATH)
    remaining_records: list[dict[str, Any]] = []
    completed_records: list[dict[str, Any]] = []

    for record in active_records:
        batch_id = record.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        batch = client.messages.batch.retrieve(batch_id)
        status = batch.processing_status
        request_counts = batch.request_counts
        # Anthropic returns request_counts as an object. Convert defensively so
        # we can print progress even if SDK fields change slightly.
        if hasattr(request_counts, "model_dump"):
            counts = request_counts.model_dump()
        elif isinstance(request_counts, dict):
            counts = request_counts
        else:
            counts = {}

        processing = int(counts.get("processing", 0) or 0)
        succeeded = int(counts.get("succeeded", 0) or 0)
        errored = int(counts.get("errored", 0) or 0)
        canceled = int(counts.get("canceled", 0) or 0)
        expired = int(counts.get("expired", 0) or 0)
        total = processing + succeeded + errored + canceled + expired
        done = succeeded + errored + canceled + expired
        pct = (done / total * 100.0) if total > 0 else 0.0

        _log(
            f"[teacher_eval_claude.poll] batch_id={batch_id} status={status} "
            f"progress={done}/{total} ({pct:.1f}%) "
            f"processing={processing} succeeded={succeeded} errored={errored} "
            f"canceled={canceled} expired={expired}"
        )

        if status in ACTIVE_STATUSES:
            remaining_records.append(record)
            continue

        if status in TERMINAL_STATUSES:
            archived = dict(record)
            archived["final_status"] = status
            archived["request_counts"] = batch.request_counts
            archived["completed_at"] = datetime.now(timezone.utc).isoformat()

            if status == "ended":
                output_path = RESULT_DIR / f"{batch.id}_results.jsonl"
                RESULT_DIR.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    for result_item in batch.results:
                        response_data = extract_claude_response(result_item)
                        f.write(json.dumps({
                            "custom_id": result_item.custom_id,
                            "response_data": response_data
                        }, ensure_ascii=True) + "\n")
                archived["output_path"] = str(output_path)
            else:
                archived["output_path"] = None

            completed_records.append(archived)
            continue

        remaining_records.append(record)

    write_jsonl(ACTIVE_LOG_PATH, remaining_records)
    append_jsonl(DONE_LOG_PATH, completed_records)
    _log(
        f"[teacher_eval_claude.poll] remaining_active={len(remaining_records)} "
        f"archived={len(completed_records)} elapsed_s={time.time() - poll_started_at:.1f}"
    )

    if args.watch and remaining_records:
        time.sleep(args.poll_interval)
        poll(args)


def extract_output_payload(record: dict[str, Any]) -> dict[str, Any]:
    response_data = record.get("response_data", {})
    return {
        "response_id": response_data.get("response_id"),
        "error": response_data.get("error", False),
        "lookalike": response_data.get("lookalike", False),
        "identical": response_data.get("identical", False),
        "reasoning": response_data.get("reasoning", ""),
    }


def find_done_record(batch_id: str | None) -> dict[str, Any]:
    records = read_jsonl(DONE_LOG_PATH)
    if not records:
        raise RuntimeError(f"No done records found in {DONE_LOG_PATH}")
    if batch_id is None:
        return records[-1]
    for record in records:
        if record.get("batch_id") == batch_id:
            return record
    raise RuntimeError(f"batch_id not found in done log: {batch_id}")


def report(args: argparse.Namespace) -> None:
    started_at = time.time()
    done_record = find_done_record(args.batch_id)
    output_path = Path(done_record["output_path"]) if done_record.get("output_path") else None
    manifest_path = Path(done_record["manifest_path"])
    if output_path is None or not output_path.exists():
        raise RuntimeError(f"Output JSONL not found for batch_id={done_record.get('batch_id')}")
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest JSONL not found: {manifest_path}")

    predictions_by_custom_id: dict[str, dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                continue
            predictions_by_custom_id[custom_id] = extract_output_payload(record)

    manifest_rows = read_jsonl(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"No manifest rows found in {manifest_path}")

    y_true: list[int] = []
    y_pred: list[int] = []
    query_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_rows: list[dict[str, Any]] = []
    submitted_pairs = 0
    response_errors = 0
    unresolved_positive_rows = 0

    for row in manifest_rows:
        true_label = int(row["true_label"])
        status = str(row["status"])
        custom_id = row.get("custom_id")
        query_id = str(row["query_id"])
        pred_label = 0

        if status == "submitted" and isinstance(custom_id, str):
            submitted_pairs += 1
            payload = predictions_by_custom_id.get(custom_id)
            if payload is None:
                row["prediction_status"] = "missing_output"
            elif payload["error"]:
                row["prediction_status"] = "response_error"
                response_errors += 1
            else:
                pred_label = 1 if payload["lookalike"] else 0
                row["prediction_status"] = "ok"
                row["reasoning"] = payload["reasoning"]
                row["identical"] = payload["identical"]
        else:
            row["prediction_status"] = status

        y_true.append(true_label)
        y_pred.append(pred_label)
        query_stats[query_id]["pairs"] += 1
        if true_label == 1:
            query_stats[query_id]["positives"] += 1
        if pred_label == 1:
            query_stats[query_id]["predicted_positive"] += 1
        if true_label == 1 and status != "submitted":
            unresolved_positive_rows += 1
            missing_rows.append(
                {
                    "batch_id": done_record["batch_id"],
                    "query_id": query_id,
                    "query_image_path": row["query_image_path"],
                    "candidate_file_name": row["candidate_file_name"],
                    "source": row["source"],
                    "status": status,
                    "note": row.get("note"),
                }
            )

    metrics = {
        "batch_id": done_record["batch_id"],
        "run_name": done_record.get("run_name"),
        "model": done_record.get("model"),
        "top_k": done_record.get("top_k"),
        "similarity_type": done_record.get("similarity_type"),
        "n_queries": len({str(row["query_id"]) for row in manifest_rows}),
        "n_eval_pairs": len(manifest_rows),
        "n_submitted_pairs": submitted_pairs,
        "n_output_predictions": len(predictions_by_custom_id),
        "n_response_errors": response_errors,
        "n_unresolved_positive_rows": unresolved_positive_rows,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    REPORT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_reports = []
    if REPORT_CSV_PATH.exists():
        with REPORT_CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            existing_reports = list(csv.DictReader(f))
        existing_reports = [row for row in existing_reports if row.get("batch_id") != metrics["batch_id"]]

    fieldnames = list(metrics.keys())
    with REPORT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_reports:
            writer.writerow(row)
        writer.writerow(metrics)

    with MISSING_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "batch_id",
            "query_id",
            "query_image_path",
            "candidate_file_name",
            "source",
            "status",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in missing_rows:
            writer.writerow(row)

    _log(
        f"[teacher_eval_claude.report] batch_id={metrics['batch_id']} queries={metrics['n_queries']} "
        f"eval_pairs={metrics['n_eval_pairs']} submitted_pairs={metrics['n_submitted_pairs']} "
        f"response_errors={metrics['n_response_errors']} "
        f"unresolved_positive_rows={metrics['n_unresolved_positive_rows']}"
    )
    _log(
        f"[teacher_eval_claude.report] accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    _log(f"[teacher_eval_claude.report] report_csv={REPORT_CSV_PATH}")
    _log(f"[teacher_eval_claude.report] missing_csv={MISSING_CSV_PATH}")
    _log(f"[teacher_eval_claude.report] completed elapsed_s={time.time() - started_at:.1f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher LLM baseline evaluation on test_query_img (Claude version)")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    submit_parser = subparsers.add_parser("submit", help="Generate and submit teacher-eval batch requests")
    submit_parser.add_argument("--gt-csv", type=str, default=str(GT_CSV_PATH))
    submit_parser.add_argument("--query-dir", type=str, default=str(TEST_QUERY_DIR))
    submit_parser.add_argument("--data-dir", type=str, default="data")
    submit_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    submit_parser.add_argument("--similarity-type", type=str, default=DEFAULT_SIMILARITY_TYPE)
    submit_parser.add_argument("--model", type=str, default=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    submit_parser.add_argument("--run-name", type=str, default=None)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument(
        "--no-few-shots",
        action="store_true",
        help="Disable few-shot examples entirely",
    )
    submit_parser.add_argument(
        "--skip-missing-few-shots",
        action="store_true",
        help="If few-shot image resolution fails, continue with zero few-shots",
    )

    poll_parser = subparsers.add_parser("poll", help="Poll active teacher-eval batches and download outputs")
    poll_parser.add_argument("--watch", action="store_true")
    poll_parser.add_argument("--poll-interval", type=int, default=30)

    report_parser = subparsers.add_parser("report", help="Compute metrics from a completed teacher-eval batch")
    report_parser.add_argument("--batch-id", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command

    if command == "submit":
        submit(args)
        return
    if command == "poll":
        poll(args)
        return
    if command == "report":
        report(args)
        return
    raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()