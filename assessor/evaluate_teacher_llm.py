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
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from assessor.evaluate import IMAGE_EXTS, QueryGroundTruth, load_ground_truth
from assessor.prompt_variants import build_schema_format as build_variant_schema_format
from assessor.prompt_variants import get_prompt_variant
from data_augmentation.batch_generator import (
    COMPLETION_WINDOW,
    ENDPOINT,
    MODEL as DEFAULT_MODEL,
    build_few_shot_content,
    build_image_index,
    load_few_shots,
    load_schema_format,
    resolve_image_path,
    upload_vision_file,
)
from image_embedding.vit import get_image_embedding, load_vit_model
from utils import get_vector_id_by_file_name, search_similar_with_metadata


DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
TEST_QUERY_DIR = Path("test_query_img")
GT_CSV_PATH = TEST_QUERY_DIR / "gt_table_csv.csv"
DEFAULT_TOP_K = 50
DEFAULT_SIMILARITY_TYPE = "l2"
BASE_DIR = Path("assessor/teacher_eval")
ACTIVE_STATUSES = {
    "validating",
    "in_progress",
    "finalizing",
    "cancelling",
}
TERMINAL_STATUSES = {
    "completed",
    "failed",
    "expired",
    "cancelled",
}


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


def build_eval_paths(base_dir: Path) -> dict[str, Path]:
    return {
        "base_dir": base_dir,
        "input_dir": base_dir / "batches",
        "manifest_dir": base_dir / "manifests",
        "result_dir": base_dir / "results",
        "active_log_path": base_dir / "active_batches.jsonl",
        "done_log_path": base_dir / "done_batches.jsonl",
        "report_csv_path": base_dir / "evaluate_teacher_llm_results.csv",
        "missing_csv_path": base_dir / "evaluate_teacher_llm_missing.csv",
    }


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
    return datetime.now(timezone.utc).strftime("teacher_eval_%Y%m%dT%H%M%SZ")


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
    load_dotenv()
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    paths = build_eval_paths(Path(args.base_dir))
    run_name = args.run_name or make_run_name()
    jsonl_path = paths["input_dir"] / f"{run_name}.jsonl"
    manifest_path = paths["manifest_dir"] / f"{run_name}.jsonl"
    ensure_parent(jsonl_path)
    ensure_parent(manifest_path)

    variant = get_prompt_variant(args.prompt_variant)
    prompt = variant["prompt"]
    text_config = build_variant_schema_format(load_schema_format(), args.prompt_variant)
    few_shots = load_few_shots(args.few_shot_count)
    few_shot_content = build_few_shot_content(few_shots)
    image_index = build_image_index()
    ground_truth = load_ground_truth(Path(args.gt_csv), Path(args.query_dir), Path(args.data_dir))
    query_image_map = build_query_image_map(Path(args.query_dir))

    processor, vit_model, vit_device = load_vit_model()
    client = None if args.dry_run else OpenAI()
    upload_cache: dict[Path, str] = {}

    manifest_records: list[dict[str, Any]] = []
    written_requests = 0
    with jsonl_path.open("w", encoding="utf-8") as batch_f:
        for item in ground_truth:
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

            query_file_id: str | None = None
            if not args.dry_run:
                assert client is not None
                query_file_id = upload_vision_file(client, query_image_path, upload_cache)

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
                assert query_file_id is not None
                assert client is not None
                candidate_file_id = upload_vision_file(client, candidate_path, upload_cache)
                line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": ENDPOINT,
                    "body": {
                        "model": args.model,
                        "input": [
                            {
                                "role": "user",
                                "content": build_request_content(
                                    prompt=prompt,
                                    few_shot_content=few_shot_content,
                                    query_image_name=query_image_name_for_prompt(query_image_path),
                                    query_file_id=query_file_id,
                                    candidate_file_name=spec.candidate_file_name,
                                    candidate_file_id=candidate_file_id,
                                ),
                            }
                        ],
                        "text": text_config,
                    },
                }
                batch_f.write(json.dumps(line, ensure_ascii=True) + "\n")
                written_requests += 1

    write_jsonl(manifest_path, manifest_records)

    submitted_count = sum(1 for row in manifest_records if row["status"] == "submitted")
    gt_extra_count = sum(1 for row in manifest_records if row["source"] == "gt_extra")
    unresolved_positive_count = sum(
        1 for row in manifest_records if row["true_label"] == 1 and row["status"] != "submitted"
    )
    print(
        f"[teacher_eval.submit] run_name={run_name} queries={len(ground_truth)} "
        f"manifest_rows={len(manifest_records)} submitted_pairs={submitted_count} "
        f"written_requests={written_requests} gt_extra_rows={gt_extra_count} "
        f"unresolved_positive_rows={unresolved_positive_count}"
    )
    print(f"[teacher_eval.submit] jsonl_path={jsonl_path}")
    print(f"[teacher_eval.submit] manifest_path={manifest_path}")

    if args.dry_run:
        print("[teacher_eval.submit] dry-run enabled; batch upload skipped")
        return

    if written_requests == 0:
        print("[teacher_eval.submit] no new requests were written; skipping batch creation")
        return

    assert client is not None
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={
            "source_file": jsonl_path.name,
            "run_name": run_name,
        },
    )
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "query_count": len(ground_truth),
        "request_count": written_requests,
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "top_k": args.top_k,
            "similarity_type": args.similarity_type,
            "gt_csv": str(args.gt_csv),
        "query_dir": str(args.query_dir),
    }
    append_jsonl(paths["active_log_path"], [record])
    print(f"[teacher_eval.submit] batch_id={batch.id}")
    print(f"[teacher_eval.submit] active_log={paths['active_log_path']}")


def download_file(client: OpenAI, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id).read()
    out_path.write_bytes(content)


def poll(args: argparse.Namespace) -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    paths = build_eval_paths(Path(args.base_dir))
    client = OpenAI()
    active_records = read_jsonl(paths["active_log_path"])
    remaining_records: list[dict[str, Any]] = []
    completed_records: list[dict[str, Any]] = []

    for record in active_records:
        batch_id = record.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            continue
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"[teacher_eval.poll] batch_id={batch_id} status={status}")

        if status in ACTIVE_STATUSES:
            remaining_records.append(record)
            continue

        if status in TERMINAL_STATUSES:
            archived = dict(record)
            archived["final_status"] = status
            archived["output_file_id"] = batch.output_file_id
            archived["error_file_id"] = batch.error_file_id
            archived["completed_at"] = datetime.now(timezone.utc).isoformat()

            if batch.output_file_id:
                output_path = paths["result_dir"] / f"{batch.id}_output.jsonl"
                download_file(client, batch.output_file_id, output_path)
                archived["output_path"] = str(output_path)
            else:
                archived["output_path"] = None

            if batch.error_file_id:
                error_path = paths["result_dir"] / f"{batch.id}_error.jsonl"
                download_file(client, batch.error_file_id, error_path)
                archived["error_path"] = str(error_path)
            else:
                archived["error_path"] = None

            completed_records.append(archived)
            continue

        remaining_records.append(record)

    write_jsonl(paths["active_log_path"], remaining_records)
    append_jsonl(paths["done_log_path"], completed_records)
    print(
        f"[teacher_eval.poll] remaining_active={len(remaining_records)} "
        f"archived={len(completed_records)}"
    )

    if args.watch and remaining_records:
        time.sleep(args.poll_interval)
        poll(args)


def extract_output_payload(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response") or {}
    body = response.get("body") or {}
    output = body.get("output") or []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") != "output_text":
                continue
            text = content.get("text")
            if not isinstance(text, str):
                continue
            payload = json.loads(text)
            return {
                "response_id": body.get("id"),
                "request_id": response.get("request_id"),
                "error": bool(payload.get("error", False)),
                "score": int(payload["score"]) if payload.get("score") is not None else None,
                "lookalike": bool(payload.get("lookalike", False)),
                "identical": bool(payload.get("identical", False)),
                "reasoning": str(payload.get("reasoning", "")),
            }
    raise RuntimeError(f"No output_text found for custom_id={record.get('custom_id')}")


def find_done_record(batch_id: str | None, done_log_path: Path) -> dict[str, Any]:
    records = read_jsonl(done_log_path)
    if not records:
        raise RuntimeError(f"No done records found in {done_log_path}")
    if batch_id is None:
        return records[-1]
    for record in records:
        if record.get("batch_id") == batch_id:
            return record
    raise RuntimeError(f"batch_id not found in done log: {batch_id}")


def report(args: argparse.Namespace) -> None:
    paths = build_eval_paths(Path(args.base_dir))
    done_record = find_done_record(args.batch_id, paths["done_log_path"])
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
            if record.get("error") is not None:
                continue
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
                row["score"] = payload["score"]
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
        "prompt_variant": done_record.get("prompt_variant"),
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

    paths["report_csv_path"].parent.mkdir(parents=True, exist_ok=True)
    existing_reports = []
    if paths["report_csv_path"].exists():
        with paths["report_csv_path"].open("r", encoding="utf-8", newline="") as f:
            existing_reports = list(csv.DictReader(f))
        existing_reports = [row for row in existing_reports if row.get("batch_id") != metrics["batch_id"]]

    fieldnames = list(metrics.keys())
    with paths["report_csv_path"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_reports:
            writer.writerow(row)
        writer.writerow(metrics)

    with paths["missing_csv_path"].open("w", encoding="utf-8", newline="") as f:
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

    print(
        f"[teacher_eval.report] batch_id={metrics['batch_id']} queries={metrics['n_queries']} "
        f"eval_pairs={metrics['n_eval_pairs']} submitted_pairs={metrics['n_submitted_pairs']} "
        f"response_errors={metrics['n_response_errors']} "
        f"unresolved_positive_rows={metrics['n_unresolved_positive_rows']}"
    )
    print(
        f"[teacher_eval.report] accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    print(f"[teacher_eval.report] report_csv={paths['report_csv_path']}")
    print(f"[teacher_eval.report] missing_csv={paths['missing_csv_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher LLM baseline evaluation on test_query_img")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    submit_parser = subparsers.add_parser("submit", help="Generate and submit teacher-eval batch requests")
    submit_parser.add_argument("--gt-csv", type=str, default=str(GT_CSV_PATH))
    submit_parser.add_argument("--query-dir", type=str, default=str(TEST_QUERY_DIR))
    submit_parser.add_argument("--data-dir", type=str, default="data")
    submit_parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    submit_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    submit_parser.add_argument("--similarity-type", type=str, default=DEFAULT_SIMILARITY_TYPE)
    submit_parser.add_argument("--few-shot-count", type=int, choices=(3, 5), default=3)
    submit_parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=("a_baseline", "b_conservative", "d_scoring"),
        default="a_baseline",
    )
    submit_parser.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    submit_parser.add_argument("--run-name", type=str, default=None)
    submit_parser.add_argument("--dry-run", action="store_true")

    poll_parser = subparsers.add_parser("poll", help="Poll active teacher-eval batches and download outputs")
    poll_parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    poll_parser.add_argument("--watch", action="store_true")
    poll_parser.add_argument("--poll-interval", type=int, default=30)

    report_parser = subparsers.add_parser("report", help="Compute metrics from a completed teacher-eval batch")
    report_parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
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
