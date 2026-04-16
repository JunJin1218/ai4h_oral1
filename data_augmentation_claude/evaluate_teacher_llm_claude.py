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
from tqdm import tqdm

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from assessor.evaluate import IMAGE_EXTS, QueryGroundTruth, load_ground_truth
from data_augmentation_claude.batch_generator_claude import (
    build_image_index,
    load_few_shots,
    load_prompt,
    load_schema_format,
    resolve_image_path,
)
from data_augmentation_claude.claude_helpers import (
    create_claude_batch_request,
    extract_claude_response,
    get_or_build_claude_few_shots,
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
    _log(f"[teacher_eval_claude.submit] Starting batch submission with run_name={run_name} model={args.model}")

    _log("[teacher_eval_claude.submit] Loading configuration files...")
    prompt = load_prompt()
    # text_config = load_schema_format()
    few_shots = load_few_shots()
    _log("[teacher_eval_claude.submit] Building image index...")
    image_index = build_image_index()
    _log(f"[teacher_eval_claude.submit] Image index built with {len(image_index)} images")

    few_shot_content: list[dict[str, Any]] = []
    if args.no_few_shots:
        _log("[teacher_eval_claude.submit] Skipping few-shot examples (--no-few-shots)")
    else:
        _log("[teacher_eval_claude.submit] Preparing few-shot examples...")
        try:
            few_shot_content = get_or_build_claude_few_shots(
                few_shots,
                image_index,
                force_rebuild=args.rebuild_few_shots,
            )
            _log(f"[teacher_eval_claude.submit] Loaded {len(few_shot_content)} few-shot examples")
        except RuntimeError as exc:
            if args.skip_missing_few_shots:
                _log(
                    "[teacher_eval_claude.submit] Warning: Failed to prepare few-shots; "
                    f"continuing without few-shots. Error: {exc}"
                )
                few_shot_content = []
            else:
                raise

    _log("[teacher_eval_claude.submit] Loading ground truth data...")
    ground_truth = load_ground_truth(Path(args.gt_csv), Path(args.query_dir), Path(args.data_dir))
    query_image_map = build_query_image_map(Path(args.query_dir))
    _log(
        f"[teacher_eval_claude.submit] Loaded {len(ground_truth)} ground truth queries "
        f"and {len(query_image_map)} query images"
    )

    _log("[teacher_eval_claude.submit] Loading ViT model for image embeddings...")
    processor, vit_model, vit_device = load_vit_model()
    _log(f"[teacher_eval_claude.submit] ViT model loaded on device: {vit_device}")

    client = None if args.dry_run else Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url="https://openrouter.ai/api",
    )

    manifest_records: list[dict[str, Any]] = []
    batch_requests: list[dict[str, Any]] = []
    written_requests = 0

    total_queries = len(ground_truth)
    _log(f"[teacher_eval_claude.submit] Processing {total_queries} queries...")

    for query_idx, item in enumerate(tqdm(ground_truth, desc="Processing queries", unit="query"), start=1):
        elapsed_s = time.time() - started_at
        _log(
            f"[teacher_eval_claude.submit] Processing query {query_idx}/{total_queries} "
            f"(ID: {item.query_id}) - Elapsed: {elapsed_s:.1f}s"
        )

        query_image_path = query_image_map.get(item.query_id, item.query_image_path)
        _log(f"[teacher_eval_claude.submit] Generating embedding for query image: {query_image_path}")
        query_vector_t = get_image_embedding(
            vit_model,
            processor,
            query_image_path,
            device=vit_device,
            pooling="cls",
        )
        query_vector = query_vector_t.detach().cpu().numpy().astype("float32", copy=False)

        _log(f"[teacher_eval_claude.submit] Finding similar candidates (top-{args.top_k})...")
        candidate_specs = build_candidate_specs(
            item=item,
            query_vector=query_vector,
            image_index=image_index,
            top_k=args.top_k,
            similarity_type=args.similarity_type,
        )
        _log(f"[teacher_eval_claude.submit] Found {len(candidate_specs)} candidate pairs")

        for spec_idx, spec in enumerate(candidate_specs):
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

            if custom_id is None:
                continue

            candidate_path = Path(spec.candidate_image_path)
            _log(f"[teacher_eval_claude.submit] Creating batch request for candidate: {spec.candidate_file_name}")
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
        _log("[teacher_eval_claude.submit] dry-run enabled; full request JSONL and manifest were saved, "
            "but remote batch submission was skipped")
        return

    if written_requests == 0:
        _log("[teacher_eval_claude.submit] no new requests were written; skipping batch creation")
        return

    assert client is not None

    submit_request_chunks(
        client=client,
        batch_requests=batch_requests,
        run_name=run_name,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        model=args.model,
        chunk_size=args.chunk_size,
        query_count=len(ground_truth),
        top_k=args.top_k,
        similarity_type=args.similarity_type,
        gt_csv=str(args.gt_csv),
        query_dir=str(args.query_dir),
        submitted_from_existing_jsonl=False,
    )

    total_elapsed_s = time.time() - started_at
    _log(f"[teacher_eval_claude.submit] Batch submission completed!")
    _log(f"[teacher_eval_claude.submit] Active log: {ACTIVE_LOG_PATH}")
    _log(f"[teacher_eval_claude.submit] Total elapsed time: {total_elapsed_s:.1f}s")
    _log(f"[teacher_eval_claude.submit] Next: Run 'python evaluate_teacher_llm_claude.py poll' to monitor progress")

def submit_existing(args: argparse.Namespace) -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")

    from anthropic import Anthropic

    run_name = args.run_name
    if not run_name:
        raise RuntimeError("--run-name is required for submit-existing")

    jsonl_path = INPUT_DIR / f"{run_name}.jsonl"
    manifest_path = MANIFEST_DIR / f"{run_name}.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Saved batch JSONL not found: {jsonl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest JSONL not found: {manifest_path}")

    batch_requests: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch_requests.append(json.loads(line))

    if not batch_requests:
        raise RuntimeError(
            f"Saved batch JSONL is empty: {jsonl_path}. "
            "Make sure the dry run was generated after removing the args.dry_run skip."
        )

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url="https://openrouter.ai/api",
    )

    _log(
        f"[teacher_eval_claude.submit_existing] Submitting saved batch "
        f"run_name={run_name} requests={len(batch_requests)}"
    )

    submit_request_chunks(
        client=client,
        batch_requests=batch_requests,
        run_name=run_name,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        model=args.model,
        chunk_size=args.chunk_size,
        query_count=None,
        top_k=None,
        similarity_type=None,
        gt_csv=None,
        query_dir=None,
        submitted_from_existing_jsonl=True,
    )

    _log(f"[teacher_eval_claude.submit_existing] Submission completed for run_name={run_name}")
    _log(f"[teacher_eval_claude.submit_existing] Active log: {ACTIVE_LOG_PATH}")


def poll(args: argparse.Namespace) -> None:
    poll_started_at = time.time()
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")

    from anthropic import Anthropic
    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url="https://openrouter.ai/api",
    )

    _log("[teacher_eval_claude.poll] Loading active batch records...")
    active_records = read_jsonl(ACTIVE_LOG_PATH)
    _log(f"[teacher_eval_claude.poll] Found {len(active_records)} active batch(es) to monitor")

    if not active_records:
        _log("[teacher_eval_claude.poll] No active batches found - nothing to poll")
        return

    remaining_records: list[dict[str, Any]] = []
    completed_records: list[dict[str, Any]] = []

    total_batches = len(active_records)
    _log(f"[teacher_eval_claude.poll] Starting to poll {total_batches} batch(es)...")

    for batch_idx, record in enumerate(active_records, start=1):
        batch_id = record.get("batch_id")
        elapsed_s = time.time() - poll_started_at

        if not isinstance(batch_id, str) or not batch_id:
            _log(f"[teacher_eval_claude.poll] Skipping invalid batch record {batch_idx}/{total_batches}")
            continue

        _log(
            f"[teacher_eval_claude.poll] Checking batch {batch_idx}/{total_batches} "
            f"(ID: {batch_id}) - Elapsed: {elapsed_s:.1f}s"
        )

        try:
            _log(f"[teacher_eval_claude.poll] Retrieving batch status from API...")
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            request_counts = batch.request_counts

            # Convert request_counts to dict for progress tracking
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
                f"[teacher_eval_claude.poll] Batch {batch_id} status: {status} "
                f"Progress: {done}/{total} ({pct:.1f}%) "
                f"[Processing: {processing}, Succeeded: {succeeded}, Errored: {errored}, "
                f"Canceled: {canceled}, Expired: {expired}]"
            )

            if status in ACTIVE_STATUSES:
                remaining_records.append(record)
                _log(f"[teacher_eval_claude.poll] Batch {batch_id} still active - will check again later")
                continue

            if status in TERMINAL_STATUSES:
                _log(f"[teacher_eval_claude.poll] Batch {batch_id} reached terminal status: {status}")
                archived = dict(record)
                archived["final_status"] = status
                archived["request_counts"] = batch.request_counts
                archived["completed_at"] = datetime.now(timezone.utc).isoformat()

                if status == "ended":
                    _log(f"[teacher_eval_claude.poll] Batch {batch_id} completed successfully! Downloading results...")
                    output_path = RESULT_DIR / f"{batch.id}_results.jsonl"
                    RESULT_DIR.mkdir(parents=True, exist_ok=True)

                    with output_path.open("w", encoding="utf-8") as f:
                        for result_item in tqdm(batch.results, desc=f"Saving results for batch {batch_id}", unit="result"):
                            response_data = extract_claude_response(result_item)
                            f.write(json.dumps({
                                "custom_id": result_item.custom_id,
                                "response_data": response_data
                            }, ensure_ascii=True) + "\n")

                    archived["output_path"] = str(output_path)
                    _log(f"[teacher_eval_claude.poll] Results saved to {output_path}")
                else:
                    archived["output_path"] = None
                    _log(f"[teacher_eval_claude.poll] Batch {batch_id} ended with status {status} - no results to save")

                completed_records.append(archived)
                _log(f"[teacher_eval_claude.poll] Batch {batch_id} archived as completed")
                continue

        except Exception as exc:
            _log(f"[teacher_eval_claude.poll] Error polling batch {batch_id}: {exc}")
            record["error"] = str(exc)
            remaining_records.append(record)
            continue

    # Save updated logs
    _log("[teacher_eval_claude.poll] Updating batch logs...")
    write_jsonl(ACTIVE_LOG_PATH, remaining_records)
    append_jsonl(DONE_LOG_PATH, completed_records)

    elapsed_total = time.time() - poll_started_at
    _log(
        f"[teacher_eval_claude.poll] Polling completed. "
        f"Remaining active: {len(remaining_records)}, "
        f"Archived: {len(completed_records)}, "
        f"Total elapsed: {elapsed_total:.1f}s"
    )

    if args.watch and remaining_records:
        _log(f"[teacher_eval_claude.poll] Watch mode enabled - sleeping for {args.poll_interval}s before next poll...")
        time.sleep(args.poll_interval)
        poll(args)
    elif not remaining_records:
        _log("[teacher_eval_claude.poll] All batches completed! Ready to run report.")
    else:
        _log(f"[teacher_eval_claude.poll] {len(remaining_records)} batch(es) still processing. Run poll again later or use --watch.")


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
    _log("[teacher_eval_claude.report] Starting report generation...")

    _log("[teacher_eval_claude.report] Finding completed batch record...")
    done_record = find_done_record(args.batch_id)
    output_path = Path(done_record["output_path"]) if done_record.get("output_path") else None
    manifest_path = Path(done_record["manifest_path"])

    if output_path is None or not output_path.exists():
        raise RuntimeError(f"Output JSONL not found for batch_id={done_record.get('batch_id')}")
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest JSONL not found: {manifest_path}")

    _log("[teacher_eval_claude.report] Loading prediction results...")
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
    _log(f"[teacher_eval_claude.report] Loaded {len(predictions_by_custom_id)} prediction results")

    _log("[teacher_eval_claude.report] Loading manifest data...")
    manifest_rows = read_jsonl(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"No manifest rows found in {manifest_path}")
    _log(f"[teacher_eval_claude.report] Loaded {len(manifest_rows)} manifest rows")

    _log("[teacher_eval_claude.report] Analyzing predictions and calculating metrics...")
    y_true: list[int] = []
    y_pred: list[int] = []
    query_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_rows: list[dict[str, Any]] = []
    submitted_pairs = 0
    response_errors = 0
    unresolved_positive_rows = 0

    total_rows = len(manifest_rows)
    _log(f"[teacher_eval_claude.report] Processing {total_rows} evaluation pairs...")

    for row_idx, row in enumerate(tqdm(manifest_rows, desc="Analyzing predictions", unit="pair"), start=1):
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

    _log("[teacher_eval_claude.report] Calculating performance metrics...")
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

    _log("[teacher_eval_claude.report] Saving report files...")
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

    _log("[teacher_eval_claude.report] Saving missing pairs report...")
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

    elapsed_total = time.time() - started_at
    _log(f"[teacher_eval_claude.report] Report generation completed in {elapsed_total:.1f}s")
    _log(f"[teacher_eval_claude.report] Batch ID: {metrics['batch_id']}")
    _log(f"[teacher_eval_claude.report] Queries: {metrics['n_queries']}, Evaluation pairs: {metrics['n_eval_pairs']}")
    _log(f"[teacher_eval_claude.report] Submitted pairs: {metrics['n_submitted_pairs']}, Response errors: {metrics['n_response_errors']}")
    _log(f"[teacher_eval_claude.report] Unresolved positive rows: {metrics['n_unresolved_positive_rows']}")
    _log(f"[teacher_eval_claude.report] Performance - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    _log(f"[teacher_eval_claude.report] Report saved to: {REPORT_CSV_PATH}")
    _log(f"[teacher_eval_claude.report] Missing pairs report saved to: {MISSING_CSV_PATH}")


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
    submit_parser.add_argument("--model", type=str, default=os.environ.get("ANTHROPIC_MODEL", "anthropic/claude-3-7-sonnet-20250219"))
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
    submit_parser.add_argument(
        "--rebuild-few-shots",
        action="store_true",
        help="Force rebuild of cached Claude few-shot content",
    )
    submit_parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of requests to send per remote batch"
    )

    submit_existing_parser = subparsers.add_parser(
        "submit-existing",
        help="Submit a previously generated batch JSONL without rebuilding requests"
    )
    submit_existing_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Run name of the previously generated dry-run files"
    )
    submit_existing_parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("ANTHROPIC_MODEL", "anthropic/claude-3-7-sonnet-20250219"),
    )
    submit_existing_parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of requests to send per remote batch"
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
    if command == "submit-existing":
        submit_existing(args)
        return
    if command == "poll":
        poll(args)
        return
    if command == "report":
        report(args)
        return
    raise RuntimeError(f"Unknown command: {command}")

def chunk_list(items: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def submit_request_chunks(
    *,
    client,
    batch_requests: list[dict[str, Any]],
    run_name: str,
    jsonl_path: Path,
    manifest_path: Path,
    model: str,
    chunk_size: int,
    query_count: int | None,
    top_k: int | None,
    similarity_type: str | None,
    gt_csv: str | None,
    query_dir: str | None,
    submitted_from_existing_jsonl: bool,
) -> None:
    chunks = chunk_list(batch_requests, chunk_size)
    _log(
        f"[teacher_eval_claude.submit_chunks] Split requests into "
        f"{len(chunks)} chunk(s) with chunk_size={chunk_size}"
    )

    for chunk_idx, request_chunk in enumerate(chunks, start=1):
        _log(
            f"[teacher_eval_claude.submit_chunks] Submitting chunk "
            f"{chunk_idx}/{len(chunks)} with {len(request_chunk)} request(s)"
        )
        batch = client.messages.batches.create(requests=request_chunk)

        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": f"{run_name}__chunk_{chunk_idx:03d}",
            "batch_id": batch.id,
            "jsonl_path": str(jsonl_path),
            "manifest_path": str(manifest_path),
            "query_count": query_count,
            "request_count": len(request_chunk),
            "model": model,
            "top_k": top_k,
            "similarity_type": similarity_type,
            "gt_csv": gt_csv,
            "query_dir": query_dir,
            "submitted_from_existing_jsonl": submitted_from_existing_jsonl,
            "chunk_index": chunk_idx,
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
        }
        append_jsonl(ACTIVE_LOG_PATH, [record])

        _log(f"[teacher_eval_claude.submit_chunks] Batch created successfully with ID: {batch.id}")


if __name__ == "__main__":
    main()