from __future__ import annotations

import argparse
import copy
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
    resolve_image_path,
)
from data_augmentation_claude.claude_helpers import (
    create_claude_batch_request,
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
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "anthropic/claude-3-7-sonnet-20250219")
OPENROUTER_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://openrouter.ai/api")
BASE_DIR = Path("assessor/teacher_eval_claude")
INPUT_DIR = BASE_DIR / "batches"
MANIFEST_DIR = BASE_DIR / "manifests"
RESULT_DIR = BASE_DIR / "results"
ACTIVE_LOG_PATH = BASE_DIR / "active_batches.jsonl"
DONE_LOG_PATH = BASE_DIR / "done_batches.jsonl"
REPORT_CSV_PATH = BASE_DIR / "evaluate_teacher_llm_results_claude.csv"
MISSING_CSV_PATH = BASE_DIR / "evaluate_teacher_llm_missing_claude.csv"



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
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")



def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
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



def local_output_path(run_name: str) -> Path:
    return RESULT_DIR / f"{run_name}_results.jsonl"



def local_error_path(run_name: str) -> Path:
    return RESULT_DIR / f"{run_name}_errors.jsonl"



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



def get_client():
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    from anthropic import Anthropic

    return Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )



def build_request_files(args: argparse.Namespace) -> tuple[str, Path, Path, int]:
    started_at = time.time()
    run_name = args.run_name or make_run_name()
    jsonl_path = INPUT_DIR / f"{run_name}.jsonl"
    manifest_path = MANIFEST_DIR / f"{run_name}.jsonl"
    ensure_parent(jsonl_path)
    ensure_parent(manifest_path)
    _log(f"[teacher_eval_claude.submit] Starting request generation with run_name={run_name} model={args.model}")

    prompt = load_prompt()
    few_shots = load_few_shots()
    image_index = build_image_index()
    _log(f"[teacher_eval_claude.submit] Image index built with {len(image_index)} images")

    few_shot_content: list[dict[str, Any]] = []
    if args.no_few_shots:
        _log("[teacher_eval_claude.submit] Skipping few-shot examples (--no-few-shots)")
    else:
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
                    "[teacher_eval_claude.submit] Warning: failed to prepare few-shots; continuing without them. "
                    f"Error: {exc}"
                )
                few_shot_content = []
            else:
                raise

    ground_truth = load_ground_truth(Path(args.gt_csv), Path(args.query_dir), Path(args.data_dir))
    query_image_map = build_query_image_map(Path(args.query_dir))
    _log(
        f"[teacher_eval_claude.submit] Loaded {len(ground_truth)} ground truth queries and {len(query_image_map)} query images"
    )

    processor, vit_model, vit_device = load_vit_model()
    _log(f"[teacher_eval_claude.submit] ViT model loaded on device: {vit_device}")

    manifest_records: list[dict[str, Any]] = []
    batch_requests: list[dict[str, Any]] = []

    for item in tqdm(ground_truth, desc="Processing queries", unit="query"):
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

            if custom_id is None:
                continue

            batch_request = create_claude_batch_request(
                custom_id=custom_id,
                prompt=prompt,
                query_image_path=query_image_path,
                candidate_image_path=Path(spec.candidate_image_path),
                query_file_name=query_image_name_for_prompt(query_image_path),
                candidate_file_name=spec.candidate_file_name,
                few_shot_content=few_shot_content,
                model=args.model,
            )
            batch_requests.append(batch_request)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=True) + "\n")
    write_jsonl(manifest_path, manifest_records)

    elapsed = time.time() - started_at
    _log(
        f"[teacher_eval_claude.submit] Saved {len(batch_requests)} executable requests and {len(manifest_records)} manifest rows in {elapsed:.1f}s"
    )
    _log(f"[teacher_eval_claude.submit] jsonl_path={jsonl_path}")
    _log(f"[teacher_eval_claude.submit] manifest_path={manifest_path}")
    return run_name, jsonl_path, manifest_path, len(ground_truth)



def extract_text_from_response(response: Any) -> str:
    parts = getattr(response, "content", None) or []
    texts: list[str] = []
    for part in parts:
        part_type = getattr(part, "type", None)
        if part_type == "text":
            texts.append(getattr(part, "text", ""))
    return "\n".join(t for t in texts if t).strip()



def sanitize_request_params(params: dict[str, Any]) -> dict[str, Any]:
    clean_params = copy.deepcopy(params)
    messages = clean_params.get("messages")
    if not isinstance(messages, list):
        return clean_params

    sanitized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized_messages.append(message)
            continue

        cleaned_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_value = block.get("text")
                if not isinstance(text_value, str):
                    continue
                if not text_value.strip():
                    continue
            cleaned_content.append(block)

        new_message = dict(message)
        new_message["content"] = cleaned_content
        sanitized_messages.append(new_message)

    clean_params["messages"] = sanitized_messages
    return clean_params



def parse_response_payload(response: Any) -> dict[str, Any]:
    text = extract_text_from_response(response)
    response_id = getattr(response, "id", None)

    if not text:
        return {
            "response_id": response_id,
            "error": True,
            "lookalike": False,
            "identical": False,
            "reasoning": "Empty text response returned by provider",
        }

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {
            "response_id": response_id,
            "error": True,
            "lookalike": False,
            "identical": False,
            "reasoning": f"Failed to parse JSON response: {exc}; raw={text[:500]}",
        }

    return {
        "response_id": response_id,
        "error": bool(payload.get("error", False)),
        "lookalike": bool(payload.get("lookalike", False)),
        "identical": bool(payload.get("identical", False)),
        "reasoning": str(payload.get("reasoning", "")),
    }



def invoke_request_with_retry(client: Any, request: dict[str, Any], max_retries: int, retry_base_seconds: float) -> dict[str, Any]:
    params = dict(request.get("params", {}))
    if not params:
        raise RuntimeError("Request missing params payload")
    params = sanitize_request_params(params)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(**params)
            return parse_response_payload(response)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            sleep_s = min(retry_base_seconds * (2**attempt), 30.0)
            _log(
                f"[teacher_eval_claude.local] Request {request.get('custom_id')} failed on attempt {attempt + 1}; "
                f"retrying in {sleep_s:.1f}s. Error: {exc}"
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Request failed after retries: {last_error}")



def process_saved_requests(
    *,
    client: Any,
    run_name: str,
    jsonl_path: Path,
    manifest_path: Path,
    model: str,
    query_count: int | None,
    top_k: int | None,
    similarity_type: str | None,
    gt_csv: str | None,
    query_dir: str | None,
    submitted_from_existing_jsonl: bool,
    chunk_size: int,
    max_retries: int,
    retry_base_seconds: float,
    limit: int | None,
) -> None:
    batch_requests = read_jsonl(jsonl_path)
    if not batch_requests:
        raise RuntimeError(f"Saved request JSONL is empty: {jsonl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest JSONL not found: {manifest_path}")

    output_path = local_output_path(run_name)
    error_path = local_error_path(run_name)
    ensure_parent(output_path)
    ensure_parent(error_path)

    completed_results = read_jsonl(output_path)
    completed_errors = read_jsonl(error_path)
    completed_ids = {
        *(r.get("custom_id") for r in completed_results if isinstance(r.get("custom_id"), str)),
        *(r.get("custom_id") for r in completed_errors if isinstance(r.get("custom_id"), str)),
    }

    pending_requests = [
        req for req in batch_requests
        if isinstance(req.get("custom_id"), str) and req["custom_id"] not in completed_ids
    ]
    if limit is not None:
        pending_requests = pending_requests[:limit]

    total_requests = len(batch_requests)
    _log(
        f"[teacher_eval_claude.local] run_name={run_name} total_requests={total_requests} "
        f"completed={len(completed_ids)} pending={len(pending_requests)}"
    )

    active_record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "batch_id": run_name,
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "error_path": str(error_path),
        "query_count": query_count,
        "request_count": total_requests,
        "model": model,
        "top_k": top_k,
        "similarity_type": similarity_type,
        "gt_csv": gt_csv,
        "query_dir": query_dir,
        "submitted_from_existing_jsonl": submitted_from_existing_jsonl,
        "chunk_size": chunk_size,
        "mode": "local_openrouter",
        "status": "running",
    }
    active_records = [r for r in read_jsonl(ACTIVE_LOG_PATH) if r.get("batch_id") != run_name]
    active_records.append(active_record)
    write_jsonl(ACTIVE_LOG_PATH, active_records)

    if not pending_requests:
        _log("[teacher_eval_claude.local] No pending requests left. Finalizing done log.")
    else:
        for idx, request in enumerate(tqdm(pending_requests, desc="Processing local requests", unit="req"), start=1):
            custom_id = request["custom_id"]
            try:
                response_data = invoke_request_with_retry(
                    client=client,
                    request=request,
                    max_retries=max_retries,
                    retry_base_seconds=retry_base_seconds,
                )
                append_jsonl(output_path, [{
                    "custom_id": custom_id,
                    "response_data": response_data,
                }])
            except Exception as exc:
                append_jsonl(error_path, [{
                    "custom_id": custom_id,
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }])
                _log(f"[teacher_eval_claude.local] Request failed permanently for {custom_id}: {exc}")

            if idx % max(chunk_size, 1) == 0:
                _log(
                    f"[teacher_eval_claude.local] Progress: processed {idx}/{len(pending_requests)} pending requests"
                )

    final_record = {
        "created_at": active_record["created_at"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "batch_id": run_name,
        "jsonl_path": str(jsonl_path),
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "error_path": str(error_path),
        "query_count": query_count,
        "request_count": total_requests,
        "model": model,
        "top_k": top_k,
        "similarity_type": similarity_type,
        "gt_csv": gt_csv,
        "query_dir": query_dir,
        "submitted_from_existing_jsonl": submitted_from_existing_jsonl,
        "chunk_size": chunk_size,
        "mode": "local_openrouter",
        "final_status": "ended",
    }
    write_jsonl(ACTIVE_LOG_PATH, [r for r in read_jsonl(ACTIVE_LOG_PATH) if r.get("batch_id") != run_name])
    done_records = [r for r in read_jsonl(DONE_LOG_PATH) if r.get("batch_id") != run_name]
    done_records.append(final_record)
    write_jsonl(DONE_LOG_PATH, done_records)

    _log(f"[teacher_eval_claude.local] Results saved to {output_path}")
    _log(f"[teacher_eval_claude.local] Errors saved to {error_path}")
    _log(f"[teacher_eval_claude.local] Done log updated at {DONE_LOG_PATH}")



def submit(args: argparse.Namespace) -> None:
    load_dotenv()
    run_name, jsonl_path, manifest_path, query_count = build_request_files(args)
    if args.dry_run:
        _log("[teacher_eval_claude.submit] Dry run complete; requests saved locally and not executed.")
        return

    client = get_client()
    process_saved_requests(
        client=client,
        run_name=run_name,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        model=args.model,
        query_count=query_count,
        top_k=args.top_k,
        similarity_type=args.similarity_type,
        gt_csv=str(args.gt_csv),
        query_dir=str(args.query_dir),
        submitted_from_existing_jsonl=False,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        limit=args.limit,
    )



def submit_existing(args: argparse.Namespace) -> None:
    load_dotenv()
    run_name = args.run_name
    if not run_name:
        raise RuntimeError("--run-name is required for submit-existing")

    jsonl_path = INPUT_DIR / f"{run_name}.jsonl"
    manifest_path = MANIFEST_DIR / f"{run_name}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Saved batch JSONL not found: {jsonl_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Saved manifest JSONL not found: {manifest_path}")

    client = get_client()
    process_saved_requests(
        client=client,
        run_name=run_name,
        jsonl_path=jsonl_path,
        manifest_path=manifest_path,
        model=args.model,
        query_count=None,
        top_k=None,
        similarity_type=None,
        gt_csv=None,
        query_dir=None,
        submitted_from_existing_jsonl=True,
        chunk_size=args.chunk_size,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        limit=args.limit,
    )



def poll(args: argparse.Namespace) -> None:
    _log("[teacher_eval_claude.poll] Poll is not needed in local mode.")
    active_records = read_jsonl(ACTIVE_LOG_PATH)
    if not active_records:
        _log("[teacher_eval_claude.poll] No active local runs recorded.")
        return
    _log(f"[teacher_eval_claude.poll] Active local runs: {len(active_records)}")
    for record in active_records:
        _log(
            f"[teacher_eval_claude.poll] run_name={record.get('run_name')} output_path={record.get('output_path')} status={record.get('status', 'running')}"
        )



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
    excluded_error_outputs = 0
    excluded_missing_outputs = 0
    excluded_rows_total = 0

    for row in tqdm(manifest_rows, desc="Analyzing predictions", unit="pair"):
        true_label = int(row["true_label"])
        status = str(row["status"])
        custom_id = row.get("custom_id")
        query_id = str(row["query_id"])
        pred_label = 0
        include_in_metrics = True

        if status == "submitted" and isinstance(custom_id, str):
            submitted_pairs += 1
            payload = predictions_by_custom_id.get(custom_id)
            if payload is None:
                row["prediction_status"] = "missing_output"
                if args.exclude_missing_outputs:
                    include_in_metrics = False
                    excluded_missing_outputs += 1
            elif payload["error"]:
                row["prediction_status"] = "response_error"
                response_errors += 1
                if args.exclude_error_outputs:
                    include_in_metrics = False
                    excluded_error_outputs += 1
            else:
                pred_label = 1 if payload["lookalike"] else 0
                row["prediction_status"] = "ok"
                row["reasoning"] = payload["reasoning"]
                row["identical"] = payload["identical"]
        else:
            row["prediction_status"] = status

        if include_in_metrics:
            y_true.append(true_label)
            y_pred.append(pred_label)
            query_stats[query_id]["pairs"] += 1
            if true_label == 1:
                query_stats[query_id]["positives"] += 1
            if pred_label == 1:
                query_stats[query_id]["predicted_positive"] += 1
        else:
            excluded_rows_total += 1

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

    if not y_true:
        raise RuntimeError("No rows left for metric calculation after applying exclusion options")

    metrics = {
        "batch_id": done_record["batch_id"],
        "run_name": done_record.get("run_name"),
        "model": done_record.get("model"),
        "top_k": done_record.get("top_k"),
        "similarity_type": done_record.get("similarity_type"),
        "n_queries": len({str(row["query_id"]) for row in manifest_rows}),
        "n_eval_pairs": len(manifest_rows),
        "n_pairs_used_for_metrics": len(y_true),
        "n_submitted_pairs": submitted_pairs,
        "n_output_predictions": len(predictions_by_custom_id),
        "n_response_errors": response_errors,
        "n_excluded_error_outputs": excluded_error_outputs,
        "n_excluded_missing_outputs": excluded_missing_outputs,
        "n_excluded_rows_total": excluded_rows_total,
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

    elapsed_total = time.time() - started_at
    _log(f"[teacher_eval_claude.report] Report generation completed in {elapsed_total:.1f}s")
    _log(f"[teacher_eval_claude.report] Batch ID: {metrics['batch_id']}")
    _log(f"[teacher_eval_claude.report] Queries: {metrics['n_queries']}, Evaluation pairs: {metrics['n_eval_pairs']}")
    _log(f"[teacher_eval_claude.report] Submitted pairs: {metrics['n_submitted_pairs']}, Response errors: {metrics['n_response_errors']}")
    _log(
        f"[teacher_eval_claude.report] Used for metrics: {metrics['n_pairs_used_for_metrics']} / {metrics['n_eval_pairs']} "
        f"(excluded error outputs={metrics['n_excluded_error_outputs']}, excluded missing outputs={metrics['n_excluded_missing_outputs']})"
    )
    _log(f"[teacher_eval_claude.report] Unresolved positive rows: {metrics['n_unresolved_positive_rows']}")
    _log(f"[teacher_eval_claude.report] Performance - Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    _log(f"[teacher_eval_claude.report] Report saved to: {REPORT_CSV_PATH}")
    _log(f"[teacher_eval_claude.report] Missing pairs report saved to: {MISSING_CSV_PATH}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher LLM baseline evaluation on test_query_img (Claude local/OpenRouter version)")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    submit_parser = subparsers.add_parser("submit", help="Generate request JSONL and optionally execute locally")
    submit_parser.add_argument("--gt-csv", type=str, default=str(GT_CSV_PATH))
    submit_parser.add_argument("--query-dir", type=str, default=str(TEST_QUERY_DIR))
    submit_parser.add_argument("--data-dir", type=str, default="data")
    submit_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    submit_parser.add_argument("--similarity-type", type=str, default=DEFAULT_SIMILARITY_TYPE)
    submit_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_parser.add_argument("--run-name", type=str, default=None)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--no-few-shots", action="store_true", help="Disable few-shot examples entirely")
    submit_parser.add_argument(
        "--skip-missing-few-shots",
        action="store_true",
        help="If few-shot image resolution fails, continue with zero few-shots",
    )
    submit_parser.add_argument("--rebuild-few-shots", action="store_true", help="Force rebuild of cached Claude few-shot content")
    submit_parser.add_argument("--chunk-size", type=int, default=25, help="Progress logging interval for local execution")
    submit_parser.add_argument("--max-retries", type=int, default=2)
    submit_parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    submit_parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending requests")

    submit_existing_parser = subparsers.add_parser(
        "submit-existing",
        help="Execute a previously generated request JSONL locally and store results on disk",
    )
    submit_existing_parser.add_argument("--run-name", type=str, required=True)
    submit_existing_parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    submit_existing_parser.add_argument("--chunk-size", type=int, default=25, help="Progress logging interval for local execution")
    submit_existing_parser.add_argument("--max-retries", type=int, default=2)
    submit_existing_parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    submit_existing_parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending requests")

    poll_parser = subparsers.add_parser("poll", help="Show any locally tracked active runs")
    poll_parser.add_argument("--watch", action="store_true")
    poll_parser.add_argument("--poll-interval", type=int, default=30)

    report_parser = subparsers.add_parser("report", help="Compute metrics from a completed local run")
    report_parser.add_argument("--batch-id", type=str, default=None)
    report_parser.add_argument("--exclude-error-outputs", action="store_true", help="Exclude outputs where response_data.error is true from metric calculation")
    report_parser.add_argument("--exclude-missing-outputs", action="store_true", help="Exclude submitted pairs with no matching output from metric calculation")

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        submit(args)
        return
    if args.command == "submit-existing":
        submit_existing(args)
        return
    if args.command == "poll":
        poll(args)
        return
    if args.command == "report":
        report(args)
        return
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
