"""
Claude version of batch_generator.py - Generates batch requests for Claude Messages API.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import (
    get_vector_id_by_file_name,
    reconstruct_vector_by_id,
    search_similar_with_metadata,
)
from claude_helpers import build_few_shot_content_claude, create_claude_batch_request


TOP_K = 50
MODEL = "claude-3-5-sonnet-20241022"
DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
IMAGE_ROOTS = [Path("input_img"), Path("processed_img"), Path("test_query_img")]
PROMPT_PATH = Path("data_augmentation/prompt.txt")
SCHEMA_PATH = Path("data_augmentation/schema.json")
FEW_SHOTS_PATH = Path("data_augmentation_claude/few_shots_claude.jsonl")
OUTPUT_DIR = Path("data_augmentation/batches_claude")
BATCH_LOG_PATH = Path("data_augmentation/batch_id_logs_claude.jsonl")


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "query"


def build_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for root in IMAGE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                index.setdefault(path.stem, path)
    return index


def resolve_image_path(file_name: str, image_index: dict[str, Path]) -> Path | None:
    return image_index.get(Path(file_name).stem)


def is_excluded_query_file_name(file_name: str) -> bool:
    return Path(file_name).stem.endswith("2")


def choose_random_query_file_name() -> str:
    if not DB_PATH.exists():
        raise RuntimeError(f"DB not found: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db")
        all_file_names = [str(row[0]) for row in cur.fetchall()]

    if not all_file_names:
        raise RuntimeError("image_db is empty.")

    allowed_file_names = [name for name in all_file_names if not is_excluded_query_file_name(name)]
    excluded_file_names = [name for name in all_file_names if is_excluded_query_file_name(name)]

    print(
        "[batch_generator_claude] query filter:"
        f" total={len(all_file_names)}"
        f" allowed={len(allowed_file_names)}"
        f" excluded_suffix_2={len(excluded_file_names)}"
    )
    if excluded_file_names:
        preview = ", ".join(Path(name).stem for name in excluded_file_names[:5])
        print(f"[batch_generator_claude] excluded query stem samples: {preview}")

    if not allowed_file_names:
        raise RuntimeError("No eligible query images remain after excluding stems ending with '2'.")

    query_file_name = random.choice(allowed_file_names)
    print(f"[batch_generator_claude] selected query stem: {Path(query_file_name).stem}")
    return query_file_name


def load_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Prompt file is empty: {PROMPT_PATH}")
    return prompt


def load_schema_format() -> dict:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if "type" not in payload:
        payload = {"type": "json_schema", **payload}
    return {
        "format": payload
    }


def load_few_shots() -> list[dict]:
    if not FEW_SHOTS_PATH.exists() or FEW_SHOTS_PATH.stat().st_size == 0:
        return []

    text = FEW_SHOTS_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def append_batch_log(
    *,
    batch_id: str,
    query_file_name: str,
    candidate_file_names: list[str],
    jsonl_path: Path,
    request_count: int,
) -> None:
    BATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "jsonl_path": str(jsonl_path),
        "query_image_name": query_file_name,
        "candidate_image_names": candidate_file_names,
        "request_count": request_count,
        "model": MODEL,
    }
    with BATCH_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    image_index = build_image_index()

    # Build few-shot content for Claude
    few_shot_content = build_few_shot_content_claude(few_shots, image_index) if few_shots else None

    client = Anthropic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    query_file_name = choose_random_query_file_name()
    query_path = resolve_image_path(query_file_name, image_index)
    if query_path is None:
        raise RuntimeError(f"Could not resolve query image path for {query_file_name}")

    query_vector_id = get_vector_id_by_file_name(query_file_name, db_path=DB_PATH)
    query_vector = reconstruct_vector_by_id(query_vector_id, index_path=INDEX_PATH)
    candidates = search_similar_with_metadata(
        query_vector=query_vector,
        similarity_type="l2",
        top_n=TOP_K,
        index_path=INDEX_PATH,
        db_path=DB_PATH,
        exclude_vector_id=query_vector_id,
    )

    # Create batch requests for Claude
    batch_requests = []
    written_candidate_names: list[str] = []

    for rank, cand in enumerate(candidates, start=1):
        if not cand.file_name:
            continue
        candidate_path = resolve_image_path(cand.file_name, image_index)
        if candidate_path is None:
            continue

        custom_id = f"{sanitize_name(Path(query_file_name).stem)}__rank_{rank:02d}__vec_{cand.vector_id}"

        batch_request = create_claude_batch_request(
            custom_id=custom_id,
            prompt=prompt,
            query_image_path=query_path,
            candidate_image_path=candidate_path,
            query_file_name=query_file_name,
            candidate_file_name=cand.file_name,
            few_shot_content=few_shot_content,
            model=MODEL
        )

        batch_requests.append(batch_request)
        written_candidate_names.append(cand.file_name)

    if not batch_requests:
        raise RuntimeError("No batch requests were created.")

    # Save batch requests to JSONL file
    output_path = OUTPUT_DIR / f"batch_{sanitize_name(Path(query_file_name).stem)}_claude.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request, ensure_ascii=True) + "\n")

    # Submit batch to Claude
    batch = client.messages.batch.create(requests=batch_requests)

    append_batch_log(
        batch_id=batch.id,
        query_file_name=query_file_name,
        candidate_file_names=written_candidate_names,
        jsonl_path=output_path,
        request_count=len(batch_requests),
    )

    print("query_file_name:", query_file_name)
    print("top_k:", TOP_K)
    print("few_shots:", len(few_shots))
    print("written_requests:", len(batch_requests))
    print("output_path:", output_path)
    print("batch_id:", batch.id)
    print("batch_processing_status:", batch.processing_status)
    print("batch_log_path:", BATCH_LOG_PATH)


if __name__ == "__main__":
    main()