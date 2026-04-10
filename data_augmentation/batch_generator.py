from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from utils import (
    get_vector_id_by_file_name,
    reconstruct_vector_by_id,
    search_similar_with_metadata,
)


TOP_K = 50
MODEL = "gpt-5-mini"
DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
IMAGE_ROOTS = [Path("input_img"), Path("processed_img")]
PROMPT_PATH = Path("data_augmentation/prompt.txt")
SCHEMA_PATH = Path("data_augmentation/schema.json")
FEW_SHOTS_PATH = Path("data_augmentation/few_shots.jsonl")
OUTPUT_DIR = Path("data_augmentation/batches")
COMPLETION_WINDOW = "24h"
ENDPOINT = "/v1/responses"
BATCH_LOG_PATH = Path("data_augmentation/batch_id_logs.jsonl")


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
        "[batch_generator] query filter:"
        f" total={len(all_file_names)}"
        f" allowed={len(allowed_file_names)}"
        f" excluded_suffix_2={len(excluded_file_names)}"
    )
    if excluded_file_names:
        preview = ", ".join(Path(name).stem for name in excluded_file_names[:5])
        print(f"[batch_generator] excluded query stem samples: {preview}")

    if not allowed_file_names:
        raise RuntimeError("No eligible query images remain after excluding stems ending with '2'.")

    query_file_name = random.choice(allowed_file_names)
    print(f"[batch_generator] selected query stem: {Path(query_file_name).stem}")
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


def build_few_shot_content(few_shots: list[dict]) -> list[dict]:
    content: list[dict] = []
    for idx, shot in enumerate(few_shots, start=1):
        query_file_id = shot.get("Query image")
        candidate_file_id = shot.get("Candidate image")
        query_file_name = shot.get("Query file name", "unknown_query")
        candidate_file_name = shot.get("Candidate file name", "unknown_candidate")
        output = shot.get("Output")

        if not isinstance(query_file_id, str) or not query_file_id.startswith("file-"):
            raise RuntimeError(f"few_shots[{idx}] is missing a valid 'Query image' file id.")
        if not isinstance(candidate_file_id, str) or not candidate_file_id.startswith("file-"):
            raise RuntimeError(f"few_shots[{idx}] is missing a valid 'Candidate image' file id.")
        if not isinstance(output, dict):
            raise RuntimeError(f"few_shots[{idx}] is missing an object 'Output'.")

        content.extend(
            [
                {"type": "input_text", "text": f"Example {idx}"},
                {"type": "input_text", "text": f"Example Query file name: {query_file_name}"},
                {"type": "input_image", "file_id": query_file_id},
                {"type": "input_text", "text": f"Example Candidate file name: {candidate_file_name}"},
                {"type": "input_image", "file_id": candidate_file_id},
                {"type": "input_text", "text": f"Example Output: {json.dumps(output, ensure_ascii=True)}"},
            ]
        )
    return content


def upload_vision_file(client: OpenAI, path: Path, cache: dict[Path, str]) -> str:
    if path in cache:
        return cache[path]
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    cache[path] = uploaded.id
    return uploaded.id


def append_batch_log(
    *,
    batch_id: str,
    input_file_id: str,
    query_file_name: str,
    candidate_file_names: list[str],
    jsonl_path: Path,
) -> None:
    BATCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "jsonl_path": str(jsonl_path),
        "query_image_name": query_file_name,
        "candidate_image_names": candidate_file_names,
    }
    with BATCH_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    few_shot_content = build_few_shot_content(few_shots)
    image_index = build_image_index()
    client = OpenAI()
    upload_cache: dict[Path, str] = {}
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

    query_file_id = upload_vision_file(client, query_path, upload_cache)
    output_path = OUTPUT_DIR / f"batch_{sanitize_name(Path(query_file_name).stem)}.jsonl"

    written = 0
    written_candidate_names: list[str] = []
    with output_path.open("w", encoding="utf-8") as f:
        for rank, cand in enumerate(candidates, start=1):
            if not cand.file_name:
                continue
            candidate_path = resolve_image_path(cand.file_name, image_index)
            if candidate_path is None:
                continue
            candidate_file_id = upload_vision_file(client, candidate_path, upload_cache)

            line = {
                "custom_id": f"{sanitize_name(Path(query_file_name).stem)}__rank_{rank:02d}__vec_{cand.vector_id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                            ] + few_shot_content + [
                                {"type": "input_text", "text": f"Query file name: {query_file_name}"},
                                {"type": "input_image", "file_id": query_file_id},
                                {"type": "input_text", "text": f"Candidate file name: {cand.file_name}"},
                                {"type": "input_image", "file_id": candidate_file_id},
                            ],
                        }
                    ],
                    "text": text_config,
                },
            }
            f.write(json.dumps(line, ensure_ascii=True) + "\n")
            written += 1
            written_candidate_names.append(cand.file_name)

    if written == 0:
        raise RuntimeError("No batch requests were written.")

    with output_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={
            "source_file": output_path.name,
            "query_file_name": query_file_name,
        },
    )

    append_batch_log(
        batch_id=batch.id,
        input_file_id=uploaded.id,
        query_file_name=query_file_name,
        candidate_file_names=written_candidate_names,
        jsonl_path=output_path,
    )

    print("query_file_name:", query_file_name)
    print("query_file_id:", query_file_id)
    print("top_k:", TOP_K)
    print("few_shots:", len(few_shots))
    print("written_requests:", written)
    print("output_path:", output_path)
    print("input_file_id:", uploaded.id)
    print("batch_id:", batch.id)
    print("batch_status:", batch.status)
    print("batch_log_path:", BATCH_LOG_PATH)


if __name__ == "__main__":
    main()
