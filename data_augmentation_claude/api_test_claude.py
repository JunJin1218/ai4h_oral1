"""
Claude version of api_test.py - Minimal Claude image comparison test.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic, APIStatusError

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import get_vector_id_by_file_name, reconstruct_vector_by_id, search_similar_with_metadata
from claude_helpers import build_few_shot_content_claude, build_vision_content, clean_stem, create_claude_batch_request


DEFAULT_MODEL = "anthropic/claude-3-5-sonnet"
TOP_K = 50
CANDIDATE_RANK = 3
DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
IMAGE_ROOTS = [Path("input_img"), Path("processed_img"), Path("test_query_img")]
PROMPT_PATH = Path("data_augmentation/prompt.txt")
SCHEMA_PATH = Path("data_augmentation/schema.json")
FEW_SHOTS_PATH = Path("data_augmentation/few_shots.jsonl")


def load_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise RuntimeError(f"Prompt file is empty: {PROMPT_PATH}")
    return prompt


def load_schema_format() -> dict[str, Any]:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if "type" not in payload:
        payload = {"type": "json_schema", **payload}
    return {"format": payload}


def load_few_shots() -> list[dict[str, Any]]:
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

    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for root in IMAGE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                index.setdefault(clean_stem(path.stem), path)
    return index


def resolve_image_path(file_name: str, image_index: dict[str, Path]) -> Path | None:
    return image_index.get(clean_stem(Path(file_name).stem))


def choose_random_query_file_name() -> str:
    if not DB_PATH.exists():
        raise RuntimeError(f"DB not found: {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db ORDER BY RANDOM() LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("image_db is empty.")
    return str(row[0])


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Minimal Claude image comparison test")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for errors")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    image_index = build_image_index()

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
    if len(candidates) < CANDIDATE_RANK:
        raise RuntimeError(
            f"Expected at least {CANDIDATE_RANK} candidates, found {len(candidates)}"
        )

    candidate = candidates[CANDIDATE_RANK - 1]
    if not candidate.file_name:
        raise RuntimeError(f"Candidate at rank {CANDIDATE_RANK} is missing file_name")
    candidate_path = resolve_image_path(candidate.file_name, image_index)
    if candidate_path is None:
        raise RuntimeError(f"Could not resolve candidate image path for {candidate.file_name}")

    # Build few-shot content for Claude
    few_shot_content = build_few_shot_content_claude(few_shots, image_index) if few_shots else None

    # Build content for Claude
    content = [
        {"type": "text", "text": prompt},
    ]

    if few_shot_content:
        content.extend(few_shot_content)

    content.extend([
        {"type": "text", "text": f"Query file name: {query_file_name}"},
        *build_vision_content("", query_path),
        {"type": "text", "text": f"Candidate file name: {candidate.file_name}"},
        *build_vision_content("", candidate_path),
    ])

    client = Anthropic()
    last_error: Exception | None = None
    response = None

    for attempt in range(args.max_retries + 1):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            break
        except APIStatusError as exc:
            last_error = exc
            print(f"[attempt {attempt + 1}] Claude API error")
            print("status:", exc.status_code)
            print("request_id:", exc.request_id)
            print("body:", exc.body)
            if attempt >= args.max_retries:
                raise
            time.sleep(min(2 ** attempt, 4))

    if response is None:
        raise RuntimeError(f"Claude request failed after retries: {last_error}")

    print("model:", response.model)
    print("query_file_name:", query_file_name)
    print("query:", query_path)
    print("query_vector_id:", query_vector_id)
    print("candidate_rank:", CANDIDATE_RANK)
    print("candidate_file_name:", candidate.file_name)
    print("candidate:", candidate_path)
    print("candidate_vector_id:", candidate.vector_id)
    print("candidate_retrieval_score:", candidate.score)
    print("few_shots:", len(few_shots))
    print()
    print(response.content[0].text)


if __name__ == "__main__":
    main()