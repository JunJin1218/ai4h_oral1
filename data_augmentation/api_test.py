from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import get_vector_id_by_file_name, reconstruct_vector_by_id, search_similar_with_metadata


DEFAULT_MODEL = "anthropic/claude-3-5-sonnet"
TOP_K = 50
CANDIDATE_RANK = 3
DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
IMAGE_ROOTS = [Path("input_img"), Path("processed_img")]
PROMPT_PATH = Path("data_augmentation/prompt.txt")
SCHEMA_PATH = Path("data_augmentation/schema.json")
FEW_SHOTS_PATH = Path("data_augmentation/few_shots.jsonl")
RUNS_LOG_PATH = Path("data_augmentation/runs/api_test_runs.jsonl")


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


def build_few_shot_content(few_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Temporarily disable few_shots as they use OpenAI file_id but script uses Claude
    content: list[dict[str, str]] = []
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
    # return []


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


def encode_image_base64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


def get_image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    elif suffix == ".png":
        return "image/png"
    elif suffix == ".webp":
        return "image/webp"
    elif suffix in {".gif"}:
        return "image/gif"
    else:
        return "image/jpeg"  # default


def serialize_content_blocks(content_blocks: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            serialized.append(block.model_dump())
        elif isinstance(block, dict):
            serialized.append(block)
        else:
            serialized.append({"type": type(block).__name__, "value": str(block)})
    return serialized


def extract_response_text(content_blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def append_run_log(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Minimal Anthropic image comparison test")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for 5xx errors")
    parser.add_argument(
        "--save-path",
        default=str(RUNS_LOG_PATH),
        help="Append run outputs to this JSONL file",
    )
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

    client = Anthropic( 
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api"),
    )
    print("Base URL:", os.getenv("ANTHROPIC_BASE_URL", "https://openrouter.ai/api"))
    print("Model:", args.model)
    query_data = encode_image_base64(query_path)
    query_media_type = get_image_media_type(query_path)
    candidate_data = encode_image_base64(candidate_path)
    candidate_media_type = get_image_media_type(candidate_path)

    few_shot_content = build_few_shot_content(few_shots)

    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        *few_shot_content,
        {"type": "text", "text": f"Query file name: {query_file_name}"},
        {"type": "image", "source": {"type": "base64", "media_type": query_media_type, "data": query_data}},
        {"type": "text", "text": f"Candidate file name: {candidate.file_name}"},
        {"type": "image", "source": {"type": "base64", "media_type": candidate_media_type, "data": candidate_data}},
    ]

    last_error: Exception | None = None
    response = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as exc:  # Anthropic exceptions
            last_error = exc
            print(f"[attempt {attempt + 1}] Anthropic error: {exc}")
            if attempt >= args.max_retries:
                raise
            time.sleep(min(2 ** attempt, 4))

    if response is None:
        raise RuntimeError(f"Anthropic request failed after retries: {last_error}")

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
    response_text = extract_response_text(response.content)
    serialized_content = serialize_content_blocks(response.content)
    if response_text:
        print(response_text)
    else:
        print("[No text block found in response content. Full content saved in run log.]")

    run_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": args.model,
        "model_used": response.model,
        "query_file_name": query_file_name,
        "query_path": str(query_path),
        "query_vector_id": query_vector_id,
        "candidate_rank": CANDIDATE_RANK,
        "candidate_file_name": candidate.file_name,
        "candidate_path": str(candidate_path),
        "candidate_vector_id": candidate.vector_id,
        "candidate_retrieval_score": candidate.score,
        "few_shots_count": len(few_shots),
        "response_text": response_text,
        "response_content": serialized_content,
    }
    append_run_log(run_record, Path(args.save_path))
    print(f"saved_run_log: {args.save_path}")


if __name__ == "__main__":
    main()
