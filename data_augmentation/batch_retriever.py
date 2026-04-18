from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import get_vector_id_by_file_name


BATCH_ID = "batch_69b7f1d042ec8190ab551a5ed690a8c6"
BATCH_LOG_PATH = Path("data_augmentation/batch_id_logs.jsonl")
OUTPUT_DIR = Path("data_augmentation/batch_results")
DOWNLOAD_OUTPUT = True
DOWNLOAD_ERROR = True
DB_PATH = Path("data/sqlite/ai4h.db")
TABLE_NAME = "ai_augment_feedback"


def find_batch_log(batch_id: str) -> dict | None:
    if not BATCH_LOG_PATH.exists():
        return None

    with BATCH_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("batch_id") == batch_id:
                return record
    return None


def download_file(client: OpenAI, file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.content(file_id).read()
    out_path.write_bytes(content)


def ensure_table() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                custom_id TEXT NOT NULL,
                query_image_name TEXT NOT NULL,
                candidate_image_name TEXT NOT NULL,
                query_vector_id INTEGER NOT NULL,
                candidate_vector_id INTEGER NOT NULL,
                label INTEGER NOT NULL,
                error INTEGER NOT NULL DEFAULT 0,
                identical INTEGER NOT NULL DEFAULT 0,
                reasoning TEXT NOT NULL,
                response_id TEXT,
                request_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(batch_id, custom_id)
            )
            """
        )
        conn.commit()


def parse_candidate_vector_id(custom_id: str) -> int:
    match = re.search(r"__vec_(\d+)$", custom_id)
    if match is None:
        raise RuntimeError(f"Could not parse candidate vector id from custom_id: {custom_id}")
    return int(match.group(1))


def get_file_name_by_vector_id(vector_id: int) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db WHERE vector_id = ?", (int(vector_id),))
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"vector_id not found in image_db: {vector_id}")
    return str(row[0])


def extract_output_payload(record: dict) -> tuple[str | None, str, int, int, int]:
    response = record.get("response") or {}
    body = response.get("body") or {}
    response_id = body.get("id")
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
            reasoning = payload.get("reasoning")
            lookalike = payload.get("lookalike")
            error = payload.get("error", False)
            identical = payload.get("identical", False)
            if (
                not isinstance(reasoning, str)
                or not isinstance(lookalike, bool)
                or not isinstance(error, bool)
                or not isinstance(identical, bool)
            ):
                raise RuntimeError(f"Invalid output payload for custom_id={record.get('custom_id')}")
            return response_id, reasoning, int(lookalike), int(error), int(identical)

    raise RuntimeError(f"No output_text found for custom_id={record.get('custom_id')}")


def import_output_jsonl(batch_id: str, output_path: Path, log_record: dict | None) -> int:
    if log_record is None:
        raise RuntimeError(f"Missing batch log entry for batch_id={batch_id}")
    if not output_path.exists():
        raise FileNotFoundError(f"Output JSONL not found: {output_path}")

    query_image_name = log_record.get("query_image_name")
    if not isinstance(query_image_name, str) or not query_image_name:
        raise RuntimeError(f"Invalid query_image_name in batch log for batch_id={batch_id}")
    query_vector_id = get_vector_id_by_file_name(query_image_name, db_path=DB_PATH)

    rows: list[tuple] = []
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

            candidate_vector_id = parse_candidate_vector_id(custom_id)
            candidate_image_name = get_file_name_by_vector_id(candidate_vector_id)
            response_id, reasoning, label, error, identical = extract_output_payload(record)
            response = record.get("response") or {}
            request_id = response.get("request_id")

            rows.append(
                (
                    batch_id,
                    custom_id,
                    query_image_name,
                    candidate_image_name,
                    query_vector_id,
                    candidate_vector_id,
                    label,
                    error,
                    identical,
                    reasoning,
                    response_id,
                    request_id,
                )
            )

    ensure_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {TABLE_NAME} (
                batch_id,
                custom_id,
                query_image_name,
                candidate_image_name,
                query_vector_id,
                candidate_vector_id,
                label,
                error,
                identical,
                reasoning,
                response_id,
                request_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def retrieve_batch(
    batch_id: str,
    *,
    client: OpenAI | None = None,
    download_output: bool = DOWNLOAD_OUTPUT,
    download_error: bool = DOWNLOAD_ERROR,
) -> dict:
    own_client = client is None
    if own_client:
        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing in .env")
        client = OpenAI()

    assert client is not None
    batch = client.batches.retrieve(batch_id)
    log_record = find_batch_log(batch_id)

    result = {
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": batch.input_file_id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "log_record": log_record,
        "output_path": None,
        "error_path": None,
        "imported_rows": 0,
        "table_name": TABLE_NAME,
    }

    if download_output and batch.output_file_id:
        output_path = OUTPUT_DIR / f"{batch.id}_output.jsonl"
        download_file(client, batch.output_file_id, output_path)
        imported = import_output_jsonl(batch.id, output_path, log_record)
        result["output_path"] = str(output_path)
        result["imported_rows"] = imported

    if download_error and batch.error_file_id:
        error_path = OUTPUT_DIR / f"{batch.id}_error.jsonl"
        download_file(client, batch.error_file_id, error_path)
        result["error_path"] = str(error_path)

    return result


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not BATCH_ID or BATCH_ID == "replace_me":
        raise RuntimeError("Set BATCH_ID in batch_retriever.py before running.")

    client = OpenAI()
    batch = client.batches.retrieve(BATCH_ID)
    log_record = find_batch_log(BATCH_ID)

    print("batch_id:", batch.id)
    print("status:", batch.status)
    print("endpoint:", batch.endpoint)
    print("input_file_id:", batch.input_file_id)
    print("output_file_id:", batch.output_file_id)
    print("error_file_id:", batch.error_file_id)
    print("created_at:", batch.created_at)
    print("in_progress_at:", batch.in_progress_at)
    print("completed_at:", batch.completed_at)
    print("failed_at:", batch.failed_at)
    print("expired_at:", batch.expired_at)
    print("request_counts:", batch.request_counts)
    print("metadata:", batch.metadata)

    if log_record is not None:
        print("logged_query_image_name:", log_record.get("query_image_name"))
        print("logged_candidate_count:", len(log_record.get("candidate_image_names", [])))
        print("logged_jsonl_path:", log_record.get("jsonl_path"))

    retrieved = retrieve_batch(
        BATCH_ID,
        client=client,
        download_output=DOWNLOAD_OUTPUT,
        download_error=DOWNLOAD_ERROR,
    )
    if retrieved["output_path"] is not None:
        print("saved_output_path:", retrieved["output_path"])
        print("imported_rows:", retrieved["imported_rows"])
        print("imported_table:", retrieved["table_name"])
    if retrieved["error_path"] is not None:
        print("saved_error_path:", retrieved["error_path"])


if __name__ == "__main__":
    main()
