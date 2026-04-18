"""
Claude version of batch_retriever.py - Retrieves and processes Claude message batch results.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import get_vector_id_by_file_name
from claude_helpers import extract_claude_response


BATCH_ID = "msgbatch_replace_me"  # Replace with actual batch ID
BATCH_LOG_PATH = Path("data_augmentation/batch_id_logs_claude.jsonl")
OUTPUT_DIR = Path("data_augmentation/batch_results_claude")
DOWNLOAD_OUTPUT = True
DOWNLOAD_ERROR = True
DB_PATH = Path("data/sqlite/ai4h.db")
TABLE_NAME = "ai_augment_feedback_claude"


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


def import_batch_results(batch_id: str, results: list, log_record: dict | None) -> int:
    if log_record is None:
        raise RuntimeError(f"Missing batch log entry for batch_id={batch_id}")

    query_image_name = log_record.get("query_image_name")
    if not isinstance(query_image_name, str) or not query_image_name:
        raise RuntimeError(f"Invalid query_image_name in batch log for batch_id={batch_id}")
    query_vector_id = get_vector_id_by_file_name(query_image_name, db_path=DB_PATH)

    rows: list[tuple] = []
    for result in results:
        custom_id = result.custom_id

        # Extract response data
        response_data = extract_claude_response(result)

        candidate_vector_id = parse_candidate_vector_id(custom_id)
        candidate_image_name = get_file_name_by_vector_id(candidate_vector_id)

        rows.append(
            (
                batch_id,
                custom_id,
                query_image_name,
                candidate_image_name,
                query_vector_id,
                candidate_vector_id,
                int(response_data["lookalike"]),
                int(response_data["error"]),
                int(response_data["identical"]),
                response_data["reasoning"],
                response_data["response_id"],
                custom_id,  # Using custom_id as request_id for Claude
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


def retrieve_batch_claude(
    batch_id: str,
    *,
    client: Anthropic | None = None,
    download_output: bool = DOWNLOAD_OUTPUT,
    download_error: bool = DOWNLOAD_ERROR,
) -> dict:
    own_client = client is None
    if own_client:
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
        client = Anthropic()

    assert client is not None
    batch = client.messages.batch.retrieve(batch_id)
    log_record = find_batch_log(batch_id)

    result = {
        "batch_id": batch.id,
        "processing_status": batch.processing_status,
        "request_counts": batch.request_counts,
        "created_at": batch.created_at,
        "ended_at": batch.ended_at,
        "log_record": log_record,
        "output_path": None,
        "error_path": None,
        "imported_rows": 0,
        "table_name": TABLE_NAME,
    }

    if download_output and batch.processing_status == "ended":
        # Save results to JSONL file
        output_path = OUTPUT_DIR / f"{batch.id}_results.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            for result_item in batch.results:
                f.write(json.dumps({
                    "custom_id": result_item.custom_id,
                    "result": {
                        "message": {
                            "id": result_item.result.message.id if hasattr(result_item.result, 'message') else None,
                            "content": [{"text": result_item.result.message.content[0].text}] if hasattr(result_item.result, 'message') and result_item.result.message.content else []
                        }
                    }
                }, ensure_ascii=True) + "\n")

        imported = import_batch_results(batch.id, batch.results, log_record)
        result["output_path"] = str(output_path)
        result["imported_rows"] = imported

    return result


def main() -> None:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    if not BATCH_ID or BATCH_ID == "msgbatch_replace_me":
        raise RuntimeError("Set BATCH_ID in batch_retriever_claude.py before running.")

    client = Anthropic()
    batch = client.messages.batch.retrieve(BATCH_ID)
    log_record = find_batch_log(BATCH_ID)

    print("batch_id:", batch.id)
    print("processing_status:", batch.processing_status)
    print("request_counts:", batch.request_counts)
    print("created_at:", batch.created_at)
    print("ended_at:", batch.ended_at)

    if log_record is not None:
        print("logged_query_image_name:", log_record.get("query_image_name"))
        print("logged_candidate_count:", len(log_record.get("candidate_image_names", [])))
        print("logged_jsonl_path:", log_record.get("jsonl_path"))

    retrieved = retrieve_batch_claude(
        BATCH_ID,
        client=client,
        download_output=DOWNLOAD_OUTPUT,
        download_error=DOWNLOAD_ERROR,
    )
    if retrieved["output_path"] is not None:
        print("saved_output_path:", retrieved["output_path"])
        print("imported_rows:", retrieved["imported_rows"])
        print("imported_table:", retrieved["table_name"])


if __name__ == "__main__":
    main()