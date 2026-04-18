"""
Claude version of batch_operator.py - Processes active Claude message batches.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

from batch_retriever_claude import retrieve_batch_claude


NUM_PARALLEL_BATCHES = 0
POLL_INTERVAL_SECONDS = 10
MAX_CYCLES = 10000
ACTIVE_LOG_PATH = Path("data_augmentation/batch_id_logs_claude.jsonl")
DONE_LOG_PATH = Path("data_augmentation/batch_id_logs_done_claude.jsonl")
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


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, records: list[dict]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_batch_generator(count: int) -> None:
    for idx in range(count):
        print(f"[batch_operator_claude] generating batch {idx + 1}/{count}")
        subprocess.run(
            [sys.executable, "batch_generator_claude.py"],
            check=True,
        )


def top_off_batches() -> None:
    active_records = read_jsonl(ACTIVE_LOG_PATH)
    deficit = max(0, NUM_PARALLEL_BATCHES - len(active_records))
    if deficit > 0:
        run_batch_generator(deficit)


def process_active_batches(client: Anthropic) -> tuple[list[dict], list[dict]]:
    active_records = read_jsonl(ACTIVE_LOG_PATH)
    remaining_records: list[dict] = []
    completed_records: list[dict] = []

    for record in active_records:
        batch_id = record.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            continue

        batch = client.messages.batch.retrieve(batch_id)
        status = batch.processing_status
        print(f"[batch_operator_claude] batch_id={batch_id} status={status}")

        if status in ACTIVE_STATUSES:
            remaining_records.append(record)
            continue

        if status in TERMINAL_STATUSES:
            retrieved = retrieve_batch_claude(batch_id, client=client)
            archived_record = dict(record)
            archived_record["final_status"] = status
            archived_record["request_counts"] = batch.request_counts
            archived_record["imported_rows"] = retrieved["imported_rows"]
            archived_record["output_path"] = retrieved["output_path"]
            archived_record["error_path"] = retrieved["error_path"]
            completed_records.append(archived_record)
            print(
                f"[batch_operator_claude] archived batch_id={batch_id} "
                f"imported_rows={retrieved['imported_rows']}"
            )
            continue

        remaining_records.append(record)

    return remaining_records, completed_records


def main() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    if NUM_PARALLEL_BATCHES < 0:
        raise RuntimeError("NUM_PARALLEL_BATCHES must be >= 0")

    client = Anthropic()
    print(f"[batch_operator_claude] target_parallel_batches={NUM_PARALLEL_BATCHES}")
    if NUM_PARALLEL_BATCHES > 0:
        top_off_batches()
    else:
        active_records = read_jsonl(ACTIVE_LOG_PATH)
        print(
            "[batch_operator_claude] top-off disabled; "
            f"monitoring existing active batches only (count={len(active_records)})"
        )

    cycles = 0
    while True:
        cycles += 1
        print(f"[batch_operator_claude] cycle={cycles}")
        remaining_records, completed_records = process_active_batches(client)
        write_jsonl(ACTIVE_LOG_PATH, remaining_records)
        append_jsonl(DONE_LOG_PATH, completed_records)
        if NUM_PARALLEL_BATCHES > 0:
            top_off_batches()
        elif not remaining_records:
            print("[batch_operator_claude] no active batches remain; exiting")
            break
        if MAX_CYCLES is not None and cycles >= MAX_CYCLES:
            print(f"[batch_operator_claude] reached MAX_CYCLES={MAX_CYCLES}, exiting")
            break
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()