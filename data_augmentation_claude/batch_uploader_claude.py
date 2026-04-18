"""
Claude version of batch_uploader.py - Uploads batch requests directly to Claude Messages API.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic


JSONL_PATH = Path("data_augmentation/batches_claude/batch_replace_me_claude.jsonl")


def main() -> None:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing in .env")
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Batch JSONL file not found: {JSONL_PATH}")

    # Read batch requests from JSONL file
    batch_requests = []
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch_requests.append(json.loads(line))

    if not batch_requests:
        raise RuntimeError(f"No batch requests found in {JSONL_PATH}")

    client = Anthropic()

    # Submit batch directly to Claude
    batch = client.messages.batch.create(requests=batch_requests)

    print("jsonl_path:", JSONL_PATH)
    print("request_count:", len(batch_requests))
    print("batch_id:", batch.id)
    print("processing_status:", batch.processing_status)
    print("created_at:", batch.created_at)


if __name__ == "__main__":
    main()