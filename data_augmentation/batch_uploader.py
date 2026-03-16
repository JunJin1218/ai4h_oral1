from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


JSONL_PATH = Path("data_augmentation/batches/batch_Imipramine_Accord-Front_box_Tab_10mg_E_MA_mfr_Accord_Pharmaceuticals_UK_v1_19May2021__0.jsonl")
COMPLETION_WINDOW = "24h"
ENDPOINT = "/v1/responses"


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Batch JSONL file not found: {JSONL_PATH}")

    client = OpenAI()
    with JSONL_PATH.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={
            "source_file": JSONL_PATH.name,
        },
    )

    print("jsonl_path:", JSONL_PATH)
    print("input_file_id:", uploaded.id)
    print("batch_id:", batch.id)
    print("status:", batch.status)
    print("endpoint:", batch.endpoint)
    print("completion_window:", batch.completion_window)


if __name__ == "__main__":
    main()
