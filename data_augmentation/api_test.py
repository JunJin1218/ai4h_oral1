from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIStatusError, InternalServerError, OpenAI


DEFAULT_MODEL = "gpt-5.1"


def upload_vision_file(client: OpenAI, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    return uploaded.id


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Minimal OpenAI image comparison test")
    parser.add_argument("--query", required=True, help="Path to the query image")
    parser.add_argument("--candidate", required=True, help="Path to the candidate image")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=2, help="Retry count for 5xx errors")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    query_path = Path(args.query)
    candidate_path = Path(args.candidate)

    client = OpenAI()
    query_file_id = upload_vision_file(client, query_path)
    candidate_file_id = upload_vision_file(client, candidate_path)

    request_input: Any = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Compare these two medication package images. "
                        "Keep the answer short. "
                        "Say whether they look alike and give a brief reason."
                    ),
                },
                {"type": "input_text", "text": "Query image"},
                {"type": "input_image", "file_id": query_file_id},
                {"type": "input_text", "text": "Candidate image"},
                {"type": "input_image", "file_id": candidate_file_id},
            ],
        }
    ]

    last_error: Exception | None = None
    response = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.responses.create(
                model=args.model,
                input=request_input,
            )
            break
        except InternalServerError as exc:
            last_error = exc
            print(f"[attempt {attempt + 1}] OpenAI 5xx error")
            print("status:", exc.status_code)
            print("request_id:", exc.request_id)
            print("body:", exc.body)
            if attempt >= args.max_retries:
                raise
            time.sleep(min(2 ** attempt, 4))
        except APIStatusError as exc:
            print("OpenAI API status error")
            print("status:", exc.status_code)
            print("request_id:", exc.request_id)
            print("body:", exc.body)
            raise

    if response is None:
        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

    print("model:", response.model)
    print("query:", query_path)
    print("query_file_id:", query_file_id)
    print("candidate:", candidate_path)
    print("candidate_file_id:", candidate_file_id)
    print()
    print(response.output_text)


if __name__ == "__main__":
    main()
