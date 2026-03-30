from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import re
import tempfile
import shutil

import pandas as pd
from dotenv import load_dotenv
from openai import APIStatusError, InternalServerError, OpenAI


# Allow imports from project root if needed later
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
)



DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_CANDIDATES = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "data_augmentation" / "prompt.txt"
SCHEMA_PATH = PROJECT_ROOT / "data_augmentation" / "schema.json"
FEW_SHOTS_PATH = PROJECT_ROOT / "data_augmentation" / "few_shots.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data_augmentation" / "batch_results"
QUERY_IMAGE_ROOT = PROJECT_ROOT / "data" / "images"
CANDIDATE_IMAGE_ROOT = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\Cleaned Dataset v2")

# Change this if your Excel file is elsewhere
DEFAULT_EXCEL_PATH = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\ai4h_oral1\data\Ground Truth Values.xlsx")

# Your local image folder
IMAGE_ROOT = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\Cleaned Dataset v2")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def norm_name(s: str) -> str:
    return " ".join(Path(str(s)).stem.strip().lower().split())


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


def build_few_shot_content(few_shots: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def build_image_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")

    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            key = norm_name(path.stem)
            index.setdefault(key, path)
    return index


def resolve_image_path(name: str, image_index: dict[str, Path]) -> Path | None:
    return image_index.get(norm_name(name))


def load_queries_from_excel(excel_path: Path) -> list[dict[str, Any]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Required orientation for each sheet
    required_orientations = {
        "1 Acarbose tab front YSP": "back",
        "2 Aciclovir Medovir Front": "back",
        "3 Antacid Beacons front": "back",
        "4 Folic acid sunward back": "front",
        "5 Fluoxetine 10 APO bottle fron": "front-box",
        "6 Clomipramine 25 front": "back",
        "7 Amiodarone 200": "back",
        "8 Loratadine 10 Front": "back",
        "9 Thalidomide 50 back": "front",
        "10 Telmisartan 80 Intas box": "front-box",
        "11 Dextromethorphan ICM bottle": "front-box",
        "12 Rifampicin 300 Medochemie ba": "front",
        "13 Gliclazide sunward front": "back",
        "14 Olanzapine 10 actavis box": "front-box",
        "15 Panadeine Back": "front",
        "16 Lacteol Forte sachet": "front",
        "35 Telmisartan": "back",
        "49 Rivaroxaban 2.5": "back",
        "50 aspirin": "front",
    }

    xl = pd.ExcelFile(excel_path)
    out: list[dict[str, Any]] = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "drug" not in df.columns:
            continue

        if sheet_name not in required_orientations:
            continue

        required_orientation = required_orientations[sheet_name].strip().lower()

        if "orientation" not in df.columns:
            continue

        df["orientation"] = df["orientation"].astype(str).str.strip().str.lower()

        filtered_df = df[df["orientation"] == required_orientation]

        candidate_names = []
        for value in filtered_df["drug"].dropna().tolist():
            s = str(value).strip()
            if s:
                candidate_names.append(s)

        if not candidate_names:
            continue

        out.append(
            {
                "query_name": sheet_name.strip(),
                "candidate_names": candidate_names,
                "required_orientation": required_orientation,
            }
        )

    return out

def upload_vision_file(client: OpenAI, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()

    # If extension is already lowercase, upload normally
    if suffix == path.suffix:
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="vision")
        return uploaded.id

    # Otherwise, create a temp file with lowercase extension
    temp_path = Path(tempfile.gettempdir()) / (path.stem + suffix)

    shutil.copyfile(path, temp_path)

    try:
        with temp_path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="vision")
        return uploaded.id
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_request_input(
    prompt: str,
    few_shot_content: list[dict[str, str]],
    query_name: str,
    query_file_id: str,
    candidate_name: str,
    candidate_file_id: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        *few_shot_content,
        {"type": "input_text", "text": f"Query file name: {query_name}"},
        {"type": "input_image", "file_id": query_file_id},
        {"type": "input_text", "text": f"Candidate file name: {candidate_name}"},
        {"type": "input_image", "file_id": candidate_file_id},
        {
            "type": "input_text",
            "text": (
                "Evaluate whether this candidate image is a lookalike of the query image. "
                "Return a JSON object for this candidate only with keys: "
                "error, reasoning, lookalike."
            ),
        },
    ]

    return [{"role": "user", "content": content}]


def call_openai_with_retries(
    client: OpenAI,
    model: str,
    request_input: list[dict[str, Any]],
    text_config: dict[str, Any],
    max_retries: int,
):
    last_error: Exception | None = None
    response = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=request_input,
                text=text_config,
            )
            return response
        except InternalServerError as exc:
            last_error = exc
            print(f"[attempt {attempt + 1}] OpenAI 5xx error")
            print("status:", exc.status_code)
            print("request_id:", exc.request_id)
            print("body:", exc.body)
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 4))
        except APIStatusError as exc:
            print("OpenAI API status error")
            print("status:", exc.status_code)
            print("request_id:", exc.request_id)
            print("body:", exc.body)
            raise

    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def parse_json_output(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_output": text}


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="LLM image comparison test from Excel sheets")
    parser.add_argument("--excel-path", type=str, default=str(DEFAULT_EXCEL_PATH))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--sheet-name", type=str, default=None, help="Run only one sheet/query")
    parser.add_argument("--limit-queries", type=int, default=0, help="Run only first N queries, 0 = all")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    excel_path = Path(args.excel_path)

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    few_shot_content = build_few_shot_content(few_shots)
    query_image_index = build_image_index(QUERY_IMAGE_ROOT)
    candidate_image_index = build_image_index(CANDIDATE_IMAGE_ROOT)
    all_queries = load_queries_from_excel(excel_path)
    print("Loaded queries:", len(all_queries))

    if args.sheet_name:
        all_queries = [q for q in all_queries if q["query_name"] == args.sheet_name]
        if not all_queries:
            raise RuntimeError(f"Sheet not found or invalid: {args.sheet_name}")

    if args.limit_queries > 0:
        all_queries = all_queries[: args.limit_queries]

    if not all_queries:
        raise RuntimeError("No valid queries loaded from Excel.")

    client = OpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    total_candidate_comparisons = 0
    total_lookalikes = 0

    for q_idx, item in enumerate(all_queries, start=1):
        query_name = item["query_name"]
        candidate_names = item["candidate_names"] if args.max_candidates <= 0 else item["candidate_names"][: args.max_candidates]

        print(f"\n=== Query {q_idx}/{len(all_queries)} ===")
        print("candidate_count:", len(candidate_names))

        query_path = resolve_image_path(query_name, query_image_index)
        if query_path is None:
            print(f"[skip] query image not found for sheet name: {query_name}")
            summary.append(
                {
                    "query_name": query_name,
                    "status": "skipped",
                    "reason": "query image not found",
                }
            )
            continue

        resolved_candidates: list[tuple[str, Path]] = []
        missing_candidates: list[str] = []

        for name in candidate_names:
            path = resolve_image_path(name, candidate_image_index)
            if path is None:
                missing_candidates.append(name)
            else:
                resolved_candidates.append((name, path))

        if not resolved_candidates:
            print(f"[skip] no candidate images found for query: {query_name}")
            summary.append(
                {
                    "query_name": query_name,
                    "status": "skipped",
                    "reason": "no candidate images found",
                }
            )
            continue

        query_file_id = upload_vision_file(client, query_path)

        candidate_results: list[dict[str, Any]] = []
        query_lookalike_count = 0

        for candidate_name, candidate_path in resolved_candidates:
            candidate_file_id = upload_vision_file(client, candidate_path)

            request_input = build_request_input(
                prompt=prompt,
                few_shot_content=few_shot_content,
                query_name=query_name,
                query_file_id=query_file_id,
                candidate_name=candidate_name,
                candidate_file_id=candidate_file_id,
            )

            response = call_openai_with_retries(
                client=client,
                model=args.model,
                request_input=request_input,
                text_config=text_config,
                max_retries=args.max_retries,
            )

            output_text = response.output_text
            parsed_output = parse_json_output(output_text)

            is_lookalike = bool(parsed_output.get("lookalike", False))
            if is_lookalike:
                query_lookalike_count += 1

            candidate_results.append(
                {
                    "candidate_name": candidate_name,
                    "candidate_path": str(candidate_path),
                    "output_text": output_text,
                    "parsed_output": parsed_output,
                }
            )

        result = {
            "query_name": query_name,
            "query_path": str(query_path),
            "missing_candidates": missing_candidates,
            "few_shots_count": len(few_shots),
            "model": args.model,
            "query_total_candidates": len(resolved_candidates),
            "query_lookalike_count": query_lookalike_count,
            "candidate_results": candidate_results,
        
}
        total_candidate_comparisons += len(resolved_candidates)
        total_lookalikes += query_lookalike_count
        safe_name = re.sub(r"[\\/]", "_", query_name)
        out_path = OUTPUT_DIR / f"{safe_name}_output.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        print("saved:", out_path)


        summary.append(
            {
                "query_name": query_name,
                "status": "ok",
                "output_file": str(out_path),
                "resolved_candidates": len(resolved_candidates),
                "missing_candidates": len(missing_candidates),
                "query_lookalike_count": query_lookalike_count,
            }
        )

    overall_summary = {
    "total_queries_run": len(summary),
    "total_candidate_comparisons": total_candidate_comparisons,
    "total_lookalikes": total_lookalikes,
    "lookalike_rate": (
        total_lookalikes / total_candidate_comparisons
        if total_candidate_comparisons > 0 else 0
    ),
    "queries": summary,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()