from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import APIStatusError, InternalServerError, OpenAI

# Allow imports from project root if needed later
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import get_vector_id_by_file_name, reconstruct_vector_by_id, search_similar_with_metadata

warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed",
    category=UserWarning,
)

DEFAULT_MODEL = "gpt-5.1"
TOP_K = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "data_augmentation" / "prompt.txt"
SCHEMA_PATH = PROJECT_ROOT / "data_augmentation" / "schema.json"
FEW_SHOTS_PATH = PROJECT_ROOT / "data_augmentation" / "few_shots.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_augmentation" / "batch_results_faiss_eval"

DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "ai4h.db"
INDEX_PATH = PROJECT_ROOT / "data" / "faiss" / "embeddings.index"

QUERY_IMAGE_ROOT = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\ai4h_oral1\data\images")
CANDIDATE_IMAGE_ROOT = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\Cleaned Dataset v2")
DEFAULT_EXCEL_PATH = Path(r"C:\Users\Asus\T5 - SDS\T8-AI4H\ai4h_oral1\data\Ground Truth Values.xlsx")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def norm_name(s: str) -> str:
    return " ".join(Path(str(s)).stem.strip().lower().split())


def norm_sheet_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


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


def get_single_required_orientations() -> dict[str, list[str]]:
    return {
        norm_sheet_name("1 Acarbose tab front YSP"): ["back"],
        norm_sheet_name("2 Aciclovir Medovir Front"): ["back"],
        norm_sheet_name("3 Antacid Beacons front"): ["back"],
        norm_sheet_name("4 Folic acid sunward back"): ["front"],
        norm_sheet_name("5 Fluoxetine 10 APO bottle fron"): ["front-box"],
        norm_sheet_name("6 Clomipramine 25 front"): ["back"],
        norm_sheet_name("7 Amiodarone 200"): ["back"],
        norm_sheet_name("8 Loratadine 10 Front"): ["back"],
        norm_sheet_name("9 Thalidomide 50 back"): ["front"],
        norm_sheet_name("10 Telmisartan 80 Intas box"): ["front-box"],
        norm_sheet_name("11 Dextromethorphan ICM bottle"): ["front-box"],
        norm_sheet_name("12 Rifampicin 300 Medochemie ba"): ["front"],
        norm_sheet_name("13 Gliclazide sunward front"): ["back"],
        norm_sheet_name("14 Olanzapine 10 actavis box"): ["front-box"],
        norm_sheet_name("15 Panadeine Back"): ["front"],
        norm_sheet_name("16 Lacteol Forte sachet"): ["front"],
        norm_sheet_name("35 Telmisartan"): ["back"],
        norm_sheet_name("49 Rivaroxaban 2.5"): ["back"],
        norm_sheet_name("50 aspirin"): ["front"],
    }


def load_queries_from_excel(excel_path: Path) -> list[dict[str, Any]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    required_orientations = get_single_required_orientations()
    xl = pd.ExcelFile(excel_path)
    out: list[dict[str, Any]] = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "drug" not in df.columns or "orientation" not in df.columns:
            continue

        sheet_key = norm_sheet_name(sheet_name)
        if sheet_key not in required_orientations:
            continue

        allowed_orientations = [x.strip().lower() for x in required_orientations[sheet_key]]
        df["orientation"] = df["orientation"].astype(str).str.strip().str.lower()

        filtered_df = df[df["orientation"].isin(allowed_orientations)]

        ground_truth_candidate_names: list[str] = []
        for value in filtered_df["drug"].dropna().tolist():
            s = str(value).strip()
            if s:
                ground_truth_candidate_names.append(s)

        if not ground_truth_candidate_names:
            continue

        out.append(
            {
                "query_name": sheet_name.strip(),
                "ground_truth_candidate_names": ground_truth_candidate_names,
                "required_orientations": allowed_orientations,
            }
        )

    return out


def upload_vision_file(client: OpenAI, path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()

    if suffix == path.suffix:
        with path.open("rb") as f:
            uploaded = client.files.create(file=f, purpose="vision")
        return uploaded.id

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

    parser = argparse.ArgumentParser(
        description="FAISS top-50 L2 retrieval + teacher LLM evaluation for 19 drug queries"
    )
    parser.add_argument("--excel-path", type=str, default=str(DEFAULT_EXCEL_PATH))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder to save per-query JSON results and summary",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    excel_path = Path(args.excel_path)
    output_dir = Path(args.output_dir)

    prompt = load_prompt()
    text_config = load_schema_format()
    few_shots = load_few_shots()
    few_shot_content = build_few_shot_content(few_shots)

    query_image_index = build_image_index(QUERY_IMAGE_ROOT)
    candidate_image_index = build_image_index(CANDIDATE_IMAGE_ROOT)
    all_queries = load_queries_from_excel(excel_path)

    if not all_queries:
        raise RuntimeError("No valid queries loaded from Excel.")

    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_tp = 0
    overall_fp = 0
    overall_tn = 0
    overall_fn = 0

    summary: list[dict[str, Any]] = []

    for item in all_queries:
        query_name = item["query_name"]
        required_orientations = item["required_orientations"]
        ground_truth_names = item["ground_truth_candidate_names"]
        ground_truth_norm = {norm_name(x) for x in ground_truth_names}

        query_path = resolve_image_path(query_name, query_image_index)
        if query_path is None:
            summary.append(
                {
                    "query_name": query_name,
                    "status": "skipped",
                    "reason": "query image not found",
                }
            )
            continue

        # Use actual resolved image filename for DB / FAISS lookup
        query_lookup_name = query_path.name

        try:
            query_vector_id = get_vector_id_by_file_name(query_lookup_name, db_path=DB_PATH)
            query_vector = reconstruct_vector_by_id(query_vector_id, index_path=INDEX_PATH)

            retrieved_candidates = search_similar_with_metadata(
                query_vector=query_vector,
                similarity_type="l2",
                top_n=TOP_K,
                index_path=INDEX_PATH,
                db_path=DB_PATH,
                exclude_vector_id=query_vector_id,
            )
        except Exception as exc:
            summary.append(
                {
                    "query_name": query_name,
                    "status": "skipped",
                    "reason": f"FAISS retrieval failed: {exc}",
                }
            )
            continue

        if not retrieved_candidates:
            summary.append(
                {
                    "query_name": query_name,
                    "status": "skipped",
                    "reason": "no FAISS candidates retrieved",
                }
            )
            continue

        query_file_id = upload_vision_file(client, query_path)

        query_tp = 0
        query_fp = 0
        query_tn = 0
        query_fn = 0
        candidate_results: list[dict[str, Any]] = []

        for rank, candidate in enumerate(retrieved_candidates, start=1):
            candidate_file_name = getattr(candidate, "file_name", None)
            candidate_vector_id = getattr(candidate, "vector_id", None)
            candidate_score = getattr(candidate, "score", None)

            if not candidate_file_name:
                candidate_results.append(
                    {
                        "rank": rank,
                        "error": True,
                        "reason": "candidate missing file_name",
                    }
                )
                continue

            candidate_path = resolve_image_path(candidate_file_name, candidate_image_index)
            if candidate_path is None:
                candidate_results.append(
                    {
                        "rank": rank,
                        "candidate_file_name": candidate_file_name,
                        "candidate_vector_id": candidate_vector_id,
                        "candidate_score_l2": candidate_score,
                        "error": True,
                        "reason": "candidate image path not found",
                    }
                )
                continue

            actual_positive = norm_name(candidate_file_name) in ground_truth_norm

            try:
                candidate_file_id = upload_vision_file(client, candidate_path)

                request_input = build_request_input(
                    prompt=prompt,
                    few_shot_content=few_shot_content,
                    query_name=query_name,
                    query_file_id=query_file_id,
                    candidate_name=candidate_file_name,
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
                predicted_positive = bool(parsed_output.get("lookalike", False))

                if predicted_positive and actual_positive:
                    query_tp += 1
                elif predicted_positive and not actual_positive:
                    query_fp += 1
                elif not predicted_positive and actual_positive:
                    query_fn += 1
                else:
                    query_tn += 1

                candidate_results.append(
                    {
                        "rank": rank,
                        "candidate_file_name": candidate_file_name,
                        "candidate_path": str(candidate_path),
                        "candidate_vector_id": candidate_vector_id,
                        "candidate_score_l2": candidate_score,
                        "actual_positive": actual_positive,
                        "predicted_positive": predicted_positive,
                        "output_text": output_text,
                        "parsed_output": parsed_output,
                    }
                )

            except Exception as exc:
                candidate_results.append(
                    {
                        "rank": rank,
                        "candidate_file_name": candidate_file_name,
                        "candidate_path": str(candidate_path),
                        "candidate_vector_id": candidate_vector_id,
                        "candidate_score_l2": candidate_score,
                        "actual_positive": actual_positive,
                        "error": True,
                        "reason": str(exc),
                    }
                )

        metrics = compute_metrics(query_tp, query_fp, query_tn, query_fn)

        result = {
            "query_name": query_name,
            "query_path": str(query_path),
            "query_lookup_name": query_lookup_name,
            "required_orientations": required_orientations,
            "ground_truth_candidate_names": ground_truth_names,
            "faiss_top_k": TOP_K,
            "similarity_type": "l2",
            "few_shots_count": len(few_shots),
            "model": args.model,
            "tp": query_tp,
            "fp": query_fp,
            "tn": query_tn,
            "fn": query_fn,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "candidate_results": candidate_results,
        }

        overall_tp += query_tp
        overall_fp += query_fp
        overall_tn += query_tn
        overall_fn += query_fn

        safe_name = re.sub(r"[\\/]", "_", query_name)
        out_path = output_dir / f"{safe_name}_output.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved:", out_path)

        summary.append(
            {
                "query_name": query_name,
                "status": "ok",
                "required_orientations": required_orientations,
                "output_file": str(out_path),
                "tp": query_tp,
                "fp": query_fp,
                "tn": query_tn,
                "fn": query_fn,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )

    overall_metrics = compute_metrics(overall_tp, overall_fp, overall_tn, overall_fn)

    overall_summary = {
        "faiss_top_k": TOP_K,
        "similarity_type": "l2",
        "total_queries_run": len(summary),
        "overall_tp": overall_tp,
        "overall_fp": overall_fp,
        "overall_tn": overall_tn,
        "overall_fn": overall_fn,
        "overall_precision": overall_metrics["precision"],
        "overall_recall": overall_metrics["recall"],
        "overall_f1": overall_metrics["f1"],
        "overall_accuracy": overall_metrics["accuracy"],
        "queries": summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()