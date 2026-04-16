"""
Helper functions for Claude integration.
Migrated from OpenAI to Anthropic Claude.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic

CLAUDE_FEW_SHOT_CACHE_PATH = Path("data_augmentation_claude/few_shots_claude.json")

def encode_image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 for Claude vision API."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """Get MIME type from image extension."""
    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff"
    }
    return media_types.get(ext, "image/jpeg")


def build_vision_content(prompt: str, image_path: Path) -> List[Dict[str, Any]]:
    """Build message content with image for Claude."""
    image_data = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        },
        {"type": "text", "text": prompt}
    ]


def clean_stem(stem: str) -> str:
    return stem.lstrip("0123456789 ").strip()

def build_few_shot_content_claude(few_shots, image_index):
    content = []
    for idx, shot in enumerate(few_shots, start=1):
        query_file_name = shot.get("Query file name", "unknown_query")
        candidate_file_name = shot.get("Candidate file name", "unknown_candidate")
        output = shot.get("Output")

        if not isinstance(output, dict):
            raise RuntimeError(f"few_shots[{idx}] is missing an object 'Output'.")

        # query_key = clean_stem(Path(query_file_name).stem)
        # candidate_key = clean_stem(Path(candidate_file_name).stem)

        # query_path = image_index.get(query_key)
        # candidate_path = image_index.get(candidate_key)

        query_path = resolve_few_shot_image_path(query_file_name, image_index)
        candidate_path = resolve_few_shot_image_path(candidate_file_name, image_index)

        if query_path is None:
            raise RuntimeError(
                f"Query image not found in index: {query_file_name} (lookup path: {query_path})"
            )
        if candidate_path is None:
            raise RuntimeError(
                f"Candidate image not found in index: {candidate_file_name} (lookup path: {candidate_path})"
            )

        content.extend([
            {"type": "text", "text": f"Example {idx}"},
            {"type": "text", "text": f"Example Query file name: {query_file_name}"},
            *build_vision_content("", query_path),
            {"type": "text", "text": f"Example Candidate file name: {candidate_file_name}"},
            *build_vision_content("", candidate_path),
            {"type": "text", "text": f"Example Output: {json.dumps(output, ensure_ascii=True)}"},
        ])
    return content


def create_claude_batch_request(
    custom_id: str,
    prompt: str,
    query_image_path: Path,
    candidate_image_path: Path,
    query_file_name: str,
    candidate_file_name: str,
    few_shot_content: List[Dict[str, Any]] | None = None,
    model: str = "claude-3-5-sonnet-20241022"
) -> Dict[str, Any]:
    """Create a batch request for Claude Messages API."""

    content = [
        {"type": "text", "text": prompt},
    ]

    if few_shot_content:
        content.extend(few_shot_content)

    content.extend([
        {"type": "text", "text": f"Query file name: {query_file_name}"},
        *build_vision_content("", query_image_path),
        {"type": "text", "text": f"Candidate file name: {candidate_file_name}"},
        *build_vision_content("", candidate_image_path),
    ])

    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": content
            }]
        }
    }


def extract_claude_response(result: Any) -> Dict[str, Any]:
    """Extract structured response from Claude batch result."""
    try:
        content = result.result.message.content[0].text
        payload = json.loads(content)
        return {
            "response_id": result.result.message.id,
            "error": bool(payload.get("error", False)),
            "lookalike": bool(payload.get("lookalike", False)),
            "identical": bool(payload.get("identical", False)),
            "reasoning": str(payload.get("reasoning", "")),
        }
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        return {
            "response_id": getattr(result, 'result', {}).get('message', {}).get('id', 'unknown'),
            "error": True,
            "lookalike": False,
            "identical": False,
            "reasoning": f"Failed to parse response: {str(e)}",
        }

def save_claude_few_shot_cache(content: List[Dict[str, Any]], cache_path: Path = CLAUDE_FEW_SHOT_CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=True, indent=2)


def load_claude_few_shot_cache(cache_path: Path = CLAUDE_FEW_SHOT_CACHE_PATH) -> List[Dict[str, Any]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Claude few-shot cache not found: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"Claude few-shot cache must contain a list: {cache_path}")
    return data


def get_or_build_claude_few_shots(
    few_shots: list[dict[str, Any]],
    image_index: dict[str, Path],
    *,
    cache_path: Path = CLAUDE_FEW_SHOT_CACHE_PATH,
    force_rebuild: bool = False,
) -> List[Dict[str, Any]]:
    if not force_rebuild and cache_path.exists():
        return load_claude_few_shot_cache(cache_path)

    content = build_few_shot_content_claude(few_shots, image_index)
    save_claude_few_shot_cache(content, cache_path)
    return content

def resolve_few_shot_image_path(file_name: str, image_index: dict[Path, Path] | dict[str, Path]) -> Path | None:
    raw_stem = Path(file_name).stem
    cleaned_stem = clean_stem(raw_stem)

    # 1. exact raw match
    path = image_index.get(raw_stem)
    if path is not None:
        return path

    # 2. exact cleaned match
    path = image_index.get(cleaned_stem)
    if path is not None:
        return path

    # 3. compare against cleaned indexed stems
    for stem, candidate_path in image_index.items():
        if clean_stem(stem) == cleaned_stem:
            return candidate_path

    return None