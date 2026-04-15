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

        query_key = clean_stem(Path(query_file_name).stem)
        candidate_key = clean_stem(Path(candidate_file_name).stem)

        query_path = image_index.get(query_key)
        candidate_path = image_index.get(candidate_key)

        if query_path is None:
            raise RuntimeError(
                f"Query image not found in index: {query_file_name} (lookup key: {query_key})"
            )
        if candidate_path is None:
            raise RuntimeError(
                f"Candidate image not found in index: {candidate_file_name} (lookup key: {candidate_key})"
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