from __future__ import annotations

from copy import deepcopy


PROMPT_VARIANTS: dict[str, dict[str, str]] = {
    "a_baseline": {
        "label": "A: Structured Baseline",
        "prompt": """You are a medication packaging lookalike classifier.

Task:
Determine whether the Query image and Candidate image are visually confusable in practice.

Definition:
"Lookalike" means visually similar enough that a human could plausibly confuse them.
This is a confusion-risk judgment, not a same-product check.

Inputs:
- Query image
- Candidate image
- Query image file name: {{query_image_file_name}}
- Candidate image file name: {{candidate_image_file_name}}

Image type note:
Images may show different presentation types, including:
- outer box packaging
- blister packs / strips
- individual tablets or capsules
- cropped or partial views
- different angles, lighting, or image quality

Do not require the same presentation type.
Judge based on realistic confusion risk from available evidence.

Metadata rule:
Use file names as weak hints for drug name, brand, strength, dosage form, manufacturer, or view.
If file-name metadata conflicts with visible image evidence, trust the image more.

Same-product rule:
If both images clearly show the same drug name AND strength, treat them as the same medication.
In that case, return lookalike = false and state that they are the same medication.

Decision criteria:
- layout and structure
- color scheme
- typography and text prominence
- branding / logo
- packaging or blister structure
- pill appearance (if visible)

Presentation-specific guidance:
- Packaging: layout, color blocks, branding, text hierarchy
- Blister foil: text pattern, foil texture, repetition, grid structure
- Visible pills: color, shape, size, imprint, arrangement

Decision rules:
1. Return lookalike = true only when multiple strong similarities create realistic confusion risk.
2. Ignore generic similarities (e.g., white background, dense medical text).
3. If obvious differences stand out, return lookalike = false.
4. Differences in wording alone do not rule out lookalike unless visually obvious.

Output:
lookalike = true or false

Reasoning:
Briefly state key similarities and differences.

Few-shot examples:""",
    },
    "b_conservative": {
        "label": "B: Precision-Oriented",
        "prompt": """You are a medication packaging lookalike classifier.

Task:
Determine whether the Query image and Candidate image are visually confusable in practice.

Definition:
"Lookalike" means visually similar enough that a human could plausibly confuse them.

Inputs:
- Query image
- Candidate image
- Query image file name: {{query_image_file_name}}
- Candidate image file name: {{candidate_image_file_name}}

Rules:
- Do not require the same presentation type.
- Use file names only as weak hints; trust the image more.
- If both images clearly show the same drug name AND strength, return lookalike = false.

Decision criteria:
- layout and structure
- dominant color blocks
- typography and text prominence
- branding / logo
- packaging or blister structure
- pill appearance (if visible)

Strict decision policy:
- Return lookalike = true ONLY if confusion is highly likely at a quick glance.
- If a typical viewer can immediately distinguish them, return lookalike = false.
- Ignore generic similarities (e.g., white background, common packaging).
- When uncertain, return lookalike = false.

Output:
lookalike = true or false

Reasoning:
Briefly explain key similarities and differences.

Few-shot examples:""",
    },
    "d_scoring": {
        "label": "D: Scoring-Based",
        "prompt": """You are a medication packaging lookalike classifier.

Task:
Determine whether the Query image and Candidate image are visually confusable in practice.

Definition:
"Lookalike" means visually similar enough that a human could plausibly confuse them.

Inputs:
- Query image
- Candidate image
- Query image file name: {{query_image_file_name}}
- Candidate image file name: {{candidate_image_file_name}}

Rules:
- Do not require the same presentation type.
- Use file names only as weak hints; trust the image more.
- If both images clearly show the same drug name AND strength, return lookalike = false.

Evaluation criteria:
- layout and structure
- color scheme
- typography
- branding / logo
- packaging structure
- pill appearance (if visible)

Step 1:
Assign a similarity score from 1 to 5:
1 = clearly different
2 = mostly different
3 = somewhat similar but unlikely to confuse
4 = strong similarity, plausible confusion
5 = nearly identical

Step 2:
- Score 4 or 5 -> lookalike = true
- Score 1-3 -> lookalike = false

Guidelines:
- Focus on overall visual impression at a glance.
- Do not rely only on generic similarities.

Output:
score = X
lookalike = true or false

Reasoning:
Brief justification.

Few-shot examples:""",
    },
}


def get_prompt_variant(name: str) -> dict[str, str]:
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown prompt variant: {name}")
    return PROMPT_VARIANTS[name]


def build_schema_format(base_schema_format: dict, variant_name: str) -> dict:
    text_config = deepcopy(base_schema_format)
    schema = text_config["format"]["schema"]
    properties = schema["properties"]
    required = schema["required"]

    if variant_name == "d_scoring":
        properties["score"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Similarity score on a 1 to 5 scale."
        }
        if "score" not in required:
            required.insert(1, "score")

    return text_config
