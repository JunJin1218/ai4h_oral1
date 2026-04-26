# Please use Linux (WSL)

# Lookalike medication identification (AI for Healthcare)

Identify and distinguish lookalike medications from images. Uses **ViT** for image embeddings and **model_v3** (Assessor MLP) for lookalike scoring, trained on ground-truth labels with both lookalike and non-lookalike examples (class-balanced loss).

> **Note on repository branches:** The main branch contains the stable workflow documented in this README. Other branches are primarily used for ongoing testing and experimentation, especially around LLM-based evaluation and annotation generation. For the standard lookalike medication identification workflow, please refer to the main branch unless a specific experimental branch is required.
---

## Setup

```bash
uv sync --extra cu124   # optional, for GPU
```

---

## Pipeline

- **Cosine similarity** = retrieval only (top-k candidates per query). Not the lookalike decision.
- **Assessor (model_v3)** = decision model. Outputs score in [0, 1]; threshold 0.5 → “lookalike” / “non-lookalike”.

The API embeds the query, retrieves top-k by cosine, then runs the assessor on those pairs.

---

## Model_v3

| Item | Value |
|------|--------|
| **Location** | `assessor/model_v3/` |
| **Checkpoint** | `assessor/model_v3/assessor.pt` |
| **Config** | `assessor/model_v3/config.json` |

**Architecture:** Two 1280-d embeddings → L2-normalize, concat `[ |e1−e2|, e1⊙e2 ]` → MLP 2560→1024→256→1 (ReLU, dropout 0.2) → sigmoid.

---

## Annotations and reformatting

Use the **streamlined format**: one sheet per query, columns **"Lookalike"** and **"Non lookalike"**, candidate names in rows.

Convert `lookalike_annotation.xlsx` to this format:

```bash
uv run python scripts/reformat_lookalike_to_testdata.py
```

Output: `data/Annotation/data_reformatted.xlsx`. Check that names match your embeddings:

```bash
uv run python scripts/show_annotation_pairs.py --annotation-file data/Annotation/data_reformatted.xlsx
```

---

## Train model_v3

From project root:

```bash
uv run python scripts/train_model_v3.py --annotation-file data/Annotation/data_reformatted.xlsx
```

Optional: `--epochs 25`, `--annotation-file data/Annotation/test-data.xlsx`. Saves to `assessor/model_v3/`.

---

## Run the API and UI

**1. Start the API** (from project root):

**PowerShell:**
```powershell
$env:ASSESSOR_MODEL_DIR="assessor/model_v3"
uv run uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8000
```

**Linux/macOS:**
```bash
ASSESSOR_MODEL_DIR=assessor/model_v3 uv run uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8000
```

**2. Start the frontend** (new terminal):

```bash
cd web_ui/frontend
npm install
npm run dev
```

**3. Open** [http://localhost:5173](http://localhost:5173). Upload a query image and click **Scan Database**. Results show lookalikes (score ≥ 0.5) and non-lookalikes (score < 0.5).

Verify the backend is using v3: `curl http://localhost:8000/config` → `"assessor_model_dir": "assessor/model_v3"`.

---

## Evaluate and compare models

**Evaluate model_v3 on test data:**
```bash
uv run python scripts/eval_on_test_data.py --model-dir assessor/model_v3 --test-data data/Annotation/data_reformatted.xlsx
```

**Compare two models:**
```bash
uv run python scripts/compare_models.py --model-a assessor/model --model-b assessor/model_v3
```

---

## Fixing wrong classifications

When the UI shows a medication in the wrong list:

1. Add the corrected (query, candidate, label) to `data/Annotation/data_reformatted.xlsx` (or test-data.xlsx) with the right Lookalike / Non lookalike labels.
2. Retrain: `uv run python scripts/train_model_v3.py --annotation-file data/Annotation/data_reformatted.xlsx`
3. Restart the API and re-check.

If the model predicts almost everything as lookalike, check the training log for **label_counts** and ensure you have enough non-lookalikes (label 0); `train_model_v3.py` already uses `--balance-weights`.

---

## Summary

| Goal | Command |
|------|---------|
| Reformat lookalike_annotation → streamlined | `uv run python scripts/reformat_lookalike_to_testdata.py` |
| Train model_v3 | `uv run python scripts/train_model_v3.py --annotation-file data/Annotation/data_reformatted.xlsx` |
| Run API with v3 | `ASSESSOR_MODEL_DIR=assessor/model_v3 uv run uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8000` |
| Run frontend | `cd web_ui/frontend && npm install && npm run dev` → http://localhost:5173 |
| Eval v3 | `uv run python scripts/eval_on_test_data.py --model-dir assessor/model_v3 --test-data data/Annotation/data_reformatted.xlsx` |
| Fix wrong predictions | Add corrections to annotation, then retrain v3 |

---

## Production

- **Frontend:** `cd web_ui/frontend && npm run build` → serve `dist/` with any static host; set API URL (e.g. `VITE_API_URL`) to your deployed backend.
- **Backend:** Run uvicorn with `ASSESSOR_MODEL_DIR=assessor/model_v3` pointing at your trained model.
