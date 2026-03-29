# Please use Linux (WSL)

# Lookalike medication identification (AI for Healthcare)

Identify and distinguish lookalike medications from images. Uses **ViT** for image embeddings and **model_v3** (Assessor MLP) for lookalike scoring, trained on ground-truth labels with both lookalike and non-lookalike examples (class-balanced loss).

---

## Improvements over previous implementation

- **No online RL:** Model is trained from **curated annotation files** (batch supervised learning). Closing the app does not lose data; the model is not updated live from user clicks, so abuse or mistakes do not corrupt the model until you explicitly add them to annotations and retrain.
- **Persistent model:** Weights are saved to disk (`assessor/model_v3/assessor.pt`). Restarting the API loads the same model; no in-memory-only state.
- **Simplified pipeline:** One query → **retrieve top-k by cosine** (not the full dataset) → run the **assessor only on those k** candidates. More efficient and easier to reason about.
- **Better data handling:** Annotation formats supported (streamlined sheet-per-query, reformat script from lookalike_annotation). Training reports **label_counts** and supports **class-balanced loss** to handle imbalance.
- **Evaluation:** We use **precision** (and recall, F1) as primary metrics; prof is fine with this. Eval scripts and training logs report them.
- **Human feedback path:** Corrections are added to annotation files and retrained in batch (export → review → retrain). Optional later: AI-generated labels to augment small human feedback; if that doesn’t work, we rely on human feedback only.

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

**3. Open** [http://localhost:5173](http://localhost:5173). Upload a query image and click **Scan Database**. Results show lookalikes (score ≥ 0.5) and non-lookalikes (score < 0.5), **with thumbnail images** for each candidate when source images are available (see “Lookalike images in the UI” below).

Verify the backend is using v3: `curl http://localhost:8000/config` → `"assessor_model_dir": "assessor/model_v3"`.

---

## Lookalike images in the UI

When you upload a query image and run **Scan Database**, the Output section shows your **query image** and each candidate as a **card with thumbnail image**, name, and score. Candidate images are resolved from `input_img/` (or `INPUT_IMG_DIR`) and `data/images/` using the same stem as the embedding (e.g. `4049.pt` → `4049.jpg`). If no image is found, the card shows “No image”.

---

## Reinforcement learning (feedback)

You can improve the assessor from **human or AI feedback** without editing Excel:

1. **Collect feedback** – POST to `/feedback` with `query_image` (file), `candidate_name` (string), and `is_lookalike` (true/false). The API saves the query embedding under `data/feedback/embeddings/` and appends a line to `data/feedback/feedback.jsonl`. Data is **persistent** (no reset on close).

2. **Train from feedback** – Run:
   ```bash
   uv run python train_from_feedback.py --resume
   ```
   This loads `data/feedback/feedback.jsonl`, resolves candidate embeddings, and **fine-tunes** the assessor. The updated model is saved to `assessor/model` (or `--out-dir`). Restart the API to use the new weights. Options: `--feedback-dir`, `--epochs`, `--lr`, `--out-dir`.

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
| RL: train from feedback | Collect via POST /feedback, then `uv run python train_from_feedback.py --resume` |

---

## Production

- **Frontend:** `cd web_ui/frontend && npm run build` → serve `dist/` with any static host; set API URL (e.g. `VITE_API_URL`) to your deployed backend.
- **Backend:** Run uvicorn with `ASSESSOR_MODEL_DIR=assessor/model_v3` pointing at your trained model.

---

## Features you can add (aligned with consensus)

| Priority | Feature | Why |
|----------|---------|-----|
| **Checkoff 1** | **Working prototype** | You already have it: upload → scan → lookalike/non-lookalike lists. Add a one-page “Demo / Checkoff 1” in README or a `docs/CHECKOFF1.md` with steps to run and what to show. |
| **Checkoff 1** | **Precision as primary metric** | Already computed; make it explicit in README and in eval script output (e.g. print precision first, or add a one-line “Primary metric: precision”). |
| **High** | **Export corrections from UI** | Button “Export corrections” that downloads a CSV (query, candidate, label) for the pairs the user marked wrong. Data is saved to file; you review before adding to annotations and retraining. Avoids “close and lose” and limits abuse. |
| **High** | **Annotation validation script** | Script that checks annotation file: label balance, missing embeddings, duplicate pairs. Run before training. Helps “we add more data handling” and fail fast. |
| **Medium** | **Human feedback flow doc** | Short doc: “Human feedback: export corrections → add to data_reformatted.xlsx → retrain.” Backup plan if AI feedback is dropped. |
| **Medium** | **Primary metric in eval** | In `eval_on_test_data.py` and `compare_models.py`, print a line like “Primary metric (precision): 0.82” so it’s clear. |
| **Later** | **AI-assisted labels (optional)** | Pipeline to generate (query, candidate, label) from an LLM or teacher model for unlabeled pairs; append to annotation and retrain. Document cost/token tradeoffs; ask prof for budget if needed. |
| **Later** | **Pharmacist feedback** | If you add any “feedback” UI (e.g. thumbs up/down or “Wrong list”), same export-to-CSV flow so pharmacist input is persisted and reviewed before training. |

No **online** RL in the API (no live gradient updates). RL is done **offline**: collect feedback via POST /feedback, then run `train_from_feedback.py` to update the model; data is persisted and the model is saved to disk.
