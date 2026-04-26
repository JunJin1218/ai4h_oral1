
# README: Teacher LLM Claude Evaluation Scripts

This README documents the two evaluation scripts in `data_augmentation_claude/`:

- `evaluate_teacher_llm_claude_individual.py`
- `evaluate_teacher_llm_claude_batch_api.py`

Both scripts are used to evaluate a teacher LLM on medication packaging lookalike detection. They share the same core evaluation idea:

1. load query images and ground-truth labels
2. retrieve top-k candidate images using embeddings
3. create one request per query-candidate pair
4. collect model outputs
5. compare the model's `lookalike` prediction against the ground-truth label
6. compute accuracy, precision, recall, and F1

---

## 1. Script summary

### `evaluate_teacher_llm_claude_individual.py`
This script runs requests **individually** and stores results locally. This also the script used to generate evaluation scores for our teacher LLM tests.

Main characteristics:
- builds request JSONL and manifest JSONL
- can execute requests directly through `client.messages.create(...)`
- supports resuming from saved request JSONL
- stores results in local JSONL files
- stores hard failures in a separate error JSONL
- supports report options to exclude:
  - `error=true` outputs
  - missing outputs

Main commands:
- `submit`
- `submit-existing`
- `poll`
- `report`

This script is suited for local, resumable execution where requests are processed one by one.

### `evaluate_teacher_llm_claude_batch_api.py`
This script uses the **batch API workflow** and will **not** work with OpenRouter API keys.

Main characteristics:
- builds request JSONL and manifest JSONL
- splits requests into chunks
- submits chunks as remote batches
- polls remote batch status
- downloads batch results when complete
- computes metrics from completed batch output

Main commands:
- `submit`
- `submit-existing`
- `poll`
- `report`

This script is suited for workflows where remote batch submission and polling are available.

---

## 2. Shared inputs and required files

Both scripts expect the following project assets to exist.

### Ground truth and query images
- `test_query_img/gt_table_csv.csv`
- `test_query_img/`

### Embedding and retrieval files
- `data/sqlite/ai4h.db`
- `data/faiss/embeddings.index`

### Prompt and few-shot files
These are loaded through helper functions in `data_augmentation_claude/`.
Typical files include:
- prompt text file
- few-shot JSON / JSONL file
- Claude helper utilities
- batch generator utilities

### Output directories
Both scripts write under:
- `assessor/teacher_eval_claude/batches/`
- `assessor/teacher_eval_claude/manifests/`
- `assessor/teacher_eval_claude/results/`
- `assessor/teacher_eval_claude/active_batches.jsonl`
- `assessor/teacher_eval_claude/done_batches.jsonl`
- `assessor/teacher_eval_claude/evaluate_teacher_llm_results_claude.csv`
- `assessor/teacher_eval_claude/evaluate_teacher_llm_missing_claude.csv`

---

## 3. Python dependencies

The scripts import the following packages directly:

- `python-dotenv`
- `scikit-learn`
- `tqdm`
- `anthropic`

They also depend on project modules such as:
- `assessor.evaluate`
- `data_augmentation_claude.batch_generator_claude`
- `data_augmentation_claude.claude_helpers`
- `image_embedding.vit`
- `utils`

You will also need the libraries required by those imported project modules, which may include model and embedding dependencies used elsewhere in the repository.

To install the necessary packages for the teacher evaluations scripts use:

```bash
uv add python-dotenv scikit-learn tqdm anthropic
```

Make sure to `uv sync` before running either scripts.

---

## 4. Environment variables

Both scripts require:

- `ANTHROPIC_API_KEY` (for our purposes, we are using an OpenRouter key instead of an Anthropic key)

The individual script also reads optional environment variables:
- `ANTHROPIC_MODEL`
- `ANTHROPIC_BASE_URL`

Defaults in `evaluate_teacher_llm_claude_individual.py`:
- model: `anthropic/claude-3-7-sonnet-20250219`
- base URL: `https://openrouter.ai/api`

The batch API script also reads `ANTHROPIC_MODEL` as a default model value. However, the current OpenRouter key setup is not supported for the batch API script. Hence, to use it, change the OpenRouter key to an Anthropic key.

---

## 5. How request generation works

In both scripts, `submit` builds the evaluation request set from scratch.

High-level flow:
- load prompt
- load few-shots
- build image index
- load ground truth
- embed each query image with ViT
- retrieve top-k similar candidates
- create a manifest row for each query-candidate pair
- create one Claude request per submitted pair
- save:
  - request JSONL
  - manifest JSONL

Each executable request contains:
- `custom_id`
- `params.model`
- `params.max_tokens`
- `params.messages`

Each request corresponds to **one query image + one candidate image**.

---

## 6. Manifest and result matching

Both scripts rely on the manifest to provide the ground-truth label for each pair.

Each submitted pair has a `custom_id`, typically shaped like:

```text
<run_name>__query_<query_id>__vec_<vector_id>
```

During reporting:
- the manifest provides `true_label`
- the result JSONL provides the model output for the same `custom_id`
- the predicted label is taken from `lookalike`
- metrics are computed by comparing predicted labels against `true_label`

---

## 7. Commands for `evaluate_teacher_llm_claude_individual.py`

Use these commands if you want the individual/local execution workflow.

### Generate requests and immediately execute locally
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit
```

### Generate requests only, without executing
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit --dry-run
```

### Generate with an explicit run name
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit \
  --run-name teacher_eval_claude_20260416T085614Z
```

### Resume or execute a previously saved request JSONL
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit-existing \
  --run-name teacher_eval_claude_20260416T085614Z
```

### Process only the first N pending requests
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit-existing \
  --run-name teacher_eval_claude_20260416T085614Z \
  --limit 10
```

### Run without few-shots
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit \
  --no-few-shots
```

### Rebuild cached few-shots
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit \
  --rebuild-few-shots
```

### Continue even if few-shot image resolution fails
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit \
  --skip-missing-few-shots
```

### Change top-k retrieval size
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit \
  --top-k 5
```

### Change retry settings
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit-existing \
  --run-name teacher_eval_claude_20260416T085614Z \
  --max-retries 4 \
  --retry-base-seconds 3
```

### Show locally tracked active runs
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py poll
```

### Run report normally
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py report \
  --batch-id teacher_eval_claude_20260416T085614Z
```

### Run report excluding `error=true` outputs
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py report \
  --batch-id teacher_eval_claude_20260416T085614Z \
  --exclude-error-outputs
```

### Run report excluding `error=true` outputs and missing outputs
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py report \
  --batch-id teacher_eval_claude_20260416T085614Z \
  --exclude-error-outputs \
  --exclude-missing-outputs
```

### Important note on `batch-id` in the individual script
In the individual script, `batch_id` is effectively the same as the run name. It is used to look up the run in the local done log.

---

## 8. Commands for `evaluate_teacher_llm_claude_batch_api.py`

Use these commands if you want the remote batch workflow.

### Generate requests and submit remote batches
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit
```

### Generate requests only, without remote submission
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit --dry-run
```

### Generate with explicit run name
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit \
  --run-name teacher_eval_claude_20260416T085614Z
```

### Submit a previously generated JSONL without rebuilding
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit-existing \
  --run-name teacher_eval_claude_20260416T085614Z
```

### Change batch chunk size
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit-existing \
  --run-name teacher_eval_claude_20260416T085614Z \
  --chunk-size 50
```

### Poll active remote batches once
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py poll
```

### Poll continuously
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py poll --watch
```

### Poll with custom interval
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py poll \
  --watch \
  --poll-interval 30
```

### Run report on a completed batch
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py report \
  --batch-id <completed_batch_id>
```

---

## 9. Output files

### Individual script
Typical outputs:
- `assessor/teacher_eval_claude/batches/<run_name>.jsonl`
- `assessor/teacher_eval_claude/manifests/<run_name>.jsonl`
- `assessor/teacher_eval_claude/results/<run_name>_results.jsonl`
- `assessor/teacher_eval_claude/results/<run_name>_errors.jsonl`

It also updates:
- `active_batches.jsonl`
- `done_batches.jsonl`
- `evaluate_teacher_llm_results_claude.csv`
- `evaluate_teacher_llm_missing_claude.csv`

### Batch API script
Typical outputs:
- `assessor/teacher_eval_claude/batches/<run_name>.jsonl`
- `assessor/teacher_eval_claude/manifests/<run_name>.jsonl`
- `assessor/teacher_eval_claude/results/<batch_id>_results.jsonl`

It also updates:
- `active_batches.jsonl`
- `done_batches.jsonl`
- `evaluate_teacher_llm_results_claude.csv`
- `evaluate_teacher_llm_missing_claude.csv`

---

## 10. Meaning of the report outputs

The report compares:
- ground truth from the manifest (`true_label`)
- predicted `lookalike` from the result JSONL

Metrics:
- **Accuracy**: overall fraction of correct predictions
- **Precision**: among predicted lookalikes, how many were truly lookalikes
- **Recall**: among true lookalikes, how many were found
- **F1**: harmonic mean of precision and recall

Other counts:
- **Submitted pairs**: pairs that were actually executable and sent
- **Response errors**: rows where a response existed but was unusable or marked as error
- **Unresolved positive rows**: ground-truth positive pairs that were not submitted at all

### Missing pairs CSV
`evaluate_teacher_llm_missing_claude.csv` stores unresolved **positive** pairs that were not submitted successfully. It is mainly a coverage/debugging file.

---

## 11. Continue/Resume script behavior

### Individual script
Resume behavior is based on the contents of:
- `<run_name>_results.jsonl`
- `<run_name>_errors.jsonl`

If a `custom_id` already exists in either file, it is treated as completed and will be skipped on rerun.

This means:
- deleting or renaming the error JSONL causes failed hard-error requests to be retried
- removing `error=true` rows from the results JSONL causes those soft-failure requests to be retried

### Batch API script
Resume behavior depends on the remote batch submission and done/active logs, rather than local per-request completion.

---

## 12. Typical workflows

### Workflow A: Individual/local execution
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit --dry-run
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py submit-existing --run-name <run_name>
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_individual.py report --batch-id <run_name> --exclude-error-outputs
```

### Workflow B: Batch API execution
```bash
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit --dry-run
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py submit-existing --run-name <run_name>
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py poll --watch
uv run python data_augmentation_claude/evaluate_teacher_llm_claude_batch_api.py report --batch-id <batch_id>
```

---

## 13. Notes

- `submit` does not require `--run-name`; one will be generated automatically if omitted.
- `submit-existing` requires `--run-name` because it needs to load existing request and manifest files.
- In the individual script, the request JSONL already contains the prompt and image payloads, so `submit-existing` does not need to rebuild them.
- In the individual script, `chunk-size` is used as a local progress logging interval, not as a remote batch size.
