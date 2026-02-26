"""
FastAPI backend for lookalike comparison.

Pipeline:
- Cosine similarity is used only for retrieval (top-k candidates per query), not as the
  lookalike decision. The assessor model is the decision model, trained on ground truth.
- find_lookalikes: embed query -> retrieve top-k by cosine -> run assessor on those k only.
Run from project root: uv run uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8000

To use a different assessor (e.g. after comparing models): set env ASSESSOR_MODEL_DIR=assessor/model_new
"""
import os
import re
import time
from pathlib import Path
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lookalike API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded models (loaded on first /compare request)
_vit_processor = None
_vit_model = None
_vit_device = None
_assessor_model = None
_assessor_cfg = None


def _ensure_models():
    global _vit_processor, _vit_model, _vit_device, _assessor_model, _assessor_cfg
    if _assessor_model is None:
        t0 = time.perf_counter()
        from image_embedding.vit import load_vit_model, get_image_embedding
        from assessor.model import load_assessor
        _vit_processor, _vit_model, _vit_device = load_vit_model()
        model_dir = os.environ.get("ASSESSOR_MODEL_DIR", "assessor/model")
        _assessor_model, _assessor_cfg = load_assessor(device=_vit_device, in_dir=model_dir)
        _assessor_model.eval()
        print(f"[lookalike] Models loaded in {time.perf_counter() - t0:.1f}s (device: {_vit_device}, assessor: {model_dir})")


THRESHOLD = 0.5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    """Return model config for the UI (e.g. decision threshold, which assessor is used)."""
    model_dir = os.environ.get("ASSESSOR_MODEL_DIR", "assessor/model")
    return {"threshold": THRESHOLD, "assessor_model_dir": model_dir}


@app.post("/compare")
async def compare(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    """
    Compare two images; returns lookalike score in [0, 1] (1 = lookalike).
    """
    from image_embedding.vit import get_image_embedding

    t_total = time.perf_counter()
    _ensure_models()

    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if image_a.content_type not in allowed or image_b.content_type not in allowed:
        raise HTTPException(400, "Both files must be images (JPEG, PNG, WebP, GIF).")

    with tempfile.NamedTemporaryFile(suffix=Path(image_a.filename or "a").suffix, delete=False) as fa:
        fa.write(await image_a.read())
        path_a = fa.name
    with tempfile.NamedTemporaryFile(suffix=Path(image_b.filename or "b").suffix, delete=False) as fb:
        fb.write(await image_b.read())
        path_b = fb.name

    try:
        t0 = time.perf_counter()
        emb_a = get_image_embedding(
            _vit_model, _vit_processor, path_a, device=_vit_device, pooling="cls"
        )
        emb_b = get_image_embedding(
            _vit_model, _vit_processor, path_b, device=_vit_device, pooling="cls"
        )
        print(f"[lookalike] /compare embeddings: {time.perf_counter() - t0:.2f}s")
    finally:
        Path(path_a).unlink(missing_ok=True)
        Path(path_b).unlink(missing_ok=True)

    emb_a = emb_a.to(_vit_device)
    emb_b = emb_b.to(_vit_device)
    if emb_a.shape != (_assessor_cfg.embedding_size,) or emb_b.shape != (_assessor_cfg.embedding_size,):
        raise HTTPException(500, "Embedding size mismatch with assessor.")

    t0 = time.perf_counter()
    with torch.no_grad():
        score = _assessor_model(emb_a, emb_b)
    print(f"[lookalike] /compare assessor: {time.perf_counter() - t0:.2f}s")
    print(f"[lookalike] /compare total: {time.perf_counter() - t_total:.2f}s")

    score_val = float(score.item())
    return {
        "score": round(score_val, 4),
        "lookalike": score_val > THRESHOLD,
        "threshold": THRESHOLD,
    }


def _collect_embedding_paths(data_dir: Path) -> list[Path]:
    """All .pt embedding paths under data/, excluding Annotation."""
    out: list[Path] = []
    for p in data_dir.rglob("*.pt"):
        if "Annotation" in p.parts:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.name)


@app.post("/find_lookalikes")
async def find_lookalikes(
    query_image: UploadFile = File(...),
    data_dir: str = "data",
    batch_size: int = 64,
    top_k: int = 0,  # 0 = compare with full catalog (all 6K+); else cap at this many
    exclude_query_and_capsules: bool = True,
):
    """
    Pipeline: (1) Embed query. (2) Retrieve top-k candidates by cosine similarity (O(n)).
    (3) Run assessor (decision model) only on those k pairs. Cosine is retrieval only;
    the assessor score is the lookalike decision.
    By default the catalog excludes paths containing 'query' or 'capsules' (e.g. query images, ind. capsules).
    """
    from image_embedding.vit import get_image_embedding
    from torch.nn.functional import normalize

    t_total = time.perf_counter()
    _ensure_models()

    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if query_image.content_type not in allowed:
        raise HTTPException(400, "File must be an image (JPEG, PNG, WebP, GIF).")

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise HTTPException(400, f"Data directory not found: {data_dir}")

    t0 = time.perf_counter()
    pt_paths = _collect_embedding_paths(data_path)
    if exclude_query_and_capsules:
        pt_paths = [p for p in pt_paths if "query" not in p.as_posix().lower() and "capsules" not in p.as_posix().lower()]
    print(f"[lookalike] /find_lookalikes catalog discovery: {time.perf_counter() - t0:.2f}s ({len(pt_paths)} .pt files)")

    if not pt_paths:
        return {
            "lookalikes": [],
            "not_lookalikes": [],
            "threshold": THRESHOLD,
            "total_candidates": 0,
            "message": "No embedding files found under data/ (excluding Annotation).",
        }

    top_k_used = len(pt_paths) if top_k <= 0 else min(top_k, len(pt_paths))

    with tempfile.NamedTemporaryFile(
        suffix=Path(query_image.filename or "query").suffix, delete=False
    ) as f:
        f.write(await query_image.read())
        path_query = f.name

    try:
        t0 = time.perf_counter()
        query_emb = get_image_embedding(
            _vit_model, _vit_processor, path_query, device=_vit_device, pooling="cls"
        )
        print(f"[lookalike] /find_lookalikes query embedding: {time.perf_counter() - t0:.2f}s")
    finally:
        Path(path_query).unlink(missing_ok=True)

    query_emb = query_emb.to(_vit_device)
    if query_emb.shape != (_assessor_cfg.embedding_size,):
        raise HTTPException(500, "Embedding size mismatch with assessor.")
    query_emb_norm = normalize(query_emb.unsqueeze(0), p=2, dim=-1)

    def _name(p: Path) -> str:
        try:
            rel = p.relative_to(data_path)
            s = str(rel.with_suffix("")).replace("\\", " / ")
            # Drop leading "embeddings / " or "embeddings/" so UI shows drug name only
            s = re.sub(r"^embeddings\s*/\s*", "", s, flags=re.IGNORECASE).lstrip()
            return s or p.stem
        except ValueError:
            return p.stem

    # Step 1: Retrieve top-k by cosine similarity (O(n))
    t0 = time.perf_counter()
    retrieval: list[tuple[str, Path, float]] = []
    for i in range(0, len(pt_paths), batch_size):
        batch_paths = pt_paths[i : i + batch_size]
        batch_tensors: list[torch.Tensor] = []
        valid_names: list[str] = []
        valid_paths: list[Path] = []
        for p in batch_paths:
            try:
                t = torch.load(p, map_location=_vit_device, weights_only=True)
            except Exception:
                continue
            if torch.is_tensor(t) and t.shape == (_assessor_cfg.embedding_size,):
                batch_tensors.append(t.float())
                valid_names.append(_name(p))
                valid_paths.append(p)
        if not batch_tensors:
            continue
        batch = torch.stack(batch_tensors).to(_vit_device)
        batch_norm = normalize(batch, p=2, dim=-1)
        cos_sims = (query_emb_norm @ batch_norm.T).squeeze(0)
        for k in range(len(valid_names)):
            retrieval.append((valid_names[k], valid_paths[k], float(cos_sims[k].item())))
    retrieval.sort(key=lambda x: -x[2])
    top_k_actual = min(top_k_used, len(retrieval))
    retrieval = retrieval[:top_k_actual]
    print(f"[lookalike] /find_lookalikes retrieval top-{top_k_actual} by cosine: {time.perf_counter() - t0:.2f}s")

    # Step 2: Run assessor (decision model) only on top-k
    t0 = time.perf_counter()
    results: list[tuple[str, float, float]] = []
    query_emb_1 = query_emb_norm.squeeze(0)
    for name, p, cos_sim in retrieval:
        emb = torch.load(p, map_location=_vit_device, weights_only=True)
        if not torch.is_tensor(emb) or emb.shape != (_assessor_cfg.embedding_size,):
            continue
        cand = emb.float().to(_vit_device)
        with torch.no_grad():
            score = _assessor_model(query_emb_1, cand)
        results.append((name, round(float(cos_sim), 4), round(float(score.item()), 4)))
    print(f"[lookalike] /find_lookalikes assessor on {len(results)} candidates: {time.perf_counter() - t0:.2f}s")
    print(f"[lookalike] /find_lookalikes total: {time.perf_counter() - t_total:.2f}s")

    results.sort(key=lambda x: -x[2])
    lookalikes = [{"name": n, "cosine_sim": c, "score": s} for n, c, s in results if s >= THRESHOLD]
    not_lookalikes = [{"name": n, "cosine_sim": c, "score": s} for n, c, s in results if s < THRESHOLD]
    # Top 100 by score (strongest lookalikes), bottom 100 (least similar)
    top_100 = [{"name": n, "cosine_sim": c, "score": s} for n, c, s in results[:100]]
    bottom_100 = [{"name": n, "cosine_sim": c, "score": s} for n, c, s in results[-100:]]

    return {
        "lookalikes": lookalikes,
        "not_lookalikes": not_lookalikes,
        "top_100": top_100,
        "bottom_100": bottom_100,
        "threshold": THRESHOLD,
        "total_candidates": len(results),
        "retrieval_top_k": top_k_actual,
    }
