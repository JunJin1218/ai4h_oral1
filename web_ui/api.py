"""
FastAPI backend for online labeling + training loop.

Run from project root:
  uv run uvicorn web_ui.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from image_embedding.vit import get_image_embedding, load_vit_model
from train_online import OnlineTrainer, build_feedback_items
from utils import (
    get_file_name_by_vector_id,
    get_vector_id_by_file_name,
    reconstruct_vector_by_id,
    search_similar_with_metadata,
)

app = FastAPI(title="Lookalike Online Trainer API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
IMAGE_ROOTS = [Path("input_img")]
DEFAULT_ASSESSOR_CHECKPOINT_PATH = Path(
    "assessor/model_ai_feedback_h2048_1024_512/assessor_best_f1.pt"
)
ASSESSOR_CHECKPOINT_PATH = Path(
    os.environ.get("ASSESSOR_CHECKPOINT_PATH", str(DEFAULT_ASSESSOR_CHECKPOINT_PATH))
)
AI_AUGMENT_TABLE = "ai_augment_feedback"


class SessionInitRequest(BaseModel):
    similarity_type: str = Field(default="l2")
    top_k: int = Field(default=20, ge=1, le=200)
    per_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    per_beta: float = Field(default=0.4, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=1, le=512)
    train_steps: int = Field(default=1, ge=1, le=20)
    min_buffer_size: int = Field(default=32, ge=1, le=5000)
    recent_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    hydrate_from_feedback: bool = False
    hydrate_limit: int = Field(default=2000, ge=1, le=50000)


class LabelItem(BaseModel):
    vector_id: int
    label: int


class SubmitRequest(BaseModel):
    query_vector_id: int
    labels: list[LabelItem]


@dataclass
class SessionState:
    similarity_type: str = "l2"
    top_k: int = 20
    batch_size: int = 32
    train_steps: int = 1
    min_buffer_size: int = 32
    recent_ratio: float = 0.5
    current_query_vector_id: int | None = None
    current_query_file_name: str | None = None
    current_candidate_vector_ids: list[int] | None = None


_lock = Lock()
_trainer: OnlineTrainer | None = None
_session = SessionState()
_image_index: dict[str, Path] | None = None
_vit_bundle: tuple | None = None


def _get_trainer() -> OnlineTrainer:
    global _trainer
    if _trainer is None:
        _trainer = OnlineTrainer(
            model_checkpoint_path=ASSESSOR_CHECKPOINT_PATH,
            db_path=DB_PATH,
            index_path=INDEX_PATH,
        )
        print(f"[web_ui.api] assessor model dir: {_trainer.model_dir}")
        print(f"[web_ui.api] assessor checkpoint: {_trainer.model_checkpoint}")
    return _trainer


def _build_image_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    for root in IMAGE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                index.setdefault(path.stem, path)
    return index


def _get_vit_bundle():
    global _vit_bundle
    if _vit_bundle is None:
        _vit_bundle = load_vit_model()
    return _vit_bundle


def _resolve_image_path_from_embedding_file(file_name: str) -> Path | None:
    global _image_index
    if _image_index is None:
        _image_index = _build_image_index()
    stem = Path(file_name).stem
    return _image_index.get(stem)


def _random_query_file_name() -> str:
    if not DB_PATH.exists():
        raise HTTPException(400, "DB not found. Build image_db first.")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_name FROM image_db ORDER BY RANDOM() LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise HTTPException(400, "image_db is empty.")
    return str(row[0])


def _ai_augment_table_exists(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (AI_AUGMENT_TABLE,),
    )
    return cur.fetchone() is not None


def _make_batch_payload() -> dict:
    trainer = _get_trainer()
    query_file = _random_query_file_name()
    query_vector_id = get_vector_id_by_file_name(query_file, db_path=DB_PATH)
    query_vector = reconstruct_vector_by_id(query_vector_id, index_path=INDEX_PATH)

    candidates = trainer.retrieve_candidates(
        query_file_name=query_file,
        top_k=_session.top_k,
        similarity_type=_session.similarity_type,
    )
    scores = trainer.score_candidates(query_vector, candidates)

    _session.current_query_vector_id = query_vector_id
    _session.current_query_file_name = query_file
    _session.current_candidate_vector_ids = [x.vector_id for x in candidates]

    query_img = _resolve_image_path_from_embedding_file(query_file)
    candidate_payload = []
    for cand, score in zip(candidates, scores):
        img = _resolve_image_path_from_embedding_file(cand.file_name or "")
        candidate_payload.append(
            {
                "vector_id": cand.vector_id,
                "file_name": cand.file_name,
                "retrieval_score": round(float(cand.score), 6),
                "model_score": round(float(score), 6),
                "image_url": f"/online/image/{cand.vector_id}" if img else None,
            }
        )

    return {
        "query": {
            "vector_id": query_vector_id,
            "file_name": query_file,
            "image_url": f"/online/image/{query_vector_id}" if query_img else None,
        },
        "candidates": candidate_payload,
        "session": {
            "similarity_type": _session.similarity_type,
            "top_k": _session.top_k,
            "buffer_size": len(_get_trainer().replay),
        },
    }


def _predict_candidates_from_image(
    image: Image.Image,
    *,
    top_k: int,
    similarity_type: str,
    threshold: float,
) -> dict:
    trainer = _get_trainer()
    processor, vit_model, vit_device = _get_vit_bundle()
    query_vector_t = get_image_embedding(
        vit_model,
        processor,
        image,
        device=vit_device,
        pooling="cls",
    )
    query_vector = query_vector_t.detach().cpu().numpy().astype("float32", copy=False)

    candidates = search_similar_with_metadata(
        query_vector=query_vector,
        similarity_type=similarity_type,
        top_n=top_k,
        index_path=INDEX_PATH,
        db_path=DB_PATH,
        exclude_vector_id=None,
    )
    scores = trainer.score_candidates(query_vector, candidates)

    lookalikes = []
    non_lookalikes = []
    for cand, score in zip(candidates, scores):
        img = _resolve_image_path_from_embedding_file(cand.file_name or "")
        is_lookalike = float(score) >= threshold
        item = {
            "vector_id": cand.vector_id,
            "file_name": cand.file_name,
            "retrieval_score": round(float(cand.score), 6),
            "model_score": round(float(score), 6),
            "is_lookalike": is_lookalike,
            "image_url": f"/online/image/{cand.vector_id}" if img else None,
        }
        if is_lookalike:
            lookalikes.append(item)
        else:
            non_lookalikes.append(item)

    return {
        "lookalikes": lookalikes,
        "non_lookalikes": non_lookalikes,
        "meta": {
            "top_k": top_k,
            "similarity_type": similarity_type,
            "threshold": threshold,
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/online/session/init")
def init_online_session(req: SessionInitRequest) -> dict:
    with _lock:
        trainer = _get_trainer()
        _session.similarity_type = req.similarity_type
        _session.top_k = req.top_k
        _session.batch_size = req.batch_size
        _session.train_steps = req.train_steps
        _session.min_buffer_size = req.min_buffer_size
        _session.recent_ratio = req.recent_ratio

        trainer.reset_replay(per_alpha=req.per_alpha, per_beta=req.per_beta)
        hydrated = 0
        if req.hydrate_from_feedback:
            hydrated = trainer.hydrate_replay_from_online_feedback(limit=req.hydrate_limit)

        batch = _make_batch_payload()
        return {
            "message": "session initialized",
            "hydrated": hydrated,
            "per": {"alpha": trainer.replay.alpha, "beta": trainer.replay.beta},
            "batch": batch,
        }


@app.post("/inference/predict")
async def inference_predict(
    image: UploadFile = File(...),
    top_k: int = 20,
    similarity_type: str = "l2",
    threshold: float = 0.5,
) -> dict:
    if top_k <= 0 or top_k > 200:
        raise HTTPException(400, "top_k must be between 1 and 200.")
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(400, "threshold must be between 0.0 and 1.0.")
    try:
        metric = similarity_type.lower()
        if metric not in {"l2", "euclidean", "ip", "inner_product", "dot", "cosine", "cos", "cos_sim"}:
            raise HTTPException(400, "invalid similarity_type")
        content = await image.read()
        if not content:
            raise HTTPException(400, "Uploaded image is empty.")
        pil_image = Image.open(BytesIO(content)).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse uploaded image: {exc}") from exc

    with _lock:
        return _predict_candidates_from_image(
            pil_image,
            top_k=top_k,
            similarity_type=similarity_type,
            threshold=threshold,
        )


@app.get("/online/session/next")
def next_batch() -> dict:
    with _lock:
        return _make_batch_payload()


@app.post("/online/session/submit")
def submit_labels(req: SubmitRequest) -> dict:
    with _lock:
        trainer = _get_trainer()
        if _session.current_query_vector_id is None or _session.current_query_file_name is None:
            raise HTTPException(400, "No active batch. Call /online/session/init or /online/session/next.")
        if req.query_vector_id != _session.current_query_vector_id:
            raise HTTPException(400, "query_vector_id does not match current batch.")

        labels_by_vector_id: dict[int, int] = {}
        for item in req.labels:
            if item.label not in (0, 1):
                raise HTTPException(400, "label must be 0 or 1.")
            labels_by_vector_id[int(item.vector_id)] = int(item.label)

        query_vector = reconstruct_vector_by_id(req.query_vector_id, index_path=INDEX_PATH)
        candidates = trainer.retrieve_candidates(
            query_file_name=_session.current_query_file_name,
            top_k=_session.top_k,
            similarity_type=_session.similarity_type,
        )
        scores = trainer.score_candidates(query_vector, candidates)
        feedback = build_feedback_items(
            query_vector_id=req.query_vector_id,
            query_file_name=_session.current_query_file_name,
            query_vector=query_vector,
            candidates=candidates,
            labels_by_vector_id=labels_by_vector_id,
            model_scores=scores,
        )

        trainer.ingest_feedback(feedback)
        loss = trainer.train_step(
            batch_size=_session.batch_size,
            steps=_session.train_steps,
            min_buffer_size=_session.min_buffer_size,
            recent_ratio=_session.recent_ratio,
        )

        next_payload = _make_batch_payload()
        return {
            "ingested": len(feedback),
            "trained": loss is not None,
            "loss": None if loss is None else round(float(loss), 6),
            "buffer_size": len(trainer.replay),
            "next_batch": next_payload,
        }


@app.get("/online/image/{vector_id}")
def get_image(vector_id: int):
    file_name = get_file_name_by_vector_id(vector_id, db_path=DB_PATH)
    if file_name is None:
        raise HTTPException(404, "vector_id not found.")
    image_path = _resolve_image_path_from_embedding_file(file_name)
    if image_path is None or not image_path.exists():
        raise HTTPException(404, "image not found for this vector.")
    return FileResponse(path=image_path)


@app.get("/online/replay_buffer")
def get_replay_buffer(limit: int = 100) -> dict:
    if limit <= 0:
        raise HTTPException(400, "limit must be > 0")
    with _lock:
        trainer = _get_trainer()
        replay_items = trainer.replay_snapshot(limit=limit)
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM online_feedback")
            feedback_count = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM lookalike_graph")
            graph_count = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT id, label, query_vector_id, candidate_vector_id, query_file_name, candidate_file_name, retrieval_score, model_score, created_at
                FROM online_feedback
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            feedback_rows = [
                {
                    "id": int(r[0]),
                    "label": int(r[1]),
                    "query_vector_id": r[2],
                    "candidate_vector_id": r[3],
                    "query_file_name": r[4],
                    "candidate_file_name": r[5],
                    "retrieval_score": r[6],
                    "model_score": r[7],
                    "created_at": r[8],
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT lg.id, lg.base_id, lg.similar_id, ib.file_name, isb.file_name
                FROM lookalike_graph lg
                LEFT JOIN image_db ib ON ib.id = lg.base_id
                LEFT JOIN image_db isb ON isb.id = lg.similar_id
                ORDER BY lg.id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            graph_rows = [
                {
                    "id": int(r[0]),
                    "base_id": int(r[1]) if r[1] is not None else None,
                    "similar_id": int(r[2]) if r[2] is not None else None,
                    "base_file_name": r[3],
                    "similar_file_name": r[4],
                }
                for r in cur.fetchall()
            ]

        return {
            "replay": {
                "size": len(trainer.replay),
                "alpha": trainer.replay.alpha,
                "beta": trainer.replay.beta,
                "items": replay_items,
            },
            "online_feedback": {
                "count": feedback_count,
                "items": feedback_rows,
            },
            "lookalike_graph": {
                "count": graph_count,
                "items": graph_rows,
            },
        }


@app.get("/ai_feedback/queries")
def get_ai_feedback_queries(limit: int = 50) -> dict:
    if limit <= 0:
        raise HTTPException(400, "limit must be > 0")

    with sqlite3.connect(DB_PATH) as conn:
        if not _ai_augment_table_exists(conn):
            return {"queries": []}

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                query_vector_id,
                query_image_name,
                COUNT(*) AS total_count,
                SUM(label) AS lookalike_count,
                COUNT(*) - SUM(label) AS non_lookalike_count,
                MAX(id) AS latest_id
            FROM {AI_AUGMENT_TABLE}
            WHERE COALESCE(error, 0) = 0
            GROUP BY query_vector_id, query_image_name
            ORDER BY latest_id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()

    queries = []
    for row in rows:
        query_vector_id = int(row[0])
        query_image_name = str(row[1])
        query_img = _resolve_image_path_from_embedding_file(query_image_name)
        queries.append(
            {
                "query_vector_id": query_vector_id,
                "query_image_name": query_image_name,
                "total_count": int(row[2]),
                "lookalike_count": int(row[3] or 0),
                "non_lookalike_count": int(row[4] or 0),
                "image_url": f"/online/image/{query_vector_id}" if query_img else None,
            }
        )
    return {"queries": queries}


@app.get("/ai_feedback/by_query")
def get_ai_feedback_by_query(query_vector_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        if not _ai_augment_table_exists(conn):
            raise HTTPException(404, "ai_augment_feedback table not found.")

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                batch_id,
                query_image_name,
                candidate_vector_id,
                candidate_image_name,
                label,
                COALESCE(identical, 0),
                reasoning
            FROM {AI_AUGMENT_TABLE}
            WHERE query_vector_id = ?
              AND COALESCE(error, 0) = 0
            ORDER BY id DESC
            """,
            (int(query_vector_id),),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(404, "No AI feedback rows found for this query_vector_id.")

    query_image_name = str(rows[0][1])
    query_img = _resolve_image_path_from_embedding_file(query_image_name)
    lookalikes = []
    non_lookalikes = []
    batch_ids: list[str] = []
    seen_batches: set[str] = set()

    for batch_id, _, candidate_vector_id, candidate_image_name, label, identical, reasoning in rows:
        batch_id = str(batch_id)
        if batch_id not in seen_batches:
            seen_batches.add(batch_id)
            batch_ids.append(batch_id)

        candidate_vector_id = int(candidate_vector_id)
        cand_img = _resolve_image_path_from_embedding_file(str(candidate_image_name))
        item = {
            "vector_id": candidate_vector_id,
            "file_name": candidate_image_name,
            "label": int(label),
            "identical": bool(int(identical)),
            "reasoning": reasoning,
            "batch_id": batch_id,
            "image_url": f"/online/image/{candidate_vector_id}" if cand_img else None,
        }
        if int(label) == 1:
            lookalikes.append(item)
        else:
            non_lookalikes.append(item)

    return {
        "query": {
            "vector_id": int(query_vector_id),
            "file_name": query_image_name,
            "image_url": f"/online/image/{int(query_vector_id)}" if query_img else None,
        },
        "lookalikes": lookalikes,
        "non_lookalikes": non_lookalikes,
        "meta": {
            "total_count": len(rows),
            "lookalike_count": len(lookalikes),
            "non_lookalike_count": len(non_lookalikes),
            "batch_ids": batch_ids,
        },
    }
