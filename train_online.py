from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from assessor.model import AssessorConfig, AssessorMLP, load_assessor, save_assessor
from utils import (
    SimilarResult,
    find_similar_by_file_name_with_metadata,
    get_vector_id_by_file_name,
    reconstruct_vector_by_id,
)


@dataclass
class FeedbackItem:
    label: int
    query_vector: np.ndarray
    candidate_vector: np.ndarray
    query_vector_id: int | None = None
    candidate_vector_id: int | None = None
    query_file_name: str | None = None
    candidate_file_name: str | None = None
    retrieval_score: float | None = None
    model_score: float | None = None


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._items: list[FeedbackItem] = []
        self._priorities: list[float] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._items)

    def reset(self, alpha: float | None = None, beta: float | None = None) -> None:
        if alpha is not None:
            self.alpha = float(alpha)
        if beta is not None:
            self.beta = float(beta)
        self._items.clear()
        self._priorities.clear()
        self._next_index = 0

    def _default_priority(self) -> float:
        if not self._priorities:
            return 1.0
        return max(self._priorities)

    def add_many(
        self,
        items: Iterable[FeedbackItem],
        priorities: Iterable[float] | None = None,
    ) -> None:
        if priorities is None:
            prios = None
        else:
            prios = list(priorities)
        incoming = list(items)
        for i, item in enumerate(incoming):
            p = self._default_priority() if prios is None else float(max(prios[i], 1e-6))
            if len(self._items) < self.capacity:
                self._items.append(item)
                self._priorities.append(p)
            else:
                self._items[self._next_index] = item
                self._priorities[self._next_index] = p
                self._next_index = (self._next_index + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        recent_ratio: float = 0.5,
        recent_window: int = 1024,
    ) -> tuple[list[FeedbackItem], list[int], np.ndarray]:
        n = len(self._items)
        if n == 0:
            return [], [], np.zeros((0,), dtype=np.float32)

        k = min(int(batch_size), n)
        prios = np.asarray(self._priorities, dtype=np.float64)
        probs = np.power(np.maximum(prios, 1e-8), self.alpha)

        if recent_ratio > 0 and n > 1:
            recent_start = max(0, n - int(recent_window))
            recent_mask = np.zeros((n,), dtype=np.float64)
            recent_mask[recent_start:] = 1.0
            probs = probs * (1.0 + float(recent_ratio) * recent_mask)

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones((n,), dtype=np.float64) / n
        else:
            probs /= probs_sum

        indices = np.random.choice(n, size=k, replace=False, p=probs).astype(np.int64)
        sampled_probs = probs[indices]
        weights = np.power(n * np.maximum(sampled_probs, 1e-12), -self.beta)
        weights = (weights / np.max(weights)).astype(np.float32)

        items = [self._items[int(i)] for i in indices]
        return items, indices.tolist(), weights

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        for i, p in zip(indices, priorities):
            if 0 <= i < len(self._priorities):
                self._priorities[i] = float(max(p, 1e-6))

    def snapshot(self, limit: int = 100) -> list[dict]:
        if limit <= 0:
            return []
        start = max(0, len(self._items) - int(limit))
        out: list[dict] = []
        for i in range(start, len(self._items)):
            item = self._items[i]
            out.append(
                {
                    "buffer_index": i,
                    "priority": float(self._priorities[i]),
                    "label": int(item.label),
                    "query_vector_id": item.query_vector_id,
                    "candidate_vector_id": item.candidate_vector_id,
                    "query_file_name": item.query_file_name,
                    "candidate_file_name": item.candidate_file_name,
                    "retrieval_score": item.retrieval_score,
                    "model_score": item.model_score,
                }
            )
        return out


class OnlineTrainer:
    def __init__(
        self,
        *,
        model_dir: str | Path | None = None,
        db_path: str | Path = "data/sqlite/ai4h.db",
        buffer_capacity: int = 20000,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        lr: float = 1e-4,
        index_path: str | Path = "data/faiss/embeddings.index",
    ) -> None:
        if model_dir is None:
            model_dir = os.environ.get("ASSESSOR_MODEL_DIR", "assessor/model")
        self.model_dir = Path(model_dir)
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.log_dir = Path(os.environ.get("TENSORBOARD_LOG_DIR", "log")) / "online"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta=per_beta,
        )
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.train_step_count = 0

        if (self.model_dir / "assessor.pt").exists() and (self.model_dir / "config.json").exists():
            self.model, self.cfg = load_assessor(device=self.device, in_dir=self.model_dir)
        else:
            self.cfg = AssessorConfig()
            self.model = AssessorMLP(self.cfg).to(self.device)

        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCELoss(reduction="none")
        self._ensure_online_tables()

    def reset_replay(self, per_alpha: float | None = None, per_beta: float | None = None) -> None:
        self.replay.reset(alpha=per_alpha, beta=per_beta)

    def _ensure_online_tables(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS online_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label INTEGER NOT NULL,
                    query_vector_id INTEGER,
                    candidate_vector_id INTEGER,
                    query_file_name TEXT,
                    candidate_file_name TEXT,
                    retrieval_score REAL,
                    model_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS online_train_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step_count INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lookalike_graph (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    base_id INTEGER,
                    similar_id INTEGER,
                    FOREIGN KEY (base_id) REFERENCES image_db(id) ON DELETE CASCADE,
                    FOREIGN KEY (similar_id) REFERENCES image_db(id) ON DELETE CASCADE,
                    UNIQUE(base_id, similar_id)
                )
                """
            )
            conn.commit()

    def retrieve_candidates(
        self,
        query_file_name: str,
        *,
        top_k: int = 20,
        similarity_type: str = "l2",
    ) -> list[SimilarResult]:
        return find_similar_by_file_name_with_metadata(
            file_name=query_file_name,
            similarity_type=similarity_type,
            top_n=top_k,
            db_path=self.db_path,
            index_path=self.index_path,
            exclude_self=True,
        )

    @torch.no_grad()
    def score_candidates(
        self,
        query_vector: np.ndarray,
        candidates: list[SimilarResult],
    ) -> list[float]:
        if not candidates:
            return []
        q = torch.from_numpy(query_vector.astype(np.float32)).to(self.device).unsqueeze(0)
        c = torch.from_numpy(np.stack([item.vector for item in candidates]).astype(np.float32)).to(
            self.device
        )
        q_batch = q.repeat(c.shape[0], 1)
        self.model.eval()
        scores = self.model(q_batch, c).detach().cpu().numpy().tolist()
        self.model.train()
        return [float(s) for s in scores]

    def _vector_to_image_id_map(self, conn: sqlite3.Connection) -> dict[int, int]:
        cur = conn.cursor()
        cur.execute("SELECT id, vector_id FROM image_db")
        out: dict[int, int] = {}
        for image_id, vector_id in cur.fetchall():
            out[int(vector_id)] = int(image_id)
        return out

    def ingest_feedback(self, items: list[FeedbackItem]) -> None:
        if not items:
            return

        priorities = []
        pos_count = 0
        for x in items:
            if int(x.label) == 1:
                pos_count += 1
            if x.model_score is None:
                priorities.append(1.0)
            else:
                priorities.append(abs(float(x.model_score) - float(x.label)) + 1e-3)
        self.replay.add_many(items, priorities=priorities)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executemany(
                """
                INSERT INTO online_feedback (
                    label, query_vector_id, candidate_vector_id, query_file_name,
                    candidate_file_name, retrieval_score, model_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        int(x.label),
                        x.query_vector_id,
                        x.candidate_vector_id,
                        x.query_file_name,
                        x.candidate_file_name,
                        x.retrieval_score,
                        x.model_score,
                    )
                    for x in items
                ],
            )

            vec_to_img = self._vector_to_image_id_map(conn)
            for x in items:
                if x.query_vector_id is None or x.candidate_vector_id is None:
                    continue
                if x.query_vector_id not in vec_to_img or x.candidate_vector_id not in vec_to_img:
                    continue
                a = vec_to_img[x.query_vector_id]
                b = vec_to_img[x.candidate_vector_id]
                base_id, similar_id = (a, b) if a < b else (b, a)
                if int(x.label) == 1:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO lookalike_graph (base_id, similar_id)
                        VALUES (?, ?)
                        """,
                        (base_id, similar_id),
                    )
                else:
                    conn.execute(
                        """
                        DELETE FROM lookalike_graph WHERE base_id = ? AND similar_id = ?
                        """,
                        (base_id, similar_id),
                    )

            conn.commit()

        neg_count = len(items) - pos_count
        self.writer.add_scalar("online/feedback_ingested", len(items), self.train_step_count)
        self.writer.add_scalar("online/feedback_positive", pos_count, self.train_step_count)
        self.writer.add_scalar("online/feedback_negative", neg_count, self.train_step_count)
        self.writer.add_scalar("online/replay_size", len(self.replay), self.train_step_count)
        self.writer.flush()

    def train_step(
        self,
        *,
        batch_size: int = 32,
        steps: int = 1,
        min_buffer_size: int = 64,
        recent_ratio: float = 0.5,
    ) -> float | None:
        if len(self.replay) < min_buffer_size:
            return None

        losses: list[float] = []
        for _ in range(max(1, steps)):
            batch, indices, is_weights = self.replay.sample(
                batch_size=batch_size,
                recent_ratio=recent_ratio,
            )
            if not batch:
                return None

            q = np.stack([x.query_vector for x in batch]).astype(np.float32)
            c = np.stack([x.candidate_vector for x in batch]).astype(np.float32)
            y = np.asarray([x.label for x in batch], dtype=np.float32)

            q_t = torch.from_numpy(q).to(self.device)
            c_t = torch.from_numpy(c).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)
            w_t = torch.from_numpy(is_weights).to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(q_t, c_t)
            per_sample_loss = self.criterion(pred, y_t)
            loss = (per_sample_loss * w_t).sum() / torch.clamp(w_t.sum(), min=1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                new_prios = torch.abs(pred - y_t).detach().cpu().numpy()
            self.replay.update_priorities(indices, (new_prios + 1e-3).tolist())
            losses.append(float(loss.item()))

        avg_loss = float(sum(losses) / len(losses))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO online_train_log (step_count, batch_size, loss) VALUES (?, ?, ?)",
                (steps, batch_size, avg_loss),
            )
            conn.commit()
        self.train_step_count += max(1, steps)
        self.writer.add_scalar("online/loss", avg_loss, self.train_step_count)
        self.writer.add_scalar("online/replay_size", len(self.replay), self.train_step_count)
        self.writer.add_scalar("online/batch_size", batch_size, self.train_step_count)
        self.writer.add_scalar("online/steps_per_update", steps, self.train_step_count)
        self.writer.flush()
        save_assessor(self.model, self.cfg, out_dir=self.model_dir)
        return avg_loss

    def hydrate_replay_from_online_feedback(self, limit: int = 5000) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT label, query_vector_id, candidate_vector_id, query_file_name, candidate_file_name, model_score
                FROM online_feedback
                WHERE query_vector_id IS NOT NULL AND candidate_vector_id IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cur.fetchall()

        rows.reverse()
        items: list[FeedbackItem] = []
        priorities: list[float] = []
        for label, qid, cid, qname, cname, model_score in rows:
            try:
                qv = reconstruct_vector_by_id(int(qid), index_path=self.index_path)
                cv = reconstruct_vector_by_id(int(cid), index_path=self.index_path)
            except Exception:
                continue
            items.append(
                FeedbackItem(
                    label=int(label),
                    query_vector=qv,
                    candidate_vector=cv,
                    query_vector_id=int(qid),
                    candidate_vector_id=int(cid),
                    query_file_name=qname,
                    candidate_file_name=cname,
                    model_score=model_score,
                )
            )
            if model_score is None:
                priorities.append(1.0)
            else:
                priorities.append(abs(float(model_score) - float(label)) + 1e-3)

        self.replay.add_many(items, priorities=priorities)
        return len(items)

    def replay_snapshot(self, limit: int = 100) -> list[dict]:
        return self.replay.snapshot(limit=limit)


def build_feedback_items(
    *,
    query_vector_id: int,
    query_file_name: str,
    query_vector: np.ndarray,
    candidates: list[SimilarResult],
    labels_by_vector_id: dict[int, int],
    model_scores: list[float] | None = None,
) -> list[FeedbackItem]:
    score_map = model_scores if model_scores is not None else [None] * len(candidates)
    out: list[FeedbackItem] = []
    for cand, model_score in zip(candidates, score_map):
        if cand.vector_id not in labels_by_vector_id:
            continue
        out.append(
            FeedbackItem(
                label=int(labels_by_vector_id[cand.vector_id]),
                query_vector=query_vector.copy(),
                candidate_vector=cand.vector.copy(),
                query_vector_id=query_vector_id,
                candidate_vector_id=cand.vector_id,
                query_file_name=query_file_name,
                candidate_file_name=cand.file_name,
                retrieval_score=cand.score,
                model_score=model_score,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Online training pipeline for lookalike assessor")
    parser.add_argument("--query-file-name", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--similarity-type", type=str, default="l2")
    parser.add_argument(
        "--labels-json",
        type=str,
        default=None,
        help='JSON path: [{"vector_id": 123, "label": 1}, ...]',
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--min-buffer-size", type=int, default=32)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    args = parser.parse_args()

    trainer = OnlineTrainer(per_alpha=args.per_alpha, per_beta=args.per_beta)
    loaded = trainer.hydrate_replay_from_online_feedback(limit=2000)
    print(f"Hydrated replay from online_feedback: {loaded}")

    candidates = trainer.retrieve_candidates(
        query_file_name=args.query_file_name,
        top_k=args.top_k,
        similarity_type=args.similarity_type,
    )
    if not candidates:
        print("No candidates found.")
        return

    query_vector_id = get_vector_id_by_file_name(args.query_file_name, db_path=trainer.db_path)
    query_vector = reconstruct_vector_by_id(query_vector_id, index_path=trainer.index_path)
    model_scores = trainer.score_candidates(query_vector=query_vector, candidates=candidates)

    print(f"Retrieved {len(candidates)} candidates.")
    for idx, (cand, score) in enumerate(zip(candidates, model_scores), start=1):
        print(
            f"{idx:02d}. vector_id={cand.vector_id} retrieval={cand.score:.6f} "
            f"model={score:.4f} file={cand.file_name}"
        )

    if not args.labels_json:
        print("Dry run complete. Provide --labels-json to ingest feedback and train.")
        return

    payload = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
    labels_by_vector_id = {int(x["vector_id"]): int(x["label"]) for x in payload}
    feedback = build_feedback_items(
        query_file_name=args.query_file_name,
        query_vector_id=query_vector_id,
        query_vector=query_vector,
        candidates=candidates,
        labels_by_vector_id=labels_by_vector_id,
        model_scores=model_scores,
    )
    trainer.ingest_feedback(feedback)
    print(f"Ingested {len(feedback)} feedback rows into replay + sqlite/graph.")

    loss = trainer.train_step(
        batch_size=args.batch_size,
        steps=args.steps,
        min_buffer_size=args.min_buffer_size,
    )
    if loss is None:
        print(
            f"Train skipped: replay size={len(trainer.replay)} < min_buffer_size={args.min_buffer_size}"
        )
    else:
        print(f"Train complete. avg_loss={loss:.6f}")


if __name__ == "__main__":
    main()
