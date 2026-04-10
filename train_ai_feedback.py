"""
Train the assessor directly from ai_augment_feedback in SQLite.

Usage:
  uv run python -m train_ai_feedback
  uv run python -m train_ai_feedback --epochs 20 --batch-size 64
  uv run python -m train_ai_feedback --out-dir assessor/model_ai_feedback
  uv run python -m train_ai_feedback --resume
  uv run python -m train_ai_feedback --include-identical
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from assessor.model import AssessorConfig, AssessorMLP, load_assessor, save_assessor
from utils import reconstruct_vector_by_id


DB_PATH = Path("data/sqlite/ai4h.db")
INDEX_PATH = Path("data/faiss/embeddings.index")
TABLE_NAME = "ai_augment_feedback"


@dataclass(frozen=True)
class AIFeedbackRow:
    query_vector_id: int
    candidate_vector_id: int
    query_file_name: str
    candidate_file_name: str
    label: int
    identical: int


class PairIdDataset(Dataset):
    def __init__(
        self,
        pairs: list[AIFeedbackRow],
        *,
        index_path: Path,
        embedding_size: int = 1280,
    ) -> None:
        self.pairs = pairs
        self.index_path = index_path
        self.embedding_size = embedding_size
        self.cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_emb(self, vector_id: int) -> torch.Tensor:
        if vector_id not in self.cache:
            vec = reconstruct_vector_by_id(vector_id, index_path=self.index_path)
            t = torch.from_numpy(vec).float()
            if t.ndim != 1 or t.shape[0] != self.embedding_size:
                raise ValueError(f"Invalid vector shape for vector_id={vector_id}: {tuple(t.shape)}")
            self.cache[vector_id] = t
        return self.cache[vector_id]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.pairs[idx]
        q = self._load_emb(row.query_vector_id)
        c = self._load_emb(row.candidate_vector_id)
        y = torch.tensor(float(row.label), dtype=torch.float32)
        return q, c, y


def load_ai_feedback_rows(
    *,
    db_path: Path,
    include_identical: bool,
    min_rows_per_query: int,
) -> list[AIFeedbackRow]:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                query_vector_id,
                candidate_vector_id,
                query_image_name,
                candidate_image_name,
                label,
                COALESCE(identical, 0)
            FROM {TABLE_NAME}
            WHERE COALESCE(error, 0) = 0
              AND query_vector_id IS NOT NULL
              AND candidate_vector_id IS NOT NULL
            ORDER BY id
            """
        )
        rows = [
            AIFeedbackRow(
                query_vector_id=int(qid),
                candidate_vector_id=int(cid),
                query_file_name=str(qname),
                candidate_file_name=str(cname),
                label=int(label),
                identical=int(identical),
            )
            for qid, cid, qname, cname, label, identical in cur.fetchall()
        ]

    raw_count = len(rows)
    if not include_identical:
        rows = [row for row in rows if row.identical == 0]
    post_identical_count = len(rows)

    by_query: dict[int, list[AIFeedbackRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_vector_id].append(row)

    filtered: list[AIFeedbackRow] = []
    dropped_queries = 0
    for query_vector_id, group in by_query.items():
        if len(group) < min_rows_per_query:
            dropped_queries += 1
            continue
        filtered.extend(group)

    print(
        f"[train_ai_feedback] loaded_rows={raw_count} "
        f"after_identical_filter={post_identical_count} "
        f"after_min_rows_filter={len(filtered)} "
        f"dropped_queries={dropped_queries}"
    )
    return filtered


def split_by_query(
    rows: list[AIFeedbackRow],
    *,
    train_ratio: float,
    seed: int,
) -> tuple[list[AIFeedbackRow], list[AIFeedbackRow]]:
    by_query: dict[int, list[AIFeedbackRow]] = defaultdict(list)
    for row in rows:
        by_query[row.query_vector_id].append(row)

    query_ids = sorted(by_query.keys())
    if not query_ids:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(query_ids)

    if len(query_ids) == 1:
        return list(by_query[query_ids[0]]), []

    n_train = int(len(query_ids) * train_ratio)
    n_train = min(max(n_train, 1), len(query_ids) - 1)
    train_query_ids = set(query_ids[:n_train])

    train_rows: list[AIFeedbackRow] = []
    val_rows: list[AIFeedbackRow] = []
    for query_id, group in by_query.items():
        if query_id in train_query_ids:
            train_rows.extend(group)
        else:
            val_rows.extend(group)
    return train_rows, val_rows


def label_counts(rows: list[AIFeedbackRow]) -> Counter:
    return Counter(int(row.label) for row in rows)


def parse_hidden_sizes(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("--hidden-sizes must contain at least one integer")
    hidden_sizes = tuple(int(x) for x in parts)
    if any(x <= 0 for x in hidden_sizes):
        raise ValueError("--hidden-sizes values must be > 0")
    return hidden_sizes


def model_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_metric_checkpoint(
    *,
    model: torch.nn.Module,
    cfg: AssessorConfig,
    out_dir: Path,
    metric_name: str,
    epoch: int,
    metric_value: float,
) -> None:
    filename = f"assessor_best_{metric_name}.pt"
    save_assessor(model, cfg, out_dir=out_dir, filename=filename)
    print(
        f"  saved best-{metric_name} checkpoint: epoch={epoch} "
        f"{metric_name}={metric_value:.4f} -> {out_dir / filename}"
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    rows: list[AIFeedbackRow],
    *,
    cfg: AssessorConfig,
    device: torch.device,
    batch_size: int,
    index_path: Path,
    threshold: float = 0.5,
) -> dict[str, object]:
    if not rows:
        return {
            "n": 0,
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "cm": None,
        }

    dataset = PairIdDataset(rows, index_path=index_path, embedding_size=cfg.embedding_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = torch.nn.BCELoss()

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    n_batches = 0

    for e1, e2, labels in loader:
        e1 = e1.to(device)
        e2 = e2.to(device)
        labels = labels.to(device)

        scores = model(e1, e2)
        loss = criterion(scores, labels)
        total_loss += float(loss.item())
        n_batches += 1

        probs = scores.detach().cpu().tolist()
        lbls = labels.detach().cpu().tolist()
        y_true.extend(int(v) for v in lbls)
        y_pred.extend(1 if p > threshold else 0 for p in probs)

    avg_loss = total_loss / max(n_batches, 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": len(y_true),
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train assessor from ai_augment_feedback in SQLite")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--index-path", type=str, default=str(INDEX_PATH))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", type=str, default="assessor/model_ai_feedback")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--balance-weights", action="store_true")
    parser.add_argument("--include-identical", action="store_true")
    parser.add_argument("--min-rows-per-query", type=int, default=1)
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes override, e.g. 2048,1024,256",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout in AssessorConfig",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.resume and (args.hidden_sizes is not None or args.dropout is not None):
        raise ValueError("--resume cannot be combined with --hidden-sizes or --dropout")

    db_path = Path(args.db_path)
    index_path = Path(args.index_path)
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = AssessorConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    else:
        cfg = AssessorConfig()

    hidden_sizes_override = parse_hidden_sizes(args.hidden_sizes)
    if hidden_sizes_override is not None or args.dropout is not None:
        cfg = AssessorConfig(
            embedding_size=cfg.embedding_size,
            hidden_sizes=hidden_sizes_override if hidden_sizes_override is not None else cfg.hidden_sizes,
            dropout=args.dropout if args.dropout is not None else cfg.dropout,
        )

    if args.resume and (out_dir / "assessor.pt").exists():
        model, _ = load_assessor(device=device, in_dir=out_dir)
        print("Resuming from existing checkpoint")
    else:
        model = AssessorMLP(cfg).to(device)
        print("Training from scratch")

    print(
        f"Model config: embedding_size={cfg.embedding_size} "
        f"hidden_sizes={cfg.hidden_sizes} dropout={cfg.dropout}"
    )
    print(f"Model parameters: {model_param_count(model):,}")

    rows = load_ai_feedback_rows(
        db_path=db_path,
        include_identical=args.include_identical,
        min_rows_per_query=args.min_rows_per_query,
    )
    if not rows:
        raise RuntimeError("No usable ai_augment_feedback rows found.")

    train_rows, val_rows = split_by_query(rows, train_ratio=args.train_ratio, seed=args.seed)
    if not train_rows:
        raise RuntimeError("No training rows after query split.")

    train_counts = label_counts(train_rows)
    val_counts = label_counts(val_rows)
    train_query_count = len({row.query_vector_id for row in train_rows})
    val_query_count = len({row.query_vector_id for row in val_rows})
    print("\nSplit: ai_feedback by query")
    print(f"  train queries: {train_query_count}, rows: {len(train_rows)}, label_counts={dict(train_counts)}")
    print(f"  val queries:   {val_query_count}, rows: {len(val_rows)}, label_counts={dict(val_counts)}")

    n0 = train_counts.get(0, 0)
    n1 = train_counts.get(1, 0)
    if args.balance_weights and n0 > 0 and n1 > 0:
        neg_weight = n1 / max(n0, 1)
        pos_weight = n0 / max(n1, 1)
        print(
            f"  Class weights: neg_weight={neg_weight:.3f} "
            f"pos_weight={pos_weight:.3f} (counts 0={n0}, 1={n1})"
        )
    else:
        neg_weight = 1.0
        pos_weight = 1.0

    train_dataset = PairIdDataset(train_rows, index_path=index_path, embedding_size=cfg.embedding_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss(reduction="none")

    model.train()
    best_metrics: dict[str, tuple[float, int] | None] = {
        "accuracy": None,
        "f1": None,
        "recall": None,
    }
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for e1, e2, labels in pbar:
            e1 = e1.to(device)
            e2 = e2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            score = model(e1, e2)
            loss_per_item = criterion(score, labels)
            if args.balance_weights and n0 > 0 and n1 > 0:
                weights = torch.where(labels == 1, pos_weight, neg_weight)
                loss = (weights * loss_per_item).mean()
            else:
                loss = loss_per_item.mean()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} avg_loss={avg_loss:.4f}")

        if val_rows:
            metrics = evaluate(
                model,
                val_rows,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                index_path=index_path,
                threshold=args.threshold,
            )
            cm = metrics["cm"]
            print(
                f"  val: n={metrics['n']} loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"cm=[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]"
            )
            epoch_num = epoch + 1
            for metric_name in ("accuracy", "f1", "recall"):
                metric_value = float(metrics[metric_name])
                best = best_metrics[metric_name]
                if best is None or metric_value > best[0]:
                    best_metrics[metric_name] = (metric_value, epoch_num)
                    save_metric_checkpoint(
                        model=model,
                        cfg=cfg,
                        out_dir=out_dir,
                        metric_name=metric_name,
                        epoch=epoch_num,
                        metric_value=metric_value,
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_assessor(model, cfg, out_dir=out_dir)
    print(f"Saved model to {out_dir}")
    for metric_name in ("accuracy", "f1", "recall"):
        best = best_metrics[metric_name]
        if best is not None:
            best_value, best_epoch = best
            print(
                f"Best validation checkpoint ({metric_name}): epoch={best_epoch} "
                f"{metric_name}={best_value:.4f} -> {out_dir / f'assessor_best_{metric_name}.pt'}"
            )


if __name__ == "__main__":
    main()
