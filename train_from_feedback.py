"""
Train (fine-tune) the assessor from human/AI feedback collected via POST /feedback.

Reads data/feedback/feedback.jsonl: each line is
  {"query_emb_path": "embeddings/<uuid>.pt", "candidate_name": "...", "label": 0|1, ...}
Resolves candidate_name to embedding path, loads pairs, and runs BCE training.
Model is saved to --out-dir so it persists (no reset on close).

Usage:
  uv run python train_from_feedback.py
  uv run python train_from_feedback.py --epochs 5 --lr 5e-4 --out-dir assessor/model
  uv run python train_from_feedback.py --feedback-dir data/feedback --resume
"""
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from assessor.model import (
    AssessorConfig,
    AssessorMLP,
    load_assessor,
    save_assessor,
)
from assessor.test_lookalike import find_embedding_path


class FeedbackPairDataset(Dataset):
    """Dataset of (e1, e2, label) from feedback log: query emb path + candidate name."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path, int]],
        embedding_size: int = 1280,
    ):
        self.pairs = pairs
        self.embedding_size = embedding_size
        self.cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_emb(self, path: Path) -> torch.Tensor:
        key = str(path)
        if key not in self.cache:
            t = torch.load(path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(t) or t.shape[-1] != self.embedding_size:
                raise ValueError(f"Invalid embedding {path}: shape {getattr(t, 'shape', None)}")
            self.cache[key] = t.float()
        return self.cache[key]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self.pairs[idx]
        e1 = self._load_emb(path1)
        e2 = self._load_emb(path2)
        return e1, e2, torch.tensor(label, dtype=torch.float32)


def load_feedback_pairs(
    feedback_dir: Path,
    data_dir: Path,
) -> list[tuple[Path, Path, int]]:
    """
    Read feedback.jsonl and resolve to (query_emb_path, candidate_emb_path, label).
    query_emb_path is under feedback_dir; candidate is resolved via find_embedding_path.
    """
    log_path = feedback_dir / "feedback.jsonl"
    if not log_path.exists():
        return []

    pairs: list[tuple[Path, Path, int]] = []
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        rel = record.get("query_emb_path") or record.get("query_embedding_path")
        candidate_name = record.get("candidate_name")
        label = record.get("label")
        if rel is None or not candidate_name or label not in (0, 1):
            continue
        query_path = feedback_dir / rel
        if not query_path.is_file():
            continue
        cand_path = find_embedding_path(candidate_name, data_dir)
        if cand_path is None:
            continue
        pairs.append((query_path, cand_path, label))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Train assessor from feedback log")
    parser.add_argument("--feedback-dir", type=str, default="data/feedback", help="Directory containing feedback.jsonl and embeddings/")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root for resolving candidate embeddings")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (small for fine-tune)")
    parser.add_argument("--out-dir", type=str, default="assessor/model", help="Where to save the updated model")
    parser.add_argument("--resume", action="store_true", help="Load existing assessor from --out-dir and fine-tune")
    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = load_feedback_pairs(feedback_dir, data_dir)
    if len(pairs) == 0:
        print("No feedback pairs found. Use POST /feedback to collect (query_image, candidate_name, is_lookalike).")
        return

    print(f"Loaded {len(pairs)} feedback pairs from {feedback_dir / 'feedback.jsonl'}")

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = AssessorConfig(**cfg_dict)
    else:
        cfg = AssessorConfig()

    if args.resume and (out_dir / "assessor.pt").exists():
        model, _ = load_assessor(device=device, in_dir=out_dir)
        print("Resuming from existing checkpoint (fine-tune)")
    else:
        model = AssessorMLP(cfg).to(device)
        print("Training from scratch (no checkpoint found)")

    dataset = FeedbackPairDataset(pairs, embedding_size=cfg.embedding_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for e1, e2, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            e1, e2 = e1.to(device), e2.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            score = model(e1, e2).unsqueeze(1)
            loss = criterion(score, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_assessor(model, cfg, out_dir=out_dir)
    print(f"Saved model to {out_dir}. Restart the API to use the updated assessor.")


if __name__ == "__main__":
    main()
