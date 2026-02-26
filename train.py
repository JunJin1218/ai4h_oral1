"""
Train (or fine-tune) the assessor on labeled pairs from annotation Excel files.

Uses annotations to build (query_embedding, candidate_embedding, label) pairs,
then trains the assessor MLP with binary cross-entropy.

Usage:
  # Train from scratch (or continue from existing checkpoint)
  uv run python train.py

  # Fine-tune existing model (load assessor/model/ first)
  uv run python train.py --resume

  # Source-based split (train/val/test from different Excel files)
  uv run python train.py --split by_source

  # Disjoint-by-query split (no data leakage: validation queries unseen in training)
  uv run python train.py --split by_query

  # Lookalike annotation only, 80%% train / 20%% test (by query; no leakage)
  uv run python train.py --split lookalike_80_20

  # Options
  uv run python train.py --epochs 20 --batch-size 32 --lr 1e-3 --resume
"""
from pathlib import Path
import argparse
import random
from collections import Counter, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from assessor.model import (
    AssessorConfig,
    AssessorMLP,
    load_assessor,
    save_assessor,
)
from assessor.test_lookalike import load_annotations, load_annotations_streamlined, find_embedding_path


class PairDataset(Dataset):
    """Dataset of (e1, e2, label) from embedding paths."""

    def __init__(self, pairs: list[tuple[Path, Path, int]], embedding_size: int = 1280):
        self.pairs = pairs
        self.embedding_size = embedding_size
        self.cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_emb(self, path: Path) -> torch.Tensor:
        key = str(path)
        if key not in self.cache:
            t = torch.load(path, map_location="cpu")
            if not torch.is_tensor(t) or t.shape[-1] != self.embedding_size:
                raise ValueError(f"Invalid embedding {path}: shape {getattr(t, 'shape', None)}")
            self.cache[key] = t.float()
        return self.cache[key]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path1, path2, label = self.pairs[idx]
        e1 = self._load_emb(path1)
        e2 = self._load_emb(path2)
        return e1, e2, torch.tensor(label, dtype=torch.float32)


def _resolve_pairs(
    raw_pairs: list[tuple[str, str, int]],
    *,
    data_dir: Path,
) -> list[tuple[Path, Path, int]]:
    resolved: list[tuple[Path, Path, int]] = []
    for q_name, c_name, label in tqdm(
        raw_pairs,
        desc="Resolving embeddings",
        mininterval=1.0,
        miniters=50,
    ):
        p1 = find_embedding_path(q_name, data_dir)
        p2 = find_embedding_path(c_name, data_dir)
        if p1 is not None and p2 is not None:
            resolved.append((p1, p2, label))
    return resolved


def collect_all_raw_pairs(data_dir: Path) -> list[tuple[str, str, int]]:
    """
    Load all (query_name, candidate_name, label) from annotation Excel files.
    Used for disjoint-by-query split so we can group by query before resolving.
    """
    annotation_dir = data_dir / "Annotation"
    sources: list[Path] = [
        annotation_dir / "lookalike_annotation.xlsx",
    ]
    raw: list[tuple[str, str, int]] = []
    for excel_path in sources:
        if not excel_path.exists():
            continue
        pairs = load_annotations(excel_path)
        print(f"Loaded {len(pairs)} raw pairs from {excel_path.name}")
        raw.extend(pairs)
    return raw


def group_pairs_by_query(
    raw_pairs: list[tuple[str, str, int]],
) -> dict[str, list[tuple[str, str, int]]]:
    """Group (query, candidate, label) by query name. Each query appears in only one group."""
    by_query: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for q, c, label in raw_pairs:
        by_query[q].append((q, c, label))
    return dict(by_query)


def split_by_query_disjoint(
    raw_pairs: list[tuple[str, str, int]],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """
    Split so that validation uses only query drugs never seen in training (no leakage).
    - Group all pairs by query image.
    - Assign 80% of query drugs (and all their pairs) to train, 20% to val.
    - Returns (train_raw_pairs, val_raw_pairs).
    """
    by_query = group_pairs_by_query(raw_pairs)
    queries = sorted(by_query.keys())
    n = len(queries)
    if n == 0:
        return [], []

    rng = random.Random(seed)
    rng.shuffle(queries)
    n_train = max(1, int(n * train_ratio))
    train_queries = set(queries[:n_train])
    val_queries = set(queries[n_train:])

    train_raw: list[tuple[str, str, int]] = []
    val_raw: list[tuple[str, str, int]] = []
    for q, triples in by_query.items():
        if q in train_queries:
            train_raw.extend(triples)
        else:
            val_raw.extend(triples)

    return train_raw, val_raw


def collect_lookalike_only_raw(data_dir: Path) -> list[tuple[str, str, int]]:
    """
    Load (query_name, candidate_name, label) from lookalike annotation Excel only.
    Tries 'lookalike annotation.xlsx' then 'lookalike_annotation.xlsx'.
    Used for train/test 80/20 split on lookalike data only.
    """
    annotation_dir = data_dir / "Annotation"
    for name in ("lookalike annotation.xlsx", "lookalike_annotation.xlsx"):
        path = annotation_dir / name
        if path.exists():
            raw = load_annotations(path)
            if raw:
                print(f"Loaded {len(raw)} raw pairs from {path.name}")
                return raw
            # file exists but no pairs parsed; fall through to try next name
    return []


def collect_pairs_by_source(data_dir: Path) -> dict[str, list[tuple[Path, Path, int]]]:
    """
    Load annotations from known Excel files and return resolved pairs per source:
      - query_set: query_set_annotation_CGH1.xlsx (label=1)
      - uncertain_negatives: uncertain_negatives_annotation_CGH1.xlsx (label=0)
      - lookalike: lookalike annotation.xlsx (label can be 0 or 1)
    """
    annotation_dir = data_dir / "Annotation"
    lookalike_path = annotation_dir / "lookalike annotation.xlsx"
    if not lookalike_path.exists():
        lookalike_path = annotation_dir / "lookalike_annotation.xlsx"
    sources: dict[str, Path] = {
        "query_set": annotation_dir / "query_set_annotation_CGH1.xlsx",
        "uncertain_negatives": annotation_dir / "uncertain_negatives_annotation_CGH1.xlsx",
        "lookalike": lookalike_path,
    }

    out: dict[str, list[tuple[Path, Path, int]]] = {}
    for key, excel_path in sources.items():
        if not excel_path.exists():
            out[key] = []
            continue
        raw = load_annotations(excel_path)
        print(f"Loaded {len(raw)} raw pairs from {excel_path.name}")
        out[key] = _resolve_pairs(raw, data_dir=data_dir)
        print(f"Resolved {len(out[key])} pairs for source={key}")
    return out


def _label_counts(pairs: list[tuple[Path, Path, int]]) -> Counter:
    return Counter(int(lbl) for _, _, lbl in pairs)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    pairs: list[tuple[Path, Path, int]],
    *,
    cfg: AssessorConfig,
    device: torch.device,
    batch_size: int,
    threshold: float = 0.5,
) -> dict[str, object]:
    """
    Evaluate model on a resolved-pair list.
    Returns dict with metrics + confusion matrix.
    """
    if len(pairs) == 0:
        return {
            "n": 0,
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "cm": None,
        }

    dataset = PairDataset(pairs, embedding_size=cfg.embedding_size)
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
        labels = labels.to(device).unsqueeze(1)

        scores = model(e1, e2).unsqueeze(1)
        loss = criterion(scores, labels)
        total_loss += float(loss.item())
        n_batches += 1

        probs = scores.squeeze(1).detach().cpu().tolist()
        lbls = labels.squeeze(1).detach().cpu().tolist()
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


def main():
    parser = argparse.ArgumentParser(description="Train assessor on annotation pairs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Load assessor/model and fine-tune")
    parser.add_argument("--out-dir", type=str, default="assessor/model", help="Where to save checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "by_source", "by_query", "lookalike_80_20"],
        help=(
            "Dataset split strategy. "
            "'all': train on all resolved pairs. "
            "'by_source': train/val/test by Excel file. "
            "'by_query': disjoint by query drug (80%% train, 20%% val; no leakage). "
            "'lookalike_80_20': lookalike annotation.xlsx only, 80%% train / 20%% test by query (no leakage)."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="For --split by_query: fraction of query drugs used for training (default 0.8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for by_query split.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument(
        "--balance-weights",
        action="store_true",
        help="Use class-balanced BCE (upweight minority class). Use when train has many more 1s than 0s.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Use this file for training (e.g. data/Annotation/test-data.xlsx). Same 80/20 by-query split. Streamlined format supported.",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config (and optionally existing model)
    out_dir = Path(args.out_dir)
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        import json
        cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = AssessorConfig(**cfg_dict)
    else:
        cfg = AssessorConfig()

    if args.resume and (out_dir / "assessor.pt").exists():
        model, _ = load_assessor(device=device, in_dir=out_dir)
        print("Resuming from existing checkpoint (fine-tune)")
    else:
        model = AssessorMLP(cfg).to(device)
        print("Training from scratch")

    # Collect pairs that have both embeddings
    if args.split == "lookalike_80_20":
        if args.annotation_file:
            ann_path = Path(args.annotation_file)
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_path}")
            raw_all = load_annotations_streamlined(ann_path)
            if raw_all:
                print(f"Loaded {len(raw_all)} raw pairs from {ann_path.name} (streamlined format)")
            else:
                # Diagnostic: show first rows so user can fix format
                try:
                    import pandas as pd
                    df = pd.read_excel(ann_path, header=None)
                    print(f"  File has {len(df)} rows, {len(df.columns)} columns. First 3 rows:")
                    for i in range(min(3, len(df))):
                        print(f"    row {i}: {list(df.iloc[i].values)}")
                    print("  Expected: col0=query, col1=candidate, col2=label (0/1 or Lookalike/Non lookalike); or col0=query, col2=label, col3+=candidates.")
                except Exception as e:
                    print(f"  Could not read file for diagnostic: {e}")
        else:
            raw_all = collect_lookalike_only_raw(data_dir)
        if len(raw_all) == 0:
            raise RuntimeError(
                "No pairs from annotation. Check --annotation-file path and file format (or use default lookalike annotation.xlsx)."
            )
        train_raw, test_raw = split_by_query_disjoint(
            raw_all,
            train_ratio=args.train_ratio,
            val_ratio=1.0 - args.train_ratio,
            seed=args.seed,
        )
        train_pairs = _resolve_pairs(train_raw, data_dir=data_dir)
        test_pairs = _resolve_pairs(test_raw, data_dir=data_dir)
        val_pairs = []  # no separate val; we use test for final eval
        by_query = group_pairs_by_query(raw_all)
        n_queries = len(by_query)
        n_train_q = len(set(q for q, _, _ in train_raw))
        n_test_q = len(set(q for q, _, _ in test_raw))
        print("\nSplit: lookalike_80_20 (80% train / 20% test by query)")
        if args.annotation_file:
            print(f"  source: {args.annotation_file}")
        print(f"  unique query drugs: {n_queries} total -> {n_train_q} train, {n_test_q} test")
        print(f"  train: {len(train_pairs)} pairs (resolved), label_counts={dict(_label_counts(train_pairs))}")
        print(f"  test:  {len(test_pairs)} pairs (resolved), label_counts={dict(_label_counts(test_pairs))}")
        if len(train_pairs) == 0:
            # Diagnose: show why resolution failed for a few examples
            print("\n  Diagnostic (first 5 train raw pairs):")
            for i, (q_name, c_name, lbl) in enumerate(train_raw[:5]):
                p1 = find_embedding_path(q_name, data_dir)
                p2 = find_embedding_path(c_name, data_dir)
                print(f"    [{i+1}] query={q_name!r} -> {p1}")
                print(f"        candidate={c_name!r} -> {p2}")
            print("  Tip: Embeddings can be under data/embeddings/, data/query/, data/query images/, or data/. Names match by stem (case-insensitive, _0/_1 suffix allowed).")
            raise RuntimeError("No train pairs resolved. Check embedding filenames vs annotation names.")
        if len(test_pairs) == 0:
            print("  WARNING: No test pairs resolved. You will only train (no test eval).")
    elif args.split == "by_query":
        raw_all = collect_all_raw_pairs(data_dir)
        if len(raw_all) == 0:
            raise RuntimeError("No raw pairs from annotation files. Check data/Annotation/.")
        train_raw, val_raw = split_by_query_disjoint(
            raw_all,
            train_ratio=args.train_ratio,
            val_ratio=1.0 - args.train_ratio,
            seed=args.seed,
        )
        train_pairs = _resolve_pairs(train_raw, data_dir=data_dir)
        val_pairs = _resolve_pairs(val_raw, data_dir=data_dir)
        test_pairs = []
        by_query = group_pairs_by_query(raw_all)
        n_queries = len(by_query)
        n_train_q = len(set(q for q, _, _ in train_raw))
        n_val_q = len(set(q for q, _, _ in val_raw))
        print("\nSplit: by_query (disjoint query drugs; no leakage)")
        print(f"  unique query drugs: {n_queries} total -> {n_train_q} train, {n_val_q} val")
        print(f"  train: {len(train_pairs)} pairs (resolved), label_counts={dict(_label_counts(train_pairs))}")
        print(f"  val:   {len(val_pairs)} pairs (resolved), label_counts={dict(_label_counts(val_pairs))}")
        if len(train_pairs) == 0:
            raise RuntimeError("No train pairs resolved. Check embedding filenames vs annotation names.")
    else:
        by_source = collect_pairs_by_source(data_dir)
        all_pairs = by_source.get("query_set", []) + by_source.get("uncertain_negatives", []) + by_source.get("lookalike", [])
        if len(all_pairs) == 0:
            raise RuntimeError(
                "No (query, candidate, label) pairs found with matching embeddings. "
                "Check data/embeddings and annotation Excel names."
            )

        if args.split == "by_source":
            train_pairs = by_source.get("uncertain_negatives", [])
            val_pairs = by_source.get("query_set", [])
            test_pairs = by_source.get("lookalike", [])
            print("\nSplit: by_source")
            print(f"  train=uncertain_negatives: {len(train_pairs)} pairs, label_counts={dict(_label_counts(train_pairs))}")
            print(f"  val=query_set:           {len(val_pairs)} pairs, label_counts={dict(_label_counts(val_pairs))}")
            print(f"  test=lookalike:          {len(test_pairs)} pairs, label_counts={dict(_label_counts(test_pairs))}")

            # If train has only one class, training won't be meaningful; fall back to 'all' training
            train_counts = _label_counts(train_pairs)
            if len(train_counts) < 2:
                print(
                    "\n⚠️  WARNING: train split contains only one class. "
                    "Falling back to training on ALL resolved pairs to avoid a degenerate model.\n"
                    "If you want strict holdout-by-source, you must ensure train contains both label 0 and label 1.\n"
                )
                train_pairs = all_pairs
        else:
            train_pairs = all_pairs
            val_pairs = []
            test_pairs = []
            print("\nSplit: all")
            print(f"  train(all): {len(train_pairs)} pairs, label_counts={dict(_label_counts(train_pairs))}")

    print(f"\nTraining on {len(train_pairs)} pairs")

    label_counts = _label_counts(train_pairs)
    n0, n1 = label_counts.get(0, 0), label_counts.get(1, 0)
    if args.balance_weights and n0 > 0 and n1 > 0:
        # Upweight minority class so the model doesn't collapse to majority
        neg_weight = n1 / max(n0, 1)
        pos_weight = n0 / max(n1, 1)
        print(f"  Class weights (balance): neg_weight={neg_weight:.3f} pos_weight={pos_weight:.3f} (counts 0={n0}, 1={n1})")
    else:
        neg_weight = pos_weight = 1.0

    dataset = PairDataset(train_pairs, embedding_size=cfg.embedding_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss(reduction="none")

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for e1, e2, labels in pbar:
            e1 = e1.to(device)
            e2 = e2.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            score = model(e1, e2)
            loss_per_item = criterion(score.unsqueeze(1), labels)
            if args.balance_weights and n0 > 0 and n1 > 0:
                weights = torch.where(labels == 1, pos_weight, neg_weight)
                loss = (weights * loss_per_item).mean()
            else:
                loss = loss_per_item.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        if args.split in ("by_source", "by_query", "lookalike_80_20") and len(val_pairs) > 0:
            metrics = evaluate(
                model,
                val_pairs,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                threshold=args.threshold,
            )
            cm = metrics["cm"]
            print(
                f"  val: n={metrics['n']} loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                f"cm=[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_assessor(model, cfg, out_dir=out_dir)
    print(f"Saved model to {out_dir}")

    if args.split == "by_source" and len(test_pairs) > 0:
        metrics = evaluate(
            model,
            test_pairs,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        cm = metrics["cm"]
        print(
            f"\nTEST (lookalike annotation): n={metrics['n']} loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        print("Confusion Matrix (labels 0=Not, 1=Lookalike):")
        print(f"  [[{cm[0][0]},{cm[0][1]}],")
        print(f"   [{cm[1][0]},{cm[1][1]}]]")
    elif args.split == "lookalike_80_20" and len(test_pairs) > 0:
        metrics = evaluate(
            model,
            test_pairs,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        cm = metrics["cm"]
        print(
            f"\nTEST (lookalike 20%% holdout): n={metrics['n']} loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        print("Confusion Matrix (labels 0=Not, 1=Lookalike):")
        print(f"  [[{cm[0][0]},{cm[0][1]}],")
        print(f"   [{cm[1][0]},{cm[1][1]}]]")
    elif args.split == "by_query" and len(val_pairs) > 0:
        metrics = evaluate(
            model,
            val_pairs,
            cfg=cfg,
            device=device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        cm = metrics["cm"]
        print(
            f"\nVAL (unseen query drugs): n={metrics['n']} loss={metrics['loss']:.4f} "
            f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        print("Confusion Matrix (labels 0=Not, 1=Lookalike):")
        print(f"  [[{cm[0][0]},{cm[0][1]}],")
        print(f"   [{cm[1][0]},{cm[1][1]}]]")


if __name__ == "__main__":
    main()
