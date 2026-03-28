"""
MLP assessor for binary lookalike classification.

Input: e1 (query embedding), e2 (candidate embedding), each size D.
We L2-normalize then concat(|e1-e2|, e1*e2) -> 2D-dim.
Output: sigmoid score in [0, 1].
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AssessorConfig:
    embedding_size: int = 1280
    hidden_sizes: Tuple[int, ...] = (1024, 256)
    dropout: float = 0.2


def cosine_similarity(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between embedding vectors (L2-normalized then dot product).
    e1, e2: (B, D) or (D,). Returns (B,) or scalar in [-1, 1].
    """
    if e1.dim() == 1:
        e1 = e1.unsqueeze(0)
    if e2.dim() == 1:
        e2 = e2.unsqueeze(0)
    e1 = F.normalize(e1, p=2, dim=-1)
    e2 = F.normalize(e2, p=2, dim=-1)
    return (e1 * e2).sum(dim=-1)


def build_pair_features(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """
    e1/e2: (B, D) or (D,)
    returns: (B, 2D)
    """
    if e1.dim() == 1:
        e1 = e1.unsqueeze(0)
    if e2.dim() == 1:
        e2 = e2.unsqueeze(0)

    e1 = F.normalize(e1, p=2, dim=-1)
    e2 = F.normalize(e2, p=2, dim=-1)
    return torch.cat([torch.abs(e1 - e2), e1 * e2], dim=-1)


class AssessorMLP(nn.Module):
    def __init__(self, cfg: AssessorConfig) -> None:
        super().__init__()
        input_size = cfg.embedding_size * 2

        layers: list[nn.Module] = []
        in_dim = input_size
        for hidden in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        x = build_pair_features(e1, e2)
        logits = self.mlp(x)
        return torch.sigmoid(logits).squeeze(-1)


def save_assessor(
    model: AssessorMLP,
    cfg: AssessorConfig,
    out_dir: str | Path = "data/assessor",
    filename: str = "assessor.pt",
) -> Path:
    """
    Saves:
      - out_dir/filename: torch checkpoint (state_dict)
      - out_dir/config.json: model config
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / filename
    cfg_path = out_dir / "config.json"

    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return ckpt_path


def load_assessor(
    device: str | torch.device = "cpu",
    in_dir: str | Path = "data/assessor",
    filename: str = "assessor.pt",
) -> tuple[AssessorMLP, AssessorConfig]:
    """
    Loads model + cfg from data/assessor.
    """
    in_dir = Path(in_dir)
    cfg_path = in_dir / "config.json"
    ckpt_path = in_dir / filename

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = AssessorConfig(**cfg_dict)

    model = AssessorMLP(cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cfg
