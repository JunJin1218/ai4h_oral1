from pathlib import Path
import torch

from assessor.model import load_assessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2, cfg2 = load_assessor(device=device, in_dir="assessor/model")
model2.eval()

pt_files = sorted(Path("data").rglob("*.pt"))
if len(pt_files) < 2:
    raise FileNotFoundError("Need at least 2 .pt files under data/")

# 맨 위 2개
f1, f2 = pt_files[0], pt_files[1]

e1 = torch.load(f1, map_location="cpu")
e2 = torch.load(f2, map_location="cpu")

if not torch.is_tensor(e1) or not torch.is_tensor(e2):
    raise TypeError("Loaded .pt is not a torch.Tensor")
if e1.shape != (cfg2.embedding_size,) or e2.shape != (cfg2.embedding_size,):
    raise RuntimeError(
        f"Shape mismatch: e1={tuple(e1.shape)}, e2={tuple(e2.shape)}, "
        f"expected ({cfg2.embedding_size},)"
    )

e1 = e1.to(device)
e2 = e2.to(device)

with torch.no_grad():
    score = model2(e1, e2)

print(f"Using: {f1.name} vs {f2.name}")
print(f"score={float(score.item()):.4f} (1=lookalike, 0=not)")
