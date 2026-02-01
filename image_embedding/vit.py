"""
Pretrained ViT model -> embedding vectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from transformers import AutoImageProcessor, ViTModel

Pooling = Literal["cls", "mean"]


def load_vit_model(
    model_name: str = "google/vit-huge-patch14-224-in21k",
    device: str | None = None,
) -> tuple[AutoImageProcessor, ViTModel, torch.device]:
    """
    Load a pretrained ViT model + its processor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    print(f"DEVICE: {torch_device}")
    model.to(torch_device)
    model.eval()

    return processor, model, torch_device


def preprocess_image(
    image: str | Path | Image.Image,
    processor: AutoImageProcessor,
) -> dict[str, torch.Tensor]:
    """
    Preprocess an image into ViT-ready tensors.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    return processor(images=image, return_tensors="pt")


@torch.no_grad()
def get_image_embedding(
    model: ViTModel,
    processor: AutoImageProcessor,
    image: str | Path | Image.Image,
    device: torch.device | None = None,
    pooling: Pooling = "cls",
) -> torch.Tensor:
    """
    Compute an embedding vector from a single image.
    """
    inputs = preprocess_image(image, processor)

    if device is None:
        device = next(model.parameters()).device

    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (B, seq, hidden)

    if pooling == "cls":
        embedding = hidden[:, 0, :]
    elif pooling == "mean":
        embedding = hidden.mean(dim=1)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    return embedding.squeeze(0).cpu()
