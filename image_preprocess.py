"""
1. input_img에 있는 모든 이미지에 대해서
2. pretrained vit 모델 불러와서 embedding으로 변환 후 (관련 functions는 전부 image_embedding/vit.py에 있다고 치고)
3. data폴더에다가 저장 (각 이미지에 대해 .pt 저장)
"""

from pathlib import Path

import torch

from image_embedding.vit import get_image_embedding, load_vit_model


def main() -> None:
    input_dir = Path(
        r"C:\Users\Asus\T5 - SDS\T8-AI4H\Oral Dose Forms\Individual tablets capsules"
    )
    output_dir = Path(
        r"C:\Users\Asus\T5 - SDS\T8-AI4H\Individual Capsules"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_vit_model()

    image_paths = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in {".gif",".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    if not image_paths:
        print(f"No images found in {input_dir.resolve()}")
        return
    
    failed = []

    for img_path in image_paths:
        try:
            embedding = get_image_embedding(
                model, processor, img_path, device=device, pooling="cls"
            )
            out_path = output_dir / f"{img_path.stem}.pt"
            torch.save(embedding, out_path)
        except Exception as e:
            failed.append((img_path, str(e)))

    print(f"Finished. Failed images: {len(failed)}")
    for p, err in failed:
        print("FAILED:", p, err)


    # for img_path in image_paths:
    #     embedding = get_image_embedding(model, processor, img_path, device=device, pooling="cls")
    #     out_path = output_dir / f"{img_path.stem}.pt"
    #     torch.save(embedding, out_path)
    #     print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
