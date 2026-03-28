"""
1. input_img에 있는 모든 이미지에 대해서 (including subfolders)
2. pretrained vit 모델 불러와서 embedding으로 변환 후 (관련 functions는 전부 image_embedding/vit.py에 있다고 치고)
3. data/embeddings 폴더에다가 저장 (각 이미지에 대해 .pt 저장).
   Filenames match image names so annotations (Excel) can find them.
   Tip: Use the same image filenames as in your annotation Excel (e.g. 4049.jpg,
   "1 Acarbose tab front YSP.PNG") so test_lookalike can match pairs.
"""

from pathlib import Path

import torch

from image_embedding.vit import get_image_embedding, load_vit_model


def main() -> None:
<<<<<<< HEAD
    input_dir = Path(
        r"C:\Users\Asus\T5 - SDS\T8-AI4H\Oral Dose Forms\Individual tablets capsules"
    )
    output_dir = Path(
        r"C:\Users\Asus\T5 - SDS\T8-AI4H\Individual Capsules"
    )
=======
    input_dir = Path("input_img")
    output_dir = Path("data") / "embeddings"
>>>>>>> 609bbb243b37b7cb242148049d9cffc0513d8e13
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_vit_model()

    # Support both flat input_img and subfolders (e.g. Oral Dose Forms)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = sorted(
        p for p in input_dir.rglob("*")
<<<<<<< HEAD
        if p.suffix.lower() in {".gif",".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
=======
        if p.suffix.lower() in exts and p.is_file()
>>>>>>> 609bbb243b37b7cb242148049d9cffc0513d8e13
    )

    if not image_paths:
        print(f"No images found in {input_dir.resolve()}")
        return
    
    failed = []

    for img_path in image_paths:
<<<<<<< HEAD
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
=======
        embedding = get_image_embedding(model, processor, img_path, device=device, pooling="cls")
        # Keep filename unique: use stem only if flat, else include relative path
        try:
            rel = img_path.relative_to(input_dir)
            if rel.parent == Path("."):
                out_name = f"{img_path.stem}.pt"
            else:
                out_name = f"{'__'.join(rel.with_suffix('').parts)}.pt"
        except ValueError:
            out_name = f"{img_path.stem}.pt"
        out_path = output_dir / out_name
        torch.save(embedding, out_path)
        print(f"Saved: {out_path}")
>>>>>>> 609bbb243b37b7cb242148049d9cffc0513d8e13

if __name__ == "__main__":
    main()
