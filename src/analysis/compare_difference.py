import argparse
from pathlib import Path

import cv2
import numpy as np

from common import ENDOSRR_DIR, read_image


EXTENSIONS = {".png", ".jpg", ".jpeg"}


def difference_overlay(original, restored, threshold=30):
    if original.shape != restored.shape:
        raise ValueError(f"Image sizes differ: {original.shape} != {restored.shape}")

    changed = np.max(cv2.absdiff(original, restored), axis=2) > threshold
    overlay = original.copy()
    overlay[changed] = (0, 0, 255)  # BGR red
    return overlay, float(changed.mean() * 100)


def main():
    parser = argparse.ArgumentParser(description="Highlight changed pixels after inpainting.")
    parser.add_argument("--image-dir", type=Path, default=ENDOSRR_DIR / "image")
    parser.add_argument("--inpaint-dir", type=Path, default=ENDOSRR_DIR / "output/final_inpaint")
    parser.add_argument("--output-dir", type=Path, default=ENDOSRR_DIR / "output/difference")
    parser.add_argument("--threshold", type=int, default=30)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if not args.image_dir.is_dir():
        parser.error(f"Image directory not found: {args.image_dir}")

    image_paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not image_paths:
        parser.error(f"No images found in: {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for image_path in image_paths:
        restored_path = args.inpaint_dir / image_path.name
        if not restored_path.is_file():
            print(f"Skipping {image_path.name}: restored image not found")
            continue

        original = read_image(image_path)
        restored = read_image(restored_path)
        overlay, changed_pct = difference_overlay(original, restored, args.threshold)

        output_path = args.output_dir / image_path.name
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"Failed to save: {output_path}")
        print(f"{image_path.name}: changed pixels {changed_pct:.2f}%")
        processed += 1

    if processed == 0:
        raise FileNotFoundError("No matching original/restored image pairs found")


if __name__ == "__main__":
    main()
