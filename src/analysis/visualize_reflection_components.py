import argparse
from pathlib import Path

import cv2

from common import ENDOSRR_DIR, RESULTS_DIR, find_pngs, read_binary_mask, read_image


def draw_components(image, mask):
    if image.shape[:2] != mask.shape:
        raise ValueError(f"Image and mask sizes differ: {image.shape[:2]} != {mask.shape}")

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype("uint8"), connectivity=8
    )
    result = image.copy()

    # BGR: small (red), medium (yellow), large (green).
    for region_id in range(1, count):
        x, y, width, height, area = stats[region_id]
        if area <= 10:
            color = (0, 0, 255)
        elif area <= 100:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), color, 1)

    return result, count - 1


def main():
    parser = argparse.ArgumentParser(description="Draw boxes around predicted mask components.")
    parser.add_argument("--image-dir", type=Path, default=ENDOSRR_DIR / "image")
    parser.add_argument("--mask-dir", type=Path, default=ENDOSRR_DIR / "output/final_mask")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "component_visualizations")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in find_pngs(args.mask_dir):
        image = read_image(args.image_dir / mask_path.name)
        mask = read_binary_mask(mask_path)
        result, count = draw_components(image, mask)

        output_path = args.output_dir / mask_path.name
        if not cv2.imwrite(str(output_path), result):
            raise OSError(f"Failed to save: {output_path}")
        print(f"{mask_path.name}: {count} regions")


if __name__ == "__main__":
    main()
