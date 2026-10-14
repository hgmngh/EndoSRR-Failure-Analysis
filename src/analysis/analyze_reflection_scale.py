import argparse
from pathlib import Path

import cv2
import pandas as pd

from common import ENDOSRR_DIR, RESULTS_DIR, find_pngs, read_binary_mask


def analyze_masks(mask_dir):
    rows = []

    for path in find_pngs(mask_dir):
        mask = read_binary_mask(path).astype("uint8")
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        image_area = mask.size

        for region_id in range(1, count):
            area = int(stats[region_id, cv2.CC_STAT_AREA])
            rows.append({
                "image": path.name,
                "region_id": region_id,
                "area_px": area,
                "area_ratio_pct": area / image_area * 100,
            })

    return pd.DataFrame(rows, columns=["image", "region_id", "area_px", "area_ratio_pct"])


def main():
    parser = argparse.ArgumentParser(description="Measure connected components in predicted masks.")
    parser.add_argument("--mask-dir", type=Path, default=ENDOSRR_DIR / "output/final_mask")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "reflection_scale.csv")
    args = parser.parse_args()

    result = analyze_masks(args.mask_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Regions: {len(result)} | Saved: {args.output}")


if __name__ == "__main__":
    main()
